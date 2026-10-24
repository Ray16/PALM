"""The low-rank leakage-minimizing splitter (registered as ``lowrank``).

Graph-free: factorize ``S ~= B B^T`` (Nyström), then minimize cross-split leakage
in the r-dim factor space with balanced-Lloyd restarts + a monotone FM polish.
Memory is linear, O(n·r); time is empirically near-linear in n at bounded rank,
iterations and FM moves (see the README "Complexity" section for the exact terms).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from lowrank.splitters.base import BaseSplitter, SplitResult, SplitSpec, register
from lowrank.splitters.common.feature_preparation import feature_matrix_from_dict
from lowrank.splitters.common.leakage_metrics import scaled_lpi
from lowrank.splitters.common.pairwise_similarity import check_metric
from lowrank.splitters.common.split_naming import assign_split_names

from lowrank.splitters.common.balanced_assignment import target_sizes

from .grouped import grouped_split, must_link_groups, resolve_tau
from .nystrom import LANDMARKS, nystrom_features
from .objective import factor_leakage
from .optimize import balanced_lloyd, fm_polish, interpolate_to_random
from .diagnostics import sampled_leakage, positive_integer

logger = logging.getLogger(__name__)

_LEAKAGE_MAX_N = 100_000


def _check_int(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"{name} must be an integer >= 1; got {v!r}")


def resolve_size_tolerance(p, spec: SplitSpec):
    """Validate the shared Params and return ``(lloyd_slack, fm_epsilon)``.

    ``allowed_size_imbalance``: ``None`` -> use ``balance_slack`` (default zero);
    ``0`` -> exact sizes end to end (FM uses size-preserving swaps);
    ``x > 0`` -> ``(1 ± x)`` corridor for both. ``spec.epsilon`` does not
    implicitly relax the low-rank splitter's default size guarantee.
    """
    if p.rank != "auto":
        _check_int("rank", p.rank)
    _check_int("n_restarts", p.n_restarts)
    _check_int("n_iter", p.n_iter)
    if not isinstance(p.score_exact_leakage, (bool, np.bool_)):
        raise ValueError("score_exact_leakage must be a boolean")
    if p.landmark not in LANDMARKS:
        raise ValueError(f"unknown landmark method {p.landmark!r}; expected one of {LANDMARKS}")
    if p.hardness is not None:
        h = p.hardness
        if isinstance(h, bool) or not np.isfinite(float(h)) or not 0.0 <= float(h) <= 1.0:
            raise ValueError(f"hardness must be None or a finite value in [0, 1]; got {h!r}")
    grouped = getattr(p, "must_link", None) is not None or getattr(p, "groups", None) is not None
    if grouped:
        if p.hardness is not None and float(p.hardness) < 1.0:
            raise ValueError("must_link/groups cannot be combined with hardness < 1: "
                             "point randomization would break retained groups")
    if getattr(p, "must_link", None) is not None:
        if not 0.0 < p.max_group_frac <= 1.0:
            raise ValueError("max_group_frac must be in (0, 1]")
        _check_int("must_link_k", p.must_link_k)
        if p.must_link_backend not in ("exact", "tiled", "nndescent", "auto"):
            raise ValueError("must_link_backend must be 'exact', 'tiled', 'nndescent' or 'auto'")
        _check_int("must_link_row_block", p.must_link_row_block)
        _check_int("must_link_ref_block", p.must_link_ref_block)
        _check_int("must_link_candidates", p.must_link_candidates)
        _check_int("must_link_n_jobs", p.must_link_n_jobs)
        if p.must_link != "auto" and not np.isfinite(float(p.must_link)):
            raise ValueError("must_link must be 'auto' or a finite threshold")
    if (p.allowed_size_imbalance is not None and p.balance_slack != 0.0
            and p.allowed_size_imbalance != p.balance_slack):
        raise ValueError(
            "allowed_size_imbalance and balance_slack must match when both are set")
    size_tolerance = (p.allowed_size_imbalance
                      if p.allowed_size_imbalance is not None else p.balance_slack)
    if not 0.0 <= size_tolerance < 1.0:
        raise ValueError("allowed_size_imbalance must be in [0, 1)")
    return size_tolerance, size_tolerance


def validate_nonempty_targets(n, splits):
    if np.any(target_sizes(n, splits) == 0):
        raise ValueError("requested split ratios produce an empty fold at this dataset size")


@register("lowrank")
class LowRankSplitter(BaseSplitter):
    description = "Nyström low-rank factorization + balanced-Lloyd + FM (graph-free, O(n·r) memory)"
    arity = "1d"

    @dataclass
    class Params:
        rank: object = 256                  # positive integer or "auto"
        rank_min: int = 32
        rank_max: int = 512
        rank_tolerance: float = 0.05
        rank_probe_pairs: int = 2048
        factor_block_size: int = 2048
        factor_memory_mb: Optional[float] = None
        eigen_rtol: float = 1e-8
        metric: Optional[str] = None
        landmark: str = "kmeans++"          # kmeans++ | uniform
        n_restarts: int = 4
        n_iter: int = 25
        fm: bool = True
        # Flat single-move FM is O(n·r) per move and converges in ~n/100 moves from a
        # Lloyd init (~12s at n=1M); the old 200k cap shipped *un-polished* Lloyd splits
        # at scale, leaving ~4% leakage on the table (archive/multilevel/multilevel_fm.py).
        # A multilevel V-cycle recovers the same leakage ~48× slower — not worth it (a
        # documented negative result, see archive/multilevel/). So we keep flat FM and
        # just raise the cap. >2M still skips (keeps a hard time bound).
        fm_max_n: int = 2_000_000
        # Maximum fractional deviation from each target split size. This is the
        # clearer public name; balance_slack remains as a compatibility alias.
        allowed_size_imbalance: Optional[float] = None
        balance_slack: float = 0.0
        # Empirical hardness dial: None/1 = best-found optimized split, 0 = random.
        # Label swaps preserve counts. Predictive difficulty is dataset- and
        # model-specific, not a guaranteed ordering of individual partitions.
        hardness: Optional[float] = None
        # Tail-aware extensions (lowrank/grouped.py, docs/FINDINGS.md Step 9):
        # kernel_power p optimizes Σ cross sim**p — p>1 sharpens the kernel toward
        # strong pairs so whole families/series move together; must_link = tau (float,
        # or "auto" = 0.6 tanimoto / 0.995 pair-sim quantile otherwise) forms capped
        # groups. Missing edges, refused merges and repair prevent a universal pair
        # threshold guarantee. A tested molecular setting is power=3, tau=0.4
        # (tanimoto) — see comparisons/FINDINGS.md "Tail-aware low-rank".
        kernel_power: int = 1
        must_link: Optional[object] = None
        # Optional externally defined groups: complete entity-id -> group mapping.
        # Strict retention applies to supplied/constructed groups, not all pairs.
        groups: Optional[dict] = None
        group_policy: str = "count_first"    # count_first | strict
        group_feasibility_seconds: float = 5.0
        max_group_frac: float = 0.5
        must_link_k: int = 32
        # what a group does when the cap stops a merge: "freeze" = stop growing at that
        # similarity (per-group effective tau; avoids cap-sized chunks of one chained
        # component), "skip" = skip that merge only (original rule). See must_link_groups.
        must_link_cap: str = "freeze"
        # Optional packed-bit approximate search; edges are verified exactly.
        # Approximate recall must be measured for the intended dataset.
        must_link_backend: str = "exact"
        must_link_candidates: int = 64
        must_link_n_jobs: int = 1
        must_link_row_block: int = 256
        must_link_ref_block: int = 8192
        # Exact diagnostic is quadratic and can dominate splitting time.
        score_exact_leakage: bool = True
        # None preserves the legacy flag; new workflows should use "auto".
        leakage_diagnostic: Optional[str] = None   # none | exact | sampled | auto
        diagnostic_pairs: int = 8192
        # Experimental normalized global + positive local pairwise penalty.
        # Off by default; compare against the existing powered/grouped method.
        local_weight: float = 0.0
        local_k: int = 32
        local_threshold: float = 0.4
        local_backend: str = "tiled"     # tiled | nndescent | auto
        neighbour_audit_queries: int = 32
        neighbour_minimum_recall: float = 0.98

    @classmethod
    def param_schema(cls):
        schema = super().param_schema()
        schema["properties"]["rank"] = {"oneOf": [{"type": "integer", "minimum": 1},
                                                   {"const": "auto"}], "default": 256}
        schema["properties"]["groups"] = {"type": ["object", "null"], "default": None}
        return schema

    def _diagnose(self, X, labels, metric, seed):
        p = self.params
        mode = p.leakage_diagnostic
        if mode is None:
            mode = "exact" if p.score_exact_leakage and len(X) <= _LEAKAGE_MAX_N else "none"
        if mode == "auto":
            mode = "exact" if len(X) <= 2000 else "sampled"
        result = {"leakage": None, "leakage_diagnostic": mode}
        if mode == "exact":
            result["leakage"] = round(scaled_lpi(X, labels, metric=metric), 6)
        elif mode == "sampled":
            result["leakage_sampled"] = sampled_leakage(X, labels, metric,
                sample_pairs_count=p.diagnostic_pairs, seed=104729 + seed)
        return result

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        size_tolerance, fm_eps = resolve_size_tolerance(p, spec)
        if p.leakage_diagnostic not in (None, "none", "exact", "sampled", "auto"):
            raise ValueError("leakage_diagnostic must be none, exact, sampled or auto")
        positive_integer("diagnostic_pairs", p.diagnostic_pairs)
        if p.group_policy not in ("count_first", "strict"):
            raise ValueError("group_policy must be count_first or strict")
        if not np.isfinite(p.group_feasibility_seconds) or p.group_feasibility_seconds <= 0:
            raise ValueError("group_feasibility_seconds must be finite and positive")
        if p.groups is not None and p.must_link is not None:
            raise ValueError("supply groups or must_link, not both")
        if not np.isfinite(p.local_weight) or p.local_weight < 0:
            raise ValueError("local_weight must be finite and non-negative")
        positive_integer("local_k", p.local_k)
        if p.local_weight and (p.must_link is not None or p.groups is not None):
            raise ValueError("experimental local_weight cannot yet be combined with groups/must_link")
        if p.metric is not None:
            check_metric(p.metric)

        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        validate_nonempty_targets(n, spec.splits)
        B, metric, factor_info = nystrom_features(X, rank=p.rank, metric=p.metric,
                                     landmark=p.landmark, seed=spec.seed,
                                     kernel_power=p.kernel_power, return_info=True,
                                     rank_min=p.rank_min, rank_max=p.rank_max,
                                     rank_tolerance=p.rank_tolerance, rank_probe_pairs=p.rank_probe_pairs,
                                     block_size=p.factor_block_size, factor_memory_mb=p.factor_memory_mb,
                                     eigen_rtol=p.eigen_rtol)
        logger.info("  Low-rank: n=%d rank=%d metric=%s", n, B.shape[1], metric)
        optimization_B, local_term, local_info = B, None, None
        if p.local_weight:
            from .local import local_graph, normalize_components, SparseTerm
            from .neighbours import audited_neighbours
            neighbours, backend_info = audited_neighbours(X, metric, p.local_k, backend=p.local_backend,
                candidates=max(p.must_link_candidates, p.local_k + 1), seed=spec.seed,
                n_jobs=p.must_link_n_jobs, audit_queries=p.neighbour_audit_queries,
                minimum_recall=p.neighbour_minimum_recall, strong_threshold=p.local_threshold,
                row_block=p.must_link_row_block, reference_block=p.must_link_ref_block)
            graph, graph_info = local_graph(X, metric, k=p.local_k, threshold=p.local_threshold,
                                           neighbours=neighbours)
            optimization_B, local_term, normalization = normalize_components(B, graph, p.local_weight)
            local_state = SparseTerm(local_term)
            local_info = dict(graph=graph_info, normalization=normalization, backend=backend_info, experimental=True)

        def objective(labels):
            value = factor_leakage(optimization_B, labels, len(spec.splits))
            return value + local_state.cross(labels) if local_term is not None else value

        best_labels, best_obj = None, np.inf
        for r in range(p.n_restarts):
            labels = balanced_lloyd(optimization_B, spec.splits, n_iter=p.n_iter, seed=spec.seed + r,
                                    balance_slack=size_tolerance, local=local_term)
            obj = objective(labels)
            if obj < best_obj:
                best_obj, best_labels = obj, labels
        logger.info("  Best-of-%d Lloyd leakage=%.1f", p.n_restarts, best_obj)

        moves = 0
        if p.fm and n <= p.fm_max_n:
            best_labels, moves = fm_polish(optimization_B, best_labels, spec.splits,
                                          epsilon=fm_eps, local=local_term,
                                          tol=1e-9 if local_term is not None else 1e-6)
            best_obj = objective(best_labels)
            logger.info("  FM polish: %d moves, leakage=%.1f", moves, best_obj)

        if p.must_link is not None or p.groups is not None:
            return self._split_grouped(ids, X, B, metric, spec, fm_eps, t0,
                                       init=best_labels, factor_info=factor_info)

        if p.hardness is not None:                 # controllable-hardness dial
            best_labels = interpolate_to_random(best_labels, p.hardness, seed=spec.seed)
            best_obj = objective(best_labels)

        assignment = assign_split_names(ids, best_labels, spec.splits, spec.names)
        diagnostics = self._diagnose(X, best_labels, metric, spec.seed)
        if local_info is not None:
            local_info.update(composite_objective=float(best_obj),
                              global_factor_ratio=float(factor_leakage(optimization_B, best_labels, len(spec.splits))),
                              weighted_local_graph_ratio=float(local_state.cross(best_labels)))
            diagnostics["local_objective"] = local_info
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            rank=int(B.shape[1]), factor_leakage=round(float(
                                factor_leakage(B, best_labels, len(spec.splits)) if local_term is not None else best_obj), 3),
                            fm_moves=int(moves), factorization=factor_info, **diagnostics)

    def _split_grouped(self, ids, X, B, metric, spec, eps, t0, init=None, factor_info=None):
        p = self.params
        tau = resolve_tau(p.must_link, metric, X, seed=spec.seed) if p.must_link is not None else None
        smallest = int(target_sizes(len(ids), spec.splits).min())
        max_group = max(1, int(p.max_group_frac * smallest))
        nbrs = None
        backend_info = None
        if p.must_link is not None and p.must_link_backend != "exact":
            from .neighbours import audited_neighbours
            nbrs, backend_info = audited_neighbours(X, metric, p.must_link_k, backend=p.must_link_backend,
                candidates=p.must_link_candidates, seed=spec.seed, n_jobs=p.must_link_n_jobs,
                audit_queries=p.neighbour_audit_queries, minimum_recall=p.neighbour_minimum_recall,
                strong_threshold=tau, row_block=p.must_link_row_block, reference_block=p.must_link_ref_block)
        if p.groups is not None:
            if not isinstance(p.groups, dict) or set(p.groups) != set(ids):
                raise ValueError("groups must map every entity id exactly once")
            codes = {}
            try:
                gid = np.array([codes.setdefault(p.groups[i], len(codes)) for i in ids])
            except TypeError as exc:
                raise ValueError("group labels must be hashable") from exc
            group_source = "supplied"
        else:
            gid = must_link_groups(X, metric, tau, max_group, k=p.must_link_k,
                                   cap_rule=p.must_link_cap, nbrs=nbrs)
            group_source = "capped_topk"
        labels, info = grouped_split(B, gid, spec.splits, eps, n_restarts=p.n_restarts,
                                     n_iter=p.n_iter, seed=spec.seed, fm=p.fm,
                                     init_labels=init, group_policy=p.group_policy,
                                     feasibility_seconds=p.group_feasibility_seconds)
        logger.info("  Grouped low-rank: tau=%s groups=%d (%d pts grouped, %d broken) "
                    "leakage=%.1f", tau, info["n_groups"], info["grouped_points"],
                    info["broken_groups"], info["factor_leakage"])
        if p.hardness is not None:
            labels = interpolate_to_random(labels, p.hardness, seed=spec.seed)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        n = len(ids)
        diagnostics = self._diagnose(X, labels, metric, spec.seed)
        if backend_info is not None:
            diagnostics["neighbour_backend"] = backend_info
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            rank=int(B.shape[1]), must_link_tau=round(tau, 4) if tau is not None else None,
                            group_source=group_source, group_contract_scope="supplied_or_constructed_groups",
                            max_group=max_group if tau is not None else None,
                            factorization=factor_info, **diagnostics,
                            must_link_backend=p.must_link_backend,
                            **{k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in info.items()})
