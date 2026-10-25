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
from lowrank.splitters.common.split_naming import assign_split_names

from lowrank.splitters.common.balanced_assignment import target_sizes

from .grouped import grouped_split, must_link_groups, resolve_tau
from .nystrom import LANDMARKS, nystrom_features
from .objective import factor_leakage
from .optimize import optimize_restarts, interpolate_to_random
from .diagnostics import sampled_leakage, positive_integer
from .factors import validate_fold_weights

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
        raise ValueError(
            f"unknown landmark method {p.landmark!r}; expected one of {LANDMARKS}"
        )
    if p.hardness is not None:
        h = p.hardness
        if (
            isinstance(h, bool)
            or not np.isfinite(float(h))
            or not 0.0 <= float(h) <= 1.0
        ):
            raise ValueError(
                f"hardness must be None or a finite value in [0, 1]; got {h!r}"
            )
    grouped = (
        getattr(p, "must_link", None) is not None
        or getattr(p, "groups", None) is not None
    )
    if grouped:
        if p.hardness is not None and float(p.hardness) < 1.0:
            raise ValueError(
                "must_link/groups cannot be combined with hardness < 1: "
                "point randomization would break retained groups"
            )
    if getattr(p, "must_link", None) is not None:
        if not 0.0 < p.max_group_frac <= 1.0:
            raise ValueError("max_group_frac must be in (0, 1]")
        _check_int("must_link_k", p.must_link_k)
        if p.must_link_backend not in ("exact", "tiled", "nndescent", "auto"):
            raise ValueError(
                "must_link_backend must be 'exact', 'tiled', 'nndescent' or 'auto'"
            )
        _check_int("must_link_row_block", p.must_link_row_block)
        _check_int("must_link_ref_block", p.must_link_ref_block)
        _check_int("must_link_candidates", p.must_link_candidates)
        _check_int("must_link_n_jobs", p.must_link_n_jobs)
        if p.must_link != "auto" and not np.isfinite(float(p.must_link)):
            raise ValueError("must_link must be 'auto' or a finite threshold")
    if (
        p.allowed_size_imbalance is not None
        and p.balance_slack != 0.0
        and p.allowed_size_imbalance != p.balance_slack
    ):
        raise ValueError(
            "allowed_size_imbalance and balance_slack must match when both are set"
        )
    size_tolerance = (
        p.allowed_size_imbalance
        if p.allowed_size_imbalance is not None
        else p.balance_slack
    )
    if not 0.0 <= size_tolerance < 1.0:
        raise ValueError("allowed_size_imbalance must be in [0, 1)")
    return size_tolerance, size_tolerance


def validate_nonempty_targets(n, splits):
    if np.any(target_sizes(n, splits) == 0):
        raise ValueError(
            "requested split ratios produce an empty fold at this dataset size"
        )


@register("lowrank")
class LowRankSplitter(BaseSplitter):
    description = "Nyström low-rank factorization + balanced-Lloyd + FM (graph-free, O(n·r) memory)"
    arity = "1d"

    @dataclass
    class Params:
        rank: object = 256  # positive integer or "auto"
        rank_min: int = 32
        rank_max: int = 512
        rank_tolerance: float = 0.05
        rank_probe_pairs: Optional[int] = None
        factor_block_size: int = 2048
        factor_memory_mb: Optional[float] = None
        eigen_rtol: float = 1e-8
        metric: Optional[str] = None
        landmark: str = "kmeans++"  # kmeans++ | uniform
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
        group_policy: str = "count_first"  # count_first | strict
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
        leakage_diagnostic: Optional[str] = None  # none | exact | sampled | auto
        diagnostic_pairs: int = 8192
        # Experimental normalized global + positive local pairwise penalty.
        # Off by default; compare against the existing powered/grouped method.
        local_weight: float = 0.0
        local_k: int = 32
        local_threshold: float = 0.4
        local_backend: str = "tiled"  # tiled | nndescent | auto
        neighbour_audit_queries: int = 32
        neighbour_minimum_recall: float = 0.98
        refine_restarts: Optional[int] = 2
        fold_weights: Optional[object] = None
        rank_check: str = "pair_mae"
        on_rank_failure: str = "warn"
        factor_backend: str = "nystrom"

    @classmethod
    def param_schema(cls):
        schema = super().param_schema()
        schema["properties"]["rank"] = {
            "oneOf": [{"type": "integer", "minimum": 1}, {"const": "auto"}],
            "default": 256,
        }
        schema["properties"]["groups"] = {"type": ["object", "null"], "default": None}
        schema["properties"]["fold_weights"] = {
            "type": ["array", "null"],
            "default": None,
            "items": {"type": "array", "items": {"type": "number"}},
        }
        for field, choices in (
            ("rank_check", ["pair_mae", "cut_error"]),
            ("on_rank_failure", ["warn", "raise", "return"]),
            ("factor_backend", ["nystrom", "auto"]),
        ):
            schema["properties"][field]["enum"] = choices
        return schema

    def _diagnose(self, X, labels, metric, seed):
        p = self.params
        mode = p.leakage_diagnostic
        if mode is None:
            mode = (
                "exact"
                if p.score_exact_leakage and len(X) <= _LEAKAGE_MAX_N
                else "none"
            )
        if mode == "auto":
            mode = "exact" if len(X) <= 2000 else "sampled"
        result = {"leakage": None, "leakage_diagnostic": mode}
        if mode == "exact":
            value = scaled_lpi(X, labels, metric=metric)
            result["leakage"] = round(value, 6) if value is not None else None
            result["leakage_status"] = (
                "exact" if value is not None else "undefined_nonpositive_kernel_mass"
            )
            if p.fold_weights is not None:
                result["weighted_leakage"] = scaled_lpi(
                    X, labels, metric=metric, fold_weights=p.fold_weights
                )
        elif mode == "sampled":
            result["leakage_sampled"] = sampled_leakage(
                X,
                labels,
                metric,
                sample_pairs_count=p.diagnostic_pairs,
                seed=104729 + seed,
            )
        return result

    def _prepare_groups(self, ids, X, metric, spec):
        p = self.params
        if p.groups is not None:
            if not isinstance(p.groups, dict) or set(p.groups) != set(ids):
                raise ValueError("groups must map every entity id exactly once")
            codes = {}
            try:
                values = [p.groups[i] for i in ids]
                if any(
                    v is None
                    or isinstance(v, (float, np.floating))
                    and not np.isfinite(v)
                    for v in values
                ):
                    raise ValueError("group labels must be non-missing and finite")
                gid = np.array([codes.setdefault(v, len(codes)) for v in values])
            except TypeError as exc:
                raise ValueError("group labels must be hashable") from exc
            return gid, dict(
                group_source="supplied", must_link_tau=None, max_group=None
            )
        tau = resolve_tau(p.must_link, metric, X, seed=spec.seed)
        if not (-1 if metric == "cosine" else 0) <= tau <= 1:
            raise ValueError(
                "must_link threshold is outside the kernel's similarity range"
            )
        max_group = max(
            1, int(p.max_group_frac * target_sizes(len(ids), spec.splits).min())
        )
        nbrs, backend = None, None
        if p.must_link_backend != "exact":
            from .neighbours import audited_neighbours

            nbrs, backend = audited_neighbours(
                X,
                metric,
                p.must_link_k,
                backend=p.must_link_backend,
                candidates=p.must_link_candidates,
                seed=spec.seed,
                n_jobs=p.must_link_n_jobs,
                audit_queries=p.neighbour_audit_queries,
                minimum_recall=p.neighbour_minimum_recall,
                strong_threshold=tau,
                row_block=p.must_link_row_block,
                reference_block=p.must_link_ref_block,
            )
        gid = must_link_groups(
            X,
            metric,
            tau,
            max_group,
            k=p.must_link_k,
            cap_rule=p.must_link_cap,
            nbrs=nbrs,
        )
        info = dict(
            group_source="capped_topk", must_link_tau=float(tau), max_group=max_group
        )
        if backend is not None:
            info["neighbour_backend"] = backend
        return gid, info

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        start = time.perf_counter()
        tolerance, _ = resolve_size_tolerance(p, spec)
        if p.leakage_diagnostic not in (None, "none", "exact", "sampled", "auto"):
            raise ValueError("leakage_diagnostic must be none, exact, sampled or auto")
        positive_integer("diagnostic_pairs", p.diagnostic_pairs)
        positive_integer("local_k", p.local_k)
        if p.refine_restarts is not None:
            positive_integer("refine_restarts", p.refine_restarts)
        if p.group_policy not in ("count_first", "strict"):
            raise ValueError("group_policy must be count_first or strict")
        if (
            not np.isfinite(p.group_feasibility_seconds)
            or p.group_feasibility_seconds <= 0
        ):
            raise ValueError("group_feasibility_seconds must be finite and positive")
        if p.groups is not None and p.must_link is not None:
            raise ValueError("supply groups or must_link, not both")
        if not np.isfinite(p.local_weight) or p.local_weight < 0:
            raise ValueError("local_weight must be finite and non-negative")
        grouped = p.groups is not None or p.must_link is not None
        if p.local_weight and grouped:
            raise ValueError(
                "experimental local_weight cannot yet be combined with groups/must_link"
            )
        W = validate_fold_weights(p.fold_weights, len(spec.splits))
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        validate_nonempty_targets(len(ids), spec.splits)
        candidate = {}
        group_data, graph_data = None, None

        def optimize_candidate(B, metric):
            # Cache X-dependent work across rank trials; audit samples never
            # influence assignment proposals or optimizer settings.
            nonlocal group_data, graph_data
            optimization_B, local_term, local_info = B, None, None
            if p.local_weight:
                from .local import local_graph, normalize_components, SparseTerm
                from .neighbours import audited_neighbours

                if graph_data is None:
                    nbrs, backend = audited_neighbours(
                        X,
                        metric,
                        p.local_k,
                        backend=p.local_backend,
                        candidates=max(p.must_link_candidates, p.local_k + 1),
                        seed=spec.seed,
                        n_jobs=p.must_link_n_jobs,
                        audit_queries=p.neighbour_audit_queries,
                        minimum_recall=p.neighbour_minimum_recall,
                        strong_threshold=p.local_threshold,
                        row_block=p.must_link_row_block,
                        reference_block=p.must_link_ref_block,
                    )
                    graph, info = local_graph(
                        X,
                        metric,
                        k=p.local_k,
                        threshold=p.local_threshold,
                        neighbours=nbrs,
                    )
                    graph_data = graph, info, backend
                graph, info, backend = graph_data
                optimization_B, local_term, normalization = normalize_components(
                    B, graph, p.local_weight
                )
                local_info = dict(
                    graph=info,
                    normalization=normalization,
                    backend=backend,
                    experimental=True,
                )
            labels, _, optimization = optimize_restarts(
                optimization_B,
                spec.splits,
                seed=spec.seed,
                n_restarts=p.n_restarts,
                n_iter=p.n_iter,
                balance_slack=tolerance,
                fm=p.fm and len(ids) <= p.fm_max_n,
                refine_restarts=p.refine_restarts,
                local=local_term,
                fold_weights=p.fold_weights,
                tol=1e-9 if p.local_weight else 1e-6,
            )
            fields = dict(optimization=optimization, fm_moves=optimization["fm_moves"])
            if grouped:
                if group_data is None:
                    group_data = self._prepare_groups(ids, X, metric, spec)
                gid, group_info = group_data
                labels, info = grouped_split(
                    B,
                    gid,
                    spec.splits,
                    tolerance,
                    n_restarts=p.n_restarts,
                    n_iter=p.n_iter,
                    seed=spec.seed,
                    fm=p.fm,
                    init_labels=labels,
                    group_policy=p.group_policy,
                    feasibility_seconds=p.group_feasibility_seconds,
                    fold_weights=p.fold_weights,
                    refine_restarts=p.refine_restarts,
                )
                fields.update(
                    group_info,
                    **info,
                    group_contract_scope="supplied_or_constructed_groups",
                    must_link_backend=p.must_link_backend,
                )
            if p.hardness is not None:
                labels = interpolate_to_random(labels, p.hardness, seed=spec.seed)
            if local_info is not None:
                term = SparseTerm(local_term)
                global_cost = factor_leakage(
                    optimization_B, labels, len(spec.splits), p.fold_weights
                )
                local_info.update(
                    composite_objective=global_cost + term.cross(labels),
                    global_factor_ratio=global_cost,
                    weighted_local_graph_ratio=term.cross(labels),
                )
                fields["local_objective"] = local_info
            fields["factor_leakage"] = factor_leakage(B, labels, len(spec.splits))
            fields["weighted_factor_leakage"] = factor_leakage(
                B, labels, len(spec.splits), p.fold_weights
            )
            candidate.update(labels=labels, fields=fields)
            return labels

        callback = optimize_candidate if p.rank_check == "cut_error" else None
        B, metric, factor_info = nystrom_features(
            X,
            rank=p.rank,
            metric=p.metric,
            landmark=p.landmark,
            seed=spec.seed,
            kernel_power=p.kernel_power,
            return_info=True,
            rank_min=p.rank_min,
            rank_max=p.rank_max,
            rank_tolerance=p.rank_tolerance,
            rank_probe_pairs=p.rank_probe_pairs,
            block_size=p.factor_block_size,
            factor_memory_mb=p.factor_memory_mb,
            eigen_rtol=p.eigen_rtol,
            on_failure=p.on_rank_failure,
            factor_backend=p.factor_backend,
            rank_check=p.rank_check,
            _partitioner=callback,
            fold_weights=W,
        )
        if not candidate:
            optimize_candidate(B, metric)
        labels, fields = candidate["labels"], candidate["fields"]
        diagnostics = self._diagnose(X, labels, metric, spec.seed)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        return self._result(
            assignment,
            spec,
            time.perf_counter() - start,
            metric=metric,
            rank=int(B.shape[1]),
            factorization=factor_info,
            fold_weights=W.tolist(),
            **fields,
            **diagnostics,
        )
