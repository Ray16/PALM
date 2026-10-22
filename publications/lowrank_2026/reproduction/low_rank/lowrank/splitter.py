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

from splitters.base import BaseSplitter, SplitResult, SplitSpec, register
from splitters.common.feature_preparation import feature_matrix_from_dict
from splitters.common.leakage_metrics import scaled_lpi
from splitters.common.pairwise_similarity import check_metric
from splitters.common.split_naming import assign_split_names

from splitters.common.balanced_assignment import target_sizes

from .grouped import grouped_split, must_link_groups, resolve_tau
from .nystrom import LANDMARKS, nystrom_features
from .objective import factor_leakage
from .optimize import balanced_lloyd, fm_polish, interpolate_to_random

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
    if getattr(p, "must_link", None) is not None:
        if p.hardness is not None and float(p.hardness) < 1.0:
            raise ValueError("must_link cannot be combined with hardness < 1: "
                             "point randomization would break retained groups")
        if not 0.0 < p.max_group_frac <= 1.0:
            raise ValueError("max_group_frac must be in (0, 1]")
        _check_int("must_link_k", p.must_link_k)
        if p.must_link_backend not in ("exact", "tiled", "nndescent"):
            raise ValueError("must_link_backend must be 'exact', 'tiled' or 'nndescent'")
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
        rank: int = 256
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

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        size_tolerance, fm_eps = resolve_size_tolerance(p, spec)
        if p.metric is not None:
            check_metric(p.metric)

        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        validate_nonempty_targets(n, spec.splits)
        B, metric = nystrom_features(X, rank=p.rank, metric=p.metric,
                                     landmark=p.landmark, seed=spec.seed,
                                     kernel_power=p.kernel_power)
        logger.info("  Low-rank: n=%d rank=%d metric=%s", n, B.shape[1], metric)

        best_labels, best_obj = None, np.inf
        for r in range(p.n_restarts):
            labels = balanced_lloyd(B, spec.splits, n_iter=p.n_iter, seed=spec.seed + r,
                                    balance_slack=size_tolerance)
            obj = factor_leakage(B, labels, len(spec.splits))
            if obj < best_obj:
                best_obj, best_labels = obj, labels
        logger.info("  Best-of-%d Lloyd leakage=%.1f", p.n_restarts, best_obj)

        moves = 0
        if p.fm and n <= p.fm_max_n:
            best_labels, moves = fm_polish(B, best_labels, spec.splits, epsilon=fm_eps)
            best_obj = factor_leakage(B, best_labels, len(spec.splits))
            logger.info("  FM polish: %d moves, leakage=%.1f", moves, best_obj)

        if p.must_link is not None:   # point-level solution warm-starts the must-link optimizer
            return self._split_grouped(ids, X, B, metric, spec, fm_eps, t0, init=best_labels)

        if p.hardness is not None:                 # controllable-hardness dial
            best_labels = interpolate_to_random(best_labels, p.hardness, seed=spec.seed)
            best_obj = factor_leakage(B, best_labels, len(spec.splits))

        assignment = assign_split_names(ids, best_labels, spec.splits, spec.names)
        leak = (round(scaled_lpi(X, best_labels, metric=metric), 6)
                if p.score_exact_leakage and n <= _LEAKAGE_MAX_N else None)
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            rank=int(B.shape[1]), factor_leakage=round(float(best_obj), 3),
                            fm_moves=int(moves), leakage=leak)

    def _split_grouped(self, ids, X, B, metric, spec, eps, t0, init=None):
        p = self.params
        tau = resolve_tau(p.must_link, metric, X, seed=spec.seed)
        smallest = int(target_sizes(len(ids), spec.splits).min())
        max_group = max(1, int(p.max_group_frac * smallest))
        nbrs = None
        if p.must_link_backend == "tiled":
            from .neighbours import tiled_neighbours
            nbrs = tiled_neighbours(X, metric, p.must_link_k,
                        row_block=p.must_link_row_block, reference_block=p.must_link_ref_block)
        if p.must_link_backend == "nndescent":
            if metric != "tanimoto":
                raise ValueError("nndescent grouping currently supports binary Tanimoto only")
            from .neighbours import approximate_tanimoto_neighbours
            nbrs = approximate_tanimoto_neighbours(X, p.must_link_k,
                       candidates=p.must_link_candidates, seed=spec.seed,
                       n_jobs=p.must_link_n_jobs)
        gid = must_link_groups(X, metric, tau, max_group, k=p.must_link_k,
                               cap_rule=p.must_link_cap, nbrs=nbrs)
        labels, info = grouped_split(B, gid, spec.splits, eps, n_restarts=p.n_restarts,
                                     n_iter=p.n_iter, seed=spec.seed, fm=p.fm,
                                     init_labels=init)
        logger.info("  Grouped low-rank: tau=%.3f groups=%d (%d pts grouped, %d broken) "
                    "leakage=%.1f", tau, info["n_groups"], info["grouped_points"],
                    info["broken_groups"], info["factor_leakage"])
        if p.hardness is not None:
            labels = interpolate_to_random(labels, p.hardness, seed=spec.seed)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        n = len(ids)
        leak = (round(scaled_lpi(X, labels, metric=metric), 6)
                if p.score_exact_leakage and n <= _LEAKAGE_MAX_N else None)
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            rank=int(B.shape[1]), must_link_tau=round(tau, 4),
                            max_group=max_group, leakage=leak,
                            must_link_backend=p.must_link_backend,
                            **{k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in info.items()})
