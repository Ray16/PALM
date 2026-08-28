"""The low-rank leakage-minimizing splitter (registered as ``lowrank``).

Graph-free: factorize ``S ~= B B^T`` (Nyström), then minimize cross-split leakage
in the r-dim factor space with balanced-Lloyd restarts + a monotone FM polish —
O(n·r), scales to millions of rows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from PALM.splitters.base import BaseSplitter, SplitResult, SplitSpec, register
from PALM.splitters.common.feature_preparation import feature_matrix_from_dict
from PALM.splitters.common.leakage_metrics import scaled_lpi
from PALM.splitters.common.split_naming import assign_split_names

from .nystrom import nystrom_features
from .objective import factor_leakage
from .optimize import balanced_lloyd, fm_polish, interpolate_to_random

logger = logging.getLogger(__name__)

_LEAKAGE_MAX_N = 100_000


@register("lowrank")
class LowRankSplitter(BaseSplitter):
    description = "Nyström low-rank factorization + balanced-Lloyd + FM (graph-free, O(n·r))"
    arity = "1d"

    @dataclass
    class Params:
        rank: int = 256
        metric: Optional[str] = None
        landmark: str = "kmeans++"          # kmeans++ | uniform | leverage
        ridge: float = 0.0                  # W^{-1/2} regularization (fraction of λ_max)
        energy: Optional[float] = None      # adaptive rank: keep top spectral-energy fraction
        n_restarts: int = 4
        n_iter: int = 25
        fm: bool = True
        fm_max_n: int = 200_000
        # balance–leakage tradeoff knob: 0.0 = exact target sizes (default,
        # back-compatible); >0 opens a (1 ± balance_slack) size corridor that both
        # the Lloyd assignment and the FM polish may exploit to lower leakage.
        balance_slack: float = 0.0
        # controllable-hardness dial: None = hardest (fully optimized, default);
        # in [0,1], 1 = hardest, 0 = random (easiest). Interpolates the split toward
        # random, balance-preserving, so realized OOD difficulty is tunable.
        hardness: Optional[float] = None

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        B, metric = nystrom_features(X, rank=p.rank, metric=p.metric,
                                     landmark=p.landmark, seed=spec.seed,
                                     ridge=p.ridge, energy=p.energy)
        logger.info("  Low-rank: n=%d rank=%d metric=%s", n, B.shape[1], metric)

        # balance corridor: the tradeoff knob when set, else the spec's default
        # (preserves back-compatible behaviour: exact Lloyd + spec.epsilon FM).
        fm_eps = p.balance_slack if p.balance_slack > 0 else spec.epsilon
        best_labels, best_obj = None, np.inf
        for r in range(p.n_restarts):
            labels = balanced_lloyd(B, spec.splits, n_iter=p.n_iter, seed=spec.seed + r,
                                    balance_slack=p.balance_slack)
            obj = factor_leakage(B, labels, len(spec.splits))
            if obj < best_obj:
                best_obj, best_labels = obj, labels
        logger.info("  Best-of-%d Lloyd leakage=%.1f", p.n_restarts, best_obj)

        moves = 0
        if p.fm and n <= p.fm_max_n:
            best_labels, moves = fm_polish(B, best_labels, spec.splits, epsilon=fm_eps)
            best_obj = factor_leakage(B, best_labels, len(spec.splits))
            logger.info("  FM polish: %d moves, leakage=%.1f", moves, best_obj)

        if p.hardness is not None:                 # controllable-hardness dial
            best_labels = interpolate_to_random(best_labels, p.hardness, seed=spec.seed)
            best_obj = factor_leakage(B, best_labels, len(spec.splits))

        assignment = assign_split_names(ids, best_labels, spec.splits, spec.names)
        leak = round(scaled_lpi(X, best_labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            rank=int(B.shape[1]), factor_leakage=round(float(best_obj), 3),
                            fm_moves=int(moves), leakage=leak)
