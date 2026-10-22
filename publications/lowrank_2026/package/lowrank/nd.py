"""n-D (multi-component / reaction) low-rank splitter — ``lowrank_nd``.

The n-D generalization of the low-rank splitter. A record (e.g. a reaction
``rA + rB + rC -> product``) has several component *axes*; the n-D leakage is the
macro-average over axes of each axis's scaled ``L(pi)``
(:func:`splitters.common.leakage_metrics.macro_axis_lpi`).

**Key idea — minimize that objective in one shared factor space by concatenation.**
For each axis ``a`` build a per-record factor ``B_a`` with ``B_a B_aᵀ ≈ S_a`` (the
axis similarity):

- the axis similarity is defined once, by
  :func:`splitters.common.leakage_metrics.axis_features` (shared with the scorer):
  feature ``metric`` between featured values (auto-selected per axis — Tanimoto for
  fingerprints, standardized euclidean for dense descriptors — or set via
  ``axis_metrics``), identity between featureless values, 0 across the two groups.
- ``B_a = [B_feat | B_id]``: a Nyström factor of the featured rows (zero rows for
  featureless records) concatenated with a one-hot of the featureless records'
  values, so ``B_a B_aᵀ[i,j] = 1`` iff two featureless records share that component.
  The two blocks have disjoint row support, so ``B_a B_aᵀ`` reproduces the scorer's
  similarity exactly on the identity part. Singleton values contribute nothing and
  are dropped.

Scale each ``B_a`` by ``1/sqrt(total_a)`` where ``total_a`` is the axis's total
off-diagonal similarity mass, so the axis contributes its leakage *ratio*; then the
plain factor-space leakage of the **column-concatenated** ``B`` equals the sum of the
per-axis ratios (= ``n_axes · macro`` in factor space). So the *existing* 1-D
optimizer (:func:`balanced_lloyd` + :func:`fm_polish`) minimizes the n-D objective
unchanged — no new solver. Reported ``leakage`` is the *true* ``macro_axis_lpi``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from lowrank.splitters.base import BaseSplitter, SplitResult, SplitSpec, register
from lowrank.splitters.common.leakage_metrics import axis_features, macro_axis_lpi
from lowrank.splitters.common.split_naming import assign_split_names

from .nystrom import nystrom_features
from .objective import factor_leakage, realized_imbalance
from .optimize import balanced_lloyd, fm_polish, interpolate_to_random
from .splitter import resolve_size_tolerance, validate_nonempty_targets

logger = logging.getLogger(__name__)

_LEAKAGE_MAX_N = 100_000            # skip the O(n^2) macro-leakage diagnostic above this
_FM_MAX_N = 2_000_000              # match the 1-D splitter's FM cap


def _as_nd(data):
    """Coerce input to ``(records, axis_feature_maps)`` (mirrors the shared n-D coercion)."""
    if hasattr(data, "records") and hasattr(data, "axis_feature_maps"):
        return data.records, data.axis_feature_maps
    if isinstance(data, dict):
        return data["records"], data["axis_feature_maps"]
    records, afm = data           # (records, axis_feature_maps) tuple
    return records, afm


def _identity_factor(vals):
    """One-hot factor over the values, dropping singletons (they add no similarity).

    ``B[i, col(value_i)] = 1`` for values that appear >= 2 times, so ``B B^T[i,j] = 1``
    iff records i, j share that (non-unique) component value; unique values -> zero row.
    """
    codes, inv = np.unique(np.asarray(vals), return_inverse=True)
    counts = np.bincount(inv, minlength=len(codes))
    keep = np.where(counts >= 2)[0]                     # singletons contribute nothing
    if len(keep) == 0:
        return None
    col_of = -np.ones(len(codes), dtype=np.int64)
    col_of[keep] = np.arange(len(keep))
    n = len(vals)
    B = np.zeros((n, len(keep)), dtype=np.float32)
    cols = col_of[inv]
    rows = np.where(cols >= 0)[0]
    B[rows, cols[rows]] = 1.0
    return B


def _offdiag_mass(B):
    """Total off-diagonal similarity mass of ``B B^T`` = 0.5(||sum||^2 - sum||row||^2)."""
    s = B.sum(0)
    return 0.5 * float(s @ s - (B * B).sum())


def _build_nd_factor(records, afm, rank, landmark, seed, axis_metrics=None):
    """Concatenate per-axis, ratio-scaled factors into one ``B`` (n x sum r_a).

    Returns ``(B, used_axes)``. Each axis is scaled by ``1/sqrt(total_a)`` so its
    factor-space contribution is its leakage ratio; degenerate axes (no off-diagonal
    similarity) are skipped.
    """
    axis_metrics = axis_metrics or {}
    n = len(records)
    blocks, used = [], []
    for axis in afm:
        vals, F, has, metric = axis_features(records, afm[axis], axis,
                                             axis_metrics.get(axis))
        parts = []
        if F is not None and has.any():
            B_f, _ = nystrom_features(F[has], rank=rank, metric=metric,
                                      landmark=landmark, seed=seed)
            B_full = np.zeros((n, B_f.shape[1]), dtype=np.float32)
            B_full[has] = B_f
            parts.append(B_full)
        miss = np.where(~has)[0]
        if len(miss):
            B_m = _identity_factor([vals[i] for i in miss])
            if B_m is not None:
                B_id = np.zeros((n, B_m.shape[1]), dtype=np.float32)
                B_id[miss] = B_m
                parts.append(B_id)
        if not parts:
            continue
        B_a = np.hstack(parts)
        total = _offdiag_mass(B_a)
        if total <= 1e-12:                              # no similarity on this axis
            continue
        blocks.append(B_a * np.float32(1.0 / np.sqrt(total)))
        used.append(axis)
    if not blocks:
        raise ValueError("no axis carries any within-split similarity; cannot split")
    return np.hstack(blocks).astype(np.float32), used


@register("lowrank_nd")
class LowRankNDSplitter(BaseSplitter):
    description = "n-D low-rank: per-axis Nyström/identity factors, concatenated, balanced-Lloyd + FM"
    arity = "nd"

    @dataclass
    class Params:
        rank: int = 256
        landmark: str = "kmeans++"
        n_restarts: int = 4
        n_iter: int = 25
        fm: bool = True
        allowed_size_imbalance: Optional[float] = None
        balance_slack: float = 0.0          # compatibility alias
        hardness: Optional[float] = None    # controllable-OOD dial (as in 1-D)
        # axis -> similarity metric override; default auto per axis (Tanimoto for
        # fingerprints, standardized euclidean for dense descriptors)
        axis_metrics: Optional[Dict[str, str]] = None
        score_exact_leakage: bool = True

    def split(self, data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        size_tolerance, fm_eps = resolve_size_tolerance(p, spec)

        records, afm = _as_nd(data)
        n = len(records)
        validate_nonempty_targets(n, spec.splits)

        B, used_axes = _build_nd_factor(records, afm, p.rank, p.landmark, spec.seed,
                                        p.axis_metrics)
        logger.info("  lowrank_nd: n=%d axes=%d factor_dim=%d", n, len(used_axes), B.shape[1])

        best_labels, best_obj = None, np.inf
        for r in range(p.n_restarts):
            labels = balanced_lloyd(B, spec.splits, n_iter=p.n_iter, seed=spec.seed + r,
                                    balance_slack=size_tolerance)
            obj = factor_leakage(B, labels, len(spec.splits))
            if obj < best_obj:
                best_obj, best_labels = obj, labels

        moves = 0
        if p.fm and n <= _FM_MAX_N:
            best_labels, moves = fm_polish(B, best_labels, spec.splits, epsilon=fm_eps)

        if p.hardness is not None:
            best_labels = interpolate_to_random(best_labels, p.hardness, seed=spec.seed)

        labels = np.asarray(best_labels)
        assignment = assign_split_names(list(range(n)), labels, spec.splits, spec.names)

        leak, per_axis = (None, None)
        if p.score_exact_leakage and n <= _LEAKAGE_MAX_N:
            leak, per_axis = macro_axis_lpi(records, afm, labels, axis_metrics=p.axis_metrics)
            leak = round(leak, 6)
            per_axis = {a: round(v, 6) for a, v in per_axis.items()}
        return self._result(assignment, spec, time.time() - t0,
                            imbalance=round(realized_imbalance(labels, spec.splits), 4),
                            rank=int(B.shape[1]), fm_moves=int(moves), n_axes=len(used_axes),
                            leakage=leak, axis_leakage=per_axis)
