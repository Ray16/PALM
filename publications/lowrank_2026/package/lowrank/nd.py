"""n-D low-rank splitting with sparse feature and exact identity factors.

Each axis contributes a feature-kernel factor plus disjoint identity columns for
featureless entities. Identity columns are CSR: storage is linear in records,
not records times distinct identities. Featured records are standardized with
their original multiplicities, then unique prepared rows are factorized once.
Per-axis factors are scaled by off-diagonal mass and concatenated sparsely.

The common optimizer uses fold sums, row/fold scores and individual columns;
it never materializes either the full factor or kernel as a dense n-by-n array.
Optional fold-pair weights change the optimized evaluation boundary explicitly.
Exact macro-axis diagnostics retain the original kernel/identity definition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import sparse

from lowrank.splitters.base import BaseSplitter, SplitResult, SplitSpec, register
from lowrank.splitters.common.leakage_metrics import axis_features, macro_axis_lpi
from lowrank.splitters.common.split_naming import assign_split_names

from .nystrom import nystrom_features
from .objective import factor_leakage, realized_imbalance
from .optimize import optimize_restarts, interpolate_to_random
from .factors import offdiagonal_mass, validate_fold_weights
from .diagnostics import prepare_reference
from .splitter import resolve_size_tolerance, validate_nonempty_targets

logger = logging.getLogger(__name__)

_LEAKAGE_MAX_N = 100_000  # skip the O(n^2) macro-leakage diagnostic above this
_FM_MAX_N = 2_000_000  # match the 1-D splitter's FM cap


def _as_nd(data):
    """Coerce input to ``(records, axis_feature_maps)`` (mirrors the shared n-D coercion)."""
    if hasattr(data, "records") and hasattr(data, "axis_feature_maps"):
        return data.records, data.axis_feature_maps
    if isinstance(data, dict):
        return data["records"], data["axis_feature_maps"]
    records, afm = data  # (records, axis_feature_maps) tuple
    return records, afm


def _identity_factor(vals):
    """One-hot factor over the values, dropping singletons (they add no similarity).

    ``B[i, col(value_i)] = 1`` for values that appear >= 2 times, so ``B B^T[i,j] = 1``
    iff records i, j share that (non-unique) component value; unique values -> zero row.
    """
    codes, inv = np.unique(np.asarray(vals), return_inverse=True)
    counts = np.bincount(inv, minlength=len(codes))
    keep = np.where(counts >= 2)[0]  # singletons contribute nothing
    if len(keep) == 0:
        return None
    col_of = -np.ones(len(codes), dtype=np.int64)
    col_of[keep] = np.arange(len(keep))
    n = len(vals)
    cols = col_of[inv]
    rows = np.where(cols >= 0)[0]
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols[rows])), shape=(n, len(keep))
    )


def _offdiag_mass(B):
    """Total off-diagonal similarity mass of ``B B^T`` = 0.5(||sum||^2 - sum||row||^2)."""
    return offdiagonal_mass(B)


def _build_nd_factor(
    records,
    afm,
    rank,
    landmark,
    seed,
    axis_metrics=None,
    factor_options=None,
    return_info=False,
):
    """Concatenate per-axis, ratio-scaled factors into one ``B`` (n x sum r_a).

    Returns ``(B, used_axes)``. Each axis is scaled by ``1/sqrt(total_a)`` so its
    factor-space contribution is its leakage ratio; degenerate axes (no off-diagonal
    similarity) are skipped.
    """
    axis_metrics = axis_metrics or {}
    n = len(records)
    blocks, used, diagnostics = [], [], {}
    for axis in afm:
        vals, F, has, metric = axis_features(
            records, afm[axis], axis, axis_metrics.get(axis)
        )
        parts = []
        if F is not None and has.any():
            # Preserve record-weighted standardization, then factor each unique
            # prepared row once. Identity and continuous features stay separate.
            prepared = prepare_reference(F[has], metric).numpy()
            unique, inverse = np.unique(prepared, axis=0, return_inverse=True)
            B_f, _, info = nystrom_features(
                unique,
                rank=rank,
                metric=metric,
                landmark=landmark,
                seed=seed,
                _prepared=True,
                return_info=True,
                **(factor_options or {}),
            )
            coo = sparse.csr_matrix(B_f)[inverse].tocoo()
            rows = np.flatnonzero(has)
            parts.append(
                sparse.csr_matrix(
                    (coo.data, (rows[coo.row], coo.col)), shape=(n, B_f.shape[1])
                )
            )
            diagnostics[axis] = dict(
                unique_feature_rows=len(unique),
                featured_records=int(has.sum()),
                factorization=info,
            )
        miss = np.where(~has)[0]
        if len(miss):
            B_m = _identity_factor([vals[i] for i in miss])
            if B_m is not None:
                coo = B_m.tocoo()
                parts.append(
                    sparse.csr_matrix(
                        (coo.data, (miss[coo.row], coo.col)), shape=(n, B_m.shape[1])
                    )
                )
        if not parts:
            continue
        B_a = sparse.hstack(parts, format="csr")
        total = _offdiag_mass(B_a)
        if total <= 1e-12:  # no similarity on this axis
            continue
        blocks.append(B_a * np.float32(1.0 / np.sqrt(total)))
        used.append(axis)
    if not blocks:
        raise ValueError("no axis carries any within-split similarity; cannot split")
    B = sparse.hstack(blocks, format="csr", dtype=np.float32)
    if return_info:
        return (
            B,
            used,
            dict(
                axes=diagnostics,
                factor_nnz=int(B.nnz),
                storage_bytes=int(B.data.nbytes + B.indices.nbytes + B.indptr.nbytes),
                storage="CSR",
            ),
        )
    return B, used


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
        balance_slack: float = 0.0  # compatibility alias
        hardness: Optional[float] = None  # controllable-OOD dial (as in 1-D)
        # axis -> similarity metric override; default auto per axis (Tanimoto for
        # fingerprints, standardized euclidean for dense descriptors)
        axis_metrics: Optional[Dict[str, str]] = None
        score_exact_leakage: bool = True
        refine_restarts: Optional[int] = None
        fold_weights: Optional[object] = None
        factor_backend: str = "nystrom"
        on_rank_failure: str = "warn"

    @classmethod
    def param_schema(cls):
        from .splitter import LowRankSplitter

        schema = super().param_schema()
        shared = LowRankSplitter.param_schema()["properties"]
        for name in ("rank", "fold_weights", "factor_backend", "on_rank_failure"):
            schema["properties"][name] = shared[name]
        return schema

    def split(self, data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        size_tolerance, fm_eps = resolve_size_tolerance(p, spec)

        records, afm = _as_nd(data)
        n = len(records)
        validate_nonempty_targets(n, spec.splits)

        W = validate_fold_weights(p.fold_weights, len(spec.splits))
        B, used_axes, factor_info = _build_nd_factor(
            records,
            afm,
            p.rank,
            p.landmark,
            spec.seed,
            p.axis_metrics,
            factor_options=dict(
                factor_backend=p.factor_backend, on_failure=p.on_rank_failure
            ),
            return_info=True,
        )
        logger.info(
            "  lowrank_nd: n=%d axes=%d factor_dim=%d", n, len(used_axes), B.shape[1]
        )

        best_labels, _, optimization = optimize_restarts(
            B,
            spec.splits,
            seed=spec.seed,
            n_restarts=p.n_restarts,
            n_iter=p.n_iter,
            balance_slack=size_tolerance,
            fm=p.fm and n <= _FM_MAX_N,
            refine_restarts=p.refine_restarts,
            fold_weights=p.fold_weights,
        )

        if p.hardness is not None:
            best_labels = interpolate_to_random(best_labels, p.hardness, seed=spec.seed)

        labels = np.asarray(best_labels)
        assignment = assign_split_names(list(range(n)), labels, spec.splits, spec.names)

        leak, per_axis, weighted_leak = (None, None, None)
        if p.score_exact_leakage and n <= _LEAKAGE_MAX_N:
            leak, per_axis = macro_axis_lpi(
                records, afm, labels, axis_metrics=p.axis_metrics
            )
            leak = round(leak, 6)
            per_axis = {a: round(v, 6) for a, v in per_axis.items()}
            if p.fold_weights is not None:
                weighted_leak, _ = macro_axis_lpi(
                    records, afm, labels, axis_metrics=p.axis_metrics, fold_weights=W
                )
        return self._result(
            assignment,
            spec,
            time.time() - t0,
            imbalance=round(realized_imbalance(labels, spec.splits), 4),
            rank=int(B.shape[1]),
            optimization=optimization,
            fm_moves=optimization["fm_moves"],
            n_axes=len(used_axes),
            factorization=factor_info,
            fold_weights=W.tolist(),
            weighted_leakage=weighted_leak,
            factor_leakage=factor_leakage(B, labels, len(spec.splits)),
            weighted_factor_leakage=factor_leakage(
                B, labels, len(spec.splits), p.fold_weights
            ),
            leakage=leak,
            axis_leakage=per_axis,
        )
