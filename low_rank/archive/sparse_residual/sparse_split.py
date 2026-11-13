"""ARCHIVED — sparse k-NN residual (K ~= BBᵀ + R) and soft must-link for the low-rank splitter.

Superseded / negative (docs/FINDINGS.md Step 9, comparisons/FINDINGS.md "Tail-aware
low-rank"): the residual is a mild positive on its own (it repairs the near-duplicate
failure of kernel_power=3: synthetic family leak 0.084 -> 0.024 on near_dup), but hard
must-link groups (live ``must_link``) dominate it on every axis; the soft must-link boost
is a clear negative (real nnl_rel 0.229 vs 0.108 without it at p=3). Kept runnable:

    from sparse_split import split_sparse
    labels = split_sparse(X, [8, 1, 1], eps=0.02, seed=0, kernel_power=3, sparse_k=16)

``optimize_sparse.py`` / ``objective_sparse.py`` are frozen pre-strip snapshots of
``lowrank/optimize.py`` / ``lowrank/objective.py`` that thread ``R`` through Lloyd, FM,
the exact-size swap polish and the objective.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from lowrank.grouped import resolve_tau  # noqa: E402
from lowrank.nystrom import nystrom_features  # noqa: E402
from objective_sparse import factor_leakage  # noqa: E402
from optimize_sparse import balanced_lloyd, fm_polish  # noqa: E402
from sparse import sparse_residual  # noqa: E402
from splitters.common.feature_preparation import choose_metric  # noqa: E402


def split_sparse(X, splits, eps, seed=0, kernel_power=1, sparse_k=16, must_link=None,
                 boost=1.0, rank=256, n_restarts=4, n_iter=25, metric=None):
    """Integer block labels from the archived low-rank + sparse objective (mirrors the
    pre-strip ``LowRankSplitter`` path with ``allowed_size_imbalance=eps``)."""
    metric = metric or choose_metric(X)
    B, metric = nystrom_features(X, rank=rank, metric=metric, seed=seed,
                                 kernel_power=kernel_power)
    tau = resolve_tau(must_link, metric, X, seed=seed) if must_link is not None else None
    R = sparse_residual(X, B, metric, k=sparse_k, kernel_power=kernel_power, tau=tau,
                        boost=boost)
    best, best_obj = None, np.inf
    for r in range(n_restarts):
        lab = balanced_lloyd(B, splits, n_iter=n_iter, seed=seed + r, balance_slack=eps, R=R)
        obj = factor_leakage(B, lab, len(splits), R=R)
        if obj < best_obj:
            best, best_obj = lab, obj
    best, _ = fm_polish(B, best, splits, epsilon=eps, R=R)
    return best
