"""ARCHIVED tests for the sparse k-NN residual (archive/sparse_residual/).

    CUDA_VISIBLE_DEVICES="" python -m pytest archive/sparse_residual/test_sparse_residual.py
"""

from __future__ import annotations

import numpy as np

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(os.path.dirname(_HERE))]

from lowrank import nystrom_features  # noqa: E402
from lowrank.grouped import topk_neighbours  # noqa: E402
from objective_sparse import factor_leakage  # noqa: E402
from optimize_sparse import balanced_lloyd, fm_polish  # noqa: E402
from sparse import sparse_residual  # noqa: E402
from sparse_split import split_sparse  # noqa: E402


def _fps(n_groups=40, size=5, n_bg=200, d=512, rho=0.05, q=0.1, seed=0):
    """Binary fingerprints: near-duplicate groups (members ~0.8 Tanimoto) + random background."""
    rng = np.random.default_rng(seed)
    protos = (rng.random((n_groups, d)) < rho).astype(np.float32)
    X = []
    for p in protos:
        for _ in range(size):
            x = p.copy()
            on = x > 0
            x[on & (rng.random(d) < q)] = 0
            x[(~on) & (rng.random(d) < q * rho)] = 1
            X.append(x)
    X = np.vstack(X + [(rng.random((n_bg, d)) < rho).astype(np.float32)])
    X[X.sum(1) == 0, 0] = 1
    return X


def _tanimoto(X):
    inter = X @ X.T
    c = X.sum(1)
    return inter / (c[:, None] + c[None, :] - inter)


def test_sparse_residual_exact_on_knn_pairs():
    X = _fps()
    B, _ = nystrom_features(X, rank=32, metric="tanimoto", kernel_power=2)
    idx, sim = topk_neighbours(X, "tanimoto", k=8)
    R = sparse_residual(X, B, "tanimoto", k=8, kernel_power=2, nbrs=(idx, sim))
    S2 = _tanimoto(X) ** 2
    K = B.astype(np.float64) @ B.T.astype(np.float64) + R.toarray()
    rows = np.repeat(np.arange(len(X)), 8)
    assert np.abs(K[rows, idx.ravel()] - S2[rows, idx.ravel()]).max() < 1e-4
    assert np.allclose(R.toarray(), R.toarray().T)


def test_factor_leakage_with_R_matches_brute_force():
    X = _fps()
    B, _ = nystrom_features(X, rank=32, metric="tanimoto")
    R = sparse_residual(X, B, "tanimoto", k=8)
    lab = np.random.default_rng(0).integers(0, 3, len(X))
    K = B.astype(np.float64) @ B.T.astype(np.float64) + R.toarray()
    cross = lab[:, None] != lab[None, :]
    assert abs(factor_leakage(B, lab, 3, R=R) - 0.5 * K[cross].sum()) < 1e-2


def test_lloyd_fm_with_R_keep_sizes_and_fm_is_monotone():
    X = _fps()
    B, _ = nystrom_features(X, rank=32, metric="tanimoto", kernel_power=3)
    R = sparse_residual(X, B, "tanimoto", k=16, kernel_power=3)
    lab = balanced_lloyd(B, [8, 1, 1], seed=0, R=R)
    before = factor_leakage(B, lab, 3, R=R)
    lab2, _ = fm_polish(B, lab, [8, 1, 1], epsilon=0.0, R=R)   # exact sizes -> swaps
    assert np.array_equal(np.bincount(lab, minlength=3), np.bincount(lab2, minlength=3))
    assert factor_leakage(B, lab2, 3, R=R) <= before + 1e-3


def test_split_sparse_runs_and_keeps_sizes():
    X = _fps()
    lab = split_sparse(X, [8, 1, 1], eps=0.0, seed=0, kernel_power=3, sparse_k=16)
    assert np.bincount(lab, minlength=3).tolist() == [320, 40, 40]
