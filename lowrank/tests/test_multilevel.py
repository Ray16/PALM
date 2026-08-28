"""Tests for the multilevel FM refinement (Direction #1).

Run: `python -m PALM.lowrank.tests.test_multilevel` (plain-assert smoke run), or
via pytest. Kept in a separate file so it does not disturb `test_lowrank.py`.
"""

from __future__ import annotations

import numpy as np

from PALM.lowrank import balanced_lloyd, factor_leakage, fm_polish, nystrom_features
from PALM.lowrank.multilevel import multilevel_split
from PALM.lowrank.objective import realized_imbalance
from PALM.splitters.common.balanced_assignment import capacity_corridor


def _blobs(n=1200, d=24, k=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    X = np.vstack([centers[c] + rng.normal(size=(n // k, d)) for c in range(k)]).astype("float32")
    return X


def _flat(B, splits, seed, eps):
    """The current pipeline: best-of-4 Lloyd + single-move FM."""
    best, best_obj = None, np.inf
    for r in range(4):
        lab = balanced_lloyd(B, splits, seed=seed + r)
        o = factor_leakage(B, lab, len(splits))
        if o < best_obj:
            best_obj, best = o, lab
    lab, _ = fm_polish(B, best, splits, epsilon=eps)
    return best, lab      # (best lloyd seed, flat result)


def test_multilevel_beats_or_equals_flat():
    eps = 0.05
    X = _blobs()
    B, _ = nystrom_features(X, rank=64, seed=0)
    lloyd_seed, flat = _flat(B, [8, 2], seed=0, eps=eps)
    ml = multilevel_split(B, [8, 2], epsilon=eps, seed=0, seed_labels=lloyd_seed)
    L_flat = factor_leakage(B, flat, 2)
    L_ml = factor_leakage(B, ml, 2)
    # never worse than flat (seed_labels guarantee) — usually strictly better.
    # Relative tolerance: the weighted-FM and fm_polish paths differ only by GPU
    # float-reduction order (~1e-6 relative on leakage magnitudes in the thousands).
    assert L_ml <= L_flat * (1 + 1e-4), (L_ml, L_flat)
    print(f"  flat leakage={L_flat:.2f}  multilevel leakage={L_ml:.2f}  "
          f"({100*(L_flat-L_ml)/L_flat:+.1f}%)")


def test_multilevel_respects_corridor():
    eps = 0.05
    X = _blobs()
    B, _ = nystrom_features(X, rank=64, seed=0)
    lloyd_seed, _ = _flat(B, [8, 2], seed=0, eps=eps)
    ml = multilevel_split(B, [8, 2], epsilon=eps, seed=0, seed_labels=lloyd_seed)
    assert set(np.unique(ml)) <= {0, 1}
    assert realized_imbalance(ml, [8, 2]) <= eps + 1e-6, realized_imbalance(ml, [8, 2])


def test_multilevel_no_seed_runs():
    # runs without a seed split too (pure multilevel candidate), still corridor-valid
    X = _blobs(n=800, k=4)
    B, _ = nystrom_features(X, rank=48, seed=1)
    ml = multilevel_split(B, [7, 3], epsilon=0.05, seed=1)
    assert realized_imbalance(ml, [7, 3]) <= 0.05 + 1e-6


def test_multilevel_k3_fallback():
    # k>2 falls back to flat weighted FM from seed == monotone, corridor-valid
    X = _blobs(n=900, k=6)
    B, _ = nystrom_features(X, rank=48, seed=2)
    seed = balanced_lloyd(B, [6, 2, 2], seed=2)
    L0 = factor_leakage(B, seed, 3)
    ml = multilevel_split(B, [6, 2, 2], epsilon=0.05, seed=2, seed_labels=seed)
    assert factor_leakage(B, ml, 3) <= L0 * (1 + 1e-4)
    assert realized_imbalance(ml, [6, 2, 2]) <= 0.05 + 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} multilevel tests passed")
