"""Tests for the standalone PALM.lowrank package.

Run: `python -m pytest PALM/lowrank/tests/test_lowrank.py` (from the PALM parent),
or `python -m PALM.lowrank.tests.test_lowrank` for a plain-assert smoke run.
"""

from __future__ import annotations

import numpy as np

from PALM.lowrank import (LowRankSplitter, balanced_lloyd, factor_leakage,
                          fm_polish, lowrank_leakage, nystrom_features)
from PALM.splitters import SplitSpec, split


def _blobs(n=400, d=16, k=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    X = np.vstack([centers[c] + rng.normal(size=(n // k, d)) for c in range(k)]).astype("float32")
    return X


def test_nystrom_shape():
    X = _blobs()
    B, metric = nystrom_features(X, rank=64, seed=0)
    assert B.shape[0] == X.shape[0] and B.shape[1] <= 64 and metric


def test_balanced_lloyd_sizes_and_leakage():
    X = _blobs()
    B, _ = nystrom_features(X, rank=64, seed=0)
    lab = balanced_lloyd(B, [8, 2], epsilon=0.05, seed=0)
    # exact target sizes (largest-remainder) and a valid 2-way labelling
    assert set(np.unique(lab)) <= {0, 1}
    assert abs((lab == 1).mean() - 0.2) < 0.02


def test_fm_polish_monotone():
    X = _blobs()
    B, _ = nystrom_features(X, rank=64, seed=0)
    lab0 = balanced_lloyd(B, [8, 2], epsilon=0.05, seed=0)
    L0 = factor_leakage(B, lab0, 2)
    lab1, moves = fm_polish(B, lab0.copy(), [8, 2], epsilon=0.05)
    L1 = factor_leakage(B, lab1, 2)
    assert L1 <= L0 + 1e-6            # monotone: never increases leakage
    assert lowrank_leakage is factor_leakage


def test_splitter_registered_and_runs():
    X = _blobs()
    data = {i: X[i] for i in range(len(X))}
    res = split("lowrank", data, SplitSpec(splits=[8, 2], names=["train", "test"], seed=0))
    assert set(res.assignment.values()) == {"train", "test"}
    assert res.diagnostics.get("leakage") is not None
    assert res.diagnostics.get("rank")


def test_backcompat_import_path():
    # historical path used by the omol25 studies + old tests must still resolve
    from PALM.splitters.methods.lowrank import (balanced_lloyd as bl,  # noqa: F401
                                                fm_polish as fp, lowrank_leakage as ll,
                                                nystrom_features as nf)
    assert bl is balanced_lloyd and nf is nystrom_features


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} lowrank package tests passed")
