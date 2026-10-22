"""Tests for the standalone lowrank package.

Run: `python -m pytest tests/test_lowrank.py`
or `python -m lowrank.tests.test_lowrank` for a plain-assert smoke run.
"""

from __future__ import annotations

import numpy as np

from lowrank import (LowRankSplitter, balanced_lloyd, factor_leakage,
                          fm_polish, interpolate_to_random, lowrank_leakage,
                          nystrom_features, realized_imbalance)
from splitters import SplitSpec, split


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


def test_nystrom_landmark_variants():
    # Supported landmark strategies on the live module (kmeans++ / uniform).
    # The tight/adaptive variants (leverage / ridge / energy) were a negative result
    # and now live under archive/nystrom_variants/ (docs/FINDINGS.md Step 2).
    X = _blobs()
    for lm in ("kmeans++", "uniform"):
        B, _ = nystrom_features(X, rank=64, landmark=lm, seed=0)
        assert B.shape == (len(X), 64) and np.isfinite(B).all(), lm


def test_balance_slack_tradeoff():
    # opening the balance corridor should not INCREASE leakage vs exact sizes,
    # and should keep imbalance within the requested slack.
    X = _blobs()
    B, _ = nystrom_features(X, rank=64, seed=0)
    lab_exact = balanced_lloyd(B, [8, 2], seed=0, balance_slack=0.0)
    lab_slack = balanced_lloyd(B, [8, 2], seed=0, balance_slack=0.20)
    assert abs((lab_exact == 1).mean() - 0.2) < 0.02          # exact stays exact
    assert realized_imbalance(lab_slack, [8, 2]) <= 0.20 + 1e-6
    # more freedom -> leakage no worse (usually strictly lower)
    assert factor_leakage(B, lab_slack, 2) <= factor_leakage(B, lab_exact, 2) + 1e-6


def test_hardness_dial_monotone():
    # leakage should rise monotonically as hardness falls (split -> random),
    # while block sizes stay exact (permutation preserves them).
    X = _blobs()
    B, _ = nystrom_features(X, rank=64, seed=0)
    hard = balanced_lloyd(B, [8, 2], seed=0)
    prev = None
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):
        lab = interpolate_to_random(hard, alpha, seed=0)
        assert (lab == 1).sum() == (hard == 1).sum()           # sizes preserved exactly
        L = factor_leakage(B, lab, 2)
        if prev is not None:
            assert L >= prev - 1e-6                              # lower alpha -> more leakage
        prev = L


def test_splitter_registered_and_runs():
    X = _blobs()
    data = {i: X[i] for i in range(len(X))}
    res = split("lowrank", data, SplitSpec(splits=[8, 2], names=["train", "test"], seed=0))
    assert set(res.assignment.values()) == {"train", "test"}
    assert res.diagnostics.get("leakage") is not None
    assert res.diagnostics.get("rank")


def test_allowed_size_imbalance_alias():
    X = _blobs(n=200)
    data = {i: X[i] for i in range(len(X))}
    spec = SplitSpec(splits=[8, 2], names=["train", "test"], seed=0)
    clear = split("lowrank", data, spec, allowed_size_imbalance=0.20)
    legacy = split("lowrank", data, spec, balance_slack=0.20)
    assert clear.assignment == legacy.assignment


def test_allowed_size_imbalance_validation():
    X = _blobs(n=40)
    data = {i: X[i] for i in range(len(X))}
    spec = SplitSpec(splits=[8, 2], names=["train", "test"], seed=0)
    for params, message in [
        ({"allowed_size_imbalance": 1.0}, "must be in"),
        ({"allowed_size_imbalance": 0.1, "balance_slack": 0.2}, "must match"),
    ]:
        try:
            split("lowrank", data, spec, **params)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {params}")


def test_public_api_exports():
    # the standalone package exposes the full public API from one import
    from lowrank import (balanced_lloyd as bl,  # noqa: F401
                             fm_polish as fp, lowrank_leakage as ll,
                             nystrom_features as nf)
    assert bl is balanced_lloyd and nf is nystrom_features and ll is factor_leakage


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} lowrank package tests passed")
