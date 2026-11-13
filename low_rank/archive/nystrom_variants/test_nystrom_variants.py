"""ARCHIVED tests for the negative-result Nyström variants (docs/FINDINGS.md Step 2).

These variants (leverage-score landmarks, ridge regularization, adaptive energy-rank)
were removed from the live ``lowrank.nystrom`` module for negligible leakage benefit.
They are preserved here against the pre-strip snapshot ``nystrom_full`` so the study
stays reproducible. Run standalone:  ``python archive/nystrom_variants/test_nystrom_variants.py``
"""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))      # low_rank/ (for splitters)
sys.path.insert(0, str(_HERE))                 # local `nystrom_full` snapshot

import numpy as np

from nystrom_full import nystrom_features


def _blobs(n=1200, d=24, k=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    X = np.vstack([centers[c] + rng.normal(size=(n // k, d)) for c in range(k)]).astype("float32")
    return X


def test_nystrom_landmark_variants_and_ridge():
    X = _blobs()
    for lm in ("kmeans++", "uniform", "leverage"):
        B, _ = nystrom_features(X, rank=64, landmark=lm, seed=0)
        assert B.shape == (len(X), 64) and np.isfinite(B).all(), lm
    Bd, _ = nystrom_features(X, rank=64, landmark="kmeans++", seed=0, ridge=0.01)
    assert Bd.shape == (len(X), 64) and np.isfinite(Bd).all()


def test_nystrom_adaptive_rank():
    # blobs are ~k-dimensional: energy truncation should pick far fewer than rank cols
    X = _blobs(n=400, d=16, k=4)
    B, _ = nystrom_features(X, rank=128, energy=0.95, seed=0)
    assert B.shape[0] == len(X) and B.shape[1] < 128


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n== {len(fns)} archived nystrom-variant tests passed")
