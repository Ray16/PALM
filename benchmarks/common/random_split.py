"""Random 80/20 split makers — the one copy.

Two shapes are used across the benchmarks: an integer-label maker (0=train,
1=test) that also reports the realized test fraction, and a name-list maker
("train"/"test") for the reaction n-D scorers.
"""

from __future__ import annotations

import random

import numpy as np


def random_labels(n, seed=0, test_frac=0.2):
    """(labels[n] int64 with 1=test, realized_test_fraction). Seeded permutation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    q = int(round(n * test_frac))
    labels = np.zeros(n, dtype=np.int64)
    labels[idx[:q]] = 1
    return labels, q / n


def random_split_names(n, seed=42, test_frac=0.2):
    """List of "train"/"test" of length ``n`` (last ``test_frac`` of a shuffle)."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int((1.0 - test_frac) * n)
    lab = ["train"] * n
    for i in idx[cut:]:
        lab[i] = "test"
    return lab
