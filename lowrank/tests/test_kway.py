"""Tests for k>2 (multi-way train/val/test) support (Direction #3).

Covers the new ``corridor_assign`` k>2 branch (corridor-valid sizes, no worse than
the exact peel, exact reduction on a degenerate corridor) and that the hardness dial
preserves exact block sizes for a 3-way split at any alpha.

Run: `python -m PALM.lowrank.tests.test_kway` (from the PALM parent dir).
"""

from __future__ import annotations

import numpy as np

from PALM.splitters import SplitSpec, split
from PALM.splitters.common.balanced_assignment import (balanced_assign,
                                                       capacity_corridor, target_sizes)
from PALM.lowrank.optimize import corridor_assign


def _scores(n=300, k=3, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g)


def _sizes_of(lab, k):
    return [int((lab == c).sum().item()) for c in range(k)]


def test_corridor_kway_sizes_in_corridor():
    import torch
    n, splits = 300, [6, 2, 2]
    scores = _scores(n, len(splits))
    sizes = target_sizes(n, splits)
    caps, floors = capacity_corridor(n, splits, 0.2)
    lab = corridor_assign(scores, sizes, caps, floors)
    got = _sizes_of(lab, len(splits))
    assert sum(got) == n, got
    for c, s in enumerate(got):
        assert floors[c] <= s <= caps[c], (c, s, floors[c], caps[c])


def test_corridor_kway_not_worse_than_exact():
    # the corridor can only free points toward a preferred block -> >= exact score
    import torch
    n, splits = 400, [5, 3, 2]
    scores = _scores(n, len(splits), seed=1)
    sizes = target_sizes(n, splits)
    caps, floors = capacity_corridor(n, splits, 0.25)
    lab_c = corridor_assign(scores, sizes, caps, floors)
    lab_e = balanced_assign(scores, sizes)
    sc_c = float(scores.gather(1, lab_c[:, None]).sum())
    sc_e = float(scores.gather(1, lab_e[:, None]).sum())
    assert sc_c >= sc_e - 1e-4, (sc_c, sc_e)


def test_corridor_kway_degenerate_reproduces_exact_sizes():
    # floors == caps == sizes -> the corridor collapses to exact target sizes
    import torch
    n, splits = 250, [5, 3, 2]
    scores = _scores(n, len(splits), seed=2)
    sizes = target_sizes(n, splits)
    tight = np.asarray(sizes)
    lab = corridor_assign(scores, sizes, tight, tight)
    assert _sizes_of(lab, len(splits)) == list(sizes), (_sizes_of(lab, len(splits)), list(sizes))


def _synth_feature_data(n=360, d=16, k=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3
    X = np.vstack([centers[c] + rng.normal(size=(n // k, d)) for c in range(k)]).astype("float32")
    return {f"m{i}": X[i] for i in range(X.shape[0])}


def test_hardness_kway_preserves_block_sizes():
    # The hardness dial permutes labels (interpolate_to_random), so it must leave every
    # block's size UNCHANGED for any alpha. (The absolute sizes are set by the optimizer
    # within spec.epsilon, not necessarily the exact target ratio — that's fine; the
    # invariant here is size-invariance across alpha, at k>2.)
    splits, names = [6, 2, 2], ["train", "val", "test"]
    data = _synth_feature_data()
    ids = list(data)
    base = None
    for alpha in (0.0, 0.5, 1.0):
        res = split("lowrank", data, SplitSpec(splits, names, seed=0), hardness=alpha)
        got = [sum(1 for i in ids if res.assignment[i] == nm) for nm in names]
        assert sum(got) == len(ids), got
        if base is None:
            base = got
        assert got == base, (alpha, got, base)


def test_balance_slack_kway_runs_and_stays_in_corridor():
    splits, names = [8, 1, 1], ["train", "val", "test"]
    data = _synth_feature_data(n=360)
    ids = list(data)
    slack = 0.2
    res = split("lowrank", data, SplitSpec(splits, names, seed=0), balance_slack=slack)
    n = len(ids)
    caps, floors = capacity_corridor(n, splits, slack)
    got = [sum(1 for i in ids if res.assignment[i] == nm) for nm in names]
    assert sum(got) == n
    for c, s in enumerate(got):
        assert floors[c] <= s <= caps[c], (c, s, floors[c], caps[c])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} k-way tests passed")
