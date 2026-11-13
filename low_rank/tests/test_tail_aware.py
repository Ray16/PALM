"""Tests for the tail-aware low-rank extensions: kernel_power and hard must-link groups."""

from __future__ import annotations

import numpy as np

from lowrank import nystrom_features
from lowrank.grouped import grouped_split, must_link_groups
from lowrank.splitters import SplitSpec, split
from lowrank.splitters.common.balanced_assignment import capacity_corridor


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


def test_kernel_power_factorizes_elementwise_power():
    X = _fps(n_groups=10, n_bg=40)               # n=90 <= rank -> Nystrom is exact
    B, _ = nystrom_features(X, rank=128, metric="tanimoto", kernel_power=3)
    S3 = _tanimoto(X) ** 3
    assert np.abs(B @ B.T - S3).max() < 1e-3


def test_must_link_groups_respect_cap_and_threshold():
    X = _fps()
    gid = must_link_groups(X, "tanimoto", tau=0.6, max_group=4)
    assert np.bincount(gid).max() <= 4
    # groups only ever join pairs that are linked by a >= tau chain
    S = _tanimoto(X)
    np.fill_diagonal(S, 0)
    for g in np.unique(gid):
        m = np.nonzero(gid == g)[0]
        if len(m) > 1:
            assert (S[np.ix_(m, m)].max(1) >= 0.6).all()


def test_hard_must_link_never_cuts_a_group_and_hits_corridor():
    X = _fps()
    n = len(X)
    B, _ = nystrom_features(X, rank=64, metric="tanimoto", kernel_power=3)
    gid = must_link_groups(X, "tanimoto", tau=0.6, max_group=10)
    labels, info = grouped_split(B, gid, [8, 1, 1], epsilon=0.02, seed=0)
    caps, floors = capacity_corridor(n, [8, 1, 1], 0.02)
    sz = np.bincount(labels, minlength=3)
    assert ((sz <= caps) & (sz >= floors)).all() and info["corridor_ok"]
    if info["broken_groups"] == 0:
        for g in np.unique(gid):
            assert len(set(labels[gid == g])) == 1


def test_splitter_tail_options_end_to_end():
    X = _fps()
    fd = {i: X[i] for i in range(len(X))}
    spec = SplitSpec(splits=[8, 1, 1], names=["train", "val", "test"], seed=0, epsilon=0.02)
    S = _tanimoto(X)
    np.fill_diagonal(S, 0)
    out = {}
    for name, kw in {"plain": {}, "p3_hard": dict(kernel_power=3, must_link="auto")}.items():
        res = split("lowrank", fd, spec, allowed_size_imbalance=0.02, **kw)
        lab = np.array([res.assignment[i] for i in range(len(X))])
        cross = lab[:, None] != lab[None, :]
        out[name] = int(((S >= 0.6) & cross).sum())       # strong pairs cut by the split
    assert out["p3_hard"] <= out["plain"]
    assert out["p3_hard"] == 0


def test_freeze_cap_rule_stops_a_capped_group_from_absorbing_weak_neighbours():
    # edges: (0,1)=.9, (2,3)=.85, (1,2)=.8 would make a group of 4 > cap 3, (0,4)=.5
    idx = np.array([[1], [2], [3], [2], [0]])
    sim = np.array([[0.9], [0.8], [0.85], [0.85], [0.5]])
    X = np.zeros((5, 1))
    skip = must_link_groups(X, "tanimoto", 0.4, max_group=3, nbrs=(idx, sim), cap_rule="skip")
    freeze = must_link_groups(X, "tanimoto", 0.4, max_group=3, nbrs=(idx, sim), cap_rule="freeze")
    assert skip[4] == skip[0]                    # skip: the 0.5 edge fills the capped group
    assert freeze[4] != freeze[0]                # freeze: {0,1} stopped growing at 0.8
    assert freeze[0] == freeze[1] and freeze[2] == freeze[3] and freeze[1] != freeze[2]
