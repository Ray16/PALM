"""Tests for the standalone PALM.hypergraph package.

Run: `python -m pytest PALM/hypergraph/tests/test_hypergraph.py` (from the PALM
parent), or `python -m PALM.hypergraph.tests.test_hypergraph` for a plain-assert
smoke run. Needs the `boltz-2` env (torch + mtkahypar + rdkit + sklearn).
"""

from __future__ import annotations

import numpy as np

from PALM.hypergraph import (GraphSplitter, HypergraphSplitter, NDInput,
                             build_knn_graph, build_knn_hyperedges,
                             partition_hypergraph)
from PALM.hypergraph.knn import knn_sklearn
from PALM.hypergraph.nd_splitter import _axis_overlap, _cluster_axis
from PALM.hypergraph.partition import (balance_corridor, connectivity_objective,
                                       imbalance_of, rebalance)
from PALM.splitters import SplitSpec, list_splitters, split


def _planted_fps(n=200, dim=128, n_clusters=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 2, size=(n_clusters, dim))
    X = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        c = centers[i % n_clusters].copy()
        flip = rng.random(dim) < 0.08
        c[flip] = 1 - c[flip]
        X[i] = c
    return X


def _nd_fixture():
    records = [{"rA": f"A{i%6}", "rB": f"B{i%5}"} for i in range(40)]
    def bit(s):
        g = np.random.default_rng(s); return (g.random(64) < 0.15).astype(np.float32)
    afm = {"rA": {f"A{i}": bit(100 + i) for i in range(6)},
           "rB": {f"B{i}": bit(200 + i) for i in range(5)}}
    return records, afm


def test_build_knn_hyperedges():
    X = _planted_fps()
    edges, weights = build_knn_hyperedges(X, k=10, metric="tanimoto")
    assert len(edges) == len(X) and len(weights) == len(X)
    assert all(len(e) == 11 for e in edges)          # {node} U 10 neighbours
    assert all(w >= 1 for w in weights)              # positive integer weights


def test_build_knn_graph():
    X = _planted_fps()
    edges, weights = build_knn_graph(X, k=10, metric="tanimoto")
    assert len(edges) == len(weights) and len(edges) > 0
    assert all(a < b for a, b in edges)              # symmetrized, a<b
    assert all(w >= 1 for w in weights)


def test_partition_balance():
    X = _planted_fps()
    edges, weights = build_knn_hyperedges(X, k=10, metric="tanimoto")
    block_of, km1, imbalance = partition_hypergraph(
        len(X), edges, weights, [8, 2], seed=0, preset="deterministic")
    lab = np.asarray(block_of)
    assert set(np.unique(lab)) <= {0, 1}
    assert abs((lab == 1).mean() - 0.2) < 0.1        # within the balance corridor
    assert km1 >= 0


def test_methods_registered():
    names = list_splitters()
    for m in ["hypergraph", "graph", "hypergraph_nd", "hypergraph_nd_knn"]:
        assert m in names, f"{m} not registered"


def test_hypergraph_splitter_runs():
    X = _planted_fps(150, seed=11)
    data = {i: X[i] for i in range(len(X))}
    res = split("hypergraph", data, SplitSpec([8, 2], ["train", "test"], seed=42),
                preset="deterministic")
    assert set(res.assignment.values()) <= {"train", "test"}
    assert res.diagnostics.get("leakage") is not None
    assert res.diagnostics.get("km1") is not None


def test_graph_splitter_runs():
    X = _planted_fps(150, seed=12)
    data = {i: X[i] for i in range(len(X))}
    res = split("graph", data, SplitSpec([8, 2], ["train", "test"], seed=42),
                preset="deterministic")
    assert set(res.assignment.values()) <= {"train", "test"}
    assert res.diagnostics.get("leakage") is not None
    assert res.diagnostics.get("cut") is not None


def test_nd_methods_run():
    records, afm = _nd_fixture()
    spec = SplitSpec([8, 2], ["train", "test"], seed=42)
    r1 = split("hypergraph_nd", (records, afm), spec, sim_threshold=1.0,
               preset="deterministic")
    r2 = split("hypergraph_nd_knn", NDInput(records, afm), spec, k=10,
               preset="deterministic")
    for r in (r1, r2):
        assert len(r.assignment) == 40
        assert set(r.assignment.values()) <= {"train", "test"}
        assert r.diagnostics["km1"] >= 0


def test_determinism():
    data = {i: v for i, v in enumerate(_planted_fps(150, seed=13))}
    spec = SplitSpec([8, 2], ["train", "test"], seed=7)
    a = split("hypergraph", data, spec, preset="deterministic").assignment
    b = split("hypergraph", data, spec, preset="deterministic").assignment
    assert a == b, "hypergraph deterministic preset not reproducible"


def test_knn_sklearn_drops_self_with_duplicates():
    rng = np.random.default_rng(3)
    base = (rng.random((10, 32)) < 0.4).astype(np.float32)
    X = np.vstack([base, base, base])                # every vector triplicated
    n = len(X)
    for metric in ("euclidean", "cosine", "tanimoto"):
        idx, sim = knn_sklearn(X, 4, metric)
        assert idx.shape == (n, 4) and sim.shape == (n, 4)
        assert not any(i in idx[i] for i in range(n)), f"{metric}: self kept"
        # both duplicates of row i must be among its neighbours, at sim 1
        for i in range(n):
            dups = {(i + 10) % n, (i + 20) % n}
            assert dups <= set(idx[i].tolist()), metric
        assert np.allclose(np.sort(sim, axis=1)[:, -2:], 1.0, atol=1e-6), metric


def test_zero_similarity_edges_dropped():
    X = np.eye(6, dtype=np.float32)                  # mutually orthogonal
    X = np.vstack([X, X])                            # each row has one duplicate
    edges, weights = build_knn_graph(X, k=3, metric="cosine", use_gpu=False)
    assert sorted(edges) == [(i, i + 6) for i in range(6)]
    assert all(w == 1000 for w in weights)           # sim 1 -> WEIGHT_SCALE, no +1
    hedges, hw = build_knn_hyperedges(X, k=3, metric="cosine", use_gpu=False)
    assert all(len(e) == 2 for e in hedges)          # zero-sim neighbours are not pins


def test_partition_respects_corridor_and_final_diagnostics():
    rng = np.random.default_rng(0)
    for n in (20, 23, 37, 101):
        for splits in ([8, 2], [8, 1, 1], [7, 2, 1]):
            X = (rng.random((n, 64)) < 0.2).astype(np.float32)
            E, W = build_knn_hyperedges(X, k=5, metric="tanimoto", use_gpu=False)
            b, km1, imb, info = partition_hypergraph(
                n, E, W, splits, seed=0, preset="deterministic", return_info=True)
            lab = np.asarray(b)
            lo, hi = balance_corridor(n, splits, 0.05)
            sizes = np.bincount(lab, minlength=len(splits))
            assert ((sizes >= lo) & (sizes <= hi)).all(), (n, splits, sizes, lo, hi)
            assert info["within_corridor"]
            assert km1 == connectivity_objective(E, W, lab)
            assert imb == imbalance_of(lab, splits)


def test_rebalance_fixes_both_sides():
    n, splits = 30, [7, 2, 1]
    lab = np.zeros(n, dtype=int); lab[:2] = 1        # block 0 over, 1 and 2 under
    E = [[i, i + 1] for i in range(n - 1)]
    W = [1] * len(E)
    out, moved = rebalance(n, E, W, lab, splits, 0.05)
    lo, hi = balance_corridor(n, splits, 0.05)
    sizes = np.bincount(out, minlength=3)
    assert ((sizes >= lo) & (sizes <= hi)).all() and moved > 0
    assert rebalance(n, E, W, out, splits, 0.05)[1] == 0     # idempotent


def test_graph_diagnostics_match_final_labels():
    X = _planted_fps(120, seed=5)
    data = {i: X[i] for i in range(len(X))}
    spec = SplitSpec([8, 2], ["train", "test"], seed=1)
    res = split("graph", data, spec, preset="deterministic", use_gpu=False)
    d = res.diagnostics
    names = {"train": 0, "test": 1}
    lab = np.array([names[res.assignment[i]] for i in range(len(X))])
    edges, weights = build_knn_graph(X, k=15, metric=d["metric"], use_gpu=False)
    assert d["cut"] == connectivity_objective(edges, weights, lab)
    assert d["imbalance"] == round(imbalance_of(lab, spec.splits), 4)
    assert d["within_corridor"]


def test_axis_overlap_pairwise():
    records = [{"a": v} for v in ["x", "x", "y", "y", "z", "w"]]
    assignment = {0: "train", 1: "val", 2: "train", 3: "test", 4: "val", 5: "test"}
    ov = _axis_overlap(records, ["a"], assignment, ["train", "val", "test"])["a"]
    assert ov["n_values"] == 4
    assert ov["shared_across_splits"] == 2           # x (train,val), y (train,test)
    assert ov["shared_all_splits"] == 0
    assert ov["pairwise_shared"] == {"train|val": 1, "train|test": 1, "val|test": 0}


def test_cluster_axis_euclidean_threshold_monotone():
    rng = np.random.default_rng(1)
    values = [f"v{i}" for i in range(30)]
    feat = {v: rng.normal(size=8) for v in values}
    n_clusters = [len(set(_cluster_axis(values, feat, t).values()))
                  for t in (0.05, 0.2, 0.4, 0.8)]
    assert n_clusters == sorted(n_clusters)          # stricter sim -> more clusters
    assert n_clusters[0] < n_clusters[-1]


def test_nd_knn_identity_fallback_and_coverage():
    records, afm = _nd_fixture()
    afm = {"rA": afm["rA"], "rB": {v: None for v in afm["rB"]}}   # rB featureless
    res = split("hypergraph_nd_knn", NDInput(records, afm),
                SplitSpec([8, 2], ["train", "test"], seed=42), k=10,
                preset="deterministic")
    d = res.diagnostics
    assert d["axis_feature_coverage"] == {"rA": 1.0, "rB": 0.0}
    assert d["n_hyperedges"] == 40 + 5               # rA k-NN + one per rB value


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} hypergraph package tests passed")
