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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} hypergraph package tests passed")
