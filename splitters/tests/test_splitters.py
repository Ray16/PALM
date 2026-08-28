"""Behavior + correctness tests for the PALM splitters package.

Runnable two ways (needs the ``boltz-2`` env: torch + mtkahypar + rdkit):

    pytest PALM/splitters/tests/test_splitters.py
    CUDA_VISIBLE_DEVICES=1 python PALM/splitters/tests/test_splitters.py

Covers: the registry/discovery surface, every registered method end-to-end on
synthetic data, the JSON tool round-trip, balance corridors, determinism, and the
low-rank correctness properties (Nyström recovery, objective exactness, FM
monotonicity, brute-force optimality) ported from the original suite.
"""

import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")

from PALM.splitters import (SplitSpec, describe_splitters, get_splitter,
                            list_splitters, split)
from PALM.lowrank import (balanced_lloyd, fm_polish,
                                            lowrank_leakage, nystrom_features)
from PALM.splitters.tool import describe_splitters_tool, run_split_tool


# ── fixtures ────────────────────────────────────────────────────────────────

def planted_fps(n, dim=256, n_clusters=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 2, size=(n_clusters, dim))
    X = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        c = centers[i % n_clusters].copy()
        flip = rng.random(dim) < 0.08
        c[flip] = 1 - c[flip]
        X[i] = c
    return X


def exact_cross_tanimoto(X, labels):
    X = X.astype(np.float64)
    card = X.sum(1)
    inter = X @ X.T
    union = card[:, None] + card[None, :] - inter
    sim = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(sim, 0.0)
    lab = np.asarray(labels)
    cross = (lab[:, None] != lab[None, :]).astype(float)
    return 0.5 * float((sim * cross).sum())


def nd_fixture():
    records = [{"rA": f"A{i%6}", "rB": f"B{i%5}"} for i in range(40)]
    def bit(s):
        g = np.random.default_rng(s); return (g.random(64) < 0.15).astype(np.float32)
    afm = {"rA": {f"A{i}": bit(100 + i) for i in range(6)},
           "rB": {f"B{i}": bit(200 + i) for i in range(5)}}
    return records, afm


# ── discovery surface ───────────────────────────────────────────────────────

def test_registry_and_describe():
    names = list_splitters()
    for expected in ["hypergraph", "graph", "lowrank", "hypergraph_nd",
                     "hypergraph_nd_knn", "datasail", "scaffold"]:
        assert expected in names, f"{expected} not registered"
    for d in describe_splitters():
        assert set(d) == {"name", "description", "arity", "params_schema"}
        assert d["arity"] in ("1d", "nd")
        assert d["params_schema"]["type"] == "object"
    # types resolve (not all "string")
    lr = next(d for d in describe_splitters() if d["name"] == "lowrank")
    assert lr["params_schema"]["properties"]["rank"]["type"] == "integer"
    print("[OK] registry + describe")


def test_each_1d_method():
    fd = {i: v for i, v in enumerate(planted_fps(120, seed=11))}
    spec = SplitSpec([8, 2], ["train", "test"], seed=42)
    for method, kw in [("hypergraph", dict(preset="deterministic")),
                       ("graph", dict(preset="deterministic")),
                       ("lowrank", dict(rank=64))]:
        r = split(method, fd, spec, **kw)
        assert len(r.assignment) == 120
        assert set(r.assignment.values()) <= {"train", "test"}
        f = r.diagnostics["split_fractions"]["test"]
        assert 0.15 <= f <= 0.25, f"{method} test frac {f} outside corridor"
        assert r.diagnostics["leakage"] is not None
    # scaffold (SMILES input)
    smiles = {k: s for k, s in enumerate(
        ["c1ccccc1", "c1ccccc1C", "C1CCCCC1", "C1CCCCC1C", "CCO", "CCCO",
         "c1ccncc1", "c1ccncc1C", "CCN", "CCCN"])}
    rs = split("scaffold", smiles, SplitSpec([8, 2], ["train", "test"]))
    assert set(rs.assignment.values()) <= {"train", "test"}
    print("[OK] each 1-D method (+ scaffold)")


def test_each_nd_method():
    records, afm = nd_fixture()
    spec = SplitSpec([8, 2], ["train", "test"], seed=42)
    for method in ["hypergraph_nd", "hypergraph_nd_knn"]:
        kw = dict(preset="deterministic")
        if method == "hypergraph_nd":
            kw["sim_threshold"] = 1.0
        else:
            kw["k"] = 10
        r = split(method, (records, afm), spec, **kw)
        assert len(r.assignment) == 40
        assert set(r.assignment.values()) <= {"train", "test"}
        assert r.diagnostics["km1"] >= 0
    print("[OK] each n-D method")


def test_tool_roundtrip():
    X = planted_fps(60, seed=12)
    out = run_split_tool("hypergraph", features={str(i): X[i].tolist() for i in range(60)},
                         splits=[8, 2], names=["train", "test"], seed=42,
                         params={"preset": "deterministic"})
    json.dumps(out)                                  # must serialize
    assert set(out) == {"method", "assignment", "diagnostics"}
    assert len(out["assignment"]) == 60
    json.dumps(describe_splitters_tool())
    print("[OK] tool JSON round-trip")


def test_determinism():
    fd = {i: v for i, v in enumerate(planted_fps(150, seed=13))}
    spec = SplitSpec([8, 2], ["train", "test"], seed=7)
    a = split("hypergraph", fd, spec, preset="deterministic").assignment
    b = split("hypergraph", fd, spec, preset="deterministic").assignment
    assert a == b, "hypergraph deterministic preset not reproducible"
    c = split("lowrank", fd, spec, rank=64).assignment
    d = split("lowrank", fd, spec, rank=64).assignment
    assert c == d, "lowrank not reproducible for a fixed seed"
    print("[OK] determinism (hypergraph deterministic preset + lowrank)")


# ── low-rank correctness (ported) ───────────────────────────────────────────

def test_factorization_recovers_similarity():
    X = planted_fps(400, seed=1)
    B, metric = nystrom_features(X, rank=256, seed=0)
    assert metric == "tanimoto"
    rng = np.random.default_rng(2)
    ii = rng.integers(0, 400, 3000); jj = rng.integers(0, 400, 3000)
    m = ii != jj; ii, jj = ii[m], jj[m]
    card = X.sum(1)
    inter = (X[ii] * X[jj]).sum(1)
    true = inter / (card[ii] + card[jj] - inter + 1e-9)
    approx = (B[ii] * B[jj]).sum(1)
    corr = np.corrcoef(true, approx)[0, 1]
    rmse = np.sqrt(np.mean((true - approx) ** 2))
    assert corr > 0.9 and rmse < 0.05, f"corr={corr} rmse={rmse}"
    print(f"[OK] Nyström recovers Tanimoto: corr={corr:.3f} rmse={rmse:.3f}")


def test_objective_matches_factor_space():
    X = planted_fps(300, seed=3)
    B, _ = nystrom_features(X, rank=200, seed=0)
    rng = np.random.default_rng(4)
    lab = np.array([0] * 240 + [1] * 60); rng.shuffle(lab)
    obj = lowrank_leakage(B, lab, 2)
    G = B @ B.T; np.fill_diagonal(G, 0.0)
    cross = (lab[:, None] != lab[None, :]).astype(float)
    exact = 0.5 * float((G * cross).sum())
    assert abs(obj - exact) / max(abs(exact), 1.0) < 1e-4
    print(f"[OK] objective == factor-space cross: {obj:.2f} vs {exact:.2f}")


def test_lloyd_balance():
    X = planted_fps(1000, seed=5)
    B, _ = nystrom_features(X, rank=256, seed=0)
    lab = balanced_lloyd(B, [8, 2], epsilon=0.05, seed=0)
    sizes = np.bincount(lab, minlength=2)
    assert sizes.tolist() == [800, 200], sizes.tolist()
    print(f"[OK] Lloyd exact balance: {sizes.tolist()}")


def test_fm_monotone_and_balanced():
    X = planted_fps(600, seed=6)
    B, _ = nystrom_features(X, rank=256, seed=0)
    rng = np.random.default_rng(7)
    lab0 = np.array([0] * 480 + [1] * 120); rng.shuffle(lab0)
    L0 = lowrank_leakage(B, lab0, 2)
    lab1, moves = fm_polish(B, lab0.copy(), [8, 2], epsilon=0.05)
    L1 = lowrank_leakage(B, lab1, 2)
    sizes = np.bincount(lab1, minlength=2)
    assert L1 <= L0 + 1e-4, f"FM not monotone: {L0}->{L1}"
    assert 114 <= sizes[1] <= 126, sizes.tolist()
    print(f"[OK] FM monotone: {L0:.1f}->{L1:.1f} in {moves} moves")


def test_vs_bruteforce_tiny():
    X = planted_fps(12, dim=64, n_clusters=2, seed=8)
    best = np.inf
    for combo in itertools.combinations(range(1, 12), 6):
        lab = np.zeros(12, dtype=int); lab[list(combo)] = 1
        best = min(best, exact_cross_tanimoto(X, lab))
    B, _ = nystrom_features(X, rank=12, seed=0)
    fm_best = np.inf
    for s in range(20):
        lab = balanced_lloyd(B, [1, 1], epsilon=0.0, seed=s)
        lab, _ = fm_polish(B, lab, [1, 1], epsilon=0.0)
        assert np.bincount(lab, minlength=2).tolist() == [6, 6]
        fm_best = min(fm_best, exact_cross_tanimoto(X, lab))
    assert abs(fm_best - best) < 1e-6, f"{fm_best} != {best}"
    print(f"[OK] reaches brute-force optimum: {fm_best:.4f}")


def test_end_to_end_beats_random():
    X = planted_fps(800, seed=9)
    fd = {i: X[i] for i in range(800)}
    rng = np.random.default_rng(0)
    r = np.array([0] * 640 + [1] * 160); rng.shuffle(r)
    L_rand = exact_cross_tanimoto(X, r)
    res = split("lowrank", fd, SplitSpec([8, 2], ["train", "test"], seed=0), rank=256, n_restarts=4)
    lab = np.array([0 if res.assignment[i] == "train" else 1 for i in range(800)])
    L_lr = exact_cross_tanimoto(X, lab)
    n_test = int((lab == 1).sum())
    assert L_lr < L_rand, f"{L_lr} !< {L_rand}"
    assert 152 <= n_test <= 168, n_test
    print(f"[OK] lowrank beats random: {L_lr:.1f} < {L_rand:.1f}")


ALL = [test_registry_and_describe, test_each_1d_method, test_each_nd_method,
       test_tool_roundtrip, test_determinism, test_factorization_recovers_similarity,
       test_objective_matches_factor_space, test_lloyd_balance,
       test_fm_monotone_and_balanced, test_vs_bruteforce_tiny,
       test_end_to_end_beats_random]

if __name__ == "__main__":
    for t in ALL:
        t()
    print("\nALL SPLITTER TESTS PASSED")
