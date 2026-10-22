"""Tests for the near-duplicate ("tail") leakage diagnostic (eval/tail_report.py)."""
from __future__ import annotations

import numpy as np

from eval.tail_report import format_report, tail_report


def _disjoint_binary():
    """Train uses bits [0:64], test uses bits [64:128] -> zero cross Tanimoto.

    128-dim binary rows so `choose_metric` picks tanimoto.
    """
    rng = np.random.default_rng(0)
    fd, asn = {}, {}
    for k in range(10):
        v = np.zeros(128, dtype=np.float32)
        v[rng.choice(64, 12, replace=False)] = 1.0
        fd[f"a{k}"] = v
        asn[f"a{k}"] = "train"
    for k in range(4):
        v = np.zeros(128, dtype=np.float32)
        v[64 + rng.choice(64, 12, replace=False)] = 1.0
        fd[f"t{k}"] = v
        asn[f"t{k}"] = "test"
    return fd, asn


def test_no_cross_near_twins():
    fd, asn = _disjoint_binary()
    rep = tail_report(fd, asn, tau=0.5)
    assert rep["metric"] == "tanimoto"
    assert rep["n_cross_pairs"] == 0
    assert rep["held_out_contam_frac"] == 0.0
    assert rep["max_cross_sim"] == 0.0
    assert rep["splits"] == {"test": 4, "train": 10}


def test_detects_a_planted_duplicate():
    fd, asn = _disjoint_binary()
    fd["t0"] = fd["a0"].copy()                 # exact duplicate of a train row -> sim 1.0
    rep = tail_report(fd, asn, tau=0.7)
    assert rep["n_cross_pairs"] >= 1
    assert rep["max_cross_sim"] > 0.99
    # exactly one of four test entities is contaminated
    assert abs(rep["held_out_contam_frac"] - 0.25) < 1e-6
    assert rep["contam_frac"]["train"] > 0.0    # the train twin is contaminated too


def test_accepts_splitresult_like_object():
    fd, asn = _disjoint_binary()
    fd["t1"] = fd["a1"].copy()

    class _R:                                   # duck-typed SplitResult
        assignment = asn
    rep = tail_report(fd, _R(), tau=0.7)
    assert rep["held_out_contam_frac"] == 0.25


def test_chunking_is_consistent():
    fd, asn = _disjoint_binary()
    fd["t2"] = fd["a2"].copy()
    small = tail_report(fd, asn, tau=0.6, chunk=2)
    big = tail_report(fd, asn, tau=0.6, chunk=1000)
    assert small["n_cross_pairs"] == big["n_cross_pairs"]
    assert small["held_out_contam_frac"] == big["held_out_contam_frac"]
    assert small["max_cross_sim"] == big["max_cross_sim"]


def test_missing_held_out_is_graceful():
    fd, asn = _disjoint_binary()
    rep = tail_report(fd, asn, tau=0.7, held_out="valid")   # no such split
    assert rep["held_out_contam_frac"] is None
    assert rep["held_out_mean_nn_cross_sim"] is None
    assert isinstance(format_report(rep), str)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n== {len(fns)} tail_report tests passed")
