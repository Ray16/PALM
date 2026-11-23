"""Tests for parallelization, methodological, and scaling improvements."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

from lowrank import (
    LowRankSplitter,
    SplitSpec,
    balanced_lloyd,
    compare_candidate_cuts,
    corridor_assign,
    factor_leakage,
    fm_polish,
    interpolate_groups_to_random,
    nystrom_features,
    split,
)
from lowrank.eval.tail_report import tail_report
from lowrank.factors import FactorOps
from lowrank.grouped import _repair, must_link_groups, topk_neighbours
from lowrank.optimize import _batched_balanced_lloyd, _peel_corridor_order, optimize_restarts
from lowrank.splitters.common.balanced_assignment import (
    _peel_exact_order,
    balanced_assign,
    capacity_corridor,
    target_sizes,
)
from lowrank.splitters.common.feature_preparation import feature_matrix_from_dict
from lowrank.target_gap import split_for_gap


def test_direct_array_and_tuple_feature_inputs_match_dict():
    rng = np.random.default_rng(17)
    X = (rng.random((60, 128)) < 0.25).astype(np.float32)
    ids = [f"mol_{i:03d}" for i in range(len(X))]
    as_dict = dict(zip(ids, X))

    spec = SplitSpec([8, 1, 1], ["train", "val", "test"], seed=2)
    r_dict = split("lowrank", as_dict, spec, rank=16, leakage_diagnostic="none")
    r_tuple = split("lowrank", (ids, X), spec, rank=16, leakage_diagnostic="none")
    r_array = split("lowrank", X, spec, rank=16, leakage_diagnostic="none")

    assert r_dict.assignment == r_tuple.assignment
    assert [r_array.assignment[i] for i in range(len(X))] == [
        r_dict.assignment[k] for k in ids
    ]
    rand_dict = split("random", as_dict, spec)
    rand_tuple = split("random", (ids, X), spec)
    assert rand_dict.assignment == rand_tuple.assignment
    with pytest.raises(ValueError, match="unique"):
        feature_matrix_from_dict((["a", "a"], np.ones((2, 4), dtype=np.float32)))


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("slack", [0.0, 0.15])
def test_batched_balanced_lloyd_matches_sequential_restarts(as_sparse, weighted, slack):
    rng = np.random.default_rng(23)
    B = rng.random((45, 9)).astype(np.float32)
    B[B < 0.4] = 0.0
    F = sparse.csr_matrix(B) if as_sparse else B
    ops = FactorOps(F)
    W = [[0, 2, 0.5], [2, 0, 1], [0.5, 1, 0]] if weighted else None

    batched = _batched_balanced_lloyd(
        ops,
        [7, 2, 1],
        n_iter=15,
        seed=5,
        n_restarts=4,
        balance_slack=slack,
        fold_weights=W,
    )
    for r, (cost, r_seed, lab) in enumerate(batched):
        seq_lab = balanced_lloyd(
            ops,
            [7, 2, 1],
            n_iter=15,
            seed=5 + r,
            balance_slack=slack,
            fold_weights=W,
        )
        assert r_seed == 5 + r
        assert np.array_equal(lab, seq_lab)
        assert cost == pytest.approx(ops.cost(seq_lab, 3, W), abs=1e-7)


def test_kmeans_parallel_and_power_landmarks_produce_valid_factors():
    rng = np.random.default_rng(11)
    X = (rng.random((120, 128)) < 0.2).astype(np.float32)
    for lm in ("kmeans++", "kmeans||"):
        B, metric, info = nystrom_features(
            X,
            rank=32,
            metric="tanimoto",
            landmark=lm,
            kernel_power=3,
            power_landmarks=True,
            return_info=True,
        )
        assert metric == "tanimoto"
        assert B.shape == (120, 32) and np.isfinite(B).all()
        assert len(set(info["landmark_indices"])) == 32


def test_cut_relative_error_rank_check_rejects_bad_cut_and_accepts_good_cut():
    a = np.zeros(16, dtype=np.float32)
    a[:8] = 1.0
    b = np.zeros(16, dtype=np.float32)
    b[7:15] = 1.0
    X = np.vstack([a] * 10 + [b] * 10)
    data = dict(enumerate(X))
    spec = SplitSpec([1, 1], ["train", "test"], seed=0)
    res = split(
        "lowrank",
        data,
        spec,
        rank="auto",
        rank_min=2,
        rank_max=8,
        metric="tanimoto",
        rank_check="cut_relative_error",
        cut_rel_tol=0.1,
        cut_abs_floor=1e-3,
        on_rank_failure="raise",
        leakage_diagnostic="none",
    )
    f_info = res.diagnostics["factorization"]
    assert f_info["selection_tolerance_met"] and f_info["audit_tolerance_met"]
    assert f_info["audit"]["acceptance"]["accepted"] is True


def test_minority_first_peel_never_scores_below_forward_peel():
    g = torch.Generator().manual_seed(42)
    scores = torch.randn(300, 3, generator=g)
    splits = [8, 1, 1]
    sizes = target_sizes(300, splits)
    caps, floors = capacity_corridor(300, splits, 0.15)

    lab_best = balanced_assign(scores, sizes)
    lab_fwd = _peel_exact_order(scores, sizes, [0, 1, 2])
    assert float(scores.gather(1, lab_best[:, None]).sum()) >= float(
        scores.gather(1, lab_fwd[:, None]).sum()
    ) - 1e-6

    cor_best = corridor_assign(scores, sizes, caps, floors)
    cor_fwd = _peel_corridor_order(scores, caps, floors, [0, 1, 2])
    assert float(scores.gather(1, cor_best[:, None]).sum()) >= float(
        scores.gather(1, cor_fwd[:, None]).sum()
    ) - 1e-6


def test_spectral_and_hybrid_init_break_degree_sorting_trap():
    # Construct two tight clusters plus a high-degree bridge hub so random 80/10/10
    # starts collapse onto the degree vector while spectral cuts cleanly across clusters.
    rng = np.random.default_rng(7)
    c1 = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32) + 0.02 * rng.normal(
        size=(40, 4)
    ).astype(np.float32)
    c2 = np.array([0.1, 1.0, 0.0, 0.0], dtype=np.float32) + 0.02 * rng.normal(
        size=(40, 4)
    ).astype(np.float32)
    c3 = np.array([0.6, 0.6, 0.2, 0.0], dtype=np.float32) + 0.02 * rng.normal(
        size=(20, 4)
    ).astype(np.float32)
    B = np.clip(np.vstack([c1, c2, c3]), 0.0, None)

    _, cost_rand, _ = optimize_restarts(
        B, [8, 1, 1], seed=0, n_restarts=4, fm=False, init="random"
    )
    _, cost_hyb, info_hyb = optimize_restarts(
        B, [8, 1, 1], seed=0, n_restarts=4, fm=False, init="hybrid"
    )
    _, cost_spec, _ = optimize_restarts(
        B, [8, 1, 1], seed=0, n_restarts=4, fm=False, init="spectral"
    )
    assert info_hyb["init"] == "hybrid"
    assert cost_hyb <= cost_rand + 1e-6
    assert min(cost_hyb, cost_spec) <= cost_rand + 1e-6


def test_stratified_splitting_preserves_class_prevalence_and_reduces_leakage():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(100, 12)).astype(np.float32)
    # Imbalanced binary class labels: 80 negatives (0), 20 positives (1)
    y = np.array([0] * 80 + [1] * 20)
    rng.shuffle(y)
    data = {i: X[i] for i in range(100)}
    strata_map = {i: int(y[i]) for i in range(100)}
    spec = SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0)

    res = split(
        "lowrank",
        data,
        spec,
        rank=16,
        strata=strata_map,
        leakage_diagnostic="exact",
    )
    assert res.diagnostics["split_counts"] == {"train": 80, "val": 10, "test": 10}
    pos_counts = {
        fold: sum(
            1 for i in range(100) if res.assignment[i] == fold and strata_map[i] == 1
        )
        for fold in ("train", "val", "test")
    }
    # 20% positive prevalence preserved in every fold (16 in train, 2 in val, 2 in test)
    assert pos_counts == {"train": 16, "val": 2, "test": 2}


def test_centroid_linkage_prevents_single_edge_bridge_chaining():
    # Two separate triangles (0,1,2) and (3,4,5) linked only by one bridge edge (2,3)
    B = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.95, 0.3],
            [0.3, 0.95],
            [0.1, 0.99],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    idx = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 3],
            [4, 2],
            [5, 3],
            [4, 3],
        ]
    )
    sim = np.array(
        [
            [0.95, 0.90],
            [0.95, 0.92],
            [0.90, 0.65],
            [0.92, 0.65],
            [0.95, 0.92],
            [0.95, 0.90],
        ],
        dtype=np.float32,
    )
    g_single = must_link_groups(
        B, "cosine", tau=0.6, max_group=6, nbrs=(idx, sim), linkage="single"
    )
    g_centroid = must_link_groups(
        B, "cosine", tau=0.6, max_group=6, nbrs=(idx, sim), B=B, linkage="centroid"
    )
    assert len(np.unique(g_single)) == 1
    assert len(np.unique(g_centroid)) == 2
    assert g_centroid[0] == g_centroid[1] == g_centroid[2]
    assert g_centroid[3] == g_centroid[4] == g_centroid[5]


def test_group_bisect_repair_and_group_hardness_interpolation():
    B = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.96, 0.08],
            [0.1, 0.95],
            [0.0, 1.0],
            [0.05, 0.98],
            [-1.0, 0.0],
            [-0.95, 0.1],
        ],
        dtype=np.float64,
    )
    members = [[0, 1, 2, 3], [4, 5], [6, 7]]
    G = np.stack([B[p].sum(0) for p in members])
    w = np.array([4, 2, 2], dtype=np.int64)
    lab = np.array([0, 0, 1], dtype=np.int64)
    # Target capacities [5, 3]: block 0 has 6 points (4+2), block 1 has 2.
    # No whole group of size 1 exists, so group of size 2 or 4 must split; bisect
    # keeps coherent sub-groups when a group of size >2 is split.
    G_out, w_out, lab_out, mem_out, broken = _repair(
        B,
        G,
        w,
        lab,
        2,
        np.array([5, 3]),
        np.array([5, 3]),
        members,
        break_mode="bisect",
    )
    assert broken == 1
    assert np.bincount(lab_out, weights=w_out, minlength=2).tolist() == [5, 3]

    # Group-preserving hardness interpolation keeps intact groups together
    gid = np.repeat(np.arange(10), 2)
    base_labels = np.repeat([0, 1], [16, 4])
    interp = interpolate_groups_to_random(
        base_labels, gid, alpha=0.5, splits=[8, 2], epsilon=0.0, seed=3
    )
    assert np.bincount(interp, minlength=2).tolist() == [16, 4]
    for g in range(10):
        assert len(set(interp[gid == g])) == 1


def test_tail_report_standardizes_euclidean_and_split_for_gap_avoids_seed_overlap(
    monkeypatch,
):
    # Scale one column by 1e6; standardized euclidean tail_report must be scale-invariant.
    rng = np.random.default_rng(5)
    X1 = rng.normal(size=(20, 4)).astype(np.float32)
    X1[10] = X1[0]
    X2 = X1.copy()
    X2[:, 0] *= 1e4
    asn = {i: ("train" if i < 10 else "test") for i in range(20)}
    r1 = tail_report(dict(enumerate(X1)), asn, tau=0.8, metric="euclidean")
    r2 = tail_report(dict(enumerate(X2)), asn, tau=0.8, metric="euclidean")
    assert r1["n_cross_pairs"] == r2["n_cross_pairs"] >= 1
    assert r1["max_cross_sim"] == pytest.approx(r2["max_cross_sim"], abs=1e-4)

    # split_for_gap must not reuse the evaluation seed inside calibrate_gap
    import lowrank.target_gap as tg

    seen_cal_seeds = []

    def fake_calibrate(fd, X, y, task_type, alphas, seeds, splits, names):
        seen_cal_seeds.extend(seeds)
        return tg.GapCalibrator(
            b=0.1, a=0.4, probe_alphas=[0.0, 0.5, 1.0], probe_gaps=[0.1, 0.3, 0.5]
        )

    monkeypatch.setattr(tg, "calibrate_gap", fake_calibrate)
    fd = {i: (rng.random(128) < 0.2).astype(np.float32) for i in range(30)}
    split_for_gap(
        fd,
        0.3,
        X=np.zeros((30, 2)),
        y=np.zeros(30),
        task_type="regression",
        seed=0,
        cal_seeds=(0, 1, 2),
    )
    assert 0 not in seen_cal_seeds


def test_topk_neighbours_uses_tiled_2d_search():
    rng = np.random.default_rng(13)
    X = (rng.random((35, 128)) < 0.25).astype(np.float32)
    idx, sim = topk_neighbours(
        X, "tanimoto", k=5, row_block=7, reference_block=11, device="cpu"
    )
    assert idx.shape == sim.shape == (35, 5)
    assert not np.any(idx == np.arange(35)[:, None])


def test_paired_candidate_cut_audit_and_stage_timings():
    # Two orthogonal clusters: partition A separates them cleanly (cut=0),
    # partition B mixes them (cut>0).
    X = np.eye(2, dtype=np.float32)[np.array([0] * 10 + [1] * 10)]
    B, _ = nystrom_features(X, rank=2, metric="tanimoto")
    lab_clean = np.array([0] * 10 + [1] * 10)
    lab_mixed = np.array([0, 1] * 10)
    comp = compare_candidate_cuts(B, X, "tanimoto", lab_clean, lab_mixed)
    assert comp["exact"] and comp["verdict"] == "a_better"
    assert comp["factor_preference"] == "a_better"
    assert comp["ranking_reversed"] is False

    res = split("lowrank", X, SplitSpec([1, 1], ["train", "test"], seed=0), rank=2)
    timings = res.diagnostics["timings"]
    for key in ("factorization_s", "optimization_s", "grouping_s", "diagnostic_s"):
        assert key in timings and timings[key] >= 0.0
    opt = res.diagnostics["optimization"]
    assert "lloyd_stop_reasons" in opt and "restart_dispersion" in opt
    assert "rejected_lloyd_proposals" in opt and "lloyd_iterations" in opt

