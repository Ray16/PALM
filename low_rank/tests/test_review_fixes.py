"""Regression tests for the external code-review findings (2026-09-29).

Each test pins one reported defect: n-D factor/metric agreement, exact sizes at
``allowed_size_imbalance=0``, nested hardness perturbation, direction-aware gap
calibration, API parameter validation, per-axis n-D metrics, corridor tolerance,
and importability of the registered feature/loader modules.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from lowrank.splitters import SplitSpec, split
from lowrank.splitters.common.balanced_assignment import capacity_corridor, target_sizes
from lowrank.splitters.common.leakage_metrics import axis_features, macro_axis_lpi
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
from lowrank.nd import _build_nd_factor
from lowrank.nystrom import nystrom_features
from lowrank.objective import factor_leakage
from lowrank.optimize import interpolate_to_random
from lowrank.target_gap import GapCalibrator


# --- 1. n-D: factor objective == reported metric with featureless values ---------- #
def _mixed_axis():
    fp = np.zeros(16, np.float32); fp[:4] = 1
    records = [{"a": v} for v in ["F", "F", "F", "M", "M", "M", "U", "V"]]
    afm = {"a": {"F": fp}}                     # M repeats with no feature; U, V singletons
    return records, afm


def test_nd_factor_matches_metric_with_missing_features():
    records, afm = _mixed_axis()
    B, used = _build_nd_factor(records, afm, rank=8, landmark="kmeans++", seed=0)
    assert used == ["a"]
    for labels in ([0, 0, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 0, 1, 0, 1, 0, 1]):
        _, per_axis = macro_axis_lpi(records, afm, labels)
        assert abs(factor_leakage(B, labels, 2) - per_axis["a"]) < 1e-3, labels  # Nyström (float32) on the featured block


def test_nd_orthogonal_features_plus_repeated_missing_is_not_rejected():
    eye = np.eye(4, dtype=np.float32)
    records = [{"a": f"F{i}"} for i in range(4)] + [{"a": "M"}] * 3
    afm = {"a": {f"F{i}": eye[i] for i in range(4)}}
    B, used = _build_nd_factor(records, afm, rank=8, landmark="kmeans++", seed=0)
    labels = [0, 0, 1, 1, 0, 1, 1]
    _, per_axis = macro_axis_lpi(records, afm, labels)
    assert used == ["a"] and per_axis["a"] > 0
    assert abs(factor_leakage(B, labels, 2) - per_axis["a"]) < 1e-5


# --- 2. allowed_size_imbalance=0 means exact sizes ---------------------------------- #
def test_exact_sizes_1d():
    rng = np.random.default_rng(0)
    data = {i: (rng.random(64) < 0.3).astype(np.float32) for i in range(10)}
    res = split("lowrank", data, SplitSpec([8, 2], ["train", "test"], seed=0),
                allowed_size_imbalance=0)
    vals = list(res.assignment.values())
    assert vals.count("train") == 8 and vals.count("test") == 2


def test_exact_sizes_nd_three_way():
    rng = np.random.default_rng(1)
    records = [{"a": f"A{rng.integers(6)}", "b": f"B{rng.integers(4)}"} for _ in range(53)]
    res = split("lowrank_nd", (records, {"a": {}, "b": {}}),
                SplitSpec([7, 2, 1], ["train", "val", "test"], seed=0),
                allowed_size_imbalance=0)
    vals = list(res.assignment.values())
    assert [vals.count(s) for s in ("train", "val", "test")] == list(target_sizes(53, [7, 2, 1]))


def test_exact_swap_polish_never_worse_than_lloyd():
    from lowrank.optimize import balanced_lloyd, fm_polish
    rng = np.random.default_rng(2)
    X = (rng.random((80, 64)) < 0.2).astype(np.float32)
    B, _ = nystrom_features(X, rank=32, metric="tanimoto")
    lab0 = balanced_lloyd(B, [8, 2], n_iter=3, seed=0)
    lab1, _ = fm_polish(B, lab0, [8, 2], epsilon=0.0)
    assert np.array_equal(np.bincount(lab0), np.bincount(lab1))
    assert factor_leakage(B, lab1, 2) <= factor_leakage(B, lab0, 2) + 1e-4


# --- 3. hardness perturbation is nested --------------------------------------------- #
def test_hardness_is_nested():
    n = 40
    base = np.arange(n)                       # distinct labels expose every swap
    prev = interpolate_to_random(base, 1.0, seed=3)
    assert np.array_equal(prev, base)
    for m in range(1, n + 1):
        cur = interpolate_to_random(base, 1.0 - m / n, seed=3)
        assert (cur != prev).sum() <= 2, m    # one extra transposition per step
        assert sorted(cur) == list(base)
        prev = cur


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.5])
def test_hardness_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        interpolate_to_random(np.zeros(10, int), bad)


# --- 5. negative-slope piecewise calibration ---------------------------------------- #
def test_negative_slope_piecewise_calibration():
    cal = GapCalibrator(b=0.8, a=-0.4, probe_alphas=[0.0, 0.5, 1.0],
                        probe_gaps=[0.8, 0.6, 0.4])
    assert cal._use_piecewise() and cal.controllable
    assert cal.achievable_range == pytest.approx((0.4, 0.8))
    inv = cal.invert(0.6)
    assert inv.controllable and inv.alpha == pytest.approx(0.5)
    assert cal.invert(0.7).alpha == pytest.approx(0.25)


def test_positive_slope_isotonic_pools_noise():
    cal = GapCalibrator(b=0.1, a=0.3, probe_alphas=[0.0, 0.5, 1.0],
                        probe_gaps=[0.1, 0.3, 0.25])
    _, gg = cal._knots()
    assert np.all(np.diff(gg) >= 0) and gg[1] == pytest.approx(0.275)


# --- 7. invalid parameters are rejected at API entry -------------------------------- #
@pytest.mark.parametrize("kw", [dict(metric="wat"), dict(landmark="wat"), dict(rank=0),
                                dict(n_restarts=0), dict(n_iter=0),
                                dict(hardness=float("nan")), dict(hardness=2.0)])
def test_invalid_params_raise(kw):
    rng = np.random.default_rng(0)
    data = {i: (rng.random(32) < 0.3).astype(np.float32) for i in range(20)}
    with pytest.raises(ValueError):
        split("lowrank", data, SplitSpec([8, 2], ["train", "test"], seed=0), **kw)


def test_unknown_metric_rejected_by_kernel():
    import torch
    with pytest.raises(ValueError):
        pairwise_similarity(torch.ones(2, 3), torch.ones(2, 3), "wat")


# --- 8. per-axis metric (dense / negative descriptors are not Tanimoto) ------------- #
def test_dense_negative_axis_gets_non_tanimoto_metric():
    rng = np.random.default_rng(4)
    desc = {f"S{i}": rng.normal(size=6).astype(np.float32) for i in range(5)}
    records = [{"solvent": f"S{rng.integers(5)}"} for _ in range(30)]
    _, F, has, metric = axis_features(records, desc, "solvent")
    assert has.all() and metric == "euclidean"
    with pytest.raises(ValueError):
        axis_features(records, desc, "solvent", metric="tanimoto")
    res = split("lowrank_nd", (records, {"solvent": desc}),
                SplitSpec([8, 2], ["train", "test"], seed=0))
    assert res.diagnostics["leakage"] is not None


# --- 9. corridor stays within the advertised tolerance ------------------------------ #
def test_corridor_within_tolerance():
    n, splits, eps = 100, [8, 1, 1], 0.05
    caps, floors = capacity_corridor(n, splits, eps)
    t = np.array([80, 10, 10])
    assert np.all(caps <= np.floor(t * (1 + eps))) and np.all(floors >= np.ceil(t * (1 - eps)))
    caps0, floors0 = capacity_corridor(n, splits, 0.0)
    assert np.array_equal(caps0, floors0) and np.array_equal(caps0, target_sizes(n, splits))


def test_corridor_contains_integer_targets():
    for n in (7, 10, 33, 101):
        for splits in ([8, 2], [7, 2, 1], [1, 1, 1]):
            caps, floors = capacity_corridor(n, splits, 0.01)
            sizes = target_sizes(n, splits)
            assert np.all(floors <= sizes) and np.all(sizes <= caps)


# --- 10. registered loader/feature modules exist ------------------------------------ #
@pytest.mark.parametrize("mod", ['lowrank.features.biomolecule_features', 'lowrank.features.gene_features',
                                 'lowrank.features.material_features', 'lowrank.datasets.reactions'])
def test_feature_modules_import(mod):
    importlib.import_module(mod)
