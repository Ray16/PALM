"""Dense-reference and degenerate-input checks for adaptive factorization."""

import json
import numpy as np
import pytest
import torch
from lowrank import nystrom_features, split, SplitSpec
from lowrank.nystrom import _kmeanspp_landmarks
from lowrank.diagnostics import prepare_reference, sample_pairs, paired_similarity
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity


@pytest.mark.parametrize("metric", ["cosine", "tanimoto", "euclidean"])
def test_degenerate_landmarks_are_distinct_and_null_space_is_removed(metric):
    X = np.ones((48, 8), dtype=np.float32)
    idx = _kmeanspp_landmarks(torch.as_tensor(X), 48, metric, 0)
    assert len(np.unique(idx)) == 48
    B, _, info = nystrom_features(X, rank=32, metric=metric, return_info=True)
    assert np.isfinite(B).all() and info["effective_rank"] == 1
    assert info["discarded_eigenvalues"] == 31
    ref = prepare_reference(X, metric)
    assert np.allclose(
        B @ B.T, pairwise_similarity(ref, ref, metric).numpy(), atol=1e-6
    )
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize(
    "metric,power", [("tanimoto", 1), ("tanimoto", 3), ("cosine", 1), ("euclidean", 1)]
)
def test_full_landmark_factor_matches_dense_kernel_across_tiles(metric, power):
    rng = np.random.default_rng(3)
    X = (
        (rng.random((37, 16)) < 0.3).astype(np.float32)
        if metric == "tanimoto"
        else rng.normal(size=(37, 7)).astype(np.float32)
    )
    B, _, info = nystrom_features(
        X,
        rank=len(X),
        metric=metric,
        kernel_power=power,
        block_size=5,
        return_info=True,
    )
    ref = prepare_reference(X, metric).double()
    expected = pairwise_similarity(ref, ref, metric).pow(power).numpy()
    assert np.allclose(B @ B.T, expected, atol=3e-6)
    assert info["row_block_size"] == 5


def test_auto_stops_early_on_low_rank_and_audits_independently():
    X = np.eye(4, dtype=np.float32)[np.arange(120) % 4]
    B, _, info = nystrom_features(
        X, rank="auto", rank_min=8, rank_max=64, metric="tanimoto", return_info=True
    )
    assert info["landmark_count"] == 8 and B.shape[1] == 4
    assert info["selection_tolerance_met"] and info["audit_tolerance_met"]
    assert info["audit"]["seed"] != info["selection_history"][0]["seed"]
    assert info["audit"]["relative_pair_mae"]["estimate"] < 1e-5
    json.dumps(info, allow_nan=False)


def test_insufficient_rank_never_claims_tolerance_met():
    rng = np.random.default_rng(4)
    X = (rng.random((150, 96)) < 0.2).astype(np.float32)
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=8,
        rank_tolerance=1e-8,
        metric="tanimoto",
        on_failure="return",
        return_info=True,
    )
    assert [r["landmark_count"] for r in info["selection_history"]] == [4, 8]
    assert not info["selection_tolerance_met"] and not info["audit_tolerance_met"]
    assert info["stop_reason"] == "rank_or_memory_limit"


def test_factor_budget_is_enforced_and_fixed_rank_is_not_silently_reduced():
    X = np.eye(100, dtype=np.float32)
    with pytest.raises(MemoryError):
        nystrom_features(X, rank=100, metric="cosine", factor_memory_mb=0.2)
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=100,
        factor_memory_mb=0.2,
        rank_tolerance=1e-9,
        metric="cosine",
        return_info=True,
    )
    assert info["rank_limit"] < 100
    assert info["estimated_array_bytes"] <= 0.2 * 1024**2


def test_zero_kernel_returns_finite_empty_factor_and_valid_split():
    X = np.zeros((20, 4), dtype=np.float32)
    B, _, info = nystrom_features(
        X, rank="auto", rank_min=4, rank_max=8, metric="tanimoto", return_info=True
    )
    assert B.shape == (20, 0) and info["effective_rank"] == 0
    assert info["audit_tolerance_met"]
    result = split(
        "lowrank",
        dict(enumerate(X)),
        SplitSpec([8, 2], ["train", "test"]),
        rank=4,
        metric="tanimoto",
        leakage_diagnostic="sampled",
    )
    assert result.diagnostics["split_counts"] == {"train": 16, "test": 4}
    assert result.diagnostics["leakage_sampled"]["estimate"] is None


@pytest.mark.parametrize("metric", ["tanimoto", "cosine", "euclidean"])
def test_aligned_pair_checks_equal_dense_reference(metric):
    rng = np.random.default_rng(8)
    X = rng.random((40, 16)).astype(np.float32)
    if metric == "tanimoto":
        X = (X < 0.4).astype(np.float32)
    ref = prepare_reference(X, metric)
    i, j = sample_pairs(40, 350, 7)
    assert np.all(i != j)
    actual = paired_similarity(ref, i, j, metric, block_size=11)
    dense = pairwise_similarity(ref.double(), ref.double(), metric).numpy()[i, j]
    assert np.allclose(actual, dense, atol=1e-7)


def test_sampled_ratio_agrees_with_exact_and_does_not_change_assignment():
    rng = np.random.default_rng(9)
    X = (rng.random((320, 64)) < 0.3).astype(np.float32)
    data, spec = dict(enumerate(X)), SplitSpec([8, 2], ["train", "test"], seed=3)
    exact = split("lowrank", data, spec, rank=16, leakage_diagnostic="exact")
    sampled = split(
        "lowrank",
        data,
        spec,
        rank=16,
        leakage_diagnostic="sampled",
        diagnostic_pairs=20000,
    )
    assert exact.assignment == sampled.assignment
    summary = sampled.diagnostics["leakage_sampled"]
    assert abs(summary["estimate"] - exact.diagnostics["leakage"]) < 0.015
    assert summary["ci95"][0] < summary["ci95"][1]
    assert sampled.diagnostics["leakage"] is None and summary["exact"] is False
    json.dumps(sampled.to_json(), allow_nan=False)


@pytest.mark.parametrize(
    "params",
    [
        dict(rank="other"),
        dict(rank=True),
        dict(rank_min=0),
        dict(rank_min=64, rank_max=32),
        dict(rank_tolerance=0),
        dict(eigen_rtol=-1),
        dict(factor_memory_mb=float("nan")),
        dict(kernel_power=1.5),
    ],
)
def test_factor_option_validation(params):
    with pytest.raises(ValueError):
        nystrom_features(np.eye(12), **params)


def test_schema_exposes_integer_or_auto_rank():
    from lowrank import LowRankSplitter

    schema = LowRankSplitter.param_schema()["properties"]["rank"]
    assert schema["oneOf"] == [{"type": "integer", "minimum": 1}, {"const": "auto"}]
