"""Visible accuracy failures, exact feature maps and reusable calibration."""

import numpy as np
import pytest
from lowrank import nystrom_features, split, SplitSpec
from lowrank.nystrom import AccuracyWarning, AccuracyError
from lowrank.neighbours import audited_neighbours
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
from lowrank.diagnostics import prepare_reference


def test_auto_rank_default_warns_even_when_information_is_not_returned():
    X = (np.random.default_rng(4).random((50, 32)) < 0.3).astype(np.float32)
    with pytest.warns(AccuracyWarning, match="not established"):
        B, metric = nystrom_features(
            X,
            rank="auto",
            rank_min=2,
            rank_max=2,
            rank_tolerance=1e-8,
            metric="tanimoto",
        )
    assert B.shape == (50, 2) and metric == "tanimoto"
    with pytest.raises(AccuracyError):
        nystrom_features(
            X,
            rank="auto",
            rank_min=2,
            rank_max=2,
            rank_tolerance=1e-8,
            metric="tanimoto",
            on_failure="raise",
        )


def test_sparse_positive_support_is_selected_instead_of_wasting_landmarks_on_zero_rows():
    X = np.zeros((2000, 2), dtype=np.float32)
    X[-4:, 0] = 1
    B, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=8,
        metric="tanimoto",
        return_info=True,
        on_failure="raise",
    )
    assert set(info["landmark_indices"]) == set(range(1996, 2000))
    assert info["audit_tolerance_met"] and info["audit"]["exact"]
    assert np.allclose(B[-4:] @ B[-4:].T, 1)
    assert not np.any(B[:-4])


def test_exact_cosine_backend_avoids_landmark_kernel_and_matches_the_declared_kernel():
    X = np.random.default_rng(4).random((300, 12)).astype(np.float32)
    B, _, info = nystrom_features(
        X, rank=32, metric="cosine", factor_backend="auto", return_info=True
    )
    assert B.shape == (300, 12) and info["landmark_count"] == 0
    assert info["factor_backend"] == "exact_cosine"
    expected = pairwise_similarity(
        prepare_reference(X, "cosine"), prepare_reference(X, "cosine"), "cosine"
    ).numpy()
    assert np.allclose(B @ B.T, expected, atol=2e-7)


def test_cut_rank_selection_uses_one_incremental_landmark_prefix():
    X = (np.random.default_rng(7).random((90, 128)) < 0.2).astype(np.float32)
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=16,
        metric="tanimoto",
        rank_tolerance=1e-10,
        on_failure="return",
        return_info=True,
    )
    for count in [4, 8]:
        _, _, prefix = nystrom_features(
            X, rank=count, metric="tanimoto", return_info=True
        )
        assert prefix["landmark_indices"] == info["landmark_indices"][:count]


def test_auto_backend_reuses_only_exact_decisions_for_identical_inputs(monkeypatch):
    import lowrank.neighbours as module

    X = np.eye(37, dtype=np.float32)
    with module._PROFILE_LOCK:
        module._EXACT_PROFILES.clear()
    _, first = audited_neighbours(X, "tanimoto", k=5, backend="auto")
    _, second = audited_neighbours(X, "tanimoto", k=5, backend="auto")
    assert not first["calibration_cache_hit"] and second["calibration_cache_hit"]
    assert second["selected_backend"] == "tiled" and not second["approximate"]
    changed = X.copy()
    changed[0, 1] = 1
    _, third = audited_neighbours(changed, "tanimoto", k=5, backend="auto")
    assert not third["calibration_cache_hit"]
    _, changed_k = audited_neighbours(X, "tanimoto", k=6, backend="auto")
    assert not changed_k["calibration_cache_hit"]


def test_public_weighted_split_preserves_counts_and_reports_matching_costs():
    X = np.random.default_rng(3).random((60, 12)).astype(np.float32)
    W = [[0, 2, 0.5], [2, 0, 1], [0.5, 1, 0]]
    result = split(
        "lowrank",
        dict(enumerate(X)),
        SplitSpec([8, 1, 1], ["train", "val", "test"]),
        rank=16,
        fold_weights=W,
        leakage_diagnostic="exact",
        refine_restarts=2,
    )
    assert result.diagnostics["split_counts"] == {"train": 48, "val": 6, "test": 6}
    assert result.diagnostics["fold_weights"] == W
    assert result.diagnostics["optimization"]["refined_restarts"] == 2
    assert np.isfinite(result.diagnostics["weighted_leakage"])


def test_schema_represents_nullable_budget_and_weight_matrix():
    from lowrank import LowRankSplitter, LowRankNDSplitter

    props = LowRankSplitter.param_schema()["properties"]
    assert props["factor_memory_mb"]["type"] == ["number", "null"]
    assert props["fold_weights"]["type"] == ["array", "null"]
    assert LowRankNDSplitter.param_schema()["properties"]["rank"] == props["rank"]
