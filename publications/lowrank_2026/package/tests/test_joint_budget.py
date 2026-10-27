"""Accuracy/cost allocation, repeated-check coverage and failure contracts."""

import json
import numpy as np
import pytest

from lowrank import AccuracyError, LowRankSplitter, SplitSpec, nystrom_features, split


def test_more_probes_establish_exact_factor_without_unnecessary_rank_growth():
    X = np.eye(4, dtype=np.float32)[np.arange(420) % 4]
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=16,
        metric="tanimoto",
        rank_probe_policy="adaptive",
        rank_probe_pairs=32,
        rank_probe_max_pairs=32768,
        return_info=True,
        on_failure="raise",
    )
    history = info["selection_history"]
    assert len(history) > 1
    assert all(h["landmark_count"] == 4 for h in history)
    assert history[0]["decision"] == "increase_probes"
    assert history[-1]["decision"] == "accept"
    assert info["selection_tolerance_met"] and info["audit_tolerance_met"]
    assert info["audit"]["seed"] not in {h["seed"] for h in history}
    assert info["audit"]["sample_pairs"] == info["selected_probe_pairs"]
    assert info["audit"]["confidence"] == 0.975
    checks = info["selection_grid_checks"]
    assert checks >= len(history)
    assert checks * (1 - info["selection_confidence"]) + 0.025 <= 0.05 + 1e-12
    json.dumps(info, allow_nan=False)


def test_exact_population_error_grows_rank_without_wasting_probe_budget():
    X = np.eye(16, dtype=np.float32)[np.arange(80) % 16]
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=2,
        rank_max=16,
        metric="tanimoto",
        rank_probe_policy="adaptive",
        rank_probe_pairs=32,
        rank_probe_max_pairs=32768,
        return_info=True,
        on_failure="raise",
    )
    history = info["selection_history"]
    assert [h["landmark_count"] for h in history] == [2, 4, 8, 16]
    assert all(h["requested_pairs"] == 32 and h["exact"] for h in history)
    assert history[0]["decision"] == "increase_rank"
    assert info["audit_tolerance_met"]


def test_joint_budget_exhaustion_is_visible_and_can_raise():
    X = np.eye(16, dtype=np.float32)[np.arange(420) % 16]
    params = dict(
        rank="auto",
        rank_min=2,
        rank_max=4,
        metric="tanimoto",
        rank_probe_policy="adaptive",
        rank_probe_pairs=16,
        rank_probe_max_pairs=32,
        return_info=True,
    )
    _, _, info = nystrom_features(X, **params, on_failure="return")
    assert not info["selection_tolerance_met"]
    assert info["stop_reason"] == "rank_probe_or_memory_limit"
    assert info["selection_history"][-1]["decision"] == "budget_exhausted"
    assert max(h["requested_pairs"] for h in info["selection_history"]) <= 32
    with pytest.raises(AccuracyError):
        nystrom_features(X, **params, on_failure="raise")


def test_public_cut_policy_preserves_counts_and_uses_actual_partition():
    X = np.eye(4, dtype=np.float32)[np.arange(80) % 4]
    result = split(
        "lowrank",
        dict(enumerate(X)),
        SplitSpec([8, 1, 1], ["train", "val", "test"]),
        rank="auto",
        rank_min=4,
        rank_max=16,
        metric="tanimoto",
        rank_check="cut_error",
        rank_probe_policy="adaptive",
        on_rank_failure="raise",
        score_exact_leakage=False,
    )
    assert result.diagnostics["split_counts"] == {"train": 64, "val": 8, "test": 8}
    assert result.diagnostics["factorization"]["audit"]["criterion"] == "cut_error"
    props = LowRankSplitter.param_schema()["properties"]
    assert props["rank_probe_policy"]["enum"] == ["fixed", "adaptive"]
    assert props["rank_probe_max_pairs"]["type"] == ["integer", "null"]


@pytest.mark.parametrize(
    "options",
    [
        {"rank_probe_policy": "unknown"},
        {"rank_probe_pairs": 0},
        {"rank_probe_max_pairs": 0},
        {"rank_probe_max_pairs": 4},
        {"rank_probe_pairs": 8, "rank_probe_max_pairs": 16},
        {"rank_probe_pairs": True},
    ],
)
def test_invalid_probe_policies_are_rejected(options):
    with pytest.raises(ValueError):
        nystrom_features(np.eye(12), **options)
