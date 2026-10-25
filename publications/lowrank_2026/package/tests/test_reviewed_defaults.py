"""Budget defaults and undefined signed-kernel normalization are explicit."""

import numpy as np
import pytest
from lowrank import (
    LowRankSplitter,
    LowRankNDSplitter,
    SplitSpec,
    split,
    nystrom_features,
)
from lowrank.splitters.common.leakage_metrics import scaled_lpi, macro_axis_lpi


def test_public_defaults_refine_a_bounded_shortlist():
    assert LowRankSplitter.Params().refine_restarts == 2
    assert LowRankNDSplitter.Params().refine_restarts == 2
    X = np.random.default_rng(5).random((40, 8)).astype(np.float32)
    result = split("lowrank", dict(enumerate(X)), leakage_diagnostic="none", rank=16)
    assert result.diagnostics["optimization"]["refined_restarts"] == 2
    exhaustive = split(
        "lowrank",
        dict(enumerate(X)),
        leakage_diagnostic="none",
        rank=16,
        refine_restarts=None,
    )
    assert exhaustive.diagnostics["optimization"]["refined_restarts"] == 4


@pytest.mark.parametrize("check,expected", [("pair_mae", 2048), ("cut_error", 32768)])
def test_default_probe_budget_tracks_the_error_criterion(check, expected):
    X = np.ones((20, 8), dtype=np.float32)
    labels = np.arange(20) % 2
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=8,
        metric="tanimoto",
        rank_check=check,
        cut_labels=labels,
        return_info=True,
        on_failure="return",
    )
    assert info["probe_pair_budget"] == expected


def test_explicit_probe_budget_is_never_overridden():
    X = np.ones((20, 8), dtype=np.float32)
    _, _, info = nystrom_features(
        X,
        rank="auto",
        rank_min=4,
        rank_max=8,
        metric="tanimoto",
        rank_check="cut_error",
        cut_labels=np.arange(20) % 2,
        rank_probe_pairs=128,
        return_info=True,
        on_failure="return",
    )
    assert info["probe_pair_budget"] == 128


def test_nonpositive_cosine_normalization_does_not_claim_zero_leakage():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    assert scaled_lpi(X, np.array([0, 0, 1, 1]), metric="cosine") is None
    result = split(
        "lowrank",
        dict(enumerate(X)),
        SplitSpec([1, 1], ["train", "test"]),
        metric="cosine",
        rank=4,
        leakage_diagnostic="exact",
    )
    assert result.diagnostics["leakage"] is None
    assert result.diagnostics["leakage_status"] == "undefined_nonpositive_kernel_mass"


def test_nd_does_not_silently_drop_a_negative_mass_axis():
    vectors = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32
    )
    records = [{"a": str(i)} for i in range(4)]
    maps = {"a": dict(zip(map(str, range(4)), vectors))}
    with pytest.raises(ValueError, match="normalization mass"):
        split(
            "lowrank_nd",
            (records, maps),
            SplitSpec([1, 1], ["train", "test"]),
            axis_metrics={"a": "cosine"},
            rank=4,
        )
    with pytest.raises(ValueError, match="normalization mass"):
        macro_axis_lpi(records, maps, [0, 0, 1, 1], axis_metrics={"a": "cosine"})
