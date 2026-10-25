"""Regression coverage for the concrete critical-review failures."""

import numpy as np
import pytest
from lowrank.diagnostics import (
    prepare_reference,
    paired_similarity,
    factor_pair_error,
    sampled_leakage,
    ratio_summary,
)
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
from lowrank.splitters.common.feature_preparation import feature_matrix_from_dict


def test_missing_all_rare_similarity_mass_cannot_pass_a_fidelity_audit():
    X = np.zeros((2000, 2), dtype=np.float32)
    X[-4:, 0] = 1
    ref = prepare_reference(X, "tanimoto")
    audit = factor_pair_error(
        np.zeros((2000, 0), dtype=np.float32), ref, "tanimoto", pairs=2048
    )
    assert audit["exact"] and audit["eligible_rows"] == 4
    assert audit["relative_pair_mae"]["estimate"] == 1
    assert audit["relative_pair_mae"]["ci95"] == [1, 1]


def test_no_observed_rare_cross_pairs_does_not_produce_zero_uncertainty():
    families = np.repeat(np.arange(10), 1000)
    X = np.eye(10, dtype=np.float32)[families]
    labels = (families >= 8).astype(int)
    labels[0], labels[8000] = labels[8000], labels[0]
    result = sampled_leakage(X, labels, "tanimoto", sample_pairs_count=8192, seed=1)
    assert result["estimate"] == 0
    assert result["ci95"][0] <= 0.0004 <= result["ci95"][1]
    assert result["ci95"][1] > 0 and not result["exact"]


def test_zero_observed_denominator_is_insufficient_information():
    result = ratio_summary(np.zeros(2048), np.zeros(2048))
    assert result["ci95"] is None and result["status"] == "insufficient_mass"


def test_known_zero_kernel_has_an_exact_offdiagonal_certificate():
    X = np.zeros((2000, 4), dtype=np.float32)
    result = factor_pair_error(
        np.zeros((2000, 0)), prepare_reference(X, "tanimoto"), "tanimoto"
    )
    assert result["exact"] and result["relative_pair_mae"]["estimate"] == 0


@pytest.mark.parametrize("scale", [1e-30, 1e-10, 1.0, 1e10])
def test_cosine_aligned_and_dense_share_normalization_at_all_scales(scale):
    X = np.array([[scale, 0], [scale, 0], [0, scale]], dtype=np.float32)
    ref = prepare_reference(X, "cosine").double()
    i, j = np.array([0, 0, 1]), np.array([1, 2, 2])
    expected = pairwise_similarity(ref, ref, "cosine").numpy()[i, j]
    assert np.allclose(
        paired_similarity(ref, i, j, "cosine"), expected, rtol=1e-12, atol=1e-40
    )


@pytest.mark.parametrize("value", [1.5, 2.0, -1.0])
def test_tanimoto_rejects_count_and_negative_features(value):
    with pytest.raises(ValueError, match="binary"):
        prepare_reference(
            np.array([[value, 0], [value, 0]], dtype=np.float32), "tanimoto"
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_input_is_not_silently_imputed(value):
    with pytest.raises(ValueError, match="finite"):
        feature_matrix_from_dict({"a": [1.0, value], "b": [0.0, 1.0]})


def test_exact_small_population_leakage_matches_dense_reference():
    X = np.eye(4, dtype=np.float32)[np.arange(40) % 4]
    labels = np.arange(40) % 3
    S = pairwise_similarity(
        prepare_reference(X, "tanimoto"), prepare_reference(X, "tanimoto"), "tanimoto"
    ).numpy()
    np.fill_diagonal(S, 0)
    exact = float(S[labels[:, None] != labels[None, :]].sum() / S.sum())
    result = sampled_leakage(X, labels, "tanimoto")
    assert result["exact"] and result["estimate"] == pytest.approx(exact)
    assert result["ci95"][0] == result["ci95"][1]
