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


@pytest.mark.parametrize("scale", [1e-30, 1e-10, 1.0, 1e10, 1e30])
def test_cosine_aligned_and_dense_share_normalization_at_all_scales(scale):
    X = np.array([[scale, 0], [scale, 0], [0, scale]], dtype=np.float32)
    ref = prepare_reference(X, "cosine").double()
    i, j = np.array([0, 0, 1]), np.array([1, 2, 2])
    expected = pairwise_similarity(ref, ref, "cosine").numpy()[i, j]
    assert np.allclose(
        paired_similarity(ref, i, j, "cosine"), expected, rtol=1e-12, atol=1e-40
    )
    ref32 = prepare_reference(X, "cosine")
    dense32 = pairwise_similarity(ref32, ref32, "cosine").numpy()[i, j]
    assert np.allclose(dense32, expected, rtol=2e-6, atol=1e-40)


def test_neighbour_scoring_obeys_the_same_binary_and_finite_input_contract():
    import json
    from lowrank.splitters.common.leakage_metrics import nn_leakage
    from lowrank.splitters.common.fiduccia_mattheyses import fiduccia_mattheyses_exact

    bad = np.full((4, 8), 1.5, dtype=np.float32)
    with pytest.raises(ValueError, match="binary"):
        nn_leakage(bad, ["train", "train", "test", "test"], n_perm=1)
    with pytest.raises(ValueError, match="binary"):
        fiduccia_mattheyses_exact([0, 0, 1, 1], bad, [1, 1], "tanimoto")
    zero = nn_leakage(np.eye(4), ["train", "train", "test", "test"], n_perm=1)
    assert zero["nnl_rel"] is None
    assert zero["relative_status"] == "undefined_zero_reference_baseline"
    json.dumps(zero, allow_nan=False)


@pytest.mark.parametrize("scale", [1.0, 1e30, 1e38])
def test_euclidean_preparation_stays_finite_and_matches_float64_statistics(scale):
    X = np.random.default_rng(4).uniform(-0.5, 0.5, size=(60, 7)).astype(
        np.float32
    ) * np.float32(scale)
    X[:, 0] = np.float32(scale)
    ref = prepare_reference(X, "euclidean").numpy()
    work = X.astype(np.float64)
    expected = (work - work.mean(0)) / (work.std(0, ddof=1) + 1e-8)
    assert np.isfinite(ref).all() and np.allclose(ref, expected, atol=2e-6, rtol=2e-5)


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


def test_small_cut_passes_mass_norm_but_fails_cut_relative_audit():
    # A near-empty factor (gets the cut 100% wrong). The cut carries only a small
    # fraction of total kernel mass, so the historical mass-normalized metric looks
    # acceptable while the cut-relative metric correctly reports ~100% error.
    a = np.zeros(15, dtype=np.float32)
    a[:8] = 1.0
    b = np.zeros(15, dtype=np.float32)
    b[7:15] = 1.0  # tanimoto(a, b) = 1/15
    X = np.vstack([a] * 5 + [b] * 5)
    labels = np.array([0] * 5 + [1] * 5)
    ref = prepare_reference(X, "tanimoto")
    result = factor_pair_error(
        np.zeros((10, 0), dtype=np.float32), ref, "tanimoto", labels=labels
    )
    assert result["exact"] and result["criterion"] == "cut_error"
    # Mass-normalized metric is small (would "pass" a naive 0.1 tolerance)...
    assert result["relative_pair_mae"]["estimate"] == pytest.approx(1 / 13, rel=1e-6)
    # ...but the cut-relative metric exposes the ~100% error on the cut itself.
    assert result["cut_relative_error"]["estimate"] == pytest.approx(1.0, rel=1e-6)
    assert result["relative_pair_mae"]["estimate"] < 0.1 < result[
        "cut_relative_error"
    ]["estimate"]
    assert result["acceptance"]["accepted"] is False
    assert result["acceptance"]["status"] == "rejected"


def test_near_zero_cut_mass_is_unresolved_not_spuriously_small():
    # Orthogonal folds (zero true cut mass) with a nonzero factor: the cut residual
    # is real but cannot be normalized by a vanishing cut mass -> unresolved.
    X = np.eye(2, dtype=np.float32)[np.array([0] * 3 + [1] * 3)]
    labels = np.array([0] * 3 + [1] * 3)
    ref = prepare_reference(X, "tanimoto")
    result = factor_pair_error(
        np.full((6, 1), 0.5, dtype=np.float32), ref, "tanimoto", labels=labels
    )
    assert result["cut_relative_error"]["estimate"] is None
    assert result["cut_absolute_error"]["upper"] > result["acceptance"]["abs_floor"]
    assert result["acceptance"]["accepted"] is False
    assert result["acceptance"]["status"] == "unresolved_low_cut_mass"


def test_zero_cut_error_is_accepted_below_absolute_floor():
    # Orthogonal folds with a factor that leaves no cut residual: the cut mass is
    # zero but the absolute cut error is zero too, so the split is accepted.
    X = np.eye(2, dtype=np.float32)[np.array([0] * 3 + [1] * 3)]
    labels = np.array([0] * 3 + [1] * 3)
    ref = prepare_reference(X, "tanimoto")
    result = factor_pair_error(
        np.zeros((6, 0), dtype=np.float32), ref, "tanimoto", labels=labels
    )
    assert result["cut_absolute_error"]["upper"] <= result["acceptance"]["abs_floor"]
    assert result["acceptance"]["accepted"] is True
    assert result["acceptance"]["status"] == "accepted_below_abs_floor"


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
