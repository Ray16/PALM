"""Independent dense references for CSR factors, weighted cuts and restart choice."""

import itertools
import numpy as np
import pytest
from scipy import sparse
from lowrank import SplitSpec, split, nystrom_features
from lowrank.factors import validate_fold_weights
from lowrank.objective import factor_leakage
from lowrank.optimize import balanced_lloyd, fm_polish, optimize_restarts
from lowrank.grouped import grouped_split
from lowrank.nd import _identity_factor, _build_nd_factor
from lowrank.splitters.common.leakage_metrics import scaled_lpi, macro_axis_lpi
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
from lowrank.diagnostics import prepare_reference


def dense_cost(B, labels, W):
    S = np.asarray(B, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T
    return 0.5 * float(np.sum(S * W[labels[:, None], labels[None, :]]))


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("epsilon", [0.0, 0.2])
def test_factor_scores_and_optimizer_match_dense_weighted_reference(
    as_sparse, weighted, epsilon
):
    rng = np.random.default_rng(9)
    B = rng.random((30, 7)).astype(np.float32)
    B[B < 0.5] = 0
    labels = np.repeat(np.arange(3), 10)
    rng.shuffle(labels)
    W = (
        np.array([[0, 2, 0.5], [2, 0, 1], [0.5, 1, 0.0]])
        if weighted
        else np.ones((3, 3)) - np.eye(3)
    )
    weights = W if weighted else None
    F = sparse.csr_matrix(B) if as_sparse else B
    assert factor_leakage(F, labels, 3, weights) == pytest.approx(
        dense_cost(B, labels, W), abs=1e-8
    )
    initial = labels.copy()
    lab = balanced_lloyd(
        F, [1, 1, 1], init_labels=labels, balance_slack=epsilon, fold_weights=weights
    )
    assert dense_cost(B, lab, W) <= dense_cost(B, labels, W) + 1e-8
    before = lab.copy()
    final, _ = fm_polish(F, lab, [1, 1, 1], epsilon=epsilon, fold_weights=weights)
    assert np.array_equal(before, lab) and np.array_equal(labels, initial)
    assert dense_cost(B, final, W) <= dense_cost(B, lab, W) + 1e-8
    assert np.all(np.bincount(final, minlength=3) >= 10 * (1 - epsilon))
    assert np.all(np.bincount(final, minlength=3) <= 10 * (1 + epsilon))


def test_restart_winner_is_selected_after_refinement_against_enumerated_optimum():
    B = np.random.default_rng(0).random((12, 5)).astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    W = np.ones((3, 3)) - np.eye(3)
    exact = np.inf
    for first in itertools.combinations(range(12), 6):
        remaining = sorted(set(range(12)) - set(first))
        for second in itertools.combinations(remaining, 3):
            lab = np.full(12, 2, dtype=int)
            lab[list(first)] = 0
            lab[list(second)] = 1
            exact = min(exact, dense_cost(B, lab, W))
    labels, cost, info = optimize_restarts(B, [6, 3, 3], n_restarts=4)
    assert info["refined_restarts"] == 4 and len(info["refinement_candidates"]) == 4
    assert cost == pytest.approx(exact, abs=1e-8)
    assert dense_cost(B, labels, W) == pytest.approx(exact, abs=1e-8)
    _, budgeted, budget_info = optimize_restarts(
        B, [6, 3, 3], n_restarts=4, refine_restarts=1
    )
    assert cost <= budgeted and budget_info["refined_restarts"] == 1


def test_identity_storage_grows_with_records_not_record_times_identity_count():
    n = 10000
    F = _identity_factor(np.repeat(np.arange(n // 2), 2))
    assert sparse.isspmatrix_csr(F) and F.shape == (n, n // 2) and F.nnz == n
    assert F.data.nbytes + F.indices.nbytes + F.indptr.nbytes < 20 * n
    labels = np.arange(n) % 2
    assert factor_leakage(F, labels, 2) == n / 2


def test_record_weighted_standardization_survives_deduplication():
    records = [
        {"a": value}
        for value, count in [("x", 17), ("y", 3), ("z", 10), ("missing", 4)]
        for _ in range(count)
    ]
    maps = {
        "a": {
            "x": np.array([1.0, 4.0]),
            "y": np.array([3.0, 9.0]),
            "z": np.array([7.0, 2.0]),
        }
    }
    B, used = _build_nd_factor(records, maps, rank=32, landmark="uniform", seed=0)
    assert sparse.isspmatrix_csr(B) and used == ["a"]
    features = np.stack([maps["a"][r["a"]] for r in records if r["a"] != "missing"])
    ref = prepare_reference(features, "euclidean").double()
    expected = np.zeros((len(records), len(records)))
    expected[:30, :30] = pairwise_similarity(ref, ref, "euclidean").numpy()
    expected[30:, 30:] = 1
    mass = (expected.sum() - np.trace(expected)) / 2
    assert np.allclose((B @ B.T).toarray(), expected / mass, atol=2e-6)


def test_weighted_strict_groups_keep_counts_and_report_actual_objective():
    B = np.random.default_rng(4).random((24, 5)).astype(np.float32)
    gid = np.repeat(np.arange(12), 2)
    W = np.array([[0, 2, 0.5], [2, 0, 1], [0.5, 1, 0.0]])
    labels, info = grouped_split(
        B, gid, [2, 1, 1], 0, group_policy="strict", fold_weights=W
    )
    assert np.bincount(labels).tolist() == [12, 6, 6]
    assert info["all_groups_preserved"]
    assert info["factor_leakage"] == pytest.approx(dense_cost(B, labels, W), abs=1e-8)


@pytest.mark.parametrize("label_dtype", [np.uint8, np.int32, np.int64])
def test_weighted_exact_scorers_use_declared_integer_fold_order(label_dtype):
    X = np.eye(4, dtype=np.float32)[np.arange(40) % 4]
    labels = (np.arange(40) % 3).astype(label_dtype)
    W = np.array([[0, 2, 0.5], [2, 0, 1], [0.5, 1, 0.0]])
    S = (X @ X.T).astype(np.float64)
    np.fill_diagonal(S, 0)
    expected = (S * W[labels[:, None], labels[None, :]]).sum() / S.sum()
    assert scaled_lpi(X, labels, "tanimoto", fold_weights=W) == pytest.approx(expected)
    records = [{"a": str(i % 4)} for i in range(40)]
    result, _ = macro_axis_lpi(records, {"a": {}}, labels, fold_weights=W)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "W", [np.ones((3, 3)), [[0, -1], [-1, 0]], [[0, 1], [2, 0]], np.zeros((2, 2))]
)
def test_invalid_fold_policies_are_rejected(W):
    with pytest.raises(ValueError, match="fold_weights"):
        validate_fold_weights(W, 3 if np.shape(W) == (3, 3) else 2)


def test_cut_rank_audit_checks_the_partition_actually_returned():
    X = np.eye(4, dtype=np.float32)[np.arange(80) % 4]
    data = dict(enumerate(X))
    spec = SplitSpec([8, 2], ["train", "test"], seed=0)
    result = split(
        "lowrank",
        data,
        spec,
        rank="auto",
        rank_min=2,
        rank_max=8,
        metric="tanimoto",
        rank_check="cut_error",
        on_rank_failure="return",
        hardness=0.6,
        leakage_diagnostic="none",
    )
    info = result.diagnostics["factorization"]
    B, _ = nystrom_features(X, rank=info["landmark_count"], metric="tanimoto", seed=0)
    labels = np.array([result.assignment[i] == "test" for i in range(len(X))])
    S = X @ X.T
    np.fill_diagonal(S, 0)
    approximate = B.astype(np.float64) @ B.astype(np.float64).T
    mask = labels[:, None] != labels[None, :]
    actual = abs(float((S[mask] - approximate[mask]).sum())) / S.sum()
    assert info["audit"]["exact"] and info["audit"]["criterion"] == "cut_error"
    assert info["audit"]["relative_pair_mae"]["estimate"] == pytest.approx(
        actual, abs=1e-8
    )
