"""Composite gains checked against a dense pairwise reference."""
import numpy as np
import pytest
from scipy import sparse
from lowrank import balanced_lloyd, fm_polish, split, SplitSpec
from lowrank.local import local_graph, normalize_components, SparseTerm


def dense_cut(matrix, labels):
    labels = np.asarray(labels)
    return .5 * matrix[labels[:, None] != labels[None, :]].sum()


def test_graph_union_excludes_self_and_normalizes_objectives():
    X = np.ones((4, 3), dtype=np.float32)
    idx = np.array([[1, 2], [0, 3], [0, 3], [1, 2]])
    sim = np.array([[.9, .5], [.9, .7], [.5, .8], [.7, .8]])
    graph, info = local_graph(X, "tanimoto", neighbours=(idx, sim), threshold=.4, k=2)
    assert graph.nnz == 8 and info["edges"] == 4 and not graph.diagonal().any()
    assert np.allclose(graph.toarray(), graph.T.toarray())
    B = np.array([[1., .1], [1., .2], [.1, 1.], [.2, 1.]], dtype=np.float32)
    scaled, penalty, norm = normalize_components(B, graph, .5)
    labels = [0, 0, 1, 1]
    expected = dense_cut(B @ B.T, labels) / norm["global_factor_mass"]
    expected += .5 * dense_cut(graph.toarray(), labels) / norm["local_mass"]
    actual = dense_cut(scaled @ scaled.T + penalty.toarray(), labels)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize("epsilon", [0., .2])
@pytest.mark.parametrize("k", [2, 3])
def test_lloyd_and_fm_are_monotone_for_dense_composite_and_preserve_capacities(epsilon, k):
    rng = np.random.default_rng(10)
    n = 60
    B = rng.random((n, 5)).astype(np.float32)
    R = rng.random((n, n)); R[R < .9] = 0; R = R + R.T; np.fill_diagonal(R, 0)
    R = sparse.csr_matrix(R)
    counts = [n // k] * k
    initial = np.repeat(np.arange(k), counts); rng.shuffle(initial)
    matrix = B @ B.T + R.toarray()
    lab = balanced_lloyd(B, counts, init_labels=initial, balance_slack=epsilon, local=R)
    assert dense_cut(matrix, lab) <= dense_cut(matrix, initial) + 1e-5
    before = lab.copy()
    final, _ = fm_polish(B, lab, counts, epsilon=epsilon, local=R)
    assert dense_cut(matrix, final) <= dense_cut(matrix, lab) + 1e-5
    assert np.array_equal(lab, before)
    actual = np.bincount(final, minlength=k)
    assert np.all(actual >= np.ceil(np.array(counts) * (1 - epsilon)))
    assert np.all(actual <= np.floor(np.array(counts) * (1 + epsilon)))
    if epsilon == 0: assert np.array_equal(actual, counts)


def test_sparse_scores_and_columns_equal_dense_reference():
    R = sparse.csr_matrix([[0, 2, 0], [2, 0, 3], [0, 3, 0]])
    term = SparseTerm(R)
    labels = np.array([0, 0, 1])
    expected = R.toarray() @ np.eye(2)[labels]
    assert np.allclose(term.block_mass(labels, 2).numpy(), expected)
    assert np.allclose(term.column(1).numpy(), R.toarray()[:, 1])
    assert term.cross(labels) == 3


def test_no_positive_edges_are_explicit():
    X = np.eye(20, dtype=np.float32)
    graph, info = local_graph(X, "tanimoto", threshold=.4)
    assert graph.nnz == 0 and info["status"] == "no_positive_edges_in_graph"


def test_public_local_variant_retains_all_rows_and_exact_counts():
    rng = np.random.default_rng(4)
    X = (rng.random((80, 64)) < .2).astype(np.float32)
    res = split("lowrank", dict(enumerate(X)), SplitSpec([8, 1, 1], ["train", "val", "test"]),
                rank=16, local_weight=.5, local_threshold=.3, leakage_diagnostic="sampled")
    assert len(res.assignment) == 80
    assert res.diagnostics["split_counts"] == {"train": 64, "val": 8, "test": 8}
    info = res.diagnostics["local_objective"]
    assert info["experimental"] and np.isfinite(info["composite_objective"])
    assert np.isclose(info["composite_objective"], info["global_factor_ratio"] + info["weighted_local_graph_ratio"])


def test_unsupported_group_combination_fails_clearly():
    with pytest.raises(ValueError, match="cannot yet be combined"):
        split("lowrank", dict(enumerate(np.eye(20))), local_weight=.5, must_link=.6)
