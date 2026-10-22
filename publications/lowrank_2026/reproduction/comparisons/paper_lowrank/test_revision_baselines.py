"""Scientific comparator invariants, including the observed 32-bit overflow."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "low_rank"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_baselines import integer_graph_weights, graph_cut, repair_graph_counts
from scipy.sparse import csr_matrix


def test_large_graph_quantization_stays_inside_native_accumulator_budget():
    values = np.full(2_000_000, .9)
    weights, scale = integer_graph_weights(values)
    assert scale < 1_000_000
    assert int(weights.sum()) <= np.iinfo(np.int32).max // 8
    assert np.max(np.abs(weights / scale - values)) <= 1 / scale


def test_graph_adapter_checks_native_cut_and_both_capacity_bounds():
    X = (np.random.default_rng(2).random((60, 64)) < .25).astype(np.float32)
    labels, info = graph_cut(X, "tanimoto", epsilon=.05, k=8)
    assert info["graph_type"] == "ordinary undirected pairwise graph"
    assert info["objective"] == "CUT"
    assert info["sum_integer_edge_weights"] <= info["integer_weight_budget"]
    assert info["native_integer_cut"] >= 0
    assert info["repaired_counts"] == [48, 6, 6]


def test_sparse_count_repair_preserves_input_labels():
    adjacency = csr_matrix(np.ones((6, 6)) - np.eye(6))
    before = np.array([0, 0, 0, 0, 0, 1])
    labels, moves = repair_graph_counts(adjacency, before, np.array([3, 3]), np.array([3, 3]))
    assert np.array_equal(before, [0, 0, 0, 0, 0, 1])
    assert np.bincount(labels).tolist() == [3, 3]
    assert moves == 2
