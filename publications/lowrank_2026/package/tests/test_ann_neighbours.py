"""ANN candidate weights are exact even though candidate recall is approximate."""
import numpy as np
import pytest

from lowrank.neighbours import approximate_tanimoto_neighbours


def test_ann_edge_weights_equal_exact_binary_jaccard():
    pytest.importorskip("pynndescent")
    rng = np.random.default_rng(3)
    X = (rng.random((80, 128)) < 0.15).astype(np.float32)
    X[1] = X[0]
    idx, sim = approximate_tanimoto_neighbours(X, k=8, candidates=24, seed=7)
    inter = (X[:, None, :] * X[idx]).sum(2)
    union = X.sum(1)[:, None] + X[idx].sum(2) - inter
    assert np.allclose(sim, inter / union)
    assert not np.any(idx == np.arange(len(X))[:, None])
    assert idx[0, 0] == 1 and sim[0, 0] == 1


def test_ann_rejects_nonbinary_features():
    pytest.importorskip("pynndescent")
    with pytest.raises(ValueError, match="binary"):
        approximate_tanimoto_neighbours(np.full((20, 4), 0.5))
