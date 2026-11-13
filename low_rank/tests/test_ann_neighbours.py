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


def test_deficient_ann_candidates_are_repaired_with_exact_distinct_neighbours(monkeypatch):
    from types import SimpleNamespace
    import sys
    from lowrank.neighbours import tiled_neighbours
    rng = np.random.default_rng(4)
    X = (rng.random((24, 64)) < .2).astype(np.float32)
    X[0] = 0  # Covers the zero-fingerprint case without treating zero/zero as one.
    candidates = np.stack([np.roll(np.arange(24), i)[:12] for i in range(24)])
    candidates[0] = -1
    candidates[1] = 1
    candidates[2] = np.tile([4, 5], 6)
    fake = SimpleNamespace(NNDescent=lambda *a, **k:
        SimpleNamespace(neighbor_graph=(candidates, np.zeros_like(candidates, dtype=float))))
    monkeypatch.setitem(sys.modules, "pynndescent", fake)
    idx, sim = approximate_tanimoto_neighbours(X, k=5, candidates=12)
    assert np.isfinite(sim).all() and not np.any(idx == np.arange(24)[:, None])
    assert np.all(np.diff(np.sort(idx, axis=1), axis=1) != 0)
    expected_i, expected_s = tiled_neighbours(X, "tanimoto", k=5, queries=[0, 1, 2], device="cpu")
    assert np.allclose(sim[:3], expected_s)
    # Equal-similarity ties can select different ids; scores must remain exact.
    inter = (X[:, None, :] * X[idx]).sum(2)
    union = X.sum(1)[:, None] + X[idx].sum(2) - inter
    expected = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    assert np.allclose(sim, expected)
