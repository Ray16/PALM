"""Backend choice gates, including rejected ANN overhead and quality failures."""
import numpy as np
import pytest
import lowrank.neighbours as module


def test_small_auto_is_exact_without_ann_import(monkeypatch):
    monkeypatch.setattr(module, "approximate_tanimoto_neighbours", lambda *a, **k:
                        pytest.fail("small auto should not invoke ANN"))
    X = np.eye(40, dtype=np.float32)
    (idx, sim), info = module.audited_neighbours(X, "tanimoto", 4)
    assert idx.shape == sim.shape == (40, 4)
    assert info["selected_backend"] == "tiled" and not info["approximate"]
    assert info["total_seconds"] > 0


def test_explicit_ann_reports_audited_recall(monkeypatch):
    rng = np.random.default_rng(0)
    X = (rng.random((40, 128)) < .2).astype(np.float32)
    exact = module.tiled_neighbours(X, "tanimoto", 5, device="cpu")
    monkeypatch.setattr(module, "approximate_tanimoto_neighbours", lambda *a, **k: exact)
    result, info = module.audited_neighbours(X, "tanimoto", 5, backend="nndescent", audit_queries=12)
    assert info["approximate"] and info["index_recall"] == 1
    assert info["audit_queries"] == 12 and info["ann_audit"]["score_recall"] == 1


def test_auto_rejects_low_recall_and_keeps_audit_and_total_cost(monkeypatch):
    n, k = 4200, 3
    X = np.ones((n, 1), dtype=np.float32)
    def exact(*args, queries=None, **kwargs):
        m = n if queries is None else len(queries)
        return np.tile([1, 2, 3], (m, 1)), np.ones((m, k), dtype=np.float32)
    monkeypatch.setattr(module, "tiled_neighbours", exact)
    monkeypatch.setattr(module, "approximate_tanimoto_neighbours", lambda *a, **kw:
        (np.tile([4, 5, 6], (n, 1)), np.ones((n, k), dtype=np.float32)))
    import time
    counter = iter([0., 1., 2., 3., 4., 5.])
    monkeypatch.setattr(time, "perf_counter", lambda: next(counter))
    _, info = module.audited_neighbours(X, "tanimoto", k, backend="auto", audit_queries=10)
    assert info["selected_backend"] == "tiled"
    assert info["ann_audit"]["index_recall"] == 0
    assert info["total_seconds"] == 5


def test_auto_does_not_change_feature_representation_to_enable_ann():
    X = np.random.default_rng(3).normal(size=(4200, 2)).astype(np.float32)
    (_, _), info = module.audited_neighbours(X, "euclidean", 3, backend="auto")
    assert info["selected_backend"] == "tiled" and "representation" in info["reason"]


def test_explicit_ann_requires_binary_tanimoto():
    with pytest.raises(ValueError, match="binary Tanimoto"):
        module.audited_neighbours(np.ones((10, 4)) * .5, "tanimoto", 3, backend="nndescent")
