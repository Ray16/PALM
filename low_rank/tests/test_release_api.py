"""Public API boundary checks added after the frozen manuscript experiments."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from lowrank.neighbours import tiled_neighbours
from lowrank.target_gap import measure_gap
from lowrank.splitters import SplitSpec, split
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity, standardize_for_metric


@pytest.mark.parametrize("names", (["train", "val", "test"], ["fit", "validation", "holdout"]))
def test_gap_measures_test_not_validation_in_three_way_protocol(monkeypatch, names):
    from lowrank import splitters
    import lowrank.eval.model_eval as evaluation
    assignment = {i: names[0] for i in range(4)}
    assignment.update({4: names[1], 5: names[2]})
    monkeypatch.setattr(splitters, "split", lambda *a, **k: SimpleNamespace(
        assignment=assignment, diagnostics={"leakage": .1}))
    seen = []

    def evaluate(X, y, task, train, test, seed=0):
        seen.append((train, test))
        return {"gen_gap": .5}

    monkeypatch.setattr(evaluation, "evaluate_gap", evaluate)
    m = measure_gap({i: np.zeros(2) for i in range(6)}, np.zeros((6, 2)), np.zeros(6),
                    "regression", .5, seeds=[0], splits=[4, 1, 1], names=names)
    assert seen == [([0, 1, 2, 3], [5])]
    assert m["gap_mean"] == .5


@pytest.mark.parametrize("metric", ("tanimoto", "cosine", "euclidean"))
def test_tiled_topk_matches_full_similarity_values_across_small_tiles(metric):
    rng = np.random.default_rng(4)
    X = ((rng.random((35, 128)) < .2).astype(np.float32) if metric == "tanimoto"
         else rng.normal(size=(35, 7)).astype(np.float32))
    idx, sim = tiled_neighbours(X, metric, k=9, row_block=6, reference_block=5, device="cpu")
    ref = standardize_for_metric(torch.as_tensor(X), metric)
    S = pairwise_similarity(ref, ref, metric)
    S.fill_diagonal_(-torch.inf)
    expected = S.topk(9, dim=1).values.numpy()
    assert np.allclose(sim, expected, atol=1e-6)
    assert np.allclose(sim, S.numpy()[np.arange(len(X))[:, None], idx], atol=1e-6)
    assert not np.any(idx == np.arange(len(X))[:, None])


def test_exact_diagnostic_can_be_disabled_without_changing_the_split():
    rng = np.random.default_rng(3)
    data = {i: (rng.random(128) < .2).astype(np.float32) for i in range(40)}
    spec = SplitSpec([8, 2], ["train", "test"], seed=0)
    a = split("lowrank", data, spec, rank=16, allowed_size_imbalance=0)
    b = split("lowrank", data, spec, rank=16, allowed_size_imbalance=0, score_exact_leakage=False)
    assert a.assignment == b.assignment
    assert a.diagnostics["leakage"] is not None and b.diagnostics["leakage"] is None


def test_tiled_group_backend_preserves_the_size_postcondition():
    rng = np.random.default_rng(2)
    data = {i: (rng.random(128) < .2).astype(np.float32) for i in range(60)}
    res = split("lowrank", data, SplitSpec([8, 1, 1], ["train", "val", "test"]),
                rank=16, allowed_size_imbalance=0, must_link=.6, must_link_backend="tiled",
                must_link_row_block=9, must_link_ref_block=11, score_exact_leakage=False)
    assert res.diagnostics["split_counts"] == {"train": 48, "val": 6, "test": 6}
    assert res.diagnostics["corridor_ok"] and res.diagnostics["leakage"] is None
