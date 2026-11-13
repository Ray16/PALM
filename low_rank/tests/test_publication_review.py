"""Behavioral regressions from the 21 October publication review."""
import numpy as np
import pytest
import torch

from lowrank.grouped import _repair
from lowrank.objective import factor_leakage
from lowrank.optimize import fm_polish
from lowrank.splitters import SplitSpec, split
from lowrank.splitters.common.balanced_assignment import target_sizes
from lowrank.splitters.common.leakage_metrics import nn_leakage


def repair_case(weights, labels, floors, caps):
    members, cursor = [], 0
    for weight in weights:
        members.append(list(range(cursor, cursor + weight)))
        cursor += weight
    B = np.random.default_rng(4).normal(size=(cursor, 3))
    G = np.stack([B[points].sum(0) for points in members])
    result = _repair(B, G, np.array(weights), np.array(labels), len(caps),
                     np.array(caps), np.array(floors), members)
    G, w, lab, groups, broken = result
    sizes = np.bincount(lab, weights=w, minlength=len(caps))
    assert np.all(sizes >= floors) and np.all(sizes <= caps)
    assert sorted(p for group in groups for p in group) == list(range(cursor))
    assert np.allclose(G, np.stack([B[points].sum(0) for points in groups]))
    return sizes, broken


@pytest.mark.parametrize("floors,caps", [([5, 5], [7, 7]), ([3, 5], [8, 8])])
def test_repair_keeps_whole_groups_when_move_exceeds_excess_or_deficit(floors, caps):
    sizes, broken = repair_case([2] * 6, [0] * 4 + [1] * 2, floors, caps)
    assert broken == 0
    assert np.array_equal(sizes, [6, 6])


def test_repair_checks_other_donors_before_breaking_a_group():
    # Largest surplus is in block 0, but only block 1 has a group that fits.
    _, broken = repair_case([6, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 2], [3, 2, 2], [6, 5, 2])
    assert broken == 0


def test_repair_reports_necessary_group_dissolution():
    _, broken = repair_case([2] * 6, [0] * 4 + [1] * 2, [7, 5], [7, 5])
    assert broken == 1


def test_cpu_fm_does_not_mutate_callers_partition(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    B = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    initial = np.array([0, 1, 0, 1], dtype=np.int64)
    before = initial.copy()
    result, moves = fm_polish(B, initial, [1, 1], epsilon=0)
    assert moves > 0
    assert np.array_equal(initial, before)
    assert factor_leakage(B, result, 2) < factor_leakage(B, before, 2)


def test_default_split_sizes_are_exact():
    rng = np.random.default_rng(0)
    data = {i: (rng.random(64) < .3).astype(np.float32) for i in range(300)}
    result = split("lowrank", data, SplitSpec([8, 1, 1], ["train", "val", "test"]), rank=16)
    actual = [list(result.assignment.values()).count(s) for s in ("train", "val", "test")]
    assert actual == target_sizes(300, [8, 1, 1]).tolist()


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("method", ["lowrank", "lowrank_nd"])
def test_ratios_that_round_to_empty_folds_raise(n, method):
    data = ({i: np.ones(8) for i in range(n)} if method == "lowrank" else
            ([{"a": str(i)} for i in range(n)], {"a": {}}))
    with pytest.raises(ValueError, match="empty fold"):
        split(method, data, SplitSpec([8, 1, 1], ["train", "val", "test"]))


def test_nn_metric_does_not_select_cuda_implicitly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = nn_leakage(np.eye(4), ["train", "train", "test", "test"], n_perm=1)
    assert result["nnl"] == 0
