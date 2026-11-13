"""Whole-group feasibility, explicit relaxation and failure semantics."""
import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from lowrank import split, SplitSpec
from lowrank.grouped import (grouped_split, _whole_group_assignment,
                              GroupInfeasibleError, GroupFeasibilityUnresolved)


def fixture(weights):
    gid = np.repeat(np.arange(len(weights)), weights)
    B = np.random.default_rng(8).normal(size=(len(gid), 4))
    return B, gid


def test_strict_finds_feasible_packing_that_greedy_cannot_repair():
    w = np.array([4, 3, 3, 2])
    lab = _whole_group_assignment(w, np.array([6, 6]), np.array([6, 6]), np.array([0, 0, 1, 1]))
    assert np.bincount(lab, weights=w).tolist() == [6, 6]
    B, gid = fixture(w)
    labels, info = grouped_split(B, gid, [1, 1], 0, group_policy="strict", n_restarts=2)
    assert np.bincount(labels).tolist() == [6, 6]
    assert info["all_groups_preserved"] and info["broken_groups"] == 0


def test_infeasible_strict_counts_are_distinguished_from_count_first():
    B, gid = fixture([2] * 6)
    with pytest.raises(GroupInfeasibleError, match="infeasible"):
        grouped_split(B, gid, [7, 5], 0, group_policy="strict")
    labels, info = grouped_split(B, gid, [7, 5], 0, group_policy="count_first")
    assert np.bincount(labels).tolist() == [7, 5]
    assert info["broken_groups"] > 0 and not info["all_groups_preserved"]


def test_explicit_corridor_can_preserve_previously_infeasible_groups():
    B, gid = fixture([2] * 6)
    labels, info = grouped_split(B, gid, [7, 5], .2, group_policy="strict")
    counts = np.bincount(labels)
    assert np.all(counts >= [6, 4]) and np.all(counts <= [8, 6])
    assert info["all_groups_preserved"] and info["broken_groups"] == 0


def test_solver_timeout_is_not_reported_as_infeasible(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k:
                        SimpleNamespace(x=None, status=1, message="time limit reached"))
    with pytest.raises(GroupFeasibilityUnresolved, match="unresolved"):
        _whole_group_assignment([2, 2, 2], [3, 3], [3, 3], [0, 0, 1])


def test_valid_timeout_incumbent_is_accepted_after_integer_capacity_checks(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k:
        SimpleNamespace(x=np.array([1., 0., 0., 1.]), status=1, message="time limit"))
    result = _whole_group_assignment([2, 2], [2, 2], [2, 2], [0, 1])
    assert result.tolist() == [0, 1]


def test_fractional_timeout_incumbent_is_rejected(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k:
        SimpleNamespace(x=np.full(4, .5), status=1, message="time limit"))
    with pytest.raises(GroupFeasibilityUnresolved):
        _whole_group_assignment([2, 2], [2, 2], [2, 2], [0, 1])


@pytest.mark.parametrize("k", [2, 3])
def test_milp_feasibility_agrees_with_exhaustive_small_packings(k):
    w = np.array([3, 3, 2, 2, 2])
    for target in ([6, 6], [7, 5]) if k == 2 else ([4, 4, 4], [5, 4, 3]):
        feasible = any(np.array_equal(np.bincount(lab, weights=w, minlength=k), target)
                       for lab in itertools.product(range(k), repeat=len(w)))
        if feasible:
            lab = _whole_group_assignment(w, target, target, np.arange(len(w)) % k)
            assert np.bincount(lab, weights=w, minlength=k).tolist() == list(target)
        else:
            with pytest.raises(GroupInfeasibleError):
                _whole_group_assignment(w, target, target, np.arange(len(w)) % k)


def test_public_supplied_groups_contract_and_complete_retention():
    rng = np.random.default_rng(6)
    data = {f"entity{i}": rng.random(8).astype(np.float32) for i in range(30)}
    groups = {key: int(key[6:]) // 3 for key in data}
    result = split("lowrank", data, SplitSpec([8, 2], ["train", "test"]),
                   rank=8, groups=groups, group_policy="strict", leakage_diagnostic="none")
    assert result.diagnostics["split_counts"] == {"train": 24, "test": 6}
    assert result.diagnostics["group_source"] == "supplied"
    assert result.diagnostics["all_groups_preserved"]
    for g in set(groups.values()):
        assert len({result.assignment[key] for key in data if groups[key] == g}) == 1


def test_missing_entity_groups_and_randomization_are_rejected():
    data = {i: np.ones(4) for i in range(20)}
    with pytest.raises(ValueError, match="every entity"):
        split("lowrank", data, groups={0: "a"}, rank=4)
    with pytest.raises(ValueError, match="hardness"):
        split("lowrank", data, groups=dict.fromkeys(data, "a"), hardness=.5, rank=4)
