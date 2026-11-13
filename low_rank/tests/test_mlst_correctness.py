"""Counterexamples discovered during the MLST evidence audit."""

import numpy as np
import pytest
import torch

from lowrank.nd import _build_nd_factor
from lowrank.nystrom import nystrom_features
from lowrank.objective import factor_leakage
from lowrank.optimize import balanced_lloyd, corridor_assign, fm_polish
from lowrank.splitters import SplitSpec, split
from lowrank.splitters.common.balanced_assignment import capacity_corridor, target_sizes


@pytest.mark.parametrize("splits", ([2, 8], [8, 2]))
@pytest.mark.parametrize("preferred", (0, 1))
def test_two_block_assignment_checks_both_corridors(splits, preferred):
    scores = torch.zeros(100, 2)
    scores[:, preferred] = 1
    caps, floors = capacity_corridor(100, splits, 0.1)
    labels = corridor_assign(scores, target_sizes(100, splits), caps, floors)
    sizes = np.bincount(labels.numpy(), minlength=2)
    assert np.all(sizes >= floors) and np.all(sizes <= caps)


def test_fm_does_not_hide_an_improving_move_behind_self_similarity():
    K = np.eye(4)
    K[:2, 2:] = 0.3
    K[2:, :2] = 0.3
    B = np.linalg.cholesky(K).astype(np.float32)
    lab = np.array([0, 0, 1, 1])
    initial = factor_leakage(B, lab, 2)
    refined, moves = fm_polish(B, lab.copy(), [1, 1], epsilon=0.5)
    assert moves > 0
    assert factor_leakage(B, refined, 2) == pytest.approx(initial / 2, abs=1e-5)


def test_single_featured_record_on_nd_axis_stays_finite():
    records = [{"a": "F"}, {"a": "M"}, {"a": "M"}]
    maps = {"a": {"F": np.array([1.0, 2.0])}}
    B, axes = _build_nd_factor(records, maps, rank=4, landmark="uniform", seed=0)
    assert axes == ["a"] and np.isfinite(B.data).all()


@pytest.mark.parametrize("power", (0, -1, 1.5, True))
def test_non_psd_kernel_powers_are_rejected(power):
    with pytest.raises(ValueError, match="kernel_power"):
        nystrom_features(np.eye(4), kernel_power=power)


def test_hardness_cannot_silently_destroy_must_link_groups():
    data = {i: np.array([i, 1.0]) for i in range(12)}
    with pytest.raises(ValueError, match="break retained groups"):
        split("lowrank", data, SplitSpec(), must_link="auto", hardness=0.5)


def test_lloyd_never_accepts_a_worse_quadratic_objective():
    # Exhaust several small positive factor geometries: synchronous reassignment
    # of self-excluded scores can improve the linear score yet worsen leakage.
    rng = np.random.default_rng(9)
    for _ in range(12):
        B = rng.uniform(0, 2, (8, 3)).astype(np.float32)
        initial = np.array([0] * 4 + [1] * 4)
        rng.shuffle(initial)
        final = balanced_lloyd(B, [1, 1], init_labels=initial, n_iter=20)
        assert factor_leakage(B, final, 2) <= factor_leakage(B, initial, 2) + 2e-5
