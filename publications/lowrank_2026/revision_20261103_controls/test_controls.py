from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "package"))
from lowrank.factors import factor_ops
from lowrank.optimize import balanced_lloyd, fm_polish
from oracles import DenseOps, independent_metrics


def test_dense_oracle_uses_identical_current_algorithm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Integer features give exact binary representable inner products: this
    # isolates oracle/algorithm equivalence without stochastic rounding ties.
    B = np.random.default_rng(18).integers(0, 4, (32, 7)).astype(np.float32)
    dense, factor = DenseOps(B @ B.T), factor_ops(B)
    initial = np.repeat(np.arange(3), [24, 4, 4])
    for slack in (0., .1):
        ld = balanced_lloyd(dense, [6,1,1], init_labels=initial, balance_slack=slack)
        lf = balanced_lloyd(factor, [6,1,1], init_labels=initial, balance_slack=slack)
        assert np.array_equal(ld, lf)
        ld, md = fm_polish(dense, ld, [6,1,1], epsilon=slack)
        lf, mf = fm_polish(factor, lf, [6,1,1], epsilon=slack)
        assert np.array_equal(ld, lf)
        assert md == mf
        assert dense.cost(ld, 3) == factor.cost(lf, 3)
        assert np.array_equal(initial, np.repeat(np.arange(3), [24,4,4]))


def test_independent_cut_errors_and_nearest_are_original_kernel():
    X = np.array([[1,0],[1,1],[0,1],[1,0]], dtype=np.float32)
    z = np.array([0,0,1,2])
    S = np.array([[1, .5, 0, 1],[.5,1,.5,.5],[0,.5,1,0],[1,.5,0,1]])
    values, vectors = np.linalg.eigh(S)
    B = vectors * np.sqrt(np.maximum(values, 0))
    r = independent_metrics(X, z, B=B)
    mask = z[:,None] != z[None,:]
    assert r['true_cut'] == pytest.approx(S[mask].sum()/2)
    assert r['mean_nearest'] == 1
    assert r['absolute_cut_error'] < 1e-12
    cube = independent_metrics(X, z, power=3, B=B)
    assert cube['true_powered_cut'] == pytest.approx((S**3)[mask].sum()/2)
    assert cube['cut_relative_error'] == pytest.approx(cube['absolute_cut_error']/cube['true_powered_cut'])


def test_zero_cut_is_explicit_not_epsilon_normalized():
    r = independent_metrics(np.eye(3), np.arange(3), B=np.eye(3))
    assert r['cut_relative_error'] is None
    assert r['cut_relative_status'] == 'zero_true_cut'
    assert r['absolute_cut_error'] == 0
