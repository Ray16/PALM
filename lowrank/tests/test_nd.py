"""Tests for the n-D low-rank splitter (``lowrank_nd``).

Run: `python -m PALM.lowrank.tests.test_nd` (plain-assert smoke run), or via pytest.
Kept separate so it does not disturb the other test modules.
"""

from __future__ import annotations

import numpy as np

from PALM.splitters import SplitSpec, split
from PALM.splitters.common.leakage_metrics import macro_axis_lpi
from PALM.splitters.common.balanced_assignment import capacity_corridor
from PALM.lowrank.objective import factor_leakage, realized_imbalance
from PALM.lowrank.nd import _build_nd_factor, _identity_factor


def _identity_nd(n=90, na=10, nb=5, seed=0):
    """Synthetic n-D set with two identity-only axes (values repeat, no features)."""
    rng = np.random.default_rng(seed)
    records = [{"a": f"A{rng.integers(na)}", "b": f"B{rng.integers(nb)}"} for _ in range(n)]
    afm = {"a": {}, "b": {}}                      # no features -> identity axes
    return records, afm


def test_identity_onehot_is_same_value_indicator():
    vals = ["x", "y", "x", "z", "y", "q"]         # q is a singleton -> dropped
    B = _identity_factor(vals)
    G = B @ B.T
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            same = (vals[i] == vals[j]) and vals.count(vals[i]) >= 2
            assert G[i, j] == (1.0 if same else 0.0), (i, j, vals[i], vals[j], G[i, j])


def test_concatenation_identity_equals_macro():
    """For identity axes (exact), factor_leakage(concat B) == n_axes * macro_axis_lpi."""
    records, afm = _identity_nd()
    B, used = _build_nd_factor(records, afm, rank=256, landmark="kmeans++", seed=0)
    rng = np.random.default_rng(1)
    labels = (rng.random(len(records)) < 0.2).astype(int)     # arbitrary 2-way split
    fl = factor_leakage(B, labels, 2)
    macro, per_axis = macro_axis_lpi(records, afm, labels)
    # concatenated factor leakage == sum of per-axis ratios == n_used_axes * macro
    assert abs(fl - sum(per_axis.values())) < 1e-4, (fl, per_axis)
    assert abs(fl - len(used) * macro) < 1e-4, (fl, len(used), macro)


def test_lowrank_nd_runs_and_balances():
    """Mixed feature + identity axes: split covers all records, sizes within epsilon."""
    rng = np.random.default_rng(2)
    n = 120
    # feature axis: 8 distinct component fingerprints, reused across records
    comps = (rng.random((8, 32)) < 0.3).astype(np.float32)
    feat_idx = rng.integers(0, 8, n)
    afm = {"reactant": {f"C{k}": comps[k] for k in range(8)},
           "solvent": {}}                          # identity axis
    records = [{"reactant": f"C{feat_idx[i]}", "solvent": f"S{rng.integers(4)}"}
               for i in range(n)]
    res = split("lowrank_nd", (records, afm),
                SplitSpec([8, 2], ["train", "test"], seed=0))
    assert set(res.assignment.values()) <= {"train", "test"}
    assert len(res.assignment) == n
    # sizes must lie inside the balance corridor fm_polish actually enforces
    n_train = sum(v == "train" for v in res.assignment.values())
    n_test = sum(v == "test" for v in res.assignment.values())
    caps, floors = capacity_corridor(n, [8, 2], 0.05)     # ordered like splits [train, test]
    assert floors[0] <= n_train <= caps[0], (n_train, floors[0], caps[0])
    assert floors[1] <= n_test <= caps[1], (n_test, floors[1], caps[1])
    assert res.diagnostics["leakage"] is not None


def test_lowrank_nd_beats_random_identity():
    """The optimizer should cut identity-axis leakage well below a random split."""
    records, afm = _identity_nd(n=200, na=15, nb=6, seed=3)
    res = split("lowrank_nd", (records, afm), SplitSpec([8, 2], ["train", "test"], seed=0))
    rng = np.random.default_rng(0)
    rand = (rng.random(len(records)) < 0.2).astype(int)
    rand_macro, _ = macro_axis_lpi(records, afm, rand)
    assert res.diagnostics["leakage"] <= rand_macro + 1e-9, (res.diagnostics["leakage"], rand_macro)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n== {len(fns)} nd tests passed")
