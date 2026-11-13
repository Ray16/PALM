"""Tests for the DataSAIL / scaffold registry adapters (coverage + grouping policy).

DataSAIL itself is not run here: its output-coverage check is a pure helper, so
missing / duplicate / unknown IDs are exercised directly.
"""

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # directory containing PALM/

from PALM.splitters import SplitSpec, split
from PALM.splitters.methods.adapters import _validate_datasail_coverage

pytest.importorskip("rdkit")

SPEC = SplitSpec([8, 2], ["train", "test"])


# ── DataSAIL coverage ───────────────────────────────────────────────────────

def test_datasail_coverage_ok():
    _validate_datasail_coverage({"a": "train", "b": "test"}, ["a", "b"], ["train", "test"])


@pytest.mark.parametrize("raw, names, pattern", [
    ({"a": "train"}, ["a", "b", "c"], r"2 IDs missing.*'b'"),
    ({"a": "train", "b": "test"}, ["a", "b", "b"], r"1 duplicate input IDs.*'b'"),
    ({"a": "train", "b": "test", "zz": "test"}, ["a", "b"], r"1 unknown IDs.*'zz'"),
    ({"a": "train", "b": "not selected"}, ["a", "b"], r"unknown split labels \['not selected'\]"),
])
def test_datasail_coverage_errors(raw, names, pattern):
    with pytest.raises(RuntimeError, match=pattern):
        _validate_datasail_coverage(raw, names, ["train", "test"])


# ── scaffold grouping policy ────────────────────────────────────────────────

def test_scaffold_invalid_smiles_raise_by_default():
    smiles = {0: "c1ccccc1", 1: "not_a_smiles", 2: "CCO", 3: ""}
    with pytest.raises(ValueError, match=r"2 of 4 SMILES"):
        split("scaffold", smiles, SPEC)


def test_scaffold_invalid_smiles_singleton_option():
    smiles = {0: "c1ccccc1", 1: "not_a_smiles", 2: "xx", 3: "CCO"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = split("scaffold", smiles, SPEC, invalid="singleton")
    assert r.diagnostics["n_invalid"] == 2
    assert r.diagnostics["n_scaffolds"] == 4          # each invalid its own group
    assert len(r.assignment) == 4


def test_scaffold_acyclic_singletons_balance():
    # 10 acyclic + 10 benzene derivatives (one generic scaffold)
    acyc = ["C" * k + "O" for k in range(1, 11)]
    ring = ["c1ccccc1" + "C" * k for k in range(10)]
    smiles = {i: s for i, s in enumerate(acyc + ring)}
    r = split("scaffold", smiles, SplitSpec([1, 1], ["train", "test"]))
    d = r.diagnostics
    assert d["n_acyclic"] == 10 and d["acyclic_policy"] == "singleton"
    assert d["n_scaffolds"] == 11                     # 10 singletons + 1 ring group
    assert d["split_counts"] == {"train": 10, "test": 10}
    assert d["balance_ok"] and not d["giant_group"]
    # ring-scaffold group never straddles splits
    assert len({r.assignment[i] for i in range(10, 20)}) == 1


def test_scaffold_shared_acyclic_giant_group_flagged():
    # 18 acyclic + 2 ring: pooled acyclic group is 90% > 80% train target + tol
    acyc = ["C" * k + "O" for k in range(1, 19)]
    smiles = {i: s for i, s in enumerate(acyc + ["c1ccccc1", "C1CCCCC1C"])}
    with pytest.warns(UserWarning, match="largest scaffold group"):
        r = split("scaffold", smiles, SPEC, acyclic="shared")
    d = r.diagnostics
    assert not d["balance_ok"] and d["giant_group"]
    assert d["largest_group"] == 18
    # default policy on the same data is balanced
    r2 = split("scaffold", smiles, SPEC)
    assert r2.diagnostics["balance_ok"]
    assert r2.diagnostics["split_counts"] == {"train": 16, "test": 4}


def test_scaffold_bad_policy_rejected():
    with pytest.raises(ValueError):
        split("scaffold", {0: "CCO"}, SPEC, acyclic="bogus")
