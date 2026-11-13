"""Robustness of the splitting pipeline and the splitters on awkward inputs.

Covers what a user can get wrong (unsafe names, invalid SMILES, duplicates,
missing values, tiny datasets, 3-way splits) and invariants every split must
keep (each entity in exactly one split, exact counts, duplicates kept
together, nothing written outside ``output_dir``).

Run from the directory containing PALM/ (CPU is enough):
    CUDA_VISIBLE_DEVICES= python -m pytest PALM/tests/test_robustness.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import PALM.splitters as ps
from PALM.benchmarks.datasail_pipeline import splitting
from PALM.benchmarks.datasail_pipeline.config import EntityConfig, PipelineConfig, load_config
from PALM.benchmarks.datasail_pipeline.pipeline import run_pipeline

DEMO = Path(__file__).resolve().parents[1] / "webapp" / "demo_data" / "esol.csv"
UNSAFE_NAMES = ["../escape", "../../../tmp/pwned", "/tmp/abs_override", "a/b", "a\\b",
                "..", ".", "", "   ", "x\0y", "a" * 101]


def _entity(name="compound"):
    return EntityConfig(name=name, type="molecule", extract_column="smiles",
                        feature_sets=["morgan_fingerprint"])


def _smiles(n):
    """``n`` distinct valid SMILES from the bundled (public) ESOL demo."""
    s = pd.read_csv(DEMO)["smiles"].drop_duplicates()
    assert len(s) >= n
    return list(s.iloc[:n])


def _run(tmp_path, rows, splits=(8, 2), names=("train", "test"), techniques=("R",)):
    """Write ``rows`` (list of dicts) to a CSV, run the pipeline from a YAML config."""
    csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = {
        "input_file": str(csv), "output_dir": str(tmp_path / "out"), "dataset_name": "robust",
        "e1": {"name": "compound", "type": "molecule", "extract": {"column": "smiles"},
               "feature_sets": ["morgan_fingerprint"]},
        "splitting": {"techniques": list(techniques), "splits": list(splits), "names": list(names)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run_pipeline(load_config(str(path)))
    out = tmp_path / "out"
    comparison = json.loads(next(out.rglob("comparison_robust.json")).read_text())
    return out, comparison


def _split_table(out, technique="R"):
    """The exported input rows with their split label, for one technique."""
    path = next(p for p in out.rglob("*_with_splits.csv") if f"{technique}_" in p.name)
    return pd.read_csv(path)


def _split_column(df, names):
    col = next(c for c in df.columns if set(df[c].dropna().astype(str)) <= set(names) and df[c].notna().any())
    return df[col].astype(str)


# ── 1. names that become paths ─────────────────────────────────────────────

@pytest.mark.parametrize("bad", UNSAFE_NAMES)
def test_unsafe_dataset_name_is_rejected(bad, tmp_path):
    with pytest.raises(ValueError):
        PipelineConfig(input_file="x.csv", output_dir=str(tmp_path), dataset_name=bad, e1=_entity())


@pytest.mark.parametrize("bad", UNSAFE_NAMES)
def test_unsafe_entity_name_is_rejected(bad):
    with pytest.raises(ValueError):
        _entity(bad)


@pytest.mark.parametrize("ok", ["compound", "my compound", "v1.2", "drug-target_1", "Verbindung_ä"])
def test_harmless_names_are_accepted(ok, tmp_path):
    PipelineConfig(input_file="x.csv", output_dir=str(tmp_path), dataset_name=ok, e1=_entity(ok))


def test_identical_entity_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="different names"):
        PipelineConfig(input_file="x.csv", output_dir=str(tmp_path), dataset_name="d",
                       e1=_entity("drug"), e2=_entity("drug"))


def test_yaml_config_with_traversal_is_rejected(tmp_path):
    cfg = {"input_file": "x.csv", "output_dir": str(tmp_path), "dataset_name": "../../outside",
           "e1": {"name": "compound", "type": "molecule", "extract": {"column": "smiles"}},
           "splitting": {"techniques": ["R"]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="path separators"):
        load_config(str(path))


# ── 2. the pipeline on awkward data ─────────────────────────────────────────

def test_pipeline_handles_invalid_duplicate_and_missing_values(tmp_path):
    good = _smiles(40)
    rows = [{"smiles": s, "y": float(i)} for i, s in enumerate(good)]
    rows += [{"smiles": s, "y": 0.0} for s in ("not_a_smiles", "C1CC(", "Q")]     # unparseable
    rows += [{"smiles": s, "y": 1.0} for s in good[:5]]                           # duplicate entities
    rows.append({"smiles": good[6], "y": np.nan})                                 # missing target
    out, comparison = _run(tmp_path, rows)
    assert "R" in comparison["techniques"] and not comparison["failed"], comparison["failed"]
    df = _split_table(out)
    labels = _split_column(df, {"train", "test"})
    assert set(labels) == {"train", "test"}
    # every row of a duplicated entity lands in the same split
    by_entity = pd.DataFrame({"smiles": df["smiles"], "split": labels}).groupby("smiles")["split"].nunique()
    assert (by_entity == 1).all(), by_entity[by_entity > 1]


def test_pipeline_three_way_split(tmp_path):
    rows = [{"smiles": s, "y": float(i)} for i, s in enumerate(_smiles(120))]
    out, comparison = _run(tmp_path, rows, splits=(70, 15, 15), names=("train", "val", "test"))
    assert "R" in comparison["techniques"] and not comparison["failed"], comparison["failed"]
    labels = _split_column(_split_table(out), {"train", "val", "test"})
    frac = labels.value_counts(normalize=True)
    assert set(frac.index) == {"train", "val", "test"}
    assert abs(frac["train"] - 0.70) < 0.1 and abs(frac["val"] - 0.15) < 0.1


def test_pipeline_tiny_dataset(tmp_path):
    rows = [{"smiles": s, "y": float(i)} for i, s in enumerate(_smiles(12))]
    out, comparison = _run(tmp_path, rows)
    assert "R" in comparison["techniques"] and not comparison["failed"], comparison["failed"]
    assert set(_split_column(_split_table(out), {"train", "test"})) == {"train", "test"}


def test_pipeline_writes_only_inside_output_dir(tmp_path):
    rows = [{"smiles": s, "y": float(i)} for i, s in enumerate(_smiles(30))]
    _run(tmp_path, rows)
    outside = [p for p in tmp_path.rglob("*")
               if p.is_file() and "out" not in p.relative_to(tmp_path).parts[:1]]
    assert sorted(p.name for p in outside) == ["config.yaml", "input.csv"]


def test_infeasible_datasail_split_reports_a_clear_error(monkeypatch):
    """DataSAIL returns (None, None, None) when its solver finds no split."""
    monkeypatch.setattr(splitting, "datasail", lambda **kw: (None, None, None))
    kwargs = {"e_data": {"a": 1, "b": 2}, "f_data": {"x": 1}, "splits": [8, 2]}
    technique, result, _, error = splitting.run_technique("I2", kwargs)
    assert result is None and "no feasible I2 split" in error and "2 e1 / 1 e2" in error
    technique, result, _, error = splitting.run_technique_1d("C1e", kwargs, use_self_inter=False)
    assert result is None and "no feasible C1e split" in error


# ── 3. splitter invariants ──────────────────────────────────────────────────

def _fingerprints(n, seed=0, dim=256):
    rng = np.random.default_rng(seed)
    return {f"m{i}": (rng.random(dim) < 0.08).astype(np.float32) for i in range(n)}


@pytest.mark.parametrize("method", ["random", "lowrank", "hypergraph"])
@pytest.mark.parametrize("splits,names", [((8, 2), ("train", "test")),
                                          ((70, 15, 15), ("train", "val", "test"))])
def test_every_entity_lands_in_exactly_one_split(method, splits, names):
    data = _fingerprints(103)                                    # odd size on purpose
    res = ps.split(method, data, ps.SplitSpec(list(splits), list(names), seed=0))
    assert set(res.assignment) == set(data)
    assert set(res.assignment.values()) <= set(names)
    counts = res.diagnostics["split_counts"]
    assert sum(counts.values()) == len(data) and all(counts[n] > 0 for n in names)


def test_lowrank_hits_requested_counts_exactly():
    data = _fingerprints(103)
    counts = ps.split("lowrank", data, ps.SplitSpec([70, 15, 15], ["train", "val", "test"],
                                                    seed=1)).diagnostics["split_counts"]
    targets = {n: 103 * f / 100 for n, f in zip(["train", "val", "test"], [70, 15, 15])}
    assert all(abs(counts[n] - targets[n]) < 1 for n in counts), (counts, targets)


def test_lowrank_with_duplicate_and_all_zero_rows():
    rng = np.random.default_rng(3)
    base = (rng.random(128) < 0.1).astype(np.float32)
    data = {f"dup{i}": base.copy() for i in range(40)}                      # identical rows
    data.update({f"zero{i}": np.zeros(128, np.float32) for i in range(20)})  # no bits set
    data.update(_fingerprints(40, seed=4, dim=128))
    res = ps.split("lowrank", data, ps.SplitSpec([8, 2], ["train", "test"], seed=0))
    assert set(res.assignment) == set(data)
    assert res.diagnostics["split_counts"] == {"train": 80, "test": 20}


@pytest.mark.parametrize("method", ["astartes", "lohi"])
def test_two_way_only_baselines_refuse_three_way_specs(method):
    data = {f"m{i}": "CCO" for i in range(30)} if method == "lohi" else _fingerprints(30)
    with pytest.raises(ValueError, match="two-way"):
        ps.split(method, data, ps.SplitSpec([70, 15, 15], ["train", "val", "test"]))


def test_split_indices_follow_input_order():
    res = ps.SplitResult(assignment={"2": "train", "10": "test", "1": "train"}, method="x")
    assert res.split_indices() == {"train": [0, 2], "test": [1]}
    assert res.split_indices(["1", "2", "10"]) == {"train": [0, 1], "test": [2]}


def test_split_spec_validation():
    with pytest.raises(ValueError):
        ps.SplitSpec([8, 2], ["train"])
    with pytest.raises(ValueError):
        ps.SplitSpec([8, -2], ["train", "test"])
