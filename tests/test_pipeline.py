"""pytest integration tests for the PALM pipeline.

Each test exercises one combination of entity types end-to-end:
  - molecule + biomolecule  (Davis DTI)
  - molecule + material     (adsorbate-surface, synthetic)
  - gene + molecule         (gene-drug pharmacogenomics)

Pipeline outputs are written to a temporary directory (not the committed
tests/output/) so a test run never overwrites checked-in results.

The unit tests at the bottom (validation, per-axis overlap, cache keys) run
in-process and do not need the integration fixture.

Run:
    pytest PALM/tests/test_pipeline.py -v
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PYBIN     = sys.executable
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))          # PALM/tests/
PALM_DIR  = os.path.dirname(TESTS_DIR)                           # PALM root
CFG_DIR  = os.path.join(TESTS_DIR, "configs")

# (config_name, dataset_name, description)
CASES = [
    ("molecule_biomolecule", "davis",             "molecule + biomolecule"),
    ("molecule_material",    "adsorbate_surface",  "molecule + material"),
    ("gene_molecule",        "gene_drug",          "gene + molecule"),
]


# ── Session-scoped fixture: prepare data + run all pipelines once ──────────

@pytest.fixture(scope="session")
def prepare_and_run(tmp_path_factory):
    """Download datasets, write configs, then run all three pipelines.

    Returns the output root; each case writes to <root>/<dataset_name>.
    """
    out_root = str(tmp_path_factory.mktemp("pipeline_output"))

    # Step 1: prepare data and configs
    result = subprocess.run(
        [PYBIN, os.path.join(TESTS_DIR, "prepare_data.py")],
        cwd=PALM_DIR, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        pytest.fail(f"prepare_data.py failed:\n{result.stderr}")

    # Step 2: run each pipeline; collect errors without short-circuiting
    failures = {}
    for config_name, dataset_name, desc in CASES:
        with open(os.path.join(CFG_DIR, f"{config_name}.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        cfg["output_dir"] = os.path.join(out_root, dataset_name)
        cfg_path = os.path.join(out_root, f"{config_name}.yaml")
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg, fh)
        result = subprocess.run(
            [PYBIN, "-m", "PALM", cfg_path],
            cwd=os.path.dirname(PALM_DIR), capture_output=True, text=True, timeout=600,
        )
        print(f"\n=== {desc} (stdout) ===\n{result.stdout}")
        if result.returncode != 0:
            failures[config_name] = result.stderr
        else:
            print(f"PASS: {desc}")

    if failures:
        msg = "\n".join(f"[{k}]\n{v}" for k, v in failures.items())
        pytest.fail(f"Pipeline(s) failed:\n{msg}")
    return out_root


# ── Per-case checks ────────────────────────────────────────────────────────

@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_split_files_exist(config_name, dataset_name, desc, prepare_and_run):
    """Split CSV files must be produced for every technique."""
    split_dir = os.path.join(prepare_and_run, dataset_name, "split_result")
    assert os.path.isdir(split_dir), f"Split dir missing: {split_dir}"
    csvs = [f for f in os.listdir(split_dir) if f.endswith(".csv")]
    assert len(csvs) > 0, f"No split CSVs in {split_dir}"


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_split_columns_and_labels(config_name, dataset_name, desc, prepare_and_run):
    """Each split CSV must have _row_id and split columns with valid labels."""
    split_dir = os.path.join(prepare_and_run, dataset_name, "split_result")
    if not os.path.isdir(split_dir):
        pytest.skip("split_result dir not found")

    valid = {"train", "test", "not selected"}
    for fname in os.listdir(split_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(split_dir, fname))
        assert "_row_id" in df.columns, f"_row_id missing in {fname}"
        assert "split"   in df.columns, f"split missing in {fname}"
        bad = set(df["split"].unique()) - valid
        assert not bad, f"Unexpected labels in {fname}: {bad}"


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_split_coverage(config_name, dataset_name, desc, prepare_and_run):
    """Published splits must be complete, non-empty, and near the 80/20 target."""
    split_dir = os.path.join(prepare_and_run, dataset_name, "split_result")
    if not os.path.isdir(split_dir):
        pytest.skip("split_result dir not found")

    for fname in os.listdir(split_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(split_dir, fname))
        assert df["split"].isin({"train", "test"}).all(), (
            f"{fname}: published split has unassigned rows"
        )
        for name, target in (("train", 0.8), ("test", 0.2)):
            frac = (df["split"] == name).mean()
            assert frac > 0, f"{fname}: '{name}' split is empty"
            assert abs(frac - target) <= 0.1, f"{fname}: {name} is {frac:.1%}"


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_comparison_status(config_name, dataset_name, desc, prepare_and_run):
    """Every technique is reported as valid or failed; failed ones are not
    published and never recommended."""
    out = os.path.join(prepare_and_run, dataset_name)
    with open(os.path.join(out, "metrics", f"comparison_{dataset_name}.json")) as fh:
        comp = json.load(fh)
    assert comp["status"], "no technique status recorded"
    published = {f[len("datasail_split_"):-len(f"_{dataset_name}.csv")]
                 for f in os.listdir(os.path.join(out, "split_result"))
                 if f.startswith("datasail_split_") and f.endswith(".csv")}
    for t, status in comp["status"].items():
        if status == "failed":
            assert t not in published, f"failed technique {t} was published"
            assert comp["failed"][t], f"no failure reason for {t}"
        else:
            assert t in published
    if comp["recommendation"] is not None:
        assert comp["status"][comp["recommendation"]] == "valid"
        assert comp["recommendation"] in comp["pareto_set"]


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_feature_files_exist(config_name, dataset_name, desc, prepare_and_run):
    """Feature CSV must be produced for each entity axis."""
    feat_root = os.path.join(prepare_and_run, dataset_name, "features", dataset_name)
    assert os.path.isdir(feat_root), f"Feature root missing: {feat_root}"

    for entity_dir in os.listdir(feat_root):
        feat_path = os.path.join(feat_root, entity_dir, "features.csv")
        assert os.path.exists(feat_path), f"Missing: {feat_path}"
        df = pd.read_csv(feat_path, index_col=0)
        assert len(df) > 0,       f"Empty feature file: {feat_path}"
        assert df.shape[1] > 0,   f"No feature columns: {feat_path}"
        assert not df.isnull().all(axis=None), f"All-NaN features: {feat_path}"


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_feature_no_allzero_rows(config_name, dataset_name, desc, prepare_and_run):
    """No entity should have an all-zero feature vector (indicates featurisation failure)."""
    feat_root = os.path.join(prepare_and_run, dataset_name, "features", dataset_name)
    if not os.path.isdir(feat_root):
        pytest.skip("feature dir not found")

    for entity_dir in os.listdir(feat_root):
        feat_path = os.path.join(feat_root, entity_dir, "features.csv")
        if not os.path.exists(feat_path):
            continue
        df = pd.read_csv(feat_path, index_col=0).fillna(0)
        all_zero = (df == 0).all(axis=1)
        zero_ids = df.index[all_zero].tolist()
        assert not zero_ids, (
            f"{feat_path}: all-zero rows for entities: {zero_ids[:5]}"
        )


# ── Unit tests (no integration fixture) ────────────────────────────────────

def _splitting(**kw):
    from PALM.benchmarks.datasail_pipeline.config import SplittingConfig
    return SplittingConfig(techniques=["R", "I2"], **kw)


def test_validate_split_rejects_incomplete_and_empty():
    from PALM.benchmarks.datasail_pipeline.pipeline import _validate_split
    ok = ["train"] * 8 + ["test"] * 2
    assert _validate_split(ok, _splitting()) == []

    partial = ["train"] * 8 + ["not selected"] * 2       # empty test, 80% coverage
    reasons = _validate_split(partial, _splitting())
    assert any("incomplete" in r for r in reasons)
    assert any("'test' is empty" in r for r in reasons)

    # allow_incomplete only waives completeness, not size/non-empty checks
    partial_ok = ["train"] * 8 + ["test"] * 2 + ["not selected"] * 5
    assert _validate_split(partial_ok, _splitting(allow_incomplete=True)) == []
    off_size = ["train"] * 5 + ["test"] * 5
    assert any("target" in r for r in _validate_split(off_size, _splitting()))


def test_incomplete_technique_marked_failed(tmp_path, monkeypatch):
    """End-to-end with a stubbed solver: an incomplete I2 split is failed,
    not exported, and not recommended; the complete R split is published."""
    from PALM.benchmarks.datasail_pipeline import pipeline as pl
    from PALM.benchmarks.datasail_pipeline.config import EntityConfig, PipelineConfig

    rng = np.random.default_rng(0)
    e1 = [f"A{i}" for i in range(10)]
    e2 = [f"B{j}" for j in range(6)]
    rows = [(a, b) for a in e1 for b in e2]
    csv = tmp_path / "inter.csv"
    pd.DataFrame(rows, columns=["a", "b"]).to_csv(csv, index=False)

    def fake_featurize(entities, entity_config):
        ids = sorted(entities)
        return pd.DataFrame(rng.normal(size=(len(ids), 4)), index=ids)

    def fake_run_splitting(e1_data, e2_data, interactions, config, failures=None):
        r = {k: ("test" if i % 5 == 0 else "train") for i, k in enumerate(interactions)}
        # double-cold style: cross-split interactions dropped, test left empty
        i2 = {k: ("train" if k[0] in e1[:6] else "not selected") for k in interactions}
        return {"R": r, "I2": i2}

    monkeypatch.setattr(pl, "_featurize", fake_featurize)
    monkeypatch.setattr(pl, "run_splitting", fake_run_splitting)
    monkeypatch.setattr(pl, "generate_split_plots", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_reduce", lambda *a, **k: None)

    cfg = PipelineConfig(
        input_file=str(csv), output_dir=str(tmp_path / "out"), dataset_name="toy",
        e1=EntityConfig(name="a", type="molecule", extract_column="a"),
        e2=EntityConfig(name="b", type="molecule", extract_column="b"),
        splitting=_splitting(),
    )
    res = pl.run_pipeline(cfg)
    assert res["valid"] == ["R"]
    assert "I2" in res["failed"]
    split_dir = tmp_path / "out" / "split_result"
    assert (split_dir / "datasail_split_R_toy.csv").exists()
    assert not (split_dir / "datasail_split_I2_toy.csv").exists()
    assert (split_dir / "failed" / "datasail_split_I2_toy.csv").exists()
    comp = json.loads((tmp_path / "out" / "metrics" / "comparison_toy.json").read_text())
    assert comp["status"] == {"R": "valid", "I2": "failed"}
    assert comp["recommendation"] == "R"
    assert "I2" not in comp["techniques"]

    # If every technique is invalid, the pipeline must not report success.
    monkeypatch.setattr(pl, "run_splitting",
                        lambda *a, **k: {"I2": fake_run_splitting(*a, **k)["I2"]})
    with pytest.raises(RuntimeError, match="No technique produced a valid split"):
        pl.run_pipeline(cfg)


def test_row_metrics_overlap_all_split_pairs():
    """Overlap is reported for every split pair and per axis (e1 and e2)."""
    from PALM.benchmarks.datasail_pipeline.metrics import compute_row_split_metrics

    feats = {k: np.array([float(i), float(i) ** 2]) for i, k in enumerate("abcdef")}
    e1 = ["a", "b", "c", "d", "e", "e"]     # "e" spans val and test only
    e2 = ["a", "a", "b", "c", "d", "f"]     # "a" spans train only
    splits = ["train", "train", "val", "val", "val", "test"]
    m = compute_row_split_metrics(
        [("e1", "x", feats, e1), ("e2", "y", feats, e2)],
        splits, ["train", "val", "test"],
    )
    p1 = m["axes"]["e1"]["overlap_pairs"]
    assert set(p1) == {"train&val", "train&test", "val&test"}
    assert p1["val&test"]["count"] == 1 and p1["train&test"]["count"] == 0
    assert m["axes"]["e1"]["multi_split_count"] == 1
    assert m["axes"]["e2"]["multi_split_count"] == 0
    # shared entity "e" -> zero NN distance on the val->test pair of axis e1
    assert m["axes"]["e1"]["nn_leakage_pairs"]["val->test"]["zero_dist_count"] == 1
    assert m["entity_overlap"]["e1_overlap"] == 1 and m["coverage"] == 1.0


def test_cache_key_covers_preprocessing(tmp_path):
    from PALM.benchmarks.datasail_pipeline.cache import (
        _cache_key, dist_spec, file_fingerprint,
    )
    data = {"a": np.zeros(2000), "b": np.ones(2000)}
    base = _cache_key(data, "e1", [], dist_spec(False, False, 20, None))
    assert base == _cache_key(data, "e1", [], dist_spec(False, False, 20, None))
    assert base != _cache_key(data, "e1", [], dist_spec(True, False, 20, None))
    assert base != _cache_key(data, "e1", [], dist_spec(False, True, 20, None))
    assert (_cache_key(data, "e1", [], dist_spec(False, True, 20, None))
            != _cache_key(data, "e1", [], dist_spec(False, True, 10, None)))

    # Values differing only in the middle of a long vector (str() elides it)
    data2 = {"a": np.zeros(2000), "b": np.ones(2000)}
    data2["a"][1000] = 1.0
    assert base != _cache_key(data2, "e1", [], dist_spec(False, False, 20, None))

    f = tmp_path / "emb.npy"
    f.write_bytes(b"one")
    fp1 = file_fingerprint(str(f))
    f.write_bytes(b"two")
    os.utime(f, ns=(1, 1))  # force a distinct mtime so the memo can't mask it
    assert fp1 != file_fingerprint(str(f))
