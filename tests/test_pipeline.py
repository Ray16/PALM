"""pytest integration tests for the PALM pipeline.

Each test exercises one combination of entity types end-to-end:
  - molecule + biomolecule  (Davis DTI)
  - molecule + material     (adsorbate-surface, synthetic)
  - gene + molecule         (gene-drug pharmacogenomics)

Run:
    pytest PALM/tests/test_pipeline.py -v
"""

import os
import subprocess
import pandas as pd
import pytest

PYBIN     = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/palm/bin/python"
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))          # PALM/tests/
PALM_DIR  = os.path.dirname(TESTS_DIR)                           # PALM root
CFG_DIR  = os.path.join(TESTS_DIR, "configs")
OUT_DIR  = os.path.join(TESTS_DIR, "output")

# (config_name, dataset_name, description)
CASES = [
    ("molecule_biomolecule", "davis",             "molecule + biomolecule"),
    ("molecule_material",    "adsorbate_surface",  "molecule + material"),
    ("gene_molecule",        "gene_drug",          "gene + molecule"),
]


# ── Session-scoped fixture: prepare data + run all pipelines once ──────────

@pytest.fixture(scope="session", autouse=True)
def prepare_and_run():
    """Download datasets, write configs, then run all three pipelines."""

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
        cfg_path = os.path.join(CFG_DIR, f"{config_name}.yaml")
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


# ── Per-case checks ────────────────────────────────────────────────────────

@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_split_files_exist(config_name, dataset_name, desc, prepare_and_run):
    """Split CSV files must be produced for every technique."""
    split_dir = os.path.join(OUT_DIR, dataset_name, "split_result")
    assert os.path.isdir(split_dir), f"Split dir missing: {split_dir}"
    csvs = [f for f in os.listdir(split_dir) if f.endswith(".csv")]
    assert len(csvs) > 0, f"No split CSVs in {split_dir}"


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_split_columns_and_labels(config_name, dataset_name, desc, prepare_and_run):
    """Each split CSV must have _row_id and split columns with valid labels."""
    split_dir = os.path.join(OUT_DIR, dataset_name, "split_result")
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
    """Train + test rows should account for ≥70 % of all rows."""
    split_dir = os.path.join(OUT_DIR, dataset_name, "split_result")
    if not os.path.isdir(split_dir):
        pytest.skip("split_result dir not found")

    for fname in os.listdir(split_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(split_dir, fname))
        assigned = df["split"].isin({"train", "test"}).sum()
        coverage = assigned / len(df)
        assert coverage >= 0.70, (
            f"{fname}: only {coverage:.1%} of rows assigned to train/test"
        )


@pytest.mark.parametrize("config_name,dataset_name,desc", CASES)
def test_feature_files_exist(config_name, dataset_name, desc, prepare_and_run):
    """Feature CSV must be produced for each entity axis."""
    feat_root = os.path.join(OUT_DIR, dataset_name, "features", dataset_name)
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
    feat_root = os.path.join(OUT_DIR, dataset_name, "features", dataset_name)
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
