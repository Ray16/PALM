#!/usr/bin/env python
"""MP Formation Energy Regression — Random vs. Cluster Split Comparison.

Downloads ~500 stable structures + formation energies from Materials Project,
featurizes with PALM's material features, splits with R (random) and C1e
(cluster-based), trains a Random Forest regressor on each split, and compares
MAE / RMSE / R² to demonstrate metric inflation from random splits.

Usage:
    cd /nfs/lambda_stor_01/homes/rzhu
    conda run -n palm python3 PALM/tests/test_mp_regression.py
"""

import csv
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure PALM package is importable (parent of PALM/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from PALM.config import EntityConfig, SplittingConfig, PipelineConfig
from PALM.pipeline import run_pipeline

MP_API_KEY = "7538q37K3CJFCqUWwqYfTiLis7ZzPp2r"

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, "data", "mp_cifs_regression")
OUTPUT_DIR = os.path.join(TESTS_DIR, "output", "mp_regression")

FEATURE_SETS = [
    "magpie_properties",
    "stoichiometry",
    "electronic",
    "bonding",
    "thermodynamic",
    "classification",
]

TARGET_COUNT = 500


# ── Step 1: Download from Materials Project ─────────────────────────────────

def download_mp_data():
    """Download ~500 stable binary/ternary compounds with formation energies."""
    labels_csv = os.path.join(DATA_DIR, "labels.csv")

    # Skip if already downloaded
    if os.path.isdir(DATA_DIR) and os.path.exists(labels_csv):
        with open(labels_csv) as f:
            n = sum(1 for _ in f) - 1
        if n >= TARGET_COUNT - 50:
            print(f"  Already have {n} structures in {DATA_DIR}, skipping download.")
            return

    os.makedirs(DATA_DIR, exist_ok=True)

    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter

    print(f"  Querying Materials Project for ~{TARGET_COUNT} stable compounds...")

    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            is_stable=True,
            num_elements=(2, 3),  # binary and ternary
            fields=[
                "material_id",
                "formula_pretty",
                "formation_energy_per_atom",
                "structure",
            ],
        )

    print(f"  Got {len(docs)} total stable binary/ternary compounds")

    # Take a random subset if we have more than needed
    if len(docs) > TARGET_COUNT:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(docs), TARGET_COUNT, replace=False)
        docs = [docs[i] for i in sorted(indices)]

    print(f"  Using {len(docs)} compounds")

    # Save CIF files and labels
    labels = []
    for doc in docs:
        mid = str(doc.material_id)
        formula = doc.formula_pretty
        e_form = doc.formation_energy_per_atom

        # Filename stem = mp-xxxxx_Formula
        stem = f"{mid}_{formula}"
        cif_path = os.path.join(DATA_DIR, f"{stem}.cif")

        # Write CIF
        writer = CifWriter(doc.structure)
        writer.write_file(cif_path)

        labels.append({
            "_row_id": stem,
            "formula": formula,
            "formation_energy_per_atom": e_form,
        })

    # Save labels CSV
    with open(labels_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["_row_id", "formula", "formation_energy_per_atom"])
        w.writeheader()
        w.writerows(labels)

    print(f"  Saved {len(labels)} CIF files + labels.csv to {DATA_DIR}")


# ── Step 2: Run PALM pipeline (R + C1e splits) ─────────────────────────────

def run_palm_pipeline():
    """Run PALM featurization + splitting with R and C1e techniques."""
    config = PipelineConfig(
        input_file=DATA_DIR,
        output_dir=OUTPUT_DIR,
        dataset_name="mp_regression",
        e1=EntityConfig(
            name="material",
            type="material",
            extract_column="formula",
            feature_sets=FEATURE_SETS,
        ),
        e2=None,
        splitting=SplittingConfig(
            techniques=["R", "C1e"],
            splits=[8, 2],
            names=["train", "test"],
            max_sec=300,
            solver="SCIP",
        ),
    )
    run_pipeline(config)
    print(f"  Pipeline output saved to {OUTPUT_DIR}")


# ── Step 3: Train regressors & compare ──────────────────────────────────────

def train_and_compare():
    """Load features + splits + labels, train RF, compare R vs C1e metrics."""
    from PALM.loaders import load_cif_dir

    # Load features (indexed by ASE formula)
    feat_path = os.path.join(OUTPUT_DIR, "features", "mp_regression", "material", "features.csv")
    feat_df = pd.read_csv(feat_path, index_col=0)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Load labels (_row_id -> formation_energy_per_atom)
    labels_path = os.path.join(DATA_DIR, "labels.csv")
    labels_df = pd.read_csv(labels_path)
    rowid_to_energy = dict(zip(labels_df["_row_id"], labels_df["formation_energy_per_atom"]))

    # Load CIF directory to get _row_id -> ASE formula mapping
    cif_df = load_cif_dir(DATA_DIR)
    rowid_to_ase_formula = dict(zip(cif_df["_row_id"], cif_df["formula"]))

    # Build master table at _row_id level
    rows = []
    for row_id, ase_formula in rowid_to_ase_formula.items():
        if ase_formula in feat_df.index and row_id in rowid_to_energy:
            rows.append({
                "_row_id": row_id,
                "ase_formula": ase_formula,
                "formation_energy": rowid_to_energy[row_id],
            })
    master_df = pd.DataFrame(rows)
    print(f"\n  Samples with both features and labels: {len(master_df)}")

    results = {}

    for technique in ["R", "C1e"]:
        # Load split assignments (_row_id -> split)
        split_path = os.path.join(
            OUTPUT_DIR, "split_result",
            f"datasail_split_{technique}_mp_regression.csv"
        )
        split_df = pd.read_csv(split_path)
        rowid_to_split = dict(zip(split_df["_row_id"], split_df["split"]))

        # Filter to samples that have split assignments
        valid = master_df[master_df["_row_id"].map(
            lambda rid: rowid_to_split.get(rid) in ("train", "test")
        )].copy()
        valid["split"] = valid["_row_id"].map(rowid_to_split)

        # Build feature matrix (look up by ASE formula)
        X_all = feat_df.loc[valid["ase_formula"].values].values
        y_all = valid["formation_energy"].values
        splits = valid["split"].values

        train_mask = splits == "train"
        test_mask = splits == "test"

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        print(f"  [{technique}] Train: {len(X_train)}, Test: {len(X_test)}")

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[technique] = {"MAE": mae, "RMSE": rmse, "R²": r2}

    # Print comparison table
    print("\n" + "=" * 65)
    print("  Random vs. Cluster Split — RF Regression on Formation Energy")
    print("=" * 65)
    print(f"  {'Split':<12} {'MAE (eV/atom)':<18} {'RMSE (eV/atom)':<18} {'R²':<10}")
    print("  " + "-" * 56)
    for tech in ["R", "C1e"]:
        m = results[tech]
        print(f"  {tech:<12} {m['MAE']:<18.4f} {m['RMSE']:<18.4f} {m['R²']:<10.4f}")
    print("  " + "-" * 56)

    # Show inflation
    if "R" in results and "C1e" in results:
        mae_diff = results["C1e"]["MAE"] - results["R"]["MAE"]
        rmse_diff = results["C1e"]["RMSE"] - results["R"]["RMSE"]
        r2_diff = results["R"]["R²"] - results["C1e"]["R²"]
        print(f"\n  Random split inflates metrics:")
        print(f"    MAE  improvement (R over C1e): {mae_diff:+.4f} eV/atom")
        print(f"    RMSE improvement (R over C1e): {rmse_diff:+.4f} eV/atom")
        print(f"    R²   inflation   (R over C1e): {r2_diff:+.4f}")
    print("=" * 65)

    return results


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    print("\n[Step 1/3] Downloading data from Materials Project")
    download_mp_data()

    print("\n[Step 2/3] Running PALM pipeline (featurize + split)")
    run_palm_pipeline()

    print("\n[Step 3/3] Training regressors & comparing splits")
    results = train_and_compare()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
