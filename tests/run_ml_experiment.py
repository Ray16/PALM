#!/usr/bin/env python
"""Train regressors on PALM splits to demonstrate inflated metrics from random splitting.

Compares Random (R) vs Cluster (C1e) splitting on molecular property prediction
using Morgan fingerprints + Random Forest.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TESTS_DIR, "output")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint bit vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array([int(fp[i]) for i in range(n_bits)])


def load_split_data(dataset_name, technique, smiles_col, target_col):
    """Load train/test data from PALM split output."""
    split_dir = os.path.join(
        OUTPUT_DIR, dataset_name, "split_result",
        f"{technique}_{dataset_name}"
    )
    train_df = pd.read_csv(os.path.join(split_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, "test.csv"))
    return train_df, test_df


def run_experiment(dataset_name, smiles_col, target_col, dataset_label):
    """Run RF regression on both R and C1e splits, return metrics."""
    print(f"\n{'='*60}")
    print(f"  {dataset_label}")
    print(f"{'='*60}")

    results = {}
    for technique in ["R", "C1e"]:
        tech_label = "Random" if technique == "R" else "Cluster (C1e)"
        print(f"\n  --- {tech_label} split ---")

        train_df, test_df = load_split_data(
            dataset_name, technique, smiles_col, target_col
        )
        print(f"    Train: {len(train_df)}, Test: {len(test_df)}")

        # Compute fingerprints
        print("    Computing fingerprints...")
        X_train = np.array([smiles_to_fingerprint(s)
                            for s in train_df[smiles_col]])
        X_test = np.array([smiles_to_fingerprint(s)
                           for s in test_df[smiles_col]])
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        # Train Random Forest
        print("    Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test  R²: {test_r2:.4f}")
        print(f"    Test RMSE: {test_rmse:.4f}")
        print(f"    Test  MAE: {test_mae:.4f}")

        results[technique] = {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "y_test": y_test,
            "y_pred": y_pred_test,
        }

    return results


def run_mp_experiment():
    """Run RF regression on MP formation energy with MAGPIE features."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from PALM.loaders import load_cif_dir

    DATA_DIR = os.path.join(TESTS_DIR, "data", "mp_cifs_regression")
    MP_OUTPUT = os.path.join(OUTPUT_DIR, "mp_regression")

    print(f"\n{'='*60}")
    print(f"  Materials Project — Formation Energy (500 materials)")
    print(f"{'='*60}")

    # Load features
    feat_path = os.path.join(MP_OUTPUT, "features", "mp_regression", "material", "features.csv")
    feat_df = pd.read_csv(feat_path, index_col=0)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Load labels
    labels_df = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
    rowid_to_energy = dict(zip(labels_df["_row_id"], labels_df["formation_energy_per_atom"]))

    # Map _row_id -> ASE formula
    cif_df = load_cif_dir(DATA_DIR)
    rowid_to_ase_formula = dict(zip(cif_df["_row_id"], cif_df["formula"]))

    # Build master table
    rows = []
    for row_id, ase_formula in rowid_to_ase_formula.items():
        if ase_formula in feat_df.index and row_id in rowid_to_energy:
            rows.append({
                "_row_id": row_id,
                "ase_formula": ase_formula,
                "formation_energy": rowid_to_energy[row_id],
            })
    master_df = pd.DataFrame(rows)
    print(f"  Samples with both features and labels: {len(master_df)}")

    results = {}
    for technique in ["R", "C1e"]:
        tech_label = "Random" if technique == "R" else "Cluster (C1e)"
        print(f"\n  --- {tech_label} split ---")

        split_path = os.path.join(
            MP_OUTPUT, "split_result",
            f"datasail_split_{technique}_mp_regression.csv"
        )
        split_df = pd.read_csv(split_path)
        rowid_to_split = dict(zip(split_df["_row_id"], split_df["split"]))

        valid = master_df[master_df["_row_id"].map(
            lambda rid: rowid_to_split.get(rid) in ("train", "test")
        )].copy()
        valid["split"] = valid["_row_id"].map(rowid_to_split)

        X_all = feat_df.loc[valid["ase_formula"].values].values
        y_all = valid["formation_energy"].values
        splits = valid["split"].values

        train_mask = splits == "train"
        test_mask = splits == "test"
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

        rf = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test  R²: {test_r2:.4f}")
        print(f"    Test RMSE: {test_rmse:.4f}")
        print(f"    Test  MAE: {test_mae:.4f}")

        results[technique] = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "y_test": y_test,
            "y_pred": y_pred_test,
        }

    return results


def plot_experiment_results(esol_results, lipo_results, save_dir,
                            mp_results=None):
    """Generate comprehensive visualization of inflated metrics."""
    os.makedirs(save_dir, exist_ok=True)

    datasets = [
        ("ESOL (Aqueous Solubility)", "log S (mol/L)", esol_results),
        ("Lipophilicity", "log D", lipo_results),
    ]
    if mp_results is not None:
        datasets.append(
            ("Materials Project (Formation Energy)", "eV/atom", mp_results),
        )

    n_rows = len(datasets)
    fig = plt.figure(figsize=(16, 6 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.4, wspace=0.35)

    for row, (title, unit, results) in enumerate(datasets):
        r_res = results["R"]
        c_res = results["C1e"]

        # Panel 1: R² comparison
        ax = fig.add_subplot(gs[row, 0])
        techs = ["Random", "Cluster"]
        r2_vals = [r_res["test_r2"], c_res["test_r2"]]
        colors = ["#94a3b8", "#10b981"]
        bars = ax.bar(techs, r2_vals, color=colors, edgecolor="white",
                      linewidth=1.5, width=0.5)
        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
                    fontsize=12)
        ax.set_ylabel("Test R²")
        ax.set_title(f"{title}\nTest R²", fontweight="bold")
        ax.set_ylim(0, max(r2_vals) * 1.2)

        # Panel 2: RMSE comparison
        ax = fig.add_subplot(gs[row, 1])
        rmse_vals = [r_res["test_rmse"], c_res["test_rmse"]]
        bars = ax.bar(techs, rmse_vals, color=colors, edgecolor="white",
                      linewidth=1.5, width=0.5)
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
                    fontsize=12)
        ax.set_ylabel(f"Test RMSE ({unit})")
        ax.set_title(f"{title}\nTest RMSE", fontweight="bold")

        # Panel 3: Predicted vs Actual scatter (overlay both splits)
        ax = fig.add_subplot(gs[row, 2])
        ax.scatter(r_res["y_test"], r_res["y_pred"], alpha=0.4, s=15,
                   c="#94a3b8", label=f"Random (R²={r_res['test_r2']:.3f})",
                   edgecolors="none")
        ax.scatter(c_res["y_test"], c_res["y_pred"], alpha=0.4, s=15,
                   c="#10b981", label=f"Cluster (R²={c_res['test_r2']:.3f})",
                   edgecolors="none")
        # Diagonal line
        all_vals = np.concatenate([
            r_res["y_test"], r_res["y_pred"],
            c_res["y_test"], c_res["y_pred"],
        ])
        lo, hi = all_vals.min() - 0.5, all_vals.max() + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Measured ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_title(f"{title}\nPredicted vs Measured", fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "Data Leakage Impact on Model Performance\n"
        "Random Forest on Morgan Fingerprints — Random vs Cluster-Based Splitting",
        fontsize=15, fontweight="bold", y=1.02,
    )
    path = os.path.join(save_dir, "ml_leakage_demonstration.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Also save a compact summary bar chart
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax_idx, (title, unit, results) in enumerate(datasets):
        ax = axes[ax_idx]
        r_res = results["R"]
        c_res = results["C1e"]

        metrics = ["Test R²", "Test MAE"]
        random_vals = [r_res["test_r2"], r_res["test_mae"]]
        cluster_vals = [c_res["test_r2"], c_res["test_mae"]]

        x = np.arange(len(metrics))
        w = 0.3
        bars1 = ax.bar(x - w/2, random_vals, w, label="Random Split",
                       color="#94a3b8", edgecolor="white", linewidth=1)
        bars2 = ax.bar(x + w/2, cluster_vals, w, label="Cluster Split",
                       color="#10b981", edgecolor="white", linewidth=1)

        for bar, val in zip(bars1, random_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        for bar, val in zip(bars2, cluster_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(title, fontweight="bold")
        ax.legend(framealpha=0.9)

    fig.suptitle(
        "Random Splitting Inflates Model Performance",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(save_dir, "ml_metrics_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save numerical results as JSON
    summary = {}
    all_results = [("esol", esol_results), ("lipophilicity", lipo_results)]
    if mp_results is not None:
        all_results.append(("mp_regression", mp_results))
    for name, results in all_results:
        summary[name] = {}
        for t in ["R", "C1e"]:
            summary[name][t] = {
                "train_size": results[t]["train_size"],
                "test_size": results[t]["test_size"],
                "train_r2": round(results[t]["train_r2"], 4),
                "test_r2": round(results[t]["test_r2"], 4),
                "test_rmse": round(results[t]["test_rmse"], 4),
                "test_mae": round(results[t]["test_mae"], 4),
            }
    path = os.path.join(save_dir, "ml_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("ML Leakage Demonstration")
    print("=" * 60)
    print("Model: Random Forest (500 trees)")
    print("Features: Morgan Fingerprints / MAGPIE descriptors")
    print("Splits: Random (R) vs Cluster-based (C1e)")

    esol_results = run_experiment(
        "esol", "smiles",
        "measured log solubility in mols per litre",
        "ESOL — Aqueous Solubility (1,128 molecules)",
    )

    lipo_results = run_experiment(
        "lipophilicity", "smiles", "exp",
        "Lipophilicity — logD (4,200 molecules)",
    )

    mp_results = run_mp_experiment()

    print("\n\nGenerating visualizations...")
    save_dir = os.path.join(OUTPUT_DIR, "leakage_demo")
    plot_experiment_results(esol_results, lipo_results, save_dir,
                            mp_results=mp_results)

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: Impact of Data Leakage on Model Performance")
    print("=" * 70)
    print(f"  {'Dataset':<20} {'Split':<10} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<10}")
    print("-" * 70)
    for name, label, results in [
        ("esol", "ESOL", esol_results),
        ("lipophilicity", "Lipophilicity", lipo_results),
        ("mp_regression", "MP Formation E.", mp_results),
    ]:
        for t, t_label in [("R", "Random"), ("C1e", "Cluster")]:
            r = results[t]
            print(f"  {label:<20} {t_label:<10} {r['test_r2']:<10.4f} "
                  f"{r['test_rmse']:<12.4f} {r['test_mae']:<10.4f}")
        infl_r2 = results["R"]["test_r2"] - results["C1e"]["test_r2"]
        print(f"  {'':20} {'Δ R²:':<10} {infl_r2:+.4f} (inflation)")
        print()
    print("=" * 70)
