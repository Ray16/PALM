"""
Validate DataSAIL dataset splits for OC22.

Assesses split quality by measuring:
  1. Pair-level leakage (adsorbate, adsorbent co-occurrence in train & test)
  2. Entity-level overlap (adsorbate and adsorbent overlap)
  3. Physical separation (nearest-neighbor distances between train/test entities)

Usage:
  # Validate a single split
  python validation.py output/split_result/datasail_split_C2.csv

  # Validate multiple splits (comparison mode)
  python validation.py output/split_result/datasail_split_R.csv output/split_result/datasail_split_C1f.csv output/split_result/datasail_split_C2.csv

  # Validate all splits in output/split_result/
  python validation.py output/split_result/datasail_split_*.csv
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Validate OC22 DataSAIL splits")
parser.add_argument("split_files", nargs="+", help="Split CSV file(s) to validate")
parser.add_argument("--e-embedding", nargs="+",
                    default=["features/oc22/adsorbate/physchem_features.csv",
                             "features/oc22/adsorbate/composition_features.csv",
                             "features/oc22/adsorbate/rdkit_descriptors_features.csv",
                             "features/oc22/adsorbate/adsorption_features.csv"],
                    help="Adsorbate embedding CSV(s) for distance computation")
parser.add_argument("--f-embedding", nargs="+",
                    default=["features/oc22/adsorbent/property_features.csv",
                             "features/oc22/adsorbent/stoichiometry_features.csv",
                             "features/oc22/adsorbent/electronic_features.csv",
                             "features/oc22/adsorbent/bonding_features.csv",
                             "features/oc22/adsorbent/thermodynamic_features.csv",
                             "features/oc22/adsorbent/catalytic_features.csv"],
                    help="Adsorbent embedding CSV(s) for distance computation")
parser.add_argument("-o", "--output", default="output/figure/split_validation.png",
                    help="Output figure path (default: output/figure/split_validation.png)")
args = parser.parse_args()

# ── Load metadata ─────────────────────────────────────────────────────────
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}
print(f"Total systems: {len(entries)}")

# ── Load embeddings ───────────────────────────────────────────────────────
def load_features(file_list, key_col):
    dfs = []
    for fpath in file_list:
        df = pd.read_csv(fpath)
        feat_cols = [c for c in df.columns if c not in ("system_id", key_col)]
        dfs.append(df[["system_id", key_col] + feat_cols])
    merged = dfs[0]
    for df in dfs[1:]:
        feat_cols = [c for c in df.columns if c not in ("system_id", key_col)]
        merged = merged.merge(df[["system_id"] + feat_cols], on="system_id")
    dedup = merged.drop_duplicates(subset=[key_col]).set_index(key_col)
    feat_cols = [c for c in dedup.columns if c != "system_id"]
    return dedup, feat_cols

ads_dedup, ads_feat_cols = load_features(args.e_embedding, "ads_symbols")
bulk_dedup, bulk_feat_cols = load_features(args.f_embedding, "bulk_symbols")

ads_scaler = StandardScaler().fit(ads_dedup[ads_feat_cols].values.astype(float))
bulk_scaler = StandardScaler().fit(bulk_dedup[bulk_feat_cols].values.astype(float))

print(f"Adsorbate embeddings: {len(ads_dedup)} entities, {len(ads_feat_cols)} features")
print(f"Adsorbent embeddings: {len(bulk_dedup)} entities, {len(bulk_feat_cols)} features")

# ── Derive technique label from filename ──────────────────────────────────
def label_from_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # Try to extract technique name: datasail_split_<TECHNIQUE>__<tags>
    m = re.match(r"datasail_split_([A-Za-z0-9]+)", base)
    if m:
        return m.group(1)
    return base

# ── Analyze each split ────────────────────────────────────────────────────
results = []

for fpath in sorted(args.split_files):
    label = label_from_path(fpath)
    split_df = pd.read_csv(fpath, dtype={"system_id": str})
    split_map = dict(zip(split_df.system_id, split_df.split))

    train_pairs, test_pairs = set(), set()
    train_ads, test_ads = set(), set()
    train_bulk, test_bulk = set(), set()
    n_train, n_test, n_not = 0, 0, 0

    for sid, info in entries.items():
        s = split_map.get(sid, "not selected")
        ads, bulk = info["ads_symbols"], info["bulk_symbols"]
        if s == "train":
            train_pairs.add((ads, bulk))
            train_ads.add(ads)
            train_bulk.add(bulk)
            n_train += 1
        elif s == "test":
            test_pairs.add((ads, bulk))
            test_ads.add(ads)
            test_bulk.add(bulk)
            n_test += 1
        else:
            n_not += 1

    pair_overlap = train_pairs & test_pairs
    ads_overlap = train_ads & test_ads
    bulk_overlap = train_bulk & test_bulk

    # Compute inter-split distances
    def compute_nn_dist(train_set, test_set, dedup_df, feat_cols, scaler):
        train_list = sorted(train_set & set(dedup_df.index))
        test_list = sorted(test_set & set(dedup_df.index))
        if len(train_list) == 0 or len(test_list) == 0:
            return np.nan, np.nan, np.nan, np.nan
        X_tr = scaler.transform(dedup_df.loc[train_list, feat_cols].values.astype(float))
        X_te = scaler.transform(dedup_df.loc[test_list, feat_cols].values.astype(float))
        D = cdist(X_te, X_tr, metric="euclidean")
        nn_dists = D.min(axis=1)
        return D.mean(), np.median(D), D.min(), nn_dists.mean()

    bulk_mean, bulk_med, bulk_min, bulk_nn = compute_nn_dist(
        train_bulk, test_bulk, bulk_dedup, bulk_feat_cols, bulk_scaler)
    ads_mean, ads_med, ads_min, ads_nn = compute_nn_dist(
        train_ads, test_ads, ads_dedup, ads_feat_cols, ads_scaler)

    r = {
        "label": label,
        "file": fpath,
        "n_train": n_train,
        "n_test": n_test,
        "n_not_selected": n_not,
        "pair_overlap": len(pair_overlap),
        "pair_overlap_pct": 100 * len(pair_overlap) / max(len(test_pairs), 1),
        "ads_overlap": len(ads_overlap),
        "bulk_overlap": len(bulk_overlap),
        "bulk_nn_mean": bulk_nn,
        "bulk_dist_mean": bulk_mean,
        "ads_nn_mean": ads_nn,
        "ads_dist_mean": ads_mean,
    }
    results.append(r)

# ── Print summary table ──────────────────────────────────────────────────
print("\n" + "=" * 100)
print("SPLIT QUALITY ASSESSMENT")
print("=" * 100)

header = (f"{'Technique':<10} {'Train':>7} {'Test':>7} {'NotSel':>7} "
          f"{'Pair OL':>8} {'Ads OL':>7} {'Bulk OL':>8} "
          f"{'Ads NN':>8} {'Bulk NN':>8}")
print(header)
print("-" * 100)

for r in results:
    ads_nn_str = f"{r['ads_nn_mean']:.2f}" if not np.isnan(r['ads_nn_mean']) else "shared"
    bulk_nn_str = f"{r['bulk_nn_mean']:.2f}" if not np.isnan(r['bulk_nn_mean']) else "shared"
    print(f"{r['label']:<10} {r['n_train']:>7,} {r['n_test']:>7,} {r['n_not_selected']:>7,} "
          f"{r['pair_overlap']:>8} {r['ads_overlap']:>7} {r['bulk_overlap']:>8} "
          f"{ads_nn_str:>8} {bulk_nn_str:>8}")

print()
print("Pair OL  = number of (adsorbate, adsorbent) pairs appearing in both train and test")
print("Ads OL   = number of adsorbates in both splits (0 = cold-adsorbate)")
print("Bulk OL  = number of adsorbents in both splits (0 = cold-adsorbent)")
print("Ads NN   = mean nearest-neighbor distance from test adsorbates to train (higher = more different)")
print("Bulk NN  = mean nearest-neighbor distance from test adsorbents to train (higher = more different)")

# ── Visualization ─────────────────────────────────────────────────────────
if len(results) < 2:
    print(f"\nOnly 1 split provided, skipping comparison chart.")
    print("Pass multiple split files to generate a grouped bar comparison.")
    exit(0)

labels = [r["label"] for r in results]
n = len(labels)
x = np.arange(n)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3,
                       left=0.08, right=0.95, top=0.93, bottom=0.06)

bar_color = "#4C72B0"
highlight_color = "#DD8452"

# ── (a) Pair-level overlap ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
vals = [r["pair_overlap"] for r in results]
colors = [highlight_color if v > 0 else bar_color for v in vals]
bars = ax1.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(max(vals), 1),
             str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha="right")
ax1.set_ylabel("Count")
ax1.set_title("(a) Pair-Level Overlap (adsorbate, adsorbent)", fontweight="bold", fontsize=13)

# ── (b) Entity-level overlap ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
width = 0.35
ads_ol = [r["ads_overlap"] for r in results]
bulk_ol = [r["bulk_overlap"] for r in results]
ax2.bar(x - width / 2, ads_ol, width, label="Adsorbate overlap",
        color="#4C72B0", edgecolor="black", linewidth=0.5)
ax2.bar(x + width / 2, bulk_ol, width, label="Adsorbent overlap",
        color="#DD8452", edgecolor="black", linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_ylabel("Count")
ax2.set_title("(b) Entity-Level Overlap", fontweight="bold", fontsize=13)
ax2.legend(frameon=True)

# ── (c) Nearest-neighbor distances ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ads_nn = [r["ads_nn_mean"] if not np.isnan(r["ads_nn_mean"]) else 0 for r in results]
bulk_nn = [r["bulk_nn_mean"] if not np.isnan(r["bulk_nn_mean"]) else 0 for r in results]
ax3.bar(x - width / 2, ads_nn, width, label="Adsorbate NN dist",
        color="#4C72B0", edgecolor="black", linewidth=0.5)
ax3.bar(x + width / 2, bulk_nn, width, label="Adsorbent NN dist",
        color="#DD8452", edgecolor="black", linewidth=0.5)
for i, (a, b) in enumerate(zip(ads_nn, bulk_nn)):
    if a > 0:
        ax3.text(i - width / 2, a + 0.05, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
    if b > 0:
        ax3.text(i + width / 2, b + 0.05, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45, ha="right")
ax3.set_ylabel("Mean NN Distance (standardized)")
ax3.set_title("(c) Physical Separation: Mean Nearest-Neighbor Distance (test -> train)",
              fontweight="bold", fontsize=13)
ax3.legend(frameon=True)

# ── (d) Split size distribution ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
trains = [r["n_train"] for r in results]
tests = [r["n_test"] for r in results]
nots = [r["n_not_selected"] for r in results]
ax4.bar(x, trains, label="Train", color="#4C72B0", edgecolor="black", linewidth=0.5)
ax4.bar(x, tests, bottom=trains, label="Test", color="#DD8452", edgecolor="black", linewidth=0.5)
ax4.bar(x, nots, bottom=[t + te for t, te in zip(trains, tests)],
        label="Not selected", color="#AAAAAA", edgecolor="black", linewidth=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, rotation=45, ha="right")
ax4.set_ylabel("Number of Systems")
ax4.set_title("(d) Split Size Distribution", fontweight="bold", fontsize=13)
ax4.legend(frameon=True)

# ── (e) Combined leakage score ───────────────────────────────────────────
# Higher = better separation. Score = ads_nn + bulk_nn (0 if shared)
ax5 = fig.add_subplot(gs[2, :])
combined = [a + b for a, b in zip(ads_nn, bulk_nn)]
colors = plt.cm.RdYlGn(np.array(combined) / max(max(combined), 1e-9))
bars = ax5.barh(x, combined, color=colors, edgecolor="black", linewidth=0.6)
for i, (bar, v) in enumerate(zip(bars, combined)):
    ax5.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
             f"{v:.2f}", ha="left", va="center", fontsize=12, fontweight="bold")
    # Annotate components
    ax5.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
             f"ads={ads_nn[i]:.2f} + bulk={bulk_nn[i]:.2f}",
             ha="center", va="center", fontsize=9, color="black", alpha=0.7)
ax5.set_yticks(x)
ax5.set_yticklabels(labels)
ax5.set_xlabel("Combined NN Distance (Adsorbate + Adsorbent)")
ax5.set_title("(e) Overall Physical Separation Score (higher = less leakage)",
              fontweight="bold", fontsize=13)
ax5.invert_yaxis()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
fig.savefig(args.output, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {args.output}")
plt.close()
