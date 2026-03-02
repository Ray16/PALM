"""
Summary dashboard for comparing DataSAIL split results across feature pairs.

Generates overview figures showing raw coverage and overlap metrics so you can
visually identify the best (method, feature pair) combination.

Outputs:
  - Per-method figures: coverage bar + overlap bar for each feature pair
  - Cross-method figure: coverage heatmap + overlap heatmap side by side

Usage:
  python 3_summary.py
  python 3_summary.py --method C1e
  python 3_summary.py -o output/summary
"""

import argparse
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Summary dashboard for DataSAIL splits")
parser.add_argument("split_files", nargs="*", default=None,
                    help="Split CSV file(s). If omitted, uses all output/split_result/datasail_split_*.csv")
parser.add_argument("--method", type=str, default=None,
                    help="Only show specific method (R, I1e, I1f, I2, C1e, C1f, C2)")
parser.add_argument("-o", "--output-dir", default="output/summary",
                    help="Output directory for summary figures")
args = parser.parse_args()

if not args.split_files:
    args.split_files = sorted(glob.glob("output/split_result/datasail_split_*.csv"))
    if not args.split_files:
        print("No split CSVs found. Pass files explicitly.")
        exit(1)

os.makedirs(args.output_dir, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
TRAIN_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"

# ── Load metadata ─────────────────────────────────────────────────────────
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}
total_systems = len(entries)
print(f"Total systems: {total_systems}")

master = {}
for sid, info in entries.items():
    master[sid] = (info["ads_symbols"], info["bulk_symbols"])

# ── Parse filename ────────────────────────────────────────────────────────
def parse_filename(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"datasail_split_([A-Za-z0-9]+)(?:__(.+))?", base)
    if not m:
        return None, "unknown", "unknown"
    method = m.group(1)
    feature_tag = m.group(2) if m.group(2) else ""
    e_feat, f_feat = "default", "default"
    if feature_tag:
        e_match = re.search(r"e_([^_]+(?:\+[^_]+)*)", feature_tag)
        f_match = re.search(r"f_(.+)", feature_tag)
        if e_match:
            e_feat = e_match.group(1)
        if f_match:
            f_feat = f_match.group(1)
    return method, e_feat, f_feat

# ── Compute metrics for each split file ───────────────────────────────────
print("Computing metrics for all split files...")
records = []

for fpath in args.split_files:
    method, e_feat, f_feat = parse_filename(fpath)
    if method is None:
        continue
    if args.method and method != args.method:
        continue

    split_df = pd.read_csv(fpath, dtype={"system_id": str})
    split_map = dict(zip(split_df["system_id"], split_df["split"]))

    train_ads, test_ads = set(), set()
    train_bulk, test_bulk = set(), set()
    train_pairs, test_pairs = set(), set()
    n_train, n_test = 0, 0

    for sid, (ads, bulk) in master.items():
        s = split_map.get(sid, "not selected")
        if s == "train":
            n_train += 1
            train_ads.add(ads)
            train_bulk.add(bulk)
            train_pairs.add((ads, bulk))
        elif s == "test":
            n_test += 1
            test_ads.add(ads)
            test_bulk.add(bulk)
            test_pairs.add((ads, bulk))

    n_assigned = n_train + n_test
    coverage_pct = 100 * n_assigned / total_systems
    ads_overlap = len(train_ads & test_ads)
    bulk_overlap = len(train_bulk & test_bulk)
    pair_overlap = len(train_pairs & test_pairs)

    records.append({
        "method": method,
        "e_feat": e_feat,
        "f_feat": f_feat,
        "label": f"{e_feat} × {f_feat}",
        "coverage_pct": coverage_pct,
        "n_train": n_train,
        "n_test": n_test,
        "n_assigned": n_assigned,
        "ads_overlap": ads_overlap,
        "bulk_overlap": bulk_overlap,
        "pair_overlap": pair_overlap,
    })

df = pd.DataFrame(records)
print(f"Computed metrics for {len(df)} splits across {df['method'].nunique()} methods")

# ── Generate per-method figures ───────────────────────────────────────────
for method_name in sorted(df["method"].unique()):
    mdf = df[df["method"] == method_name].copy()
    n_pairs = len(mdf)

    if n_pairs < 2:
        print(f"  {method_name}: only {n_pairs} pair(s), skipping")
        continue

    # Sort by coverage descending for readability
    mdf = mdf.sort_values("coverage_pct", ascending=True).reset_index(drop=True)
    labels = mdf["label"].values
    y_pos = np.arange(n_pairs)

    print(f"\nGenerating figure for method: {method_name} ({n_pairs} feature pairs)")

    fig, (ax_cov, ax_ol) = plt.subplots(1, 2, figsize=(20, max(6, n_pairs * 0.4)))
    fig.suptitle(f"Method: {method_name} — Coverage & Overlap by Feature Pair",
                 fontsize=15, fontweight="bold", y=1.02)

    # ── Panel (a): Coverage (stacked train/test) ─────────────────────────
    ax_cov.barh(y_pos, mdf["n_train"], color=TRAIN_COLOR, edgecolor="black",
                linewidth=0.5, label="Train")
    ax_cov.barh(y_pos, mdf["n_test"], left=mdf["n_train"], color=TEST_COLOR,
                edgecolor="black", linewidth=0.5, label="Test")

    # Annotate with coverage %
    for i, (_, row) in enumerate(mdf.iterrows()):
        ax_cov.text(row["n_assigned"] + total_systems * 0.005, i,
                    f"{row['coverage_pct']:.1f}%  ({row['n_train']:,}+{row['n_test']:,})",
                    va="center", fontsize=8)

    ax_cov.set_yticks(y_pos)
    ax_cov.set_yticklabels(labels, fontsize=10)
    ax_cov.set_xlabel("Number of Systems", fontsize=12)
    ax_cov.set_title("(a) Coverage: Train + Test Systems", fontweight="bold", fontsize=13)
    ax_cov.legend(fontsize=10, loc="lower right")
    ax_cov.axvline(total_systems, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_cov.set_xlim(0, total_systems * 1.25)

    # ── Panel (b): Overlap (grouped bars) ────────────────────────────────
    bar_h = 0.25
    ax_ol.barh(y_pos + bar_h, mdf["pair_overlap"], height=bar_h,
               color="#C44E52", edgecolor="black", linewidth=0.5, label="Pair OL")
    ax_ol.barh(y_pos, mdf["ads_overlap"], height=bar_h,
               color="#4C72B0", edgecolor="black", linewidth=0.5, label="Ads OL")
    ax_ol.barh(y_pos - bar_h, mdf["bulk_overlap"], height=bar_h,
               color="#DD8452", edgecolor="black", linewidth=0.5, label="Bulk OL")

    # Annotate with counts
    max_ol = max(mdf[["pair_overlap", "ads_overlap", "bulk_overlap"]].max())
    for i, (_, row) in enumerate(mdf.iterrows()):
        for val, offset in [(row["pair_overlap"], bar_h),
                            (row["ads_overlap"], 0),
                            (row["bulk_overlap"], -bar_h)]:
            if val > 0:
                ax_ol.text(val + max_ol * 0.02, i + offset, str(int(val)),
                           va="center", fontsize=8)

    ax_ol.set_yticks(y_pos)
    ax_ol.set_yticklabels(labels, fontsize=10)
    ax_ol.set_xlabel("Overlap Count", fontsize=12)
    ax_ol.set_title("(b) Overlap: Pair / Adsorbate / Adsorbent", fontweight="bold", fontsize=13)
    ax_ol.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f"summary_{method_name}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

# ── Cross-method comparison figure ────────────────────────────────────────
print(f"\n{'=' * 90}")
print("Generating cross-method comparison figure...")

all_df = df.copy()
all_df["combo_label"] = all_df["method"] + ": " + all_df["label"]

methods_sorted = sorted(all_df["method"].unique())
feat_pairs_sorted = sorted(all_df["label"].unique())

fp_idx = {fp: i for i, fp in enumerate(feat_pairs_sorted)}
m_idx = {m: j for j, m in enumerate(methods_sorted)}

# Build matrices
cov_matrix = np.full((len(feat_pairs_sorted), len(methods_sorted)), np.nan)
pair_ol_matrix = np.full_like(cov_matrix, np.nan)
ads_ol_matrix = np.full_like(cov_matrix, np.nan)
bulk_ol_matrix = np.full_like(cov_matrix, np.nan)

for _, row in all_df.iterrows():
    i = fp_idx[row["label"]]
    j = m_idx[row["method"]]
    cov_matrix[i, j] = row["coverage_pct"]
    pair_ol_matrix[i, j] = row["pair_overlap"]
    ads_ol_matrix[i, j] = row["ads_overlap"]
    bulk_ol_matrix[i, j] = row["bulk_overlap"]

# Total overlap = pair + ads + bulk (single number for the overlap heatmap)
total_ol_matrix = pair_ol_matrix + ads_ol_matrix + bulk_ol_matrix

fig = plt.figure(figsize=(22, 14))
fig.suptitle("Cross-Method Comparison — Coverage & Overlap",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(1, 2, wspace=0.25, left=0.10, right=0.92, top=0.92, bottom=0.06)

# ── Panel (a): Coverage heatmap ───────────────────────────────────────────
ax_cov = fig.add_subplot(gs[0, 0])
cmap_cov = plt.cm.YlGn  # higher coverage = greener

for i in range(len(feat_pairs_sorted)):
    for j in range(len(methods_sorted)):
        val = cov_matrix[i, j]
        if np.isnan(val):
            color = "#F0F0F0"
            norm_val = 0.5
        else:
            norm_val = val / 100.0  # coverage is 0-100%
            color = cmap_cov(norm_val)
        ax_cov.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color,
                                        edgecolor="white", linewidth=1.5))
        if not np.isnan(val):
            ax_cov.text(j + 0.5, i + 0.5, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if norm_val > 0.65 else "black")

ax_cov.set_xlim(0, len(methods_sorted))
ax_cov.set_ylim(0, len(feat_pairs_sorted))
ax_cov.set_xticks([j + 0.5 for j in range(len(methods_sorted))])
ax_cov.set_xticklabels(methods_sorted, fontsize=12, fontweight="bold")
ax_cov.set_yticks([i + 0.5 for i in range(len(feat_pairs_sorted))])
ax_cov.set_yticklabels(feat_pairs_sorted, fontsize=10)
ax_cov.invert_yaxis()
ax_cov.set_title("(a) Coverage % (higher = better)", fontweight="bold", fontsize=13)
ax_cov.tick_params(axis="both", length=0)

sm_cov = ScalarMappable(cmap=cmap_cov, norm=Normalize(vmin=0, vmax=100))
sm_cov.set_array([])
cbar_cov = plt.colorbar(sm_cov, ax=ax_cov, shrink=0.7, pad=0.02)
cbar_cov.set_label("Coverage %", fontsize=10)

# ── Panel (b): Overlap heatmap ────────────────────────────────────────────
ax_ol = fig.add_subplot(gs[0, 1])
cmap_ol = plt.cm.RdYlGn_r  # lower overlap = greener (reversed)

ol_max = np.nanmax(total_ol_matrix) if np.nanmax(total_ol_matrix) > 0 else 1

for i in range(len(feat_pairs_sorted)):
    for j in range(len(methods_sorted)):
        val = total_ol_matrix[i, j]
        p_ol = pair_ol_matrix[i, j]
        a_ol = ads_ol_matrix[i, j]
        b_ol = bulk_ol_matrix[i, j]
        if np.isnan(val):
            color = "#F0F0F0"
            norm_val = 0.5
        else:
            norm_val = val / ol_max
            color = cmap_ol(norm_val)
        ax_ol.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=1.5))
        if not np.isnan(val):
            # Show breakdown: P/A/B
            ax_ol.text(j + 0.5, i + 0.35, f"P={int(p_ol)}", ha="center", va="center",
                       fontsize=6.5, color="white" if norm_val > 0.4 else "black")
            ax_ol.text(j + 0.5, i + 0.55, f"A={int(a_ol)}", ha="center", va="center",
                       fontsize=6.5, color="white" if norm_val > 0.4 else "black")
            ax_ol.text(j + 0.5, i + 0.75, f"B={int(b_ol)}", ha="center", va="center",
                       fontsize=6.5, color="white" if norm_val > 0.4 else "black")

ax_ol.set_xlim(0, len(methods_sorted))
ax_ol.set_ylim(0, len(feat_pairs_sorted))
ax_ol.set_xticks([j + 0.5 for j in range(len(methods_sorted))])
ax_ol.set_xticklabels(methods_sorted, fontsize=12, fontweight="bold")
ax_ol.set_yticks([i + 0.5 for i in range(len(feat_pairs_sorted))])
ax_ol.set_yticklabels(feat_pairs_sorted, fontsize=10)
ax_ol.invert_yaxis()
ax_ol.set_title("(b) Overlap: P=Pair, A=Adsorbate, B=Adsorbent (lower = better)",
                fontweight="bold", fontsize=13)
ax_ol.tick_params(axis="both", length=0)

sm_ol = ScalarMappable(cmap=cmap_ol, norm=Normalize(vmin=0, vmax=ol_max))
sm_ol.set_array([])
cbar_ol = plt.colorbar(sm_ol, ax=ax_ol, shrink=0.7, pad=0.02)
cbar_ol.set_label("Total Overlap (P + A + B)", fontsize=10)

output_path = os.path.join(args.output_dir, "summary_cross_method.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {output_path}")
plt.close()

# ── Print summary table ──────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("SUMMARY TABLE: ALL (METHOD, FEATURE PAIR) COMBINATIONS")
print(f"{'=' * 100}")
print(f"{'Method':<8} {'Feature Pair':<30} {'Coverage%':>10} "
      f"{'PairOL':>8} {'AdsOL':>7} {'BulkOL':>8}")
print("-" * 100)

for _, row in all_df.sort_values(["method", "label"]).iterrows():
    print(f"{row['method']:<8} {row['label']:<30} {row['coverage_pct']:>9.1f}% "
          f"{int(row['pair_overlap']):>8} {int(row['ads_overlap']):>7} "
          f"{int(row['bulk_overlap']):>8}")

print(f"\nAll figures saved to {args.output_dir}/")
