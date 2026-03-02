"""
Visualization of DataSAIL split results for OC22.

Generates per-method figures showing different feature pair combinations.
Groups results by splitting technique (R, I1e, I1f, I2, C1e, C1f, C2) and
creates one figure per method showing all feature pairs tested.

Usage:
  # Visualize all splits in output/split_result/ (grouped by method)
  python visualization.py

  # Visualize specific method
  python visualization.py --method C1e

  # Visualize specific splits
  python visualization.py output/split_result/datasail_split_C1f__e_physchem__f_property.csv
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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualize OC22 DataSAIL splits grouped by method")
parser.add_argument("split_files", nargs="*", default=None,
                    help="Split CSV file(s). If omitted, uses all output/datasail_split_*.csv")
parser.add_argument("--method", type=str, default=None,
                    help="Only visualize specific method (R, I1e, I1f, I2, C1e, C1f, C2)")
parser.add_argument("--f-embedding", nargs="+",
                    default=["features/oc22/adsorbent/property_features.csv",
                             "features/oc22/adsorbent/stoichiometry_features.csv",
                             "features/oc22/adsorbent/electronic_features.csv",
                             "features/oc22/adsorbent/bonding_features.csv",
                             "features/oc22/adsorbent/thermodynamic_features.csv",
                             "features/oc22/adsorbent/catalytic_features.csv"],
                    help="Adsorbent embedding CSV(s) for UMAP")
parser.add_argument("-o", "--output-dir", default="output/figure",
                    help="Output directory for figures (one per method)")
args = parser.parse_args()

if not args.split_files:
    args.split_files = sorted(glob.glob("output/split_result/datasail_split_*.csv"))
    if not args.split_files:
        print("No split CSVs found in output/. Pass files explicitly.")
        exit(1)

print(f"Found {len(args.split_files)} split files")
os.makedirs(args.output_dir, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
TRAIN_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"
NOT_SEL_COLOR = "#AAAAAA"

# ── Load metadata ─────────────────────────────────────────────────────────
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}
print(f"Total systems: {len(entries)}")

# ── Load adsorbent embeddings for UMAP (computed once, shared) ────────────
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
    return merged

merged = load_features(args.f_embedding, "bulk_symbols")
bulk_dedup = merged.drop_duplicates(subset=["bulk_symbols"]).copy()
bulk_dedup["system_id"] = bulk_dedup["system_id"].astype(str)
bulk_feat_cols = [c for c in bulk_dedup.columns if c not in ("system_id", "bulk_symbols")]

print(f"Unique adsorbents for UMAP: {len(bulk_dedup)}")
print("Computing UMAP (this may take a moment)...")

X = bulk_dedup[bulk_feat_cols].values.astype(float)
X_scaled = StandardScaler().fit_transform(X)
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=42)
embedding_2d = reducer.fit_transform(X_scaled)
bulk_dedup = bulk_dedup[["bulk_symbols"]].copy()
bulk_dedup["umap_x"] = embedding_2d[:, 0]
bulk_dedup["umap_y"] = embedding_2d[:, 1]
umap_lookup = bulk_dedup.set_index("bulk_symbols")

print("UMAP done.")

# ── Parse element count from bulk formula ─────────────────────────────────
def count_elements(formula):
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    return len([e for e, _ in pairs if e != ""])

# ── Parse filename to extract method and feature pair ────────────────────
def parse_filename(path):
    """
    Parse filename like: datasail_split_C1e__e_physchem__f_property.csv
    Returns: (method, e_features, f_features, full_label)
    """
    base = os.path.splitext(os.path.basename(path))[0]

    # Pattern: datasail_split_{method}__{features}
    m = re.match(r"datasail_split_([A-Za-z0-9]+)(?:__(.+))?", base)
    if not m:
        return None, None, None, base

    method = m.group(1)
    feature_tag = m.group(2) if m.group(2) else ""

    # Parse feature tags like: e_physchem__f_property
    e_features = "default"
    f_features = "default"

    if feature_tag:
        e_match = re.search(r"e_([^_]+(?:\+[^_]+)*)", feature_tag)
        f_match = re.search(r"f_(.+)", feature_tag)
        if e_match:
            e_features = e_match.group(1)
        if f_match:
            f_features = f_match.group(1)

    full_label = f"{e_features} × {f_features}"
    return method, e_features, f_features, full_label

# ── Build master dataframe (once) ─────────────────────────────────────────
master_rows = []
for sid, info in entries.items():
    master_rows.append({
        "system_id": sid,
        "ads_symbols": info["ads_symbols"],
        "bulk_symbols": info["bulk_symbols"],
    })
master_df = pd.DataFrame(master_rows)
master_df["num_elements"] = master_df["bulk_symbols"].apply(count_elements)

# ── Group split files by method ───────────────────────────────────────────
from collections import defaultdict

method_groups = defaultdict(list)
for fpath in args.split_files:
    method, e_feat, f_feat, label = parse_filename(fpath)
    if method and (args.method is None or method == args.method):
        method_groups[method].append({
            'path': fpath,
            'e_features': e_feat,
            'f_features': f_feat,
            'label': label
        })

if not method_groups:
    print(f"No files found for method: {args.method if args.method else 'any'}")
    exit(1)

print(f"\nGrouped by method:")
for method, files in sorted(method_groups.items()):
    print(f"  {method}: {len(files)} feature pairs")

# ── Process each method ───────────────────────────────────────────────────
for method_name in sorted(method_groups.keys()):
    files = sorted(method_groups[method_name], key=lambda x: x['label'])
    n_pairs = len(files)

    print(f"\n{'=' * 70}")
    print(f"Generating figure for method: {method_name} ({n_pairs} feature pairs)")
    print(f"{'=' * 70}")

    # Layout: each feature pair gets a row of 3 panels
    fig = plt.figure(figsize=(20, 6 * n_pairs + 1))
    gs = gridspec.GridSpec(n_pairs, 3, hspace=0.45, wspace=0.3,
                           left=0.06, right=0.96, top=1 - 0.03, bottom=0.03)

    for row_idx, file_info in enumerate(files):
        fpath = file_info['path']
        label = f"{method_name}: {file_info['label']}"
        split_df = pd.read_csv(fpath, dtype={"system_id": str})
        split_map = dict(zip(split_df["system_id"].astype(str), split_df["split"]))

        # Merge split into master
        df = master_df.copy()
        df["split"] = df["system_id"].map(split_map).fillna("not selected")

        n_train = (df["split"] == "train").sum()
        n_test = (df["split"] == "test").sum()
        n_not = (df["split"] == "not selected").sum()

        train_bulks = set(df[df["split"] == "train"]["bulk_symbols"])
        test_bulks = set(df[df["split"] == "test"]["bulk_symbols"])
        bulk_overlap = len(train_bulks & test_bulks)

        train_ads = set(df[df["split"] == "train"]["ads_symbols"])
        test_ads = set(df[df["split"] == "test"]["ads_symbols"])
        ads_overlap = len(train_ads & test_ads)

        print(f"  {file_info['label']}: Train={n_train:,} Test={n_test:,} NotSel={n_not:,} "
              f"BulkOL={bulk_overlap} AdsOL={ads_overlap}")

        # ── Panel 1: UMAP ────────────────────────────────────────────────────
        ax_umap = fig.add_subplot(gs[row_idx, 0])

        # Assign split to each unique bulk
        bulk_split_map = {}
        for _, r in df.iterrows():
            b = r["bulk_symbols"]
            s = r["split"]
            if b not in bulk_split_map or bulk_split_map[b] == "not selected":
                bulk_split_map[b] = s

        umap_df = umap_lookup.copy()
        umap_df["split"] = umap_df.index.map(bulk_split_map).fillna("not selected")

        # Plot not selected first (background), then train, then test on top
        for split_name, color, alpha, size in [
            ("not selected", NOT_SEL_COLOR, 0.3, 8),
            ("train", TRAIN_COLOR, 0.5, 10),
            ("test", TEST_COLOR, 0.7, 12),
        ]:
            mask = umap_df["split"] == split_name
            if mask.sum() == 0:
                continue
            ax_umap.scatter(umap_df.loc[mask, "umap_x"], umap_df.loc[mask, "umap_y"],
                            c=color, label=f"{split_name} ({mask.sum()})",
                            s=size, alpha=alpha, edgecolors="none")

        ax_umap.set_xlabel("UMAP 1")
        ax_umap.set_ylabel("UMAP 2")
        ax_umap.set_title(f"{label} — UMAP of Adsorbent Embeddings", fontweight="bold", fontsize=10)
        ax_umap.legend(frameon=True, markerscale=2, fontsize=8, loc="best")

        # ── Panel 2: Adsorbate distribution ──────────────────────────────────
        ax_ads = fig.add_subplot(gs[row_idx, 1])
        df_assigned = df[df["split"].isin(["train", "test"])]
        ads_order = sorted(df_assigned["ads_symbols"].unique())

        if len(ads_order) > 0:
            ads_counts = df_assigned.groupby(["ads_symbols", "split"]).size().unstack(fill_value=0)
            for col in ["train", "test"]:
                if col not in ads_counts.columns:
                    ads_counts[col] = 0
            ads_counts = ads_counts.reindex(ads_order, fill_value=0)

            x_pos = np.arange(len(ads_order))
            width = 0.35
            ax_ads.bar(x_pos - width / 2, ads_counts["train"], width, label="Train",
                       color=TRAIN_COLOR, edgecolor="black", linewidth=0.5)
            ax_ads.bar(x_pos + width / 2, ads_counts["test"], width, label="Test",
                       color=TEST_COLOR, edgecolor="black", linewidth=0.5)
            ax_ads.set_xticks(x_pos)
            ax_ads.set_xticklabels(ads_order, rotation=45, ha="right")

        ax_ads.set_ylabel("Number of Systems")
        ax_ads.set_title(f"{label} — Adsorbate Distribution", fontweight="bold", fontsize=10)
        ax_ads.legend(frameon=True, fontsize=8)

        # ── Panel 3: Split sizes + overlap summary ───────────────────────────
        ax_bar = fig.add_subplot(gs[row_idx, 2])

        categories = ["Train", "Test", "Not sel.", "Bulk OL", "Ads OL"]
        values = [n_train, n_test, n_not, bulk_overlap, ads_overlap]
        bar_colors = [TRAIN_COLOR, TEST_COLOR, NOT_SEL_COLOR, "#C44E52", "#C44E52"]

        bars = ax_bar.bar(categories, values, color=bar_colors, edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, values):
            if val > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(values) * 0.02,
                            f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_bar.set_ylabel("Count")
        ax_bar.set_title(f"{label} — Split Sizes & Overlap", fontweight="bold", fontsize=10)

    # Save figure for this method
    output_path = os.path.join(args.output_dir, f"datasail_{method_name}_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

print(f"\n{'=' * 70}")
print(f"All figures saved to {args.output_dir}/")
print(f"{'=' * 70}")
