"""
Visualization of DataSAIL cold-adsorbent (C1f) split results for OC22.

Generates a multi-panel figure saved to output/datasail_split_visualization.png:
  1. Train/Test split bar chart (system counts)
  2. Adsorbate distribution across splits
  3. Number of elements per adsorbent across splits
  4. UMAP of adsorbent embeddings colored by split
  5. Adsorbate x Adsorbent heatmap (interaction coverage)
  6. Composition entropy distribution across splits
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TRAIN_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"
COLORS = {"train": TRAIN_COLOR, "test": TEST_COLOR}

# ── Load data ─────────────────────────────────────────────────────────────
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}

split_df = pd.read_csv("output/datasail_split_C1f.csv", dtype={"system_id": str})
split_map = dict(zip(split_df["system_id"].astype(str), split_df["split"]))

# Build a master dataframe
rows = []
for sid, info in entries.items():
    rows.append({
        "system_id": sid,
        "ads_symbols": info["ads_symbols"],
        "bulk_symbols": info["bulk_symbols"],
        "bulk_id": info.get("bulk_id", ""),
        "nads": info.get("nads", 0),
        "split": split_map.get(sid, "unknown"),
    })
df = pd.DataFrame(rows)
df = df[df["split"].isin(["train", "test"])]
print(f"Total systems in splits: {len(df)}")

# ── Load adsorbent embeddings for UMAP ────────────────────────────────────
adsorbent_files = [
    "embeddings/adsorbent/property_features.csv",
    "embeddings/adsorbent/stoichiometry_features.csv",
    "embeddings/adsorbent/composition_features.csv",
    "embeddings/adsorbent/fraction_features.csv",
]

dfs_feat = []
for fpath in adsorbent_files:
    d = pd.read_csv(fpath)
    feat_cols = [c for c in d.columns if c not in ("system_id", "bulk_symbols")]
    dfs_feat.append(d[["system_id", "bulk_symbols"] + feat_cols])

merged = dfs_feat[0]
for d in dfs_feat[1:]:
    feat_cols = [c for c in d.columns if c not in ("system_id", "bulk_symbols")]
    merged = merged.merge(d[["system_id"] + feat_cols], on="system_id")

# Deduplicate to unique bulks, pick one representative system per bulk
bulk_dedup = merged.drop_duplicates(subset=["bulk_symbols"]).copy()
bulk_dedup["system_id"] = bulk_dedup["system_id"].astype(str)
bulk_feat_cols = [c for c in bulk_dedup.columns if c not in ("system_id", "bulk_symbols")]

# Assign split to each unique bulk (all systems with same bulk share a split)
bulk_split = df.drop_duplicates(subset=["bulk_symbols"])[["bulk_symbols", "split"]]
bulk_dedup = bulk_dedup.merge(bulk_split, on="bulk_symbols", how="inner")

print(f"Unique bulks for UMAP: {len(bulk_dedup)}")
print("Computing UMAP (this may take a moment)...")

X = bulk_dedup[bulk_feat_cols].values.astype(float)
X_scaled = StandardScaler().fit_transform(X)
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=42)
embedding_2d = reducer.fit_transform(X_scaled)
bulk_dedup["umap_x"] = embedding_2d[:, 0]
bulk_dedup["umap_y"] = embedding_2d[:, 1]

# ── Parse element count from bulk formula ─────────────────────────────────
import re

def count_elements(formula):
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    return len([e for e, _ in pairs if e != ""])

df["num_elements"] = df["bulk_symbols"].apply(count_elements)

# Load stoichiometry features for entropy
stoich = pd.read_csv("embeddings/adsorbent/stoichiometry_features.csv",
                     dtype={"system_id": str}, usecols=["system_id", "composition_entropy"])
stoich["system_id"] = stoich["system_id"].astype(str)
# Deduplicate by bulk via df
df = df.merge(stoich, on="system_id", how="left")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3,
                       left=0.08, right=0.95, top=0.95, bottom=0.04)

# ── Panel 1: Overall split sizes ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts = df["split"].value_counts().reindex(["train", "test"])
bars = ax1.bar(counts.index, counts.values, color=[TRAIN_COLOR, TEST_COLOR],
               edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    pct = 100 * val / counts.sum()
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Systems")
ax1.set_title("(a) Train / Test Split Sizes", fontweight="bold", fontsize=14)
ax1.set_ylim(0, counts.max() * 1.2)

# ── Panel 2: Adsorbate distribution per split ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ads_order = sorted(df["ads_symbols"].unique())
ads_counts = df.groupby(["ads_symbols", "split"]).size().unstack(fill_value=0)
ads_counts = ads_counts.reindex(columns=["train", "test"]).reindex(ads_order)

x_pos = np.arange(len(ads_order))
width = 0.35
ax2.bar(x_pos - width / 2, ads_counts["train"], width, label="Train",
        color=TRAIN_COLOR, edgecolor="black", linewidth=0.5)
ax2.bar(x_pos + width / 2, ads_counts["test"], width, label="Test",
        color=TEST_COLOR, edgecolor="black", linewidth=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(ads_order, rotation=45, ha="right")
ax2.set_ylabel("Number of Systems")
ax2.set_title("(b) Adsorbate Distribution per Split", fontweight="bold", fontsize=14)
ax2.legend(frameon=True)

# ── Panel 3: Number of elements in adsorbent ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
elem_bins = range(1, df["num_elements"].max() + 2)
for split_name, color in COLORS.items():
    subset = df[df["split"] == split_name]["num_elements"]
    ax3.hist(subset, bins=elem_bins, alpha=0.65, color=color, label=split_name,
             edgecolor="black", linewidth=0.5, align="left")
ax3.set_xlabel("Number of Unique Elements in Adsorbent")
ax3.set_ylabel("Number of Systems")
ax3.set_title("(c) Adsorbent Complexity (Element Count)", fontweight="bold", fontsize=14)
ax3.legend(frameon=True)

# ── Panel 4: UMAP of adsorbent embeddings ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for split_name, color in [("train", TRAIN_COLOR), ("test", TEST_COLOR)]:
    mask = bulk_dedup["split"] == split_name
    ax4.scatter(bulk_dedup.loc[mask, "umap_x"], bulk_dedup.loc[mask, "umap_y"],
                c=color, label=split_name, s=12, alpha=0.6, edgecolors="none")
ax4.set_xlabel("UMAP 1")
ax4.set_ylabel("UMAP 2")
ax4.set_title("(d) UMAP of Adsorbent Embeddings (per unique bulk)",
              fontweight="bold", fontsize=14)
ax4.legend(frameon=True, markerscale=3)

# ── Panel 5: Adsorbate x Adsorbent heatmap (interaction coverage) ────────
ax5 = fig.add_subplot(gs[2, :])

# For each adsorbate, count how many unique adsorbents appear in train vs test
ads_bulk_train = df[df["split"] == "train"].groupby("ads_symbols")["bulk_symbols"].nunique()
ads_bulk_test = df[df["split"] == "test"].groupby("ads_symbols")["bulk_symbols"].nunique()
ads_sys_train = df[df["split"] == "train"].groupby("ads_symbols").size()
ads_sys_test = df[df["split"] == "test"].groupby("ads_symbols").size()

heatmap_data = pd.DataFrame({
    "Adsorbent (train)": ads_bulk_train,
    "Adsorbent (test)": ads_bulk_test,
    "Systems (train)": ads_sys_train,
    "Systems (test)": ads_sys_test,
}).reindex(ads_order).fillna(0).astype(int)

sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax5,
            linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.6})
ax5.set_title("(e) Interaction Coverage: Unique Adsorbents & Systems per Adsorbate per Split",
              fontweight="bold", fontsize=14)
ax5.set_ylabel("Adsorbate")
ax5.set_xlabel("")

# ── Panel 6: Composition entropy distribution ────────────────────────────
ax6 = fig.add_subplot(gs[3, 0])
for split_name, color in COLORS.items():
    subset = df[df["split"] == split_name]["composition_entropy"].dropna()
    ax6.hist(subset, bins=40, alpha=0.6, color=color, label=split_name,
             edgecolor="black", linewidth=0.5, density=True)
ax6.set_xlabel("Composition Entropy")
ax6.set_ylabel("Density")
ax6.set_title("(f) Adsorbent Composition Entropy Distribution",
              fontweight="bold", fontsize=14)
ax6.legend(frameon=True)

# ── Panel 7: Unique adsorbents per split (Venn-style summary) ────────────
ax7 = fig.add_subplot(gs[3, 1])
train_bulks = set(df[df["split"] == "train"]["bulk_symbols"])
test_bulks = set(df[df["split"] == "test"]["bulk_symbols"])
overlap = train_bulks & test_bulks
only_train = len(train_bulks - test_bulks)
only_test = len(test_bulks - train_bulks)
both = len(overlap)

categories = ["Train only", "Test only", "Overlap"]
values = [only_train, only_test, both]
bar_colors = [TRAIN_COLOR, TEST_COLOR, "#888888"]
bars = ax7.bar(categories, values, color=bar_colors, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, values):
    ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
             f"{val:,}", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax7.set_ylabel("Number of Unique Adsorbents")
ax7.set_title("(g) Adsorbent Overlap Between Splits (Cold Split Check)",
              fontweight="bold", fontsize=14)
ax7.set_ylim(0, max(values) * 1.2)

# ── Save ──────────────────────────────────────────────────────────────────
out_path = "output/datasail_split_visualization.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {out_path}")
plt.close()
