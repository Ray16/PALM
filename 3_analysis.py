"""
Deep analysis of selected DataSAIL splits for OC22.

For each selected split, computes and visualizes:
  1. UMAP projections of adsorbents and adsorbates colored by train/test
  2. Nearest-neighbor distance distributions (test → train in feature space)
  3. Per-adsorbate distribution across train/test
  4. Summary statistics table

Usage:
  # Analyze a single split
  python 3_analysis.py output/split_result/datasail_split_C2__e_rdkit__f_stoichiometry.csv

  # Compare multiple selected splits
  python 3_analysis.py output/split_result/datasail_split_C2__e_rdkit__f_stoichiometry.csv \
                        output/split_result/datasail_split_C1e__e_rdkit__f_bonding.csv

  # Custom embeddings for distance/UMAP computation
  python 3_analysis.py --e-embedding features/oc22/adsorbate/physchem_features.csv \
                       --f-embedding features/oc22/adsorbent/property_features.csv \
                       output/split_result/datasail_split_C2__e_rdkit__f_stoichiometry.csv
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import umap

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Deep analysis of selected OC22 DataSAIL splits")
parser.add_argument("split_files", nargs="+", help="Split CSV file(s) to analyze")
parser.add_argument("--e-embedding", nargs="+",
                    default=["features/oc22/adsorbate/physchem_features.csv",
                             "features/oc22/adsorbate/composition_features.csv",
                             "features/oc22/adsorbate/rdkit_descriptors_features.csv",
                             "features/oc22/adsorbate/adsorption_features.csv"],
                    help="Adsorbate embedding CSV(s) for UMAP and NN distance")
parser.add_argument("--f-embedding", nargs="+",
                    default=["features/oc22/adsorbent/property_features.csv",
                             "features/oc22/adsorbent/stoichiometry_features.csv",
                             "features/oc22/adsorbent/electronic_features.csv",
                             "features/oc22/adsorbent/bonding_features.csv",
                             "features/oc22/adsorbent/thermodynamic_features.csv",
                             "features/oc22/adsorbent/catalytic_features.csv"],
                    help="Adsorbent embedding CSV(s) for UMAP and NN distance")
parser.add_argument("-o", "--output-dir", default="output/analysis",
                    help="Output directory for figures")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
TRAIN_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"
NOT_SEL_COLOR = "#CCCCCC"

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

print("Loading adsorbate embeddings...")
ads_dedup, ads_feat_cols = load_features(args.e_embedding, "ads_symbols")
print(f"  {len(ads_dedup)} unique adsorbates, {len(ads_feat_cols)} features")

print("Loading adsorbent embeddings...")
bulk_dedup, bulk_feat_cols = load_features(args.f_embedding, "bulk_symbols")
print(f"  {len(bulk_dedup)} unique adsorbents, {len(bulk_feat_cols)} features")

# ── Scale features ────────────────────────────────────────────────────────
ads_X = ads_dedup[ads_feat_cols].values.astype(float)
bulk_X = bulk_dedup[bulk_feat_cols].values.astype(float)

ads_scaler = StandardScaler().fit(ads_X)
bulk_scaler = StandardScaler().fit(bulk_X)

ads_X_scaled = ads_scaler.transform(ads_X)
bulk_X_scaled = bulk_scaler.transform(bulk_X)

# ── Compute UMAP (once, shared across splits) ────────────────────────────
print("Computing UMAP for adsorbents...")
bulk_reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=42)
bulk_umap = bulk_reducer.fit_transform(bulk_X_scaled)
bulk_umap_df = pd.DataFrame({"umap_x": bulk_umap[:, 0], "umap_y": bulk_umap[:, 1]},
                              index=bulk_dedup.index)

print("Computing UMAP for adsorbates...")
ads_reducer = umap.UMAP(n_neighbors=min(15, len(ads_dedup) - 1), min_dist=0.3,
                         metric="euclidean", random_state=42)
ads_umap = ads_reducer.fit_transform(ads_X_scaled)
ads_umap_df = pd.DataFrame({"umap_x": ads_umap[:, 0], "umap_y": ads_umap[:, 1]},
                             index=ads_dedup.index)
print("UMAP done.")

# ── Parse label from filename ─────────────────────────────────────────────
def label_from_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"datasail_split_([A-Za-z0-9]+)(?:__(.+))?", base)
    if not m:
        return base
    method = m.group(1)
    feature_tag = m.group(2) if m.group(2) else ""
    if feature_tag:
        e_match = re.search(r"e_([^_]+(?:\+[^_]+)*)", feature_tag)
        f_match = re.search(r"f_(.+)", feature_tag)
        e = e_match.group(1) if e_match else "default"
        f = f_match.group(1) if f_match else "default"
        return f"{method}: {e} × {f}"
    return method

# ── Compute NN distances ─────────────────────────────────────────────────
def compute_nn_dists(train_set, test_set, dedup_df, feat_cols, scaler):
    """Return per-test-entity NN distances to nearest train entity."""
    train_list = sorted(train_set & set(dedup_df.index))
    test_list = sorted(test_set & set(dedup_df.index))
    if len(train_list) == 0 or len(test_list) == 0:
        return np.array([]), test_list
    X_tr = scaler.transform(dedup_df.loc[train_list, feat_cols].values.astype(float))
    X_te = scaler.transform(dedup_df.loc[test_list, feat_cols].values.astype(float))
    D = cdist(X_te, X_tr, metric="euclidean")
    nn_dists = D.min(axis=1)
    return nn_dists, test_list

# ── Analyze each split ────────────────────────────────────────────────────
split_data = []

for fpath in sorted(args.split_files):
    label = label_from_path(fpath)
    split_df = pd.read_csv(fpath, dtype={"system_id": str})
    split_map = dict(zip(split_df["system_id"], split_df["split"]))

    train_ads, test_ads = set(), set()
    train_bulk, test_bulk = set(), set()
    train_pairs, test_pairs = set(), set()
    n_train, n_test, n_not = 0, 0, 0

    for sid, info in entries.items():
        s = split_map.get(sid, "not selected")
        ads, bulk = info["ads_symbols"], info["bulk_symbols"]
        if s == "train":
            train_ads.add(ads); train_bulk.add(bulk)
            train_pairs.add((ads, bulk)); n_train += 1
        elif s == "test":
            test_ads.add(ads); test_bulk.add(bulk)
            test_pairs.add((ads, bulk)); n_test += 1
        else:
            n_not += 1

    # NN distances
    bulk_nn_dists, bulk_test_list = compute_nn_dists(
        train_bulk, test_bulk, bulk_dedup, bulk_feat_cols, bulk_scaler)
    ads_nn_dists, ads_test_list = compute_nn_dists(
        train_ads, test_ads, ads_dedup, ads_feat_cols, ads_scaler)

    # Build split assignments for UMAP coloring
    bulk_split = {}
    for sid, info in entries.items():
        s = split_map.get(sid, "not selected")
        b = info["bulk_symbols"]
        if b not in bulk_split or bulk_split[b] == "not selected":
            bulk_split[b] = s

    ads_split = {}
    for sid, info in entries.items():
        s = split_map.get(sid, "not selected")
        a = info["ads_symbols"]
        if a not in ads_split or ads_split[a] == "not selected":
            ads_split[a] = s

    # Per-adsorbate counts
    ads_counts = {}
    for sid, info in entries.items():
        s = split_map.get(sid, "not selected")
        a = info["ads_symbols"]
        if s in ("train", "test"):
            if a not in ads_counts:
                ads_counts[a] = {"train": 0, "test": 0}
            ads_counts[a][s] += 1

    split_data.append({
        "label": label,
        "n_train": n_train, "n_test": n_test, "n_not": n_not,
        "pair_overlap": len(train_pairs & test_pairs),
        "ads_overlap": len(train_ads & test_ads),
        "bulk_overlap": len(train_bulk & test_bulk),
        "bulk_nn_dists": bulk_nn_dists,
        "ads_nn_dists": ads_nn_dists,
        "bulk_split": bulk_split,
        "ads_split": ads_split,
        "ads_counts": ads_counts,
    })

    print(f"  {label}: Train={n_train:,} Test={n_test:,} NotSel={n_not:,} "
          f"PairOL={len(train_pairs & test_pairs)} "
          f"AdsOL={len(train_ads & test_ads)} BulkOL={len(train_bulk & test_bulk)}")

# ── Generate figure per split ─────────────────────────────────────────────
n_splits = len(split_data)

for sd in split_data:
    label = sd["label"]
    safe_label = re.sub(r"[^\w\-]", "_", label)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f"Split Analysis: {label}", fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.30,
                           left=0.06, right=0.96, top=0.93, bottom=0.05)

    # ── (a) UMAP Adsorbents ──────────────────────────────────────────────
    ax_bulk = fig.add_subplot(gs[0, 0])
    umap_b = bulk_umap_df.copy()
    umap_b["split"] = umap_b.index.map(sd["bulk_split"]).fillna("not selected")

    for split_name, color, alpha, size in [
        ("not selected", NOT_SEL_COLOR, 0.2, 6),
        ("train", TRAIN_COLOR, 0.5, 10),
        ("test", TEST_COLOR, 0.7, 14),
    ]:
        mask = umap_b["split"] == split_name
        if mask.sum() == 0:
            continue
        ax_bulk.scatter(umap_b.loc[mask, "umap_x"], umap_b.loc[mask, "umap_y"],
                        c=color, s=size, alpha=alpha, edgecolors="none",
                        label=f"{split_name} ({mask.sum()})")

    ax_bulk.set_xlabel("UMAP 1"); ax_bulk.set_ylabel("UMAP 2")
    ax_bulk.set_title("(a) Adsorbent UMAP", fontweight="bold", fontsize=12)
    ax_bulk.legend(frameon=True, markerscale=2, fontsize=8, loc="best")

    # ── (b) UMAP Adsorbates ──────────────────────────────────────────────
    ax_ads = fig.add_subplot(gs[0, 1])
    umap_a = ads_umap_df.copy()
    umap_a["split"] = umap_a.index.map(sd["ads_split"]).fillna("not selected")

    for split_name, color, alpha, size in [
        ("not selected", NOT_SEL_COLOR, 0.3, 15),
        ("train", TRAIN_COLOR, 0.8, 40),
        ("test", TEST_COLOR, 0.9, 50),
    ]:
        mask = umap_a["split"] == split_name
        if mask.sum() == 0:
            continue
        ax_ads.scatter(umap_a.loc[mask, "umap_x"], umap_a.loc[mask, "umap_y"],
                       c=color, s=size, alpha=alpha, edgecolors="black", linewidth=0.3,
                       label=f"{split_name} ({mask.sum()})")

    # Label adsorbate points (few enough to label)
    for entity, row in umap_a.iterrows():
        if row["split"] != "not selected":
            ax_ads.annotate(entity, (row["umap_x"], row["umap_y"]),
                            fontsize=5, ha="center", va="bottom",
                            xytext=(0, 4), textcoords="offset points")

    ax_ads.set_xlabel("UMAP 1"); ax_ads.set_ylabel("UMAP 2")
    ax_ads.set_title("(b) Adsorbate UMAP", fontweight="bold", fontsize=12)
    ax_ads.legend(frameon=True, markerscale=1.5, fontsize=8, loc="best")

    # ── (c) Per-adsorbate distribution ───────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 2])
    ac = sd["ads_counts"]
    if ac:
        ads_names = sorted(ac.keys())
        train_counts = [ac[a].get("train", 0) for a in ads_names]
        test_counts = [ac[a].get("test", 0) for a in ads_names]
        y = np.arange(len(ads_names))
        bar_h = 0.35
        ax_dist.barh(y + bar_h / 2, train_counts, bar_h, color=TRAIN_COLOR,
                      edgecolor="black", linewidth=0.4, label="Train")
        ax_dist.barh(y - bar_h / 2, test_counts, bar_h, color=TEST_COLOR,
                      edgecolor="black", linewidth=0.4, label="Test")
        ax_dist.set_yticks(y)
        ax_dist.set_yticklabels(ads_names, fontsize=8)
        ax_dist.invert_yaxis()
    ax_dist.set_xlabel("Number of Systems")
    ax_dist.set_title("(c) Adsorbate Distribution", fontweight="bold", fontsize=12)
    ax_dist.legend(frameon=True, fontsize=8)

    # ── (d) Adsorbent NN distance distribution ───────────────────────────
    ax_bnn = fig.add_subplot(gs[1, 0])
    if len(sd["bulk_nn_dists"]) > 0:
        ax_bnn.hist(sd["bulk_nn_dists"], bins=50, color=TEST_COLOR,
                     edgecolor="black", linewidth=0.4, alpha=0.8)
        mean_d = sd["bulk_nn_dists"].mean()
        med_d = np.median(sd["bulk_nn_dists"])
        ax_bnn.axvline(mean_d, color="red", linestyle="--", linewidth=1.5,
                        label=f"Mean = {mean_d:.2f}")
        ax_bnn.axvline(med_d, color="darkgreen", linestyle=":", linewidth=1.5,
                        label=f"Median = {med_d:.2f}")
        ax_bnn.legend(frameon=True, fontsize=9)
    else:
        ax_bnn.text(0.5, 0.5, "All adsorbents shared\n(NN dist = 0)",
                     ha="center", va="center", fontsize=12, transform=ax_bnn.transAxes)
    ax_bnn.set_xlabel("NN Distance (standardized)")
    ax_bnn.set_ylabel("Count (test adsorbents)")
    ax_bnn.set_title("(d) Adsorbent NN Distance: test → nearest train",
                      fontweight="bold", fontsize=12)

    # ── (e) Adsorbate NN distance distribution ───────────────────────────
    ax_ann = fig.add_subplot(gs[1, 1])
    if len(sd["ads_nn_dists"]) > 0:
        ax_ann.hist(sd["ads_nn_dists"], bins=max(10, len(sd["ads_nn_dists"]) // 2),
                     color=TEST_COLOR, edgecolor="black", linewidth=0.4, alpha=0.8)
        mean_d = sd["ads_nn_dists"].mean()
        med_d = np.median(sd["ads_nn_dists"])
        ax_ann.axvline(mean_d, color="red", linestyle="--", linewidth=1.5,
                        label=f"Mean = {mean_d:.2f}")
        ax_ann.axvline(med_d, color="darkgreen", linestyle=":", linewidth=1.5,
                        label=f"Median = {med_d:.2f}")
        ax_ann.legend(frameon=True, fontsize=9)
    else:
        ax_ann.text(0.5, 0.5, "All adsorbates shared\n(NN dist = 0)",
                     ha="center", va="center", fontsize=12, transform=ax_ann.transAxes)
    ax_ann.set_xlabel("NN Distance (standardized)")
    ax_ann.set_ylabel("Count (test adsorbates)")
    ax_ann.set_title("(e) Adsorbate NN Distance: test → nearest train",
                      fontweight="bold", fontsize=12)

    # ── (f) Summary statistics ───────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")

    bulk_nn_mean = f"{sd['bulk_nn_dists'].mean():.2f}" if len(sd["bulk_nn_dists"]) > 0 else "shared"
    bulk_nn_med = f"{np.median(sd['bulk_nn_dists']):.2f}" if len(sd["bulk_nn_dists"]) > 0 else "shared"
    ads_nn_mean = f"{sd['ads_nn_dists'].mean():.2f}" if len(sd["ads_nn_dists"]) > 0 else "shared"
    ads_nn_med = f"{np.median(sd['ads_nn_dists']):.2f}" if len(sd["ads_nn_dists"]) > 0 else "shared"

    table_data = [
        ["Train systems", f"{sd['n_train']:,}"],
        ["Test systems", f"{sd['n_test']:,}"],
        ["Not selected", f"{sd['n_not']:,}"],
        ["Coverage", f"{100 * (sd['n_train'] + sd['n_test']) / len(entries):.1f}%"],
        ["", ""],
        ["Pair overlap", str(sd["pair_overlap"])],
        ["Adsorbate overlap", str(sd["ads_overlap"])],
        ["Adsorbent overlap", str(sd["bulk_overlap"])],
        ["", ""],
        ["Adsorbent NN (mean)", bulk_nn_mean],
        ["Adsorbent NN (median)", bulk_nn_med],
        ["Adsorbate NN (mean)", ads_nn_mean],
        ["Adsorbate NN (median)", ads_nn_med],
    ]

    tbl = ax_tbl.table(cellText=table_data, colLabels=["Metric", "Value"],
                        loc="center", cellLoc="left",
                        colWidths=[0.55, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    # Style header
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif table_data[r - 1][0] == "":
            cell.set_facecolor("#F8F8F8")
            cell.set_edgecolor("#F8F8F8")
        else:
            cell.set_facecolor("#F8F8F8" if r % 2 == 0 else "white")
    ax_tbl.set_title("(f) Summary Statistics", fontweight="bold", fontsize=12)

    output_path = os.path.join(args.output_dir, f"analysis_{safe_label}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

# ── Comparison figure (if multiple splits) ────────────────────────────────
if n_splits >= 2:
    print("\nGenerating comparison figure...")
    labels = [sd["label"] for sd in split_data]
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig = plt.figure(figsize=(max(14, n * 3), 14))
    fig.suptitle("Split Comparison", fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30,
                           left=0.08, right=0.95, top=0.92, bottom=0.08)

    # ── (a) Overlap comparison ───────────────────────────────────────────
    ax_ol = fig.add_subplot(gs[0, 0])
    pair_ol = [sd["pair_overlap"] for sd in split_data]
    ads_ol = [sd["ads_overlap"] for sd in split_data]
    bulk_ol = [sd["bulk_overlap"] for sd in split_data]

    bar_w = 0.25
    ax_ol.bar(x - bar_w, pair_ol, bar_w, label="Pair OL", color="#C44E52",
              edgecolor="black", linewidth=0.5)
    ax_ol.bar(x, ads_ol, bar_w, label="Ads OL", color="#4C72B0",
              edgecolor="black", linewidth=0.5)
    ax_ol.bar(x + bar_w, bulk_ol, bar_w, label="Bulk OL", color="#DD8452",
              edgecolor="black", linewidth=0.5)
    # Annotate
    for i in range(n):
        for val, offset in [(pair_ol[i], -bar_w), (ads_ol[i], 0), (bulk_ol[i], bar_w)]:
            if val > 0:
                ax_ol.text(i + offset, val + 0.5, str(val), ha="center", va="bottom", fontsize=9)
    ax_ol.set_xticks(x)
    ax_ol.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_ol.set_ylabel("Count")
    ax_ol.set_title("(a) Entity Overlap", fontweight="bold", fontsize=13)
    ax_ol.legend(frameon=True, fontsize=9)

    # ── (b) Coverage comparison ──────────────────────────────────────────
    ax_cov = fig.add_subplot(gs[0, 1])
    trains = [sd["n_train"] for sd in split_data]
    tests = [sd["n_test"] for sd in split_data]
    nots = [sd["n_not"] for sd in split_data]
    ax_cov.bar(x, trains, label="Train", color=TRAIN_COLOR, edgecolor="black", linewidth=0.5)
    ax_cov.bar(x, tests, bottom=trains, label="Test", color=TEST_COLOR,
               edgecolor="black", linewidth=0.5)
    ax_cov.bar(x, nots, bottom=[t + te for t, te in zip(trains, tests)],
               label="Not selected", color=NOT_SEL_COLOR, edgecolor="black", linewidth=0.5)
    for i in range(n):
        cov = 100 * (trains[i] + tests[i]) / len(entries)
        ax_cov.text(i, trains[i] + tests[i] + nots[i] + len(entries) * 0.01,
                     f"{cov:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_cov.set_ylabel("Number of Systems")
    ax_cov.set_title("(b) Split Size Distribution", fontweight="bold", fontsize=13)
    ax_cov.legend(frameon=True, fontsize=9)

    # ── (c) NN distance comparison (bars) ────────────────────────────────
    ax_nn = fig.add_subplot(gs[1, 0])
    ads_nn = [sd["ads_nn_dists"].mean() if len(sd["ads_nn_dists"]) > 0 else 0
              for sd in split_data]
    bulk_nn = [sd["bulk_nn_dists"].mean() if len(sd["bulk_nn_dists"]) > 0 else 0
               for sd in split_data]

    ax_nn.bar(x - width / 2, ads_nn, width, label="Adsorbate NN",
              color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax_nn.bar(x + width / 2, bulk_nn, width, label="Adsorbent NN",
              color="#DD8452", edgecolor="black", linewidth=0.5)
    for i, (a, b) in enumerate(zip(ads_nn, bulk_nn)):
        if a > 0:
            ax_nn.text(i - width / 2, a + 0.03, f"{a:.2f}", ha="center",
                       va="bottom", fontsize=9)
        if b > 0:
            ax_nn.text(i + width / 2, b + 0.03, f"{b:.2f}", ha="center",
                       va="bottom", fontsize=9)
    ax_nn.set_xticks(x)
    ax_nn.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_nn.set_ylabel("Mean NN Distance (standardized)")
    ax_nn.set_title("(c) Physical Separation: Mean NN Distance (test → train)",
                     fontweight="bold", fontsize=13)
    ax_nn.legend(frameon=True, fontsize=9)

    # ── (d) NN distance distributions (overlaid) ─────────────────────────
    ax_box = fig.add_subplot(gs[1, 1])
    box_data = []
    box_labels = []
    box_colors = []
    for sd in split_data:
        if len(sd["bulk_nn_dists"]) > 0:
            box_data.append(sd["bulk_nn_dists"])
            box_labels.append(f"{sd['label']}\n(adsorbent)")
            box_colors.append(TEST_COLOR)
        if len(sd["ads_nn_dists"]) > 0:
            box_data.append(sd["ads_nn_dists"])
            box_labels.append(f"{sd['label']}\n(adsorbate)")
            box_colors.append(TRAIN_COLOR)

    if box_data:
        bp = ax_box.boxplot(box_data, vert=True, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax_box.set_ylabel("NN Distance (standardized)")
    ax_box.set_title("(d) NN Distance Distributions", fontweight="bold", fontsize=13)

    output_path = os.path.join(args.output_dir, "analysis_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

# ── Print summary table ──────────────────────────────────────────────────
print(f"\n{'=' * 110}")
print("SPLIT ANALYSIS SUMMARY")
print(f"{'=' * 110}")
header = (f"{'Split':<35} {'Train':>7} {'Test':>7} {'NotSel':>7} "
          f"{'PairOL':>7} {'AdsOL':>6} {'BulkOL':>7} "
          f"{'AdsNN':>7} {'BulkNN':>8}")
print(header)
print("-" * 110)
for sd in split_data:
    ads_nn_str = f"{sd['ads_nn_dists'].mean():.2f}" if len(sd["ads_nn_dists"]) > 0 else "shared"
    bulk_nn_str = f"{sd['bulk_nn_dists'].mean():.2f}" if len(sd["bulk_nn_dists"]) > 0 else "shared"
    print(f"{sd['label']:<35} {sd['n_train']:>7,} {sd['n_test']:>7,} {sd['n_not']:>7,} "
          f"{sd['pair_overlap']:>7} {sd['ads_overlap']:>6} {sd['bulk_overlap']:>7} "
          f"{ads_nn_str:>7} {bulk_nn_str:>8}")
print()
print("AdsNN  = mean NN distance from test adsorbates to nearest train (higher = more different)")
print("BulkNN = mean NN distance from test adsorbents to nearest train (higher = more different)")
print(f"\nFigures saved to {args.output_dir}/")
