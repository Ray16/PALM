"""
DataSAIL-based 2D dataset splitting for OC22.

Follows the PDBBind two-dimensional split pattern:
  e-entity = adsorbate  (analogous to ligand)
  f-entity = adsorbent  (analogous to protein/pocket)
  inter    = (adsorbate, adsorbent) pairs per system

Runs all supported techniques:
  R   = random split
  I1e = identity-cold on adsorbate
  I1f = identity-cold on adsorbent
  I2  = identity-cold on both
  C1e = cluster-cold on adsorbate
  C1f = cluster-cold on adsorbent
  C2  = cluster-cold on both (double-cold)

Usage:
  python datasail_split.py --e-embedding <file> --f-embedding <file>

  Available adsorbate embeddings (--e-embedding):
    embeddings/adsorbate/physchem_features.csv
    embeddings/adsorbate/composition_features.csv
    embeddings/adsorbate/rdkit_descriptors_features.csv

  Available adsorbent embeddings (--f-embedding):
    embeddings/adsorbent/property_features.csv
    embeddings/adsorbent/stoichiometry_features.csv
    embeddings/adsorbent/composition_features.csv
    embeddings/adsorbent/fraction_features.csv

  Multiple files can be passed to concatenate embeddings:
    python datasail_split.py \\
      --e-embedding embeddings/adsorbate/physchem_features.csv \\
                    embeddings/adsorbate/composition_features.csv \\
      --f-embedding embeddings/adsorbent/property_features.csv
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from datasail.sail import datasail

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_E = [
    "embeddings/adsorbate/physchem_features.csv",
    "embeddings/adsorbate/composition_features.csv",
    "embeddings/adsorbate/rdkit_descriptors_features.csv",
]
DEFAULT_F = [
    "embeddings/adsorbent/property_features.csv",
    "embeddings/adsorbent/stoichiometry_features.csv",
    "embeddings/adsorbent/composition_features.csv",
    "embeddings/adsorbent/fraction_features.csv",
]

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DataSAIL 2D splitting for OC22")
parser.add_argument("--e-embedding", nargs="+", default=DEFAULT_E,
                    help="Adsorbate embedding CSV(s)")
parser.add_argument("--f-embedding", nargs="+", default=DEFAULT_F,
                    help="Adsorbent embedding CSV(s)")
parser.add_argument("--f-clusters", type=int, default=200,
                    help="Number of adsorbent clusters (default: 200)")
parser.add_argument("--max-sec", type=int, default=300,
                    help="Max seconds per solver run (default: 300)")
parser.add_argument("--techniques", nargs="+",
                    default=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
                    help="Splitting techniques to run")
parser.add_argument("--splits", nargs="+", type=float, default=[8, 2],
                    help="Split ratios (default: 8 2)")
parser.add_argument("--names", nargs="+", default=["train", "test"],
                    help="Split names (default: train test)")
args = parser.parse_args()

print("=" * 70)
print("DataSAIL 2D Splitting for OC22")
print("=" * 70)
print(f"  Adsorbate embeddings (e): {args.e_embedding}")
print(f"  Adsorbent embeddings (f): {args.f_embedding}")
print(f"  Techniques: {args.techniques}")
print(f"  Split ratio: {args.splits} -> {args.names}")
print(f"  f_clusters={args.f_clusters}, max_sec={args.max_sec}")
print()

# ── Load metadata ─────────────────────────────────────────────────────────
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}
system_ids = sorted(entries.keys(), key=int)
print(f"Total systems: {len(entries)}")

# ── Helper: load & merge feature CSVs, deduplicate by key column ─────────
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
    return {name: dedup.loc[name, feat_cols].values.astype(float) for name in dedup.index}

# ── Load embeddings ───────────────────────────────────────────────────────
e_data = load_features(args.e_embedding, key_col="ads_symbols")
f_data = load_features(args.f_embedding, key_col="bulk_symbols")
print(f"Adsorbate entities (e): {len(e_data)}, feature dim: {len(next(iter(e_data.values())))}")
print(f"Adsorbent entities (f): {len(f_data)}, feature dim: {len(next(iter(f_data.values())))}")

# ── Build interaction list ────────────────────────────────────────────────
inter = []
sid_to_inter = {}
for sid in system_ids:
    ads = entries[sid]["ads_symbols"]
    bulk = entries[sid]["bulk_symbols"]
    inter.append((ads, bulk))
    sid_to_inter[sid] = (ads, bulk)
print(f"Interactions: {len(inter)} (unique pairs: {len(set(inter))})")

# ── Compute distance matrices ─────────────────────────────────────────────
# Distances normalized to [0, 1] — required for C2 solver.
# C2 computes intra_weights = 1 - distances; unnormalized values (>> 1)
# produce negative weights, making the CVXPY problem non-DQCP.
def compute_dist_matrix(data_dict):
    names = sorted(data_dict.keys())
    X = np.array([data_dict[n] for n in names])
    X_scaled = StandardScaler().fit_transform(X)
    dist = squareform(pdist(X_scaled, metric="euclidean"))
    dmax = dist.max()
    if dmax > 0:
        dist = dist / dmax
    return names, dist

e_names, e_dist = compute_dist_matrix(e_data)
f_names, f_dist = compute_dist_matrix(f_data)
print(f"Distance matrices: e={e_dist.shape}, f={f_dist.shape}")

# ── Run DataSAIL (parallel per technique) ─────────────────────────────────
print(f"\nRunning {len(args.techniques)} techniques in parallel...\n")

common_kwargs = dict(
    splits=args.splits,
    names=args.names,
    runs=1,
    solver="SCIP",
    max_sec=args.max_sec,
    e_type="O",
    e_data=e_data,
    e_dist=(e_names, e_dist),
    e_clusters=min(9, len(e_data)),
    f_type="O",
    f_data=f_data,
    f_dist=(f_names, f_dist),
    f_clusters=args.f_clusters,
    inter=inter,
    delta=0.1,
    epsilon=0.1,
)

def run_technique(technique):
    t0 = time.time()
    e_s, f_s, i_s = datasail(techniques=[technique], **common_kwargs)
    return technique, i_s[technique], time.time() - t0

all_inter_splits = {}
t_start = time.time()
with ProcessPoolExecutor(max_workers=len(args.techniques)) as pool:
    futures = {pool.submit(run_technique, t): t for t in args.techniques}
    for future in as_completed(futures):
        technique, result, elapsed = future.result()
        all_inter_splits[technique] = result
        print(f"  {technique} finished in {elapsed:.1f}s")

print(f"\nAll techniques finished in {time.time() - t_start:.1f}s (wall-clock)")

# ── Build output tag from embedding filenames ─────────────────────────────
def tag_from_files(files):
    names = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        base = base.replace("_features", "")
        names.append(base)
    return "+".join(names)

e_tag = tag_from_files(args.e_embedding)
f_tag = tag_from_files(args.f_embedding)
out_tag = f"e_{e_tag}__f_{f_tag}"

# ── Map inter_splits back to system_ids and save ─────────────────────────
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

os.makedirs("output", exist_ok=True)

for technique in sorted(all_inter_splits.keys()):
    run_result = all_inter_splits[technique][0]

    records = []
    for sid in system_ids:
        key = sid_to_inter[sid]
        split = run_result.get(key, "not selected")
        records.append({"system_id": sid, "split": split})

    out_df = pd.DataFrame(records)
    assigned = out_df[out_df["split"].isin(args.names)]

    n_train = (assigned["split"] == args.names[0]).sum()
    n_test = (assigned["split"] == args.names[1]).sum() if len(args.names) > 1 else 0
    n_not = (out_df["split"] == "not selected").sum()

    train_bulks = set(entries[s]["bulk_symbols"]
                      for s in assigned[assigned["split"] == args.names[0]]["system_id"])
    test_bulks = set(entries[s]["bulk_symbols"]
                     for s in assigned[assigned["split"] == args.names[1]]["system_id"]) if len(args.names) > 1 else set()
    train_ads = set(entries[s]["ads_symbols"]
                    for s in assigned[assigned["split"] == args.names[0]]["system_id"])
    test_ads = set(entries[s]["ads_symbols"]
                   for s in assigned[assigned["split"] == args.names[1]]["system_id"]) if len(args.names) > 1 else set()

    print(f"\n--- {technique} ---")
    print(f"  {args.names[0].capitalize()}: {n_train:,} ({100*n_train/len(system_ids):.1f}%)  |  "
          f"{args.names[1].capitalize() if len(args.names) > 1 else 'N/A'}: {n_test:,} ({100*n_test/len(system_ids):.1f}%)")
    if n_not > 0:
        print(f"  Not selected: {n_not:,}")
    print(f"  Adsorbents  - {args.names[0].capitalize()}: {len(train_bulks):,}, "
          f"{args.names[1].capitalize() if len(args.names) > 1 else 'N/A'}: {len(test_bulks):,}, "
          f"Overlap: {len(train_bulks & test_bulks):,}")
    print(f"  Adsorbates  - {args.names[0].capitalize()}: {len(train_ads)}, "
          f"{args.names[1].capitalize() if len(args.names) > 1 else 'N/A'}: {len(test_ads)}, "
          f"Overlap: {len(train_ads & test_ads)}")

    out_path = f"output/datasail_split_{technique}__{out_tag}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

print("\nDone!")
