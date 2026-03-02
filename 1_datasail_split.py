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
    features/oc22/adsorbate/physchem_features.csv
    features/oc22/adsorbate/composition_features.csv
    features/oc22/adsorbate/rdkit_descriptors_features.csv
    features/oc22/adsorbate/adsorption_features.csv

  Available adsorbent embeddings (--f-embedding):
    features/oc22/adsorbent/property_features.csv
    features/oc22/adsorbent/stoichiometry_features.csv
    features/oc22/adsorbent/electronic_features.csv
    features/oc22/adsorbent/bonding_features.csv
    features/oc22/adsorbent/thermodynamic_features.csv
    features/oc22/adsorbent/catalytic_features.csv

  Multiple files can be passed to concatenate embeddings:
    python datasail_split.py \\
      --e-embedding features/oc22/adsorbate/physchem_features.csv \\
                    features/oc22/adsorbate/composition_features.csv \\
      --f-embedding features/oc22/adsorbent/property_features.csv
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
    "features/oc22/adsorbate/physchem_features.csv",
    "features/oc22/adsorbate/composition_features.csv",
    "features/oc22/adsorbate/rdkit_descriptors_features.csv",
    "features/oc22/adsorbate/adsorption_features.csv",
]
DEFAULT_F = [
    "features/oc22/adsorbent/property_features.csv",
    "features/oc22/adsorbent/stoichiometry_features.csv",
    "features/oc22/adsorbent/electronic_features.csv",
    "features/oc22/adsorbent/bonding_features.csv",
    "features/oc22/adsorbent/thermodynamic_features.csv",
    "features/oc22/adsorbent/catalytic_features.csv",
]

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DataSAIL 2D splitting for OC22")
parser.add_argument("--e-embedding", nargs="+", default=None,
                    help="Adsorbate embedding CSV(s)")
parser.add_argument("--f-embedding", nargs="+", default=None,
                    help="Adsorbent embedding CSV(s)")
parser.add_argument("--f-clusters", type=int, default=30,
                    help="Number of adsorbent clusters (default: 30)")
parser.add_argument("--max-sec", type=int, default=300,
                    help="Max seconds per solver run (default: 300)")
parser.add_argument("--techniques", nargs="+",
                    default=["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"],
                    help="Splitting techniques to run")
parser.add_argument("--splits", nargs="+", type=float, default=[8, 2],
                    help="Split ratios (default: 8 2)")
parser.add_argument("--names", nargs="+", default=["train", "test"],
                    help="Split names (default: train test)")
parser.add_argument("--max-pair-workers", type=int, default=1,
                    help="Max parallel embedding pairs in all-pairs mode (default: 1 = sequential). "
                         "Total concurrent solvers = max-pair-workers * num-techniques.")
args = parser.parse_args()

# Determine if we should generate all pairs
using_defaults = (args.e_embedding is None and args.f_embedding is None)

if using_defaults:
    # Generate all pairwise combinations
    embedding_pairs = [(e, f) for e in DEFAULT_E for f in DEFAULT_F]
    print("=" * 70)
    print("DataSAIL 2D Splitting for OC22 - ALL PAIRS MODE")
    print("=" * 70)
    print(f"  Generating {len(embedding_pairs)} pairs from defaults:")
    print(f"    E (adsorbate): {len(DEFAULT_E)} options")
    print(f"    F (adsorbent): {len(DEFAULT_F)} options")
    print(f"  Techniques: {args.techniques}")
    print(f"  Split ratio: {args.splits} -> {args.names}")
    print(f"  f_clusters={args.f_clusters}, max_sec={args.max_sec}")
    print()
else:
    # Use specified embeddings or defaults
    e_emb = args.e_embedding if args.e_embedding else DEFAULT_E
    f_emb = args.f_embedding if args.f_embedding else DEFAULT_F
    embedding_pairs = [(e_emb, f_emb)]
    print("=" * 70)
    print("DataSAIL 2D Splitting for OC22")
    print("=" * 70)
    print(f"  Adsorbate embeddings (e): {e_emb}")
    print(f"  Adsorbent embeddings (f): {f_emb}")
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
def compute_dist_matrix(data_dict, use_cosine=False, use_pca=False, pca_components=20):
    from sklearn.decomposition import PCA
    names = sorted(data_dict.keys())
    X = np.array([data_dict[n] for n in names])

    # Apply PCA for very sparse, high-dimensional features
    if use_pca and X.shape[1] > pca_components:
        n_components = min(pca_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

    if use_cosine:
        # For sparse features, use cosine distance without scaling
        # Cosine distance is better for high-dimensional sparse data
        dist = squareform(pdist(X, metric="cosine"))
    else:
        # For dense features, use standard Euclidean with scaling
        X_scaled = StandardScaler().fit_transform(X)
        dist = squareform(pdist(X_scaled, metric="euclidean"))

    dmax = dist.max()
    if dmax > 0:
        dist = dist / dmax
    return names, dist

# ── Build output tag from embedding filenames ─────────────────────────────
def tag_from_files(files):
    names = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        base = base.replace("_features", "")
        names.append(base)
    return "+".join(names)

# ── DataSAIL technique runner (module-level for pickling) ─────────────────
def run_technique(technique, common_kwargs):
    """Run a single DataSAIL technique. Must be module-level for multiprocessing."""
    t0 = time.time()
    e_s, f_s, i_s = datasail(techniques=[technique], **common_kwargs)
    return technique, i_s[technique], time.time() - t0

# ── Main pipeline function ────────────────────────────────────────────────
def run_pipeline(e_embedding_list, f_embedding_list, pair_idx=None, total_pairs=None):
    """Run the full DataSAIL pipeline for a given embedding pair.

    Collects output lines into a buffer to avoid interleaved prints when
    multiple pairs run in parallel.  Returns the buffer as a string.
    """
    log = []          # buffered output lines
    def _log(msg=""):
        log.append(msg)

    if pair_idx is not None:
        _log(f"\n{'=' * 70}")
        _log(f"PAIR {pair_idx}/{total_pairs}")
        _log(f"{'=' * 70}")
        _log(f"  E (adsorbate): {e_embedding_list}")
        _log(f"  F (adsorbent): {f_embedding_list}")

    # Load embeddings
    e_data = load_features(e_embedding_list, key_col="ads_symbols")
    f_data = load_features(f_embedding_list, key_col="bulk_symbols")
    _log(f"  Adsorbate entities (e): {len(e_data)}, feature dim: {len(next(iter(e_data.values())))}")
    _log(f"  Adsorbent entities (f): {len(f_data)}, feature dim: {len(next(iter(f_data.values())))}")

    # Check feature sparsity BEFORE computing distance matrices
    e_features = np.array([e_data[n] for n in sorted(e_data.keys())])
    f_features = np.array([f_data[n] for n in sorted(f_data.keys())])
    e_sparsity = (e_features == 0).sum() / e_features.size
    f_sparsity = (f_features == 0).sum() / f_features.size
    _log(f"  Adsorbate feature sparsity: {e_sparsity*100:.1f}%")
    _log(f"  Adsorbent feature sparsity: {f_sparsity*100:.1f}%")

    # Use PCA and cosine distance for very sparse features (>90% sparsity)
    use_pca_e = e_sparsity > 0.9 and "C2" in args.techniques
    use_pca_f = f_sparsity > 0.9 and "C2" in args.techniques
    use_cosine_e = e_sparsity > 0.5 and not use_pca_e
    use_cosine_f = f_sparsity > 0.5 and not use_pca_f

    if use_pca_e or use_pca_f:
        _log(f"  Using PCA dimensionality reduction for C2: {'E' if use_pca_e else ''}{' F' if use_pca_f else ''}")
    if use_cosine_e or use_cosine_f:
        _log(f"  Using cosine distance for: {'E' if use_cosine_e else ''}{' F' if use_cosine_f else ''}")

    # Compute distance matrices with appropriate preprocessing
    e_names, e_dist = compute_dist_matrix(e_data, use_cosine=use_cosine_e, use_pca=use_pca_e)
    f_names, f_dist = compute_dist_matrix(f_data, use_cosine=use_cosine_f, use_pca=use_pca_f)
    _log(f"  Distance matrices: e={e_dist.shape}, f={f_dist.shape}")

    # Adaptive f_clusters based on sparsity
    if f_sparsity > 0.5:
        if f_sparsity > 0.9:
            adaptive_f_clusters = max(5, min(10, len(f_data) // 50))
        else:
            adaptive_f_clusters = min(20, len(f_data) // 20)
        _log(f"  High sparsity ({f_sparsity*100:.1f}%) detected! Reducing f_clusters: {args.f_clusters} -> {adaptive_f_clusters}")
    else:
        adaptive_f_clusters = args.f_clusters
        _log(f"  Using default f_clusters: {adaptive_f_clusters}")

    # Adaptive e_clusters and delta/epsilon for C2 with sparse features
    adaptive_e_clusters = min(9, len(e_data))
    adaptive_delta = 0.1
    adaptive_epsilon = 0.1
    adaptive_max_sec = args.max_sec

    if "C2" in args.techniques and f_sparsity > 0.5:
        adaptive_e_clusters = max(2, min(3, len(e_data) // 3))
        adaptive_delta = 0.4
        adaptive_epsilon = 0.4
        adaptive_max_sec = max(600, args.max_sec * 2)
        _log(f"  C2 constraint relaxation for sparse features:")
        _log(f"    e_clusters: {min(9, len(e_data))} -> {adaptive_e_clusters}")
        _log(f"    delta/epsilon: 0.1 -> {adaptive_delta}")
        _log(f"    max_sec: {args.max_sec} -> {adaptive_max_sec}")
        _log(f"  WARNING: C2 may still fail with high-sparsity features!")

    # Run DataSAIL (parallel per technique)
    _log(f"  Running {len(args.techniques)} techniques in parallel...")

    common_kwargs = dict(
        splits=args.splits,
        names=args.names,
        runs=1,
        solver="SCIP",
        max_sec=adaptive_max_sec,
        e_type="O",
        e_data=e_data,
        e_dist=(e_names, e_dist),
        e_clusters=adaptive_e_clusters,
        f_type="O",
        f_data=f_data,
        f_dist=(f_names, f_dist),
        f_clusters=adaptive_f_clusters,
        inter=inter,
        delta=adaptive_delta,
        epsilon=adaptive_epsilon,
    )

    all_inter_splits = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=len(args.techniques)) as pool:
        futures = {pool.submit(run_technique, t, common_kwargs): t for t in args.techniques}
        for future in as_completed(futures):
            technique, result, elapsed = future.result()
            all_inter_splits[technique] = result
            _log(f"    {technique} finished in {elapsed:.1f}s")

    _log(f"  All techniques finished in {time.time() - t_start:.1f}s (wall-clock)")

    # Build output tag
    e_tag = tag_from_files(e_embedding_list)
    f_tag = tag_from_files(f_embedding_list)
    out_tag = f"e_{e_tag}__f_{f_tag}"

    # Map inter_splits back to system_ids and save
    _log(f"\n  RESULTS FOR {out_tag}")
    _log(f"  {'-' * 68}")

    os.makedirs("output/split_result", exist_ok=True)

    for technique in sorted(all_inter_splits.keys()):
        run_result = all_inter_splits[technique][0]

        # Check if C2 failed to generate test split (known issue with sparse features)
        if technique == "C2":
            unique_splits = set(run_result.values())
            if "test" not in unique_splits or len(args.names) < 2:
                _log(f"\n  WARNING: C2 failed to generate '{args.names[1] if len(args.names) > 1 else 'test'}' split!")
                _log(f"  This can happen with:")
                _log(f"    - High feature sparsity ({f_sparsity*100:.1f}% for adsorbent)")
                _log(f"    - Imbalanced cluster sizes")
                _log(f"    - Sparse interaction patterns")
                _log(f"  The C2 constraints may be infeasible for this feature combination.")
                _log(f"  Consider using:")
                _log(f"    - I2 (identity-cold on both) instead of C2")
                _log(f"    - Lower-sparsity features (property, stoichiometry)")
                _log(f"    - Custom preprocessing (PCA, feature selection)")
                _log()

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

        _log(f"  --- {technique} ---")
        _log(f"    {args.names[0].capitalize()}: {n_train:,} ({100*n_train/len(system_ids):.1f}%)  |  "
              f"{args.names[1].capitalize() if len(args.names) > 1 else 'N/A'}: {n_test:,} ({100*n_test/len(system_ids):.1f}%)")
        if n_not > 0:
            _log(f"    Not selected: {n_not:,}")
        _log(f"    Adsorbents  - {args.names[0].capitalize()}: {len(train_bulks):,}, "
              f"{args.names[1].capitalize() if len(args.names) > 1 else 'N/A'}: {len(test_bulks):,}, "
              f"Overlap: {len(train_bulks & test_bulks):,}")
        _log(f"    Adsorbates  - {args.names[0].capitalize()}: {len(train_ads)}, "
              f"{args.names[1].capitalize() if len(args.names) > 1 else 'N/A'}: {len(test_ads)}, "
              f"Overlap: {len(train_ads & test_ads)}")

        out_path = f"output/split_result/datasail_split_{technique}__{out_tag}.csv"
        out_df.to_csv(out_path, index=False)
        _log(f"    Saved to {out_path}")

    output = "\n".join(log)
    print(output)
    return output

# ── Run pipeline for all pairs ────────────────────────────────────────────
if using_defaults:
    print(f"\nProcessing {len(embedding_pairs)} embedding pairs...\n")
    overall_start = time.time()

    if args.max_pair_workers <= 1:
        # Sequential execution
        for idx, (e_emb, f_emb) in enumerate(embedding_pairs, 1):
            run_pipeline([e_emb], [f_emb], pair_idx=idx, total_pairs=len(embedding_pairs))
    else:
        # Parallel execution across pairs
        print(f"Running up to {args.max_pair_workers} pairs in parallel "
              f"({args.max_pair_workers * len(args.techniques)} max concurrent solvers)\n")
        with ProcessPoolExecutor(max_workers=args.max_pair_workers) as pool:
            futures = {}
            for idx, (e_emb, f_emb) in enumerate(embedding_pairs, 1):
                fut = pool.submit(run_pipeline, [e_emb], [f_emb],
                                  pair_idx=idx, total_pairs=len(embedding_pairs))
                futures[fut] = idx
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n  ERROR in pair {idx}: {e}")

    print(f"\n{'=' * 70}")
    print(f"ALL PAIRS COMPLETED in {time.time() - overall_start:.1f}s")
    print(f"{'=' * 70}")
else:
    # Single run with specified or concatenated embeddings
    e_emb, f_emb = embedding_pairs[0]
    run_pipeline(e_emb if isinstance(e_emb, list) else [e_emb],
                 f_emb if isinstance(f_emb, list) else [f_emb])

print("\nDone!")
