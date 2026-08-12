"""Produce and SAVE the low-rank split of the 100k UMA-embedded subsample.

Loads the merged UMA embedding, builds the RBF Nystrom factor (same similarity as
uma_leakage.py), runs the low-rank splitter at the native train/val/test
proportions, and writes the per-structure assignment (with identifiers) plus an
80/20 train/test variant. Reports L(pi) for each vs the native split and random.

Run (palm env):  python -m PALM.benchmarks.omol25.uma_split
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.splitters.methods.lowrank import balanced_lloyd, lowrank_leakage
from PALM.benchmarks.omol25 import uma_leakage as UL   # reuse rbf_factor / median_sigma / lpi_from_factor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache_uma")
RESULTS = os.path.join(HERE, "results")
META = os.path.join(HERE, "..", "..", "data", "DataSAIL_data", "1D", "omol25", "_cache", "meta.parquet")
RANK = 256


def best_split(B, sizes, k, restarts=4):
    best_lab, best_obj = None, np.inf
    for r in range(restarts):
        lab = balanced_lloyd(B, list(sizes), epsilon=0.0, n_iter=20, seed=r)
        obj = lowrank_leakage(B, lab, k)
        if obj < best_obj:
            best_obj, best_lab = obj, lab
    return best_lab


def main():
    os.makedirs(RESULTS, exist_ok=True)
    Z = np.load(os.path.join(CACHE, "uma_emb.npy"))
    native = np.load(os.path.join(CACHE, "native.npy"))
    orig_row = np.load(os.path.join(CACHE, "orig_row.npy"))
    n = len(Z)
    assert len(native) == n == len(orig_row), (len(Z), len(native), len(orig_row))
    print(f"loaded {Z.shape} UMA emb; native counts {np.bincount(native).tolist()}", flush=True)

    # whiten (same as uma_leakage) then RBF Nystrom factor at the median-distance sigma
    Z = ((Z - Z.mean(0)) / (Z.std(0) + 1e-8)).astype(np.float32)
    sigma = UL.median_sigma(Z)
    t0 = time.time()
    B = UL.rbf_factor(Z, rank=RANK, sigma=sigma, seed=0)
    print(f"RBF factor B={B.shape} (sigma={sigma:.3f}) in {time.time()-t0:.1f}s", flush=True)

    native_sizes = np.bincount(native, minlength=3)
    lpi_native = UL.lpi_from_factor(B, native, 3)
    lpi_random = UL.lpi_from_factor(B, np.random.default_rng(2).permutation(native), 3)

    # (1) low-rank 3-way split at native proportions (directly comparable to native)
    t0 = time.time()
    lab3 = best_split(B, native_sizes.tolist(), k=3)
    lpi_lr3 = UL.lpi_from_factor(B, lab3, 3)
    print(f"[3-way] low-rank split in {time.time()-t0:.1f}s  sizes={np.bincount(lab3).tolist()} "
          f"L(pi)={lpi_lr3:.4f}", flush=True)

    # (2) low-rank 80/20 train/test split
    lab2 = best_split(B, [int(round(0.8 * n)), n - int(round(0.8 * n))], k=2)
    lpi_lr2 = UL.lpi_from_factor(B, lab2, 2)
    print(f"[80/20] low-rank split  sizes={np.bincount(lab2).tolist()}  L(pi)={lpi_lr2:.4f}", flush=True)

    # attach identifiers from meta.parquet (orig_row indexes the full merged meta)
    out = pd.DataFrame({"orig_row": orig_row, "native_split": native,
                        "lowrank_3way": lab3, "lowrank_8020": lab2})
    name3 = {0: "train", 1: "val", 2: "test"}
    name2 = {0: "train", 1: "test"}
    out["native_name"] = out["native_split"].map(name3)
    out["lowrank_3way_name"] = out["lowrank_3way"].map(name3)
    out["lowrank_8020_name"] = out["lowrank_8020"].map(name2)
    try:
        meta = pd.read_parquet(META, columns=["split", "shard", "db_id", "data_id", "natoms"])
        j = meta.iloc[orig_row].reset_index(drop=True)
        for c in ["shard", "db_id", "data_id", "natoms"]:
            out[c] = j[c].values
    except Exception as e:  # identifiers are a convenience; the split is valid without them
        print(f"[warn] could not join meta.parquet ({e}); saving without identifiers", flush=True)

    pq = os.path.join(RESULTS, "omol25_uma_lowrank_split.parquet")
    csv = os.path.join(RESULTS, "omol25_uma_lowrank_split_sample.csv")
    out.to_parquet(pq, index=False)
    out.head(20000).to_csv(csv, index=False)

    summary = {"n": int(n), "sigma": float(sigma),
               "lpi_random": lpi_random, "lpi_native": lpi_native,
               "lpi_lowrank_3way": lpi_lr3, "lpi_lowrank_8020": lpi_lr2,
               "native_sizes": native_sizes.tolist(),
               "lowrank_3way_sizes": np.bincount(lab3).tolist(),
               "lowrank_8020_sizes": np.bincount(lab2).tolist()}
    json.dump(summary, open(os.path.join(RESULTS, "omol25_uma_lowrank_split_summary.json"), "w"), indent=2)

    print("\n=== L(pi) on UMA-RBF similarity (100k, 3-way native proportions) ===")
    print(f"  random baseline        {lpi_random:.4f}")
    print(f"  existing native split  {lpi_native:.4f}")
    print(f"  low-rank 3-way         {lpi_lr3:.4f}   (reduction vs native {lpi_native-lpi_lr3:+.4f})")
    print(f"  low-rank 80/20         {lpi_lr2:.4f}")
    print(f"\nsaved:\n  {pq}\n  {csv}\n  {os.path.join(RESULTS, 'omol25_uma_lowrank_split_summary.json')}")


if __name__ == "__main__":
    main()
