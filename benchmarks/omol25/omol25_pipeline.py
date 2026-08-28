"""OMol25 leakage headline results on the FULL merged set (~9.55M structures).

Steps:
  1. Load cached features + meta; fit a non-negative column scale.
  2. Validate the factorized L(pi) estimator against exact cosine on a subsample.
  3. Build the low-rank factor B once (Nystrom, cosine) for the full set.
  4. L(pi) of the EXISTING native split (train_4M / val / test).
  5. Low-rank RE-SPLIT of the merged set at the same 3 proportions; L(pi).
  6. Save both splits (parquet + a CSV sample) for manual inspection.

Hypergraph is not run here (its O(n^2) k-NN is infeasible at ~10^7) — see
omol25_scaling.py for the low-rank-vs-hypergraph comparison across sizes.

Run (palm env):  python -m PALM.benchmarks.omol25.omol25_pipeline
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
from PALM.lowrank import balanced_lloyd
from PALM.benchmarks.omol25 import omol25_leakage as LK

RESULTS = os.path.join(os.path.dirname(__file__), "results")
RANK = 256
N_RESTARTS = 3


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    X, meta = LK.load_features()
    n = X.shape[0]
    native = meta["split"].to_numpy()                     # 0=train,1=val,2=test
    native_sizes = np.bincount(native, minlength=3)
    print(f"loaded {n:,} structures; native sizes train/val/test = {native_sizes.tolist()}", flush=True)

    scale = LK.fit_nonneg_scale(X)
    Xs = np.asarray(X, dtype=np.float32) * scale          # materialize scaled features (~4.4 GB)

    # 2. validate the factorized L(pi) estimator vs exact cosine on a subsample
    idx = np.random.default_rng(0).choice(n, size=40_000, replace=False)
    Xsub, lab_sub = Xs[idx], native[idx]
    B_sub = LK.build_factor(Xsub, rank=RANK, seed=0)
    lpi_fac = LK.lpi_from_factor(B_sub, lab_sub, 3)
    lpi_ex = LK.lpi_exact_cosine(Xsub, lab_sub)
    print(f"[validate] factorized L(pi)={lpi_fac:.4f} vs exact cosine={lpi_ex:.4f} "
          f"(|diff|={abs(lpi_fac-lpi_ex):.4f}) on 40k subsample", flush=True)

    # 3. full-set factor
    print("building full-set low-rank factor B ...", flush=True)
    B = LK.build_factor(Xs, rank=RANK, seed=0)
    print(f"  B = {B.shape} in {time.time()-t0:.0f}s", flush=True)

    # 4. existing-split leakage
    lpi_existing = LK.lpi_from_factor(B, native, 3)
    print(f"[L(pi)] EXISTING native split = {lpi_existing:.4f}", flush=True)

    # 5. low-rank re-split at the same 3 proportions (reuse B; FM auto-skipped at scale)
    best_lab, best_obj = None, np.inf
    for r in range(N_RESTARTS):
        lab = balanced_lloyd(B, native_sizes.tolist(), epsilon=0.0, n_iter=20, seed=r)
        from PALM.lowrank import lowrank_leakage
        obj = lowrank_leakage(B, lab, 3)
        if obj < best_obj:
            best_obj, best_lab = obj, lab
    lpi_lowrank = LK.lpi_from_factor(B, best_lab, 3)
    lr_sizes = np.bincount(best_lab, minlength=3)
    print(f"[L(pi)] LOW-RANK re-split   = {lpi_lowrank:.4f}   sizes={lr_sizes.tolist()}", flush=True)

    # 6. save splits for manual inspection
    name_map = {0: "train", 1: "val", 2: "test"}
    meta_out = meta.copy()
    meta_out["native_split"] = meta_out["split"].map(name_map)
    meta_out["lowrank_split"] = pd.Series(best_lab).map(name_map).to_numpy()
    meta_out = meta_out.drop(columns=["split"])
    full_path = os.path.join(RESULTS, "omol25_splits.parquet")
    meta_out.to_parquet(full_path)
    sample = meta_out.sample(n=min(20_000, n), random_state=0).sort_index()
    sample.to_csv(os.path.join(RESULTS, "omol25_splits_sample.csv"), index=True)

    summary = pd.DataFrame([
        {"split": "existing_native", "L_pi": round(lpi_existing, 4),
         "sizes": native_sizes.tolist()},
        {"split": "lowrank_resplit", "L_pi": round(lpi_lowrank, 4),
         "sizes": lr_sizes.tolist()},
    ])
    summary.to_csv(os.path.join(RESULTS, "omol25_lpi_summary.csv"), index=False)
    print("\n=== SUMMARY (L(pi), cosine over structural features; lower = less leakage) ===")
    print(summary.to_string(index=False))
    print(f"\nsaved: {full_path}\n       {RESULTS}/omol25_splits_sample.csv"
          f"\n       {RESULTS}/omol25_lpi_summary.csv\nTotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
