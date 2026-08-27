"""n-D split scaling study on the large USPTO-MCR master set.

Sweeps dataset size n over the ~34k-record near-unique 3-reactant master
(``records_large.csv``, built by ``prepare_uspto_mcr_large.py``) and, at each
size, times the per-axis k-NN n-D split and scores its macro scaled L(pi) (GPU)
against a random baseline. Produces the scaling curve for selling point #2:
split time and leakage as a function of the number of records.

Subsamples are nested prefixes of a single shuffle, so larger sizes contain the
smaller ones (clean monotone curve). Component features (Morgan FP per axis) are
computed once on the full set and reused at every size, so the reported time is
the split alone, not featurization.

Run (palm env, from PALM parent):  python -m PALM.benchmarks.reactions.benchmark_mcr_scaling
Writes benchmark/mcr_scaling_results.csv.
"""

import csv
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from PALM import reactions as R
from PALM.splitters import SplitSpec, split
from .benchmark_reactions import random_split
from .leakage_nd import macro_axis_lpi_gpu

N_JOBS = -1   # use all cores for fingerprinting


def parallel_morgan_map(values):
    """{SMILES: Morgan FP}, fingerprinted in parallel across cores (RDKit is
    single-threaded per call, so the embarrassingly-parallel unique-value list
    is split over processes)."""
    values = list(values)
    fps = Parallel(n_jobs=N_JOBS, batch_size=1024)(delayed(R._morgan)(v) for v in values)
    return dict(zip(values, fps))

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "DataSAIL_data",
                    "3D+", "uspto_mcr", "records_large.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "mcr_scaling_results.csv")
K = 25
SIZES = [1000, 2000, 5000, 10000, 20000, 34000]
SEED = 7


def main():
    df = pd.read_csv(DATA)
    axes = ["rA", "rB", "rC"]
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(df))            # one shuffle; sizes are nested prefixes
    df = df.iloc[order].reset_index(drop=True)
    records_all = [{a: str(getattr(r, a)) for a in axes} for r in df.itertuples()]

    n_uniq = sum(df[a].nunique() for a in axes)
    print(f"featurizing {n_uniq} unique reactants (once, {N_JOBS} jobs)...")
    t0 = time.time()
    afm_all = {a: parallel_morgan_map(sorted({rec[a] for rec in records_all})) for a in axes}
    print(f"  featurization: {time.time() - t0:.1f}s")

    # warm up CUDA/torch + Mt-KaHyPar so the first timed size is not charged the
    # one-time GPU context + solver initialization (~2-3 s).
    split("hypergraph_nd_knn", (records_all[:500], afm_all),
          SplitSpec([8, 2], ["train", "test"], seed=42), k=K)
    macro_axis_lpi_gpu(records_all[:500], afm_all, random_split(500))

    rows = []
    for n in SIZES:
        n = min(n, len(records_all))
        recs = records_all[:n]

        t0 = time.time()
        r_knn = split("hypergraph_nd_knn", (recs, afm_all),
                      SplitSpec([8, 2], ["train", "test"], seed=42), k=K)
        a_knn = [r_knn.assignment[i] for i in range(len(recs))]
        info = r_knn.diagnostics
        t_split = round(time.time() - t0, 2)

        k_lpi, k_axis = macro_axis_lpi_gpu(recs, afm_all, a_knn)
        r_lpi, _ = macro_axis_lpi_gpu(recs, afm_all, random_split(n))

        rows.append([n, len(axes), round(k_lpi, 4), round(r_lpi, 4),
                     round(r_lpi - k_lpi, 4), t_split, info["km1"], info["n_hyperedges"]])
        print(f"  n={n:6d}  knn L(pi)={k_lpi:.4f}  random={r_lpi:.4f}  "
              f"reduction={r_lpi-k_lpi:+.4f}  split={t_split:5.2f}s  edges={info['n_hyperedges']}")

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "n_axes", "knn_macro_lpi", "random_macro_lpi",
                    "reduction", "split_time_s", "km1", "n_hyperedges"])
        w.writerows(rows)
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
