"""Head-to-head benchmark: low-rank factorized splitter vs the graph/hypergraph
backends and DataSAIL, on the MoleculeNet 1D datasets.

All methods split in the SAME 1024-bit ECFP space the metric scores in, and are
scored with the exact ``scaled_lpi`` (ECFP/Tanimoto) leakage. Because Mt-KaHyPar
is non-deterministic under multithreading, every method is run over several
seeds and reported as mean +/- std; a ``best-of`` column reports the lowest
leakage across all methods/seeds (a split selector — free, since scoring is
cheap).

Datasets are independent, so the whole sweep is parallelized across a process
pool (one worker per dataset). Run in the `palm` env:

    python -m PALM.lowrank_split.benchmark_lowrank                 # all datasets
    python -m PALM.lowrank_split.benchmark_lowrank --workers 4 esol bace
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── configuration ──────────────────────────────────────────────────────────

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "lowrank_benchmark.csv")

# published references from ../../benchmark/moleculenet1d_results.csv + addendum
REF_HYPERGRAPH_2048 = {
    "freesolv": 0.1509, "esol": 0.1653, "clintox": 0.2150, "sider": 0.2378,
    "bace": 0.2628, "bbbp": 0.2338, "lipophilicity": 0.2546, "tox21": 0.2239,
    "qm8": 0.1818, "hiv": 0.2498, "muv": 0.2484,
}
REF_DATASAIL = {
    "freesolv": 0.1424, "esol": 0.1668, "clintox": 0.2294, "sider": 0.2360,
    "bace": 0.2387, "bbbp": 0.2319, "lipophilicity": 0.2718, "tox21": 0.2230,
    "qm8": 0.2077, "hiv": None, "muv": None,          # DataSAIL times out at 41k/93k
}
ALL_DATASETS = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
                "lipophilicity", "tox21", "qm8", "hiv", "muv"]

N_SEEDS = 4
NYSTROM_RANK = 256
FM_MAX_N = 50_000          # skip the O(n r)/move FM polish above this n (Lloyd suffices)

CSV_COLS = ["dataset", "n", "datasail_ref", "hypergraph_ref",
            "hyperedge_mean", "hyperedge_std", "graph_fm_mean", "graph_fm_std",
            "lowrank_mean", "lowrank_std", "lowrank_time_s", "best_of",
            "best_of_method", "beats_datasail"]


# ── per-dataset worker (must be top-level for pickling) ─────────────────────

def _run_one_dataset(dataset: str, gpu_id: int = 0) -> dict:
    """Run every method over N_SEEDS on one dataset; return a summary row.

    Pinned to GPU ``gpu_id`` (set before any torch import) so parallel workers
    do not contend for a single device's memory.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # must precede torch import
    # imports live inside the worker so each spawned process initializes cleanly
    import logging
    logging.disable(logging.CRITICAL)
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
    from PALM.hypergraph import run_hypergraph_split, run_graph_split
    from PALM.lowrank_split.lowrank_split import run_lowrank_split
    from PALM.benchmark.benchmark_moleculenet1d import load_smiles
    from PALM.benchmark.leakage import scaled_lpi

    def ecfp1024(smiles):
        X = np.zeros((len(smiles), 1024), dtype=np.float32)
        for i, s in enumerate(smiles):
            m = Chem.MolFromSmiles(s)
            if m is not None:
                DataStructs.ConvertToNumpyArray(
                    AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), X[i])
        return X

    smiles = load_smiles(dataset)
    n = len(smiles)
    X = ecfp1024(smiles)
    feature_data = {smiles[i]: X[i] for i in range(n)}

    def score(split_dict):
        return scaled_lpi(list(smiles), split_dict)[0]

    hyperedge, graph_fm, lowrank = [], [], []
    times = {"hyperedge": 0.0, "graph+thr+fm": 0.0, "lowrank": 0.0}   # isolated per method
    candidates = []          # (score, method) across everything, for best-of
    threads = 4
    for s in range(N_SEEDS):
        t = time.time()
        split = run_hypergraph_split(feature_data, [8, 2], ["train", "test"],
                                     k=15, preset="quality", threads=threads, seed=s)
        times["hyperedge"] += time.time() - t
        v = score(split); hyperedge.append(v); candidates.append((v, "hyperedge"))

        t = time.time()
        split = run_graph_split(feature_data, [8, 2], ["train", "test"], k=15,
                                threshold=0.3, preset="quality", threads=threads,
                                seed=s, fm=True)
        times["graph+thr+fm"] += time.time() - t
        v = score(split); graph_fm.append(v); candidates.append((v, "graph+thr+fm"))

        t = time.time()
        split = run_lowrank_split(feature_data, [8, 2], ["train", "test"],
                                  rank=NYSTROM_RANK, n_restarts=4, fm=True,
                                  fm_max_n=FM_MAX_N, seed=s)
        times["lowrank"] += time.time() - t
        v = score(split); lowrank.append(v); candidates.append((v, "lowrank"))
    lr_time = times["lowrank"] / N_SEEDS      # mean isolated low-rank time per split

    best_score, best_method = min(candidates, key=lambda t: t[0])
    ds_ref = REF_DATASAIL.get(dataset)
    return {
        "dataset": dataset, "n": n,
        "datasail_ref": ds_ref, "hypergraph_ref": REF_HYPERGRAPH_2048.get(dataset),
        "hyperedge_mean": round(float(np.mean(hyperedge)), 4),
        "hyperedge_std": round(float(np.std(hyperedge)), 4),
        "graph_fm_mean": round(float(np.mean(graph_fm)), 4),
        "graph_fm_std": round(float(np.std(graph_fm)), 4),
        "lowrank_mean": round(float(np.mean(lowrank)), 4),
        "lowrank_std": round(float(np.std(lowrank)), 4),
        "lowrank_time_s": round(lr_time, 1),
        "best_of": round(best_score, 4), "best_of_method": best_method,
        "beats_datasail": (ds_ref is None) or (best_score < ds_ref),
    }


# ── driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None,
                    help="subset of datasets (default: all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel dataset workers")
    ap.add_argument("--gpus", type=int, default=8, help="GPUs to round-robin across")
    args = ap.parse_args()
    datasets = args.datasets or ALL_DATASETS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Running {len(datasets)} datasets on {args.workers} parallel workers "
          f"across {args.gpus} GPUs, {N_SEEDS} seeds each ...\n", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")           # required for CUDA in child processes
    rows = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        # pin each dataset to its own GPU (round-robin) to avoid memory contention
        futures = {ex.submit(_run_one_dataset, ds, i % args.gpus): ds
                   for i, ds in enumerate(datasets)}
        for fut in as_completed(futures):
            ds = futures[fut]
            try:
                row = fut.result()
                rows[ds] = row
                print(f"  [done] {ds:<14} lowrank={row['lowrank_mean']:.4f}"
                      f"±{row['lowrank_std']:.3f}  best-of={row['best_of']:.4f}"
                      f" ({row['best_of_method']})", flush=True)
            except Exception as e:
                print(f"  [FAIL] {ds}: {type(e).__name__}: {e}", flush=True)
    wall = time.time() - t0

    # write CSV in canonical dataset order
    ordered = [rows[d] for d in ALL_DATASETS if d in rows]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(ordered)

    # pretty table
    print(f"\n{'dataset':<14}{'n':>7}{'DataSAIL':>9}{'hyperedge':>18}"
          f"{'graph+thr+fm':>18}{'lowrank':>18}{'best-of':>10}")
    print("-" * 94)
    for r in ordered:
        ds_ref = f"{r['datasail_ref']:.4f}" if r['datasail_ref'] is not None else "timeout"
        star = " *" if r["beats_datasail"] else "  "
        print(f"{r['dataset']:<14}{r['n']:>7}{ds_ref:>9}"
              f"{r['hyperedge_mean']:>11.4f}±{r['hyperedge_std']:.3f}"
              f"{r['graph_fm_mean']:>11.4f}±{r['graph_fm_std']:.3f}"
              f"{r['lowrank_mean']:>11.4f}±{r['lowrank_std']:.3f}"
              f"{r['best_of']:>9.4f}{star}")
    n_beat = sum(1 for r in ordered if r["beats_datasail"])
    print(f"\nbest-of beats/ties DataSAIL on {n_beat}/{len(ordered)} datasets "
          f"(timeouts count as a win). Wall time: {wall:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
