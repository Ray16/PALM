"""Wall-clock DataSAIL vs low-rank split time on the MoleculeNet datasets missing
from results/lowrank_timing.csv (that file only had esol/tox21/qm8/hiv/muv).
Same DataSAIL invocation as datasail_hiv_leakage.py (C1e technique, 8/2 split),
same low-rank config as benchmark_lowrank.py (rank=256, n_restarts=4, fm=True).
Appends rows to results/lowrank_timing.csv.

    python -m PALM.benchmarks.moleculenet.benchmark_datasail_timing
    python -m PALM.benchmarks.moleculenet.benchmark_datasail_timing bace bbbp
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging; logging.disable(logging.CRITICAL)
import numpy as np

from PALM.splitters import SplitSpec, split
from PALM.benchmarks.common.datasail import datasail_fingerprint
from PALM.benchmarks.common.datasets import load_smiles
from PALM.benchmarks.common.featurize import ecfp1024

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "results", "lowrank_timing.csv")
MISSING = ["bace", "bbbp", "clintox", "freesolv", "lipophilicity", "sider"]


def datasail_time(smiles, max_sec):
    t0 = time.time()
    datasail_fingerprint({s: s for s in smiles}, max_sec=max_sec)
    return time.time() - t0


def lowrank_time(smiles):
    X = ecfp1024(smiles)
    feature_data = {smiles[i]: X[i] for i in range(len(smiles))}
    t0 = time.time()
    split("lowrank", feature_data, SplitSpec([8, 2], ["train", "test"], seed=0),
          rank=256, n_restarts=4, fm=True, fm_max_n=50_000)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=None)
    ap.add_argument("--max-sec", type=int, default=1800,
                     help="DataSAIL per-dataset timeout")
    args = ap.parse_args()
    datasets = args.datasets or MISSING

    rows = []
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as fh:
            rows = list(csv.DictReader(fh))
    have = {r["dataset"] for r in rows}

    for ds in datasets:
        smiles = load_smiles(ds)
        n = len(smiles)
        print(f"{ds}: n={n} ...", flush=True)
        lr_s = lowrank_time(smiles)
        try:
            ds_s = datasail_time(smiles, args.max_sec)
            speedup = str(round(ds_s / lr_s))
        except Exception as e:
            print(f"  DataSAIL failed on {ds}: {type(e).__name__}: {e}", flush=True)
            ds_s, speedup = "", "timeout"
        print(f"  lowrank={lr_s:.2f}s  datasail="
              f"{ds_s if ds_s == '' else f'{ds_s:.1f}s'}  speedup={speedup}",
              flush=True)
        row = {"dataset": ds, "n": n, "lowrank_s": round(lr_s, 2),
               "datasail_s": round(ds_s, 1) if ds_s != "" else "",
               "speedup": speedup}
        if ds in have:
            rows = [row if r["dataset"] == ds else r for r in rows]
        else:
            rows.append(row)
        # write after every dataset so partial progress survives a crash
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["dataset", "n", "lowrank_s",
                                                "datasail_s", "speedup"])
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: int(r["n"])))
    print("saved", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
