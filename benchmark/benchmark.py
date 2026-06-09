"""Benchmark the PALM hypergraph backend against DataSAIL on the prepared
1D molecule datasets (data/DataSAIL_data/1D/moleculenet).

For each dataset we compute, on the SAME molecules and the SAME leakage metric
(DataSAIL's eval_split with ECFP/Tanimoto):
  - hypergraph (GPU kNN + Mt-KaHyPar)   leakage + wall-clock
  - DataSAIL C1e (cluster-based 1D)      leakage + wall-clock
  - random 80/20                         leakage (baseline)
and list the addendum's reported "DataSAIL S1" value for reference.

Notes / honesty:
  - eval_split and DataSAIL both build an O(n^2) similarity matrix, so they
    are only feasible up to ~20-30k entities; for HIV (41k) / MUV (93k) those
    steps may OOM/timeout and are reported as N/A — the hypergraph (sparse kNN)
    still runs, which is itself the scalability point.

Run (from the PALM parent dir, palm env):
    python -m PALM.benchmark.benchmark
Results stream to benchmark/results.csv.
"""

import csv
import os
import random
import signal
import sys
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
import logging
logging.disable(logging.CRITICAL)

from ..hypergraph import run_hypergraph_split
from datasail.sail import datasail
from datasail.eval import eval_split

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "DataSAIL_data", "1D", "moleculenet")
OUT = os.path.join(HERE, "results.csv")

# addendum Table 1 "DataSAIL S1" scaled L(pi), for reference
PAPER_S1 = {
    "freesolv": 0.1410, "esol": 0.1808, "clintox": 0.2303, "sider": 0.2345,
    "bace": 0.3036, "bbbp": 0.2866, "lipophilicity": 0.3027, "tox21": 0.2224,
    "qm8": 0.2918, "hiv": 0.3071, "muv": 0.3143,
}
SMILES_COL = {"bace": "mol"}
# small -> large so partial results accrue
DATASETS = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
            "lipophilicity", "tox21", "qm8", "hiv", "muv"]

DATASAIL_TIMEOUT = 600   # seconds per dataset for the DataSAIL call
EVAL_TIMEOUT = 600       # seconds for an eval_split call


class _Timeout:
    def __init__(self, sec): self.sec = sec
    def __enter__(self): signal.signal(signal.SIGALRM, self._h); signal.alarm(self.sec)
    def __exit__(self, *a): signal.alarm(0)
    def _h(self, *a): raise TimeoutError(f"exceeded {self.sec}s")


def morgan_matrix(smiles, n_bits=2048, radius=2):
    """Dense int8 Morgan matrix (memory-light vs the DataFrame path)."""
    X = np.zeros((len(smiles), n_bits), dtype=np.int8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def leakage(data, split):
    with _Timeout(EVAL_TIMEOUT):
        ratio, _, _ = eval_split("M", data, None, "ecfp", None, None, split)
    return ratio


def run_dataset(ds):
    col = SMILES_COL.get(ds, "smiles")
    df = pd.read_csv(os.path.join(DATA, f"{ds}.csv"))
    df = df.dropna(subset=[col]).drop_duplicates(subset=col).reset_index(drop=True)
    smiles = [s for s in df[col].astype(str) if s and s != "nan"]
    n = len(smiles)
    data = {s: s for s in smiles}
    row = {"dataset": ds, "n": n, "paper_datasail_s1": PAPER_S1.get(ds)}

    # ---- hypergraph (GPU kNN + Mt-KaHyPar) ----
    X = morgan_matrix(smiles)
    feature_data = {smiles[i]: X[i] for i in range(n)}
    t0 = time.time()
    hg = run_hypergraph_split(feature_data, [8, 2], ["train", "test"], k=15, preset="quality")
    row["hg_time_s"] = round(time.time() - t0, 2)
    try:
        row["hg_leakage"] = round(leakage(data, hg), 4)
    except Exception as e:
        row["hg_leakage"] = f"N/A ({type(e).__name__})"

    # ---- DataSAIL C1e ----
    t0 = time.time()
    try:
        with _Timeout(DATASAIL_TIMEOUT):
            e_s, _, _ = datasail(techniques=["C1e"], splits=[8, 2], names=["train", "test"],
                                 e_type="M", e_data=data, max_sec=DATASAIL_TIMEOUT // 2)
        ds_split = e_s["C1e"][0]
        row["ds_time_s"] = round(time.time() - t0, 2)
        row["ds_leakage"] = round(leakage(data, ds_split), 4)
    except Exception as e:
        row["ds_time_s"] = round(time.time() - t0, 2)
        row["ds_leakage"] = f"FAILED ({type(e).__name__})"

    # ---- random baseline ----
    random.seed(42)
    ids = list(data); random.shuffle(ids)
    cut = int(0.8 * n)
    rand = {**{i: "train" for i in ids[:cut]}, **{i: "test" for i in ids[cut:]}}
    try:
        row["random_leakage"] = round(leakage(data, rand), 4)
    except Exception as e:
        row["random_leakage"] = f"N/A ({type(e).__name__})"

    return row


def main():
    cols = ["dataset", "n", "hg_leakage", "ds_leakage", "random_leakage",
            "paper_datasail_s1", "hg_time_s", "ds_time_s"]
    with open(OUT, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=cols).writeheader()
    for ds in DATASETS:
        print(f"[{ds}] ...", flush=True)
        try:
            row = run_dataset(ds)
        except Exception as e:
            row = {"dataset": ds, "hg_leakage": f"ERROR ({type(e).__name__}: {str(e)[:50]})"}
        with open(OUT, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore").writerow(row)
        print(f"[{ds}] {row}", flush=True)
    print("\nDONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
