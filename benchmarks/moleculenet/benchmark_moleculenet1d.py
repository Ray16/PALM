"""Benchmark the PALM hypergraph backend against DataSAIL on the prepared
1D molecule datasets (data/DataSAIL_data/1D/moleculenet).

Single source of truth for the numbers in ``moleculenet1d_results.csv`` and the chart.
For each dataset, on the SAME molecules and the SAME leakage metric, we report:
  - hypergraph (GPU kNN + Mt-KaHyPar)   scaled L(pi) + wall-clock
  - DataSAIL C1e (cluster-based 1D)      scaled L(pi) + wall-clock (None on timeout)
  - random 80/20                         scaled L(pi) (baseline)
  - paper "DataSAIL S1"                  reference value from the addendum

Leakage is scored with ``leakage.scaled_lpi`` — a GPU, chunked reimplementation
of DataSAIL's ``eval_split`` scaled L(pi) (ECFP/Tanimoto). It is numerically
equal to ``eval_split`` (see ``--validate``) but, unlike eval_split (which builds
the full O(n^2) matrix on CPU and OOMs / raises AttributeError past ~20-40k),
it scales to 100k+ entities. That is why every method can be scored on every
dataset, including HIV (41k) and MUV (93k) where eval_split cannot.

Run (from the PALM parent dir, palm env):
    python -m PALM.benchmarks.moleculenet.benchmark_moleculenet1d            # full run -> moleculenet1d_results.csv
    python -m PALM.benchmarks.moleculenet.benchmark_moleculenet1d --validate # check scaled_lpi == eval_split
Then ``python -m PALM.benchmarks.charts.make_chart`` plots moleculenet1d_results.csv.
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

from PALM.splitters import SplitSpec, split
from PALM.benchmarks.common.datasail import datasail_fingerprint
from PALM.benchmarks.common.datasets import DATA, SMILES_COL, load_smiles  # noqa: F401
from PALM.benchmarks.common.featurize import morgan_matrix
from .leakage import scaled_lpi, validate_against_eval_split

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results", "moleculenet1d_results.csv")

# addendum Table 1 "DataSAIL S1" scaled L(pi), for reference
PAPER_S1 = {
    "freesolv": 0.1410, "esol": 0.1808, "clintox": 0.2303, "sider": 0.2345,
    "bace": 0.3036, "bbbp": 0.2866, "lipophilicity": 0.3027, "tox21": 0.2224,
    "qm8": 0.2918, "hiv": 0.3071, "muv": 0.3143,
}
# small -> large so partial results accrue
DATASETS = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
            "lipophilicity", "tox21", "qm8", "hiv", "muv"]

DATASAIL_TIMEOUT = 600   # seconds per dataset for the DataSAIL call

COLS = ["dataset", "n", "hypergraph", "hg_time", "datasail", "ds_time",
        "random", "paper_s1"]


class _Timeout:
    def __init__(self, sec): self.sec = sec
    def __enter__(self): signal.signal(signal.SIGALRM, self._h); signal.alarm(self.sec)
    def __exit__(self, *a): signal.alarm(0)
    def _h(self, *a): raise TimeoutError(f"exceeded {self.sec}s")


def _score(smiles, split):
    """scaled L(pi); None if it cannot be computed."""
    try:
        return round(scaled_lpi(list(smiles), split)[0], 4)
    except Exception:
        return None


def run_dataset(ds):
    smiles = load_smiles(ds)
    n = len(smiles)
    data = {s: s for s in smiles}
    row = {"dataset": ds, "n": n, "paper_s1": PAPER_S1.get(ds)}

    # ---- hypergraph (GPU kNN + Mt-KaHyPar) ----
    X = morgan_matrix(smiles)
    feature_data = {smiles[i]: X[i] for i in range(n)}
    t0 = time.time()
    hg = split("hypergraph", feature_data, SplitSpec([8, 2], ["train", "test"], seed=42),
               k=15, preset="quality").assignment
    row["hg_time"] = round(time.time() - t0, 2)
    row["hypergraph"] = _score(smiles, hg)

    # ---- DataSAIL C1e (None if it times out) ----
    t0 = time.time()
    try:
        with _Timeout(DATASAIL_TIMEOUT):
            ds_split = datasail_fingerprint(data, [8, 2], ["train", "test"],
                                            max_sec=DATASAIL_TIMEOUT // 2)
        row["ds_time"] = round(time.time() - t0, 2)
        row["datasail"] = _score(smiles, ds_split)
    except Exception:
        row["ds_time"] = None
        row["datasail"] = None

    # ---- random baseline ----
    random.seed(42)
    ids = list(data); random.shuffle(ids)
    cut = int(0.8 * n)
    rand = {**{i: "train" for i in ids[:cut]}, **{i: "test" for i in ids[cut:]}}
    row["random"] = _score(smiles, rand)

    return row


def validate():
    """Confirm scaled_lpi == eval_split on the smallest datasets."""
    for ds in ["freesolv", "esol"]:
        smiles = load_smiles(ds)
        random.seed(0); ids = list(smiles); random.shuffle(ids)
        cut = int(0.8 * len(ids))
        split = {**{i: "train" for i in ids[:cut]}, **{i: "test" for i in ids[cut:]}}
        ours, theirs, diff, ok = validate_against_eval_split(smiles, split)
        print(f"[{ds}] scaled_lpi={ours:.5f} eval_split={theirs:.5f} "
              f"diff={diff:.2e} {'OK' if ok else 'MISMATCH'}", flush=True)


def main():
    if "--validate" in sys.argv:
        validate()
        return
    with open(OUT, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=COLS).writeheader()
    for ds in DATASETS:
        print(f"[{ds}] ...", flush=True)
        try:
            row = run_dataset(ds)
        except Exception as e:
            row = {"dataset": ds, "hypergraph": f"ERROR ({type(e).__name__}: {str(e)[:50]})"}
        with open(OUT, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore").writerow(row)
        print(f"[{ds}] {row}", flush=True)
    print("\nDONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
