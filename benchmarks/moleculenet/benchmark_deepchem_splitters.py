"""DeepChem's standard splitters (the set benchmarked in the DataSAIL paper) on
MoleculeNet. Adds the three we were missing -- DC-Fingerprint, DC-MaxMin,
DC-Weight -- plus DeepChem's own Scaffold & Butina for direct comparability.

Two stages, because DeepChem lives in the `datasail` env but our scaled_lpi
scorer needs torch (only in `palm`):
  1. (datasail env) generate splits -> results/deepchem_splits.json
       python -m PALM.benchmarks.moleculenet.benchmark_deepchem_splitters
  2. (palm env) score with the SAME ECFP/Tanimoto scaled_lpi as every other
     method -> results/deepchem_splitters.csv
       python -m PALM.benchmarks.moleculenet.benchmark_deepchem_splitters --score
"""
import os
import sys
import csv
import json
import time

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from PALM.benchmarks.common.datasets import load_smiles
from PALM.benchmarks.common.featurize import ecfp1024

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "deepchem_splitters.csv")
SPLITS_JSON = os.path.join(HERE, "..", "results", "deepchem_splits.json")
DATASETS = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
            "lipophilicity", "tox21", "qm8", "hiv", "muv"]
# MaxMin is O(n^2) diversity picking -- skip above this to stay tractable
MAXMIN_MAX_N = 30_000


def sanitize_ids(smiles):
    """Zero-fill for DeepChem: DeepChem's splitters recompute features from SMILES
    internally and crash on unparseable ones (0.02-0.54% of these datasets). Swap
    each invalid SMILES for a placeholder so DeepChem runs; scoring still uses the
    ORIGINAL SMILES via load_smiles (scaled_lpi zero-fills the invalid ones to a
    zero ECFP vector, i.e. 0 similarity to everything), so their placement doesn't
    affect L(pi) -- consistent with how low-rank handles them. Returns (ids, n_bad).
    """
    ids, bad = [], 0
    for s in smiles:
        if Chem.MolFromSmiles(str(s)) is None:
            ids.append("C")   # placeholder; scored as a zero vector regardless
            bad += 1
        else:
            ids.append(str(s))
    return np.array(ids), bad


def generate_splits():
    """Stage 1 (datasail env): produce 0/1 test-labels per dataset x splitter."""
    import deepchem as dc
    from deepchem.splits import (FingerprintSplitter, MaxMinSplitter,
                                 MolecularWeightSplitter, ScaffoldSplitter,
                                 ButinaSplitter)
    splitters = [
        ("dc_fingerprint", FingerprintSplitter),
        ("dc_maxmin", MaxMinSplitter),
        ("dc_weight", MolecularWeightSplitter),
        ("dc_scaffold", ScaffoldSplitter),
        ("dc_butina", ButinaSplitter),
    ]
    # optional dataset subset from argv (non-flag args)
    subset = [a for a in sys.argv[1:] if not a.startswith("-")]
    datasets = subset or DATASETS
    # preserve already-computed datasets so a subset re-run doesn't wipe them
    out = json.load(open(SPLITS_JSON)) if os.path.exists(SPLITS_JSON) else {}
    for ds in datasets:
        smiles = load_smiles(ds)
        n = len(smiles)
        X = ecfp1024(smiles)
        ids, n_bad = sanitize_ids(smiles)   # zero-fill invalid SMILES for DeepChem
        dset = dc.data.NumpyDataset(X=X, ids=ids)
        out[ds] = {"n": n, "n_invalid_smiles": n_bad, "methods": {}}
        print(f"\n=== {ds}  n={n}  (zero-filled {n_bad} invalid SMILES) ===",
              flush=True)
        for name, Splitter in splitters:
            if name == "dc_maxmin" and n > MAXMIN_MAX_N:
                print(f"  {name:15s} skipped (n>{MAXMIN_MAX_N}, O(n^2))", flush=True)
                out[ds]["methods"][name] = {"status": "skipped_n"}
                continue
            try:
                t0 = time.time()
                tr, _, te = Splitter().split(dset, frac_train=0.8,
                                             frac_valid=0.0, frac_test=0.2)
                dt = time.time() - t0
                lab = np.zeros(n, dtype=int)
                lab[np.asarray(te, dtype=int)] = 1
                print(f"  {name:15s} test_frac={lab.mean():.3f}  {dt:.1f}s",
                      flush=True)
                out[ds]["methods"][name] = {"status": "ok", "time_s": round(dt, 2),
                                            "labels": lab.tolist()}
            except Exception as e:
                print(f"  {name:15s} FAILED: {type(e).__name__}: {e}", flush=True)
                out[ds]["methods"][name] = {"status": f"error:{type(e).__name__}"}
            json.dump(out, open(SPLITS_JSON, "w"))
    print("\nsaved", SPLITS_JSON, flush=True)


def score_splits():
    """Stage 2 (palm env): score the saved splits with scaled_lpi (ECFP)."""
    from PALM.benchmarks.moleculenet.leakage import scaled_lpi
    data = json.load(open(SPLITS_JSON))
    rows = []
    for ds, d in data.items():
        smiles = load_smiles(ds)
        n = d["n"]
        assert len(smiles) == n, f"{ds}: smiles order changed"
        for name, m in d["methods"].items():
            if m["status"] != "ok":
                rows.append({"dataset": ds, "n": n, "method": name, "lpi": "",
                             "test_frac": "", "time_s": "", "status": m["status"]})
                continue
            lab = np.asarray(m["labels"])
            assign = {smiles[i]: ("test" if lab[i] else "train") for i in range(n)}
            lpi = scaled_lpi(list(smiles), assign)[0]
            rows.append({"dataset": ds, "n": n, "method": name,
                         "lpi": round(float(lpi), 4),
                         "test_frac": round(float(lab.mean()), 4),
                         "time_s": m.get("time_s", ""), "status": "ok"})
            print(f"{ds:14s} {name:15s} L(pi)={lpi:.4f} test={lab.mean():.3f}",
                  flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "n", "method", "lpi",
                                           "test_frac", "time_s", "status"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved", OUT, flush=True)


if __name__ == "__main__":
    if "--score" in sys.argv:
        score_splits()
    else:
        generate_splits()
