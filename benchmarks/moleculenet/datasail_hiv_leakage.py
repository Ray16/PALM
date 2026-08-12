"""Compute DataSAIL's leakage on hiv (n=41,127). The benchmark hard-coded hiv as
a timeout, but DataSAIL does complete it (~46 min); this scores that split with
the SAME ECFP scaled_lpi the benchmark uses, so the poster figure is consistent
(DataSAIL bar for hiv, not a spurious n/a). Writes results/datasail_extra.json."""
import os, sys, json, time
sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging; logging.disable(logging.CRITICAL)
import numpy as np
from PALM.benchmarks.common.datasail import datasail_fingerprint
from PALM.benchmarks.common.datasets import load_smiles
from PALM.benchmarks.moleculenet.leakage import scaled_lpi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "datasail_extra.json")


def datasail_split(smiles, max_sec):
    return datasail_fingerprint({s: s for s in smiles}, max_sec=max_sec)


def main():
    ds = "hiv"
    smiles = load_smiles(ds)
    print(f"{ds}: {len(smiles)} molecules; running DataSAIL C1e ...", flush=True)
    t0 = time.time()
    split = datasail_split(smiles, max_sec=6000)
    dt = time.time() - t0
    assign = {s: split.get(s, "train") for s in smiles}
    lpi = scaled_lpi(smiles, assign)[0]
    test_frac = float(np.mean([assign[s] == "test" for s in smiles]))
    print(f"{ds}: DataSAIL scaled_lpi={lpi:.4f}  test_frac={test_frac:.3f}  "
          f"time={dt:.0f}s", flush=True)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev[ds] = {"datasail_lpi": round(float(lpi), 4), "test_frac": round(test_frac, 4),
                "time_s": round(dt, 1)}
    json.dump(prev, open(OUT, "w"), indent=2)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
