"""Lo-Hi (Steshin, NeurIPS 2023 D&B) 'Hi' split vs low-rank, matched fairly.

Hi solves a minimum vertex k-cut MILP over an ECFP4-Tanimoto threshold graph.
Crucially it does NOT partition the dataset -- molecules it cannot cleanly
assign are DISCARDED (train_min_frac + test_min_frac < 1.0).  Scoring Hi's
L(pi) over only its survivors against low-rank's split over all n would be
badly unfair to low-rank, because the discarded molecules are exactly the
bridging ones that carry cross-boundary similarity.

Fair protocol implemented here:
  1. run Hi, record the retained set R and Hi's realized test fraction f
  2. score Hi's L(pi) on R
  3. run low-rank on R alone, at the same f, and score on R
  4. report Hi's discard rate as its cost

One dataset per invocation so the caller can impose a wall-clock timeout; the
MILP is expected to become infeasible on the larger sets, which is a result.

    python -m PALM.benchmarks.moleculenet.benchmark_lohi --dataset freesolv
"""
import argparse
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

SIM_THRESHOLD = 0.4         # lohi default
TRAIN_MIN_FRAC = 0.7        # lohi default
TEST_MIN_FRAC = 0.1         # lohi default
NYSTROM_RANK = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    ap.add_argument("--train-min", type=float, default=TRAIN_MIN_FRAC)
    ap.add_argument("--test-min", type=float, default=TEST_MIN_FRAC)
    args = ap.parse_args()

    import logging
    logging.disable(logging.CRITICAL)
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
    import lohi_splitter as lohi
    from PALM.splitters import SplitSpec, split
    from PALM.benchmarks.common.datasets import load_smiles
    from PALM.benchmarks.moleculenet.leakage import scaled_lpi

    raw = [str(s) for s in load_smiles(args.dataset)]
    n_raw = len(raw)
    # lohi feeds every SMILES straight to the fingerprint generator, so an
    # unparseable one crashes it.  Drop them first -- scaled_lpi drops the same
    # ones, so the comparison set is identical for both methods.
    smiles = [s for s in raw if Chem.MolFromSmiles(s) is not None]
    smiles = list(dict.fromkeys(smiles))          # lohi indexes positionally
    n = len(smiles)

    # -- Hi split ----------------------------------------------------------
    t0 = time.time()
    # NB: returns *indices* into `smiles`, not SMILES strings.
    train_idx, test_idx = lohi.hi_train_test_split(
        smiles=smiles, similarity_threshold=args.threshold,
        train_min_frac=args.train_min, test_min_frac=args.test_min,
        verbose=False)
    hi_time = time.time() - t0
    train_sm = [smiles[i] for i in train_idx]
    test_sm = [smiles[i] for i in test_idx]

    retained = list(train_sm) + list(test_sm)
    n_ret = len(retained)
    hi_test_frac = len(test_sm) / n_ret
    discard_frac = 1.0 - n_ret / n

    hi_split = {s: "train" for s in train_sm}
    hi_split.update({s: "test" for s in test_sm})
    hi_lpi = scaled_lpi(retained, hi_split)[0]

    # -- low-rank on the SAME retained set at the SAME ratio ---------------
    def ecfp(sm):
        a = np.zeros(1024, dtype=np.float32)
        m = Chem.MolFromSmiles(sm)
        if m is not None:
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), a)
        return a

    fd = {s: ecfp(s) for s in retained}
    w_test = round(hi_test_frac * 100)
    t0 = time.time()
    lr_split = split("lowrank", fd,
                     SplitSpec([100 - w_test, w_test], ["train", "test"], seed=0, epsilon=0.0),
                     rank=NYSTROM_RANK, n_restarts=4, fm=True).assignment
    lr_time = time.time() - t0
    lr_lpi = scaled_lpi(retained, lr_split)[0]
    lr_test_frac = sum(1 for v in lr_split.values() if v == "test") / n_ret

    row = {
        "dataset": args.dataset, "n_raw": n_raw, "n": n, "n_retained": n_ret,
        "discard_frac": round(discard_frac, 4),
        "hi_lpi": round(float(hi_lpi), 4), "hi_time_s": round(hi_time, 2),
        "hi_test_frac": round(hi_test_frac, 4),
        "lowrank_lpi": round(float(lr_lpi), 4), "lowrank_time_s": round(lr_time, 3),
        "lowrank_test_frac": round(lr_test_frac, 4),
        "threshold": args.threshold,
    }
    out = os.path.join(RESULTS, f"lohi_{args.dataset}.json")
    with open(out, "w") as fh:
        json.dump(row, fh, indent=2)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
