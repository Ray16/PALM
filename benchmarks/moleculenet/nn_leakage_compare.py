"""Nearest-neighbor (real) data-leakage audit: low-rank vs hypergraph vs DataSAIL.

`scaled_lpi` measures aggregate cross-split similarity; this instead measures the
concrete leakage that actually inflates ML scores: for every TEST molecule, its
maximum Tanimoto similarity to any TRAIN molecule (its nearest training
neighbor). A good split pushes those neighbor similarities down and leaves no
near-duplicate / exact-duplicate straddling the split.

Reported per split, over the same 1024-bit ECFP the metric uses:
  lpi      : scaled_lpi (aggregate leakage, for reference)
  NN_mean  : mean over test of max Tanimoto to train (lower = less leakage)
  NN_med   : median of the same
  %>=0.9   : fraction of test with a near-duplicate (sim>=0.9) in train
  %>=0.99  : fraction with a ~exact duplicate in train
  #dup     : count of test with an exact duplicate (sim>=0.999) in train

Run (palm env):  python -m PALM.benchmarks.moleculenet.nn_leakage_compare
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
from PALM.splitters import SplitSpec, split
from PALM.benchmarks.common.datasail import datasail_fingerprint
from PALM.benchmarks.common.datasets import load_smiles
from PALM.benchmarks.common.featurize import ecfp1024
from PALM.benchmarks.moleculenet.leakage import scaled_lpi

DATASETS = ["esol", "bace", "tox21"]
DATASAIL_MAX_SEC = 400


def nn_leakage(X, labels):
    """Per-test max Tanimoto to any train row, summarized. labels: 0=train,1=test."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, device=dev)
    card = Xt.sum(1)
    lab = torch.as_tensor(np.asarray(labels), device=dev)
    tr, te = Xt[lab == 0], Xt[lab == 1]
    ctr, cte = card[lab == 0], card[lab == 1]
    nn = torch.empty(te.shape[0], device=dev)
    for s in range(0, te.shape[0], 4096):          # chunk test rows to bound memory
        e = min(s + 4096, te.shape[0])
        inter = te[s:e] @ tr.T
        union = cte[s:e, None] + ctr[None, :] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        nn[s:e] = sim.max(1).values
    return dict(mean=float(nn.mean()), median=float(nn.median()),
                p90=float((nn >= 0.9).float().mean()),
                p99=float((nn >= 0.99).float().mean()),
                dup=int((nn >= 0.999).sum()))


def datasail_split(smiles):
    """DataSAIL C1e split as {smiles: 'train'|'test'}; None if it fails/times out."""
    try:
        return datasail_fingerprint({s: s for s in smiles}, max_sec=DATASAIL_MAX_SEC)
    except Exception as exc:
        print(f"    DataSAIL failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def main():
    print(f"{'dataset':<9}{'method':<11}{'lpi':>7}{'NN_mean':>9}{'NN_med':>8}"
          f"{'%>=.9':>7}{'%>=.99':>8}{'#dup':>6}")
    print("-" * 65)
    for ds in DATASETS:
        smiles = load_smiles(ds)
        n = len(smiles)
        X = ecfp1024(smiles)
        fd = {smiles[i]: X[i] for i in range(n)}

        # build the four splits
        rng = np.random.default_rng(0)
        rand = np.array([0] * int(0.8 * n) + [1] * (n - int(0.8 * n)))
        rng.shuffle(rand)
        hg = split("hypergraph", fd, SplitSpec([8, 2], ["train", "test"], seed=0),
                   k=15, preset="quality").assignment
        lr = split("lowrank", fd, SplitSpec([8, 2], ["train", "test"], seed=0),
                   rank=256).assignment
        ds_split = datasail_split(smiles)

        splits = {
            "random": rand,
            "hypergraph": np.array([0 if hg[smiles[i]] == "train" else 1 for i in range(n)]),
            "lowrank": np.array([0 if lr[smiles[i]] == "train" else 1 for i in range(n)]),
        }
        if ds_split is not None:
            splits["DataSAIL"] = np.array(
                [0 if ds_split.get(smiles[i], "train") == "train" else 1 for i in range(n)])

        for name, lab in splits.items():
            L = scaled_lpi(smiles, {smiles[i]: ("train" if lab[i] == 0 else "test")
                                    for i in range(n)})[0]
            m = nn_leakage(X, lab)
            print(f"{ds:<9}{name:<11}{L:>7.3f}{m['mean']:>9.3f}{m['median']:>8.3f}"
                  f"{m['p90']*100:>6.1f}%{m['p99']*100:>7.1f}%{m['dup']:>6}", flush=True)
        print()


if __name__ == "__main__":
    main()
