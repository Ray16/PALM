"""Step 2 experiment: Nyström fidelity — landmark strategy × rank.

For each dataset and landmark strategy (kmeans++ / uniform / leverage), sweep the
rank and measure:
  - reconstruction error  ‖S − BBᵀ‖_F / ‖S‖_F  on a held-out row subsample
    (how well the factor space approximates the true kernel), and
  - the resulting split's true leakage (scaled_lpi) + runtime.

The fidelity bound says a tighter ‖S − BBᵀ‖ tightens how faithfully the optimizer's
objective tracks true leakage; this shows which strategy buys the most fidelity (and
lowest leakage) per unit rank.

    CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.nystrom_fidelity \
        --datasets moleculenet_bace moleculenet_esol qmof --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PALM.splitters import SplitSpec, split
from PALM.splitters.common.feature_preparation import choose_metric
from PALM.splitters.common.pairwise_similarity import pairwise_similarity
from PALM.data.sources import load_dataset
from PALM.lowrank import nystrom_features

HERE = os.path.dirname(__file__)
OUT_CSV = os.path.join(HERE, "nystrom_fidelity.csv")
LANDMARKS = ["kmeans++", "uniform", "leverage"]
RANKS = [32, 64, 128, 256]


def _recon_error(X, B, metric, sub=600, seed=0):
    """Relative Frobenius reconstruction error of S ~= BB^T on a row subsample."""
    import torch
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(sub, n), replace=False)
    Xt = torch.as_tensor(X[idx], dtype=torch.float32)
    S = pairwise_similarity(Xt, Xt, metric)
    Bt = torch.as_tensor(B[idx], dtype=torch.float32)
    approx = Bt @ Bt.T
    return float(torch.linalg.norm(S - approx) / torch.linalg.norm(S))


def run(datasets, seeds, limit, route):
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data:
            continue
        ids = list(b.feature_data)
        X = np.stack([b.feature_data[i] for i in ids])
        metric = choose_metric(X)
        data = b.feature_data
        print(f"[{name}] n={len(ids)} metric={metric} feature_set={b.meta.get('feature_set','default')}")
        for lm in LANDMARKS:
            for rank in RANKS:
                for seed in seeds:
                    B, _ = nystrom_features(X, rank=rank, metric=metric, landmark=lm, seed=seed)
                    err = _recon_error(X, B, metric, seed=seed)
                    res = split("lowrank", data,
                                SplitSpec([8, 2], ["train", "test"], seed=seed),
                                rank=rank, landmark=lm)
                    rows.append(dict(dataset=name, landmark=lm, rank=rank, seed=seed,
                                     recon_error=round(err, 4),
                                     leakage=res.diagnostics.get("leakage"),
                                     runtime_s=res.diagnostics.get("runtime_s")))
                last = rows[-1]
                print(f"    {lm:9s} rank={rank:4d}  recon_err={last['recon_error']}  leakage={last['leakage']}")
    return rows


def plot(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in ("recon_error", "leakage"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby(["dataset", "landmark", "rank"]).agg(
        recon_error=("recon_error", "mean"), leakage=("leakage", "mean")).reset_index()
    for metric, ylab, fname in [("recon_error", "Nyström reconstruction error ‖S−BBᵀ‖/‖S‖", "nystrom_fidelity_recon.png"),
                                ("leakage", r"split leakage $L(\pi)$", "nystrom_fidelity_leakage.png")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for (ds, lm), g in agg.groupby(["dataset", "landmark"]):
            g = g.sort_values("rank")
            ax.plot(g["rank"], g[metric], "o-", ms=4, label=f"{ds} / {lm}")
        ax.set_xscale("log", base=2); ax.set_xlabel("rank", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12); ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3, which="both"); fig.tight_layout()
        fig.savefig(os.path.join(HERE, fname), dpi=300); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["moleculenet_bace", "moleculenet_esol", "qmof"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--route", action="store_true", default=True)
    args = ap.parse_args(argv)
    rows = run(args.datasets, args.seeds, args.limit, args.route)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows)
    print(f"\n== {len(rows)} runs -> {OUT_CSV}")


if __name__ == "__main__":
    main()
