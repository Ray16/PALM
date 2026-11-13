"""Step 2 experiment: Nyström fidelity — landmark strategy × rank.

For each dataset and landmark strategy (kmeans++ / uniform / leverage), sweep the
rank and measure:
  - reconstruction error  ‖S − BBᵀ‖_F / ‖S‖_F  on a held-out row subsample
    (how well the factor space approximates the true kernel), and
  - the resulting split's true leakage (scaled_lpi) + runtime.

The fidelity bound says a tighter ‖S − BBᵀ‖ tightens how faithfully the optimizer's
objective tracks true leakage; this shows which strategy buys the most fidelity (and
lowest leakage) per unit rank.

ARCHIVED — this is a documented negative result (archive/README.md and
docs/FINDINGS.md Step 2): the tight/adaptive Nyström variants (leverage landmarks,
ridge, adaptive energy-rank) give negligible leakage benefit over plain k-means++,
and were removed from the live module. It imports the pre-strip snapshot
`nystrom_full` (kept alongside) so the variant sweep still reproduces.
Run standalone:
    python archive/nystrom_variants/nystrom_fidelity.py \
        --datasets moleculenet_bace moleculenet_esol qmof --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))      # low_rank/ (for splitters/datasets)
sys.path.insert(0, str(_HERE))                 # local `nystrom_full` snapshot

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time

from splitters.common.feature_preparation import choose_metric
from splitters.common.pairwise_similarity import pairwise_similarity
from datasets.sources import load_dataset
from nystrom_full import nystrom_features
# The assignment (balanced-Lloyd + FM) is unchanged by the archived variants, so we
# reuse the live optimizer directly on the variant factor B rather than the live
# `split("lowrank", ...)` — whose param surface no longer accepts leverage/ridge/energy.
from lowrank import balanced_lloyd, fm_polish, factor_leakage

_RESULTS = _HERE / "results"
_FIGURES = _HERE / "figures"
_RESULTS.mkdir(parents=True, exist_ok=True)
_FIGURES.mkdir(parents=True, exist_ok=True)


def result_path(name: str) -> pathlib.Path:
    return _RESULTS / name


def figure_path(name: str) -> pathlib.Path:
    return _FIGURES / name


OUT_CSV = result_path("nystrom_fidelity.csv")
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
                    t0 = time.time()
                    B, _ = nystrom_features(X, rank=rank, metric=metric, landmark=lm, seed=seed)
                    err = _recon_error(X, B, metric, seed=seed)
                    labels = balanced_lloyd(B, [8, 2], n_iter=25, seed=seed)
                    labels, _ = fm_polish(B, labels, [8, 2], epsilon=0.0)
                    leakage = float(factor_leakage(B, labels, 2))
                    rows.append(dict(dataset=name, landmark=lm, rank=rank, seed=seed,
                                     recon_error=round(err, 4),
                                     leakage=round(leakage, 4),
                                     runtime_s=round(time.time() - t0, 3)))
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
        fig.savefig(figure_path(fname), dpi=300); plt.close(fig)


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
