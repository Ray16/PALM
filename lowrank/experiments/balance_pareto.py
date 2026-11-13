"""Step 1 experiment: the leakage ↔ balance Pareto frontier.

Sweeps the low-rank splitter's ``balance_slack`` knob (0 = exact target sizes,
>0 = a (1 ± slack) size corridor the optimizer may exploit) and measures, per
dataset, the resulting leakage vs the realized size imbalance — the first
multi-objective result of the low-rank method-development track.

    CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.balance_pareto \
        --datasets moleculenet_bace moleculenet_esol qmof materials_project \
        --seeds 0 1 2 --limit 2000
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
from PALM.data.sources import load_dataset
from PALM.lowrank import realized_imbalance

HERE = os.path.dirname(__file__)
OUT_CSV = os.path.join(HERE, "balance_pareto.csv")
OUT_FIG = os.path.join(HERE, "balance_pareto.png")

SLACKS = [0.0, 0.05, 0.10, 0.20, 0.30]


def run(datasets, seeds, limit, route):
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data:
            print(f"[{name}] skip: {b.reason if not b.available else 'no features'}")
            continue
        data = b.feature_data
        ids = list(data)
        print(f"[{name}] n={len(ids)} feature_set={b.meta.get('feature_set','default')}")
        for slack in SLACKS:
            for seed in seeds:
                spec = SplitSpec(splits=[8, 2], names=["train", "test"], seed=seed)
                res = split("lowrank", data, spec, balance_slack=slack)
                lab = np.array([0 if res.assignment[i] == "train" else 1 for i in ids])
                rows.append(dict(dataset=name, n=len(ids), balance_slack=slack, seed=seed,
                                 leakage=res.diagnostics.get("leakage"),
                                 imbalance=round(realized_imbalance(lab, [8, 2]), 4),
                                 runtime_s=res.diagnostics.get("runtime_s")))
            last = rows[-1]
            print(f"    slack={slack:.2f}  leakage={last['leakage']}  imbalance={last['imbalance']}")
    return rows


def plot(rows, out=OUT_FIG):
    import pandas as pd
    df = pd.DataFrame(rows)
    df["leakage"] = pd.to_numeric(df["leakage"], errors="coerce")
    agg = df.groupby(["dataset", "balance_slack"]).agg(
        leakage=("leakage", "mean"), imbalance=("imbalance", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("imbalance")
        ax.plot(g["imbalance"], g["leakage"], "o-", label=ds, ms=5)
        for _, r in g.iterrows():
            ax.annotate(f"{r['balance_slack']:.2f}", (r["imbalance"], r["leakage"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("realized size imbalance (max relative deviation)", fontsize=12)
    ax.set_ylabel(r"leakage $L(\pi)$ (lower is better)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["moleculenet_bace", "moleculenet_esol", "qmof", "materials_project"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--route", action="store_true", default=True)
    args = ap.parse_args(argv)
    rows = run(args.datasets, args.seeds, args.limit, args.route)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows)
    print(f"\n== {len(rows)} runs -> {OUT_CSV}\n== figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
