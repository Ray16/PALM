"""Does Step 1 (balance_slack) raise the gap the way Step 3 (hardness) does?

Sweeps ``balance_slack`` and measures leakage, realized imbalance, AND the
generalization gap (via the benchmark's ``evaluate_gap``). Then overlays the
Step-1 (balance) and Step-3 (hardness) points in leakage→gap space per dataset:
if they lie on the *same* curve, leakage is the single causal channel and the two
knobs are one capability; if the balance points sit above (more gap at matched
leakage), the size imbalance adds difficulty of its own.

    CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.balance_gap \
        --datasets moleculenet_esol moleculenet_bace moleculenet_freesolv --seeds 0 1 2
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
from PALM.benchmarks.master.model_eval import evaluate_gap
from PALM.lowrank import realized_imbalance

HERE = os.path.dirname(__file__)
OUT_CSV = os.path.join(HERE, "balance_gap.csv")
OUT_FIG = os.path.join(HERE, "balance_vs_hardness_gap.png")
HARDNESS_CSV = os.path.join(HERE, "hardness_control.csv")
SLACKS = [0.0, 0.05, 0.10, 0.20, 0.30]


def run(datasets, seeds, limit, route):
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data or not b.targets or not b.task_type:
            print(f"[{name}] skip (no target/features)")
            continue
        ids = list(b.feature_data)
        X = np.stack([b.feature_data[i] for i in ids])
        y = np.array([b.targets.get(i, np.nan) for i in ids], dtype=float)
        print(f"[{name}] n={len(ids)} task={b.task_type} feature_set={b.meta.get('feature_set','default')}")
        for slack in SLACKS:
            for seed in seeds:
                res = split("lowrank", b.feature_data,
                            SplitSpec([8, 2], ["train", "test"], seed=seed), balance_slack=slack)
                tr = [j for j, i in enumerate(ids) if res.assignment[i] == "train"]
                te = [j for j, i in enumerate(ids) if res.assignment[i] == "test"]
                lab = np.array([0 if res.assignment[i] == "train" else 1 for i in ids])
                g = evaluate_gap(X, y, b.task_type, tr, te, seed=seed)
                rows.append(dict(dataset=name, balance_slack=slack, seed=seed,
                                 leakage=res.diagnostics.get("leakage"),
                                 imbalance=round(realized_imbalance(lab, [8, 2]), 4),
                                 gen_gap=g.get("gen_gap")))
            last = rows[-1]
            print(f"    slack={slack:.2f}  leakage={last['leakage']}  imbalance={last['imbalance']}  gap={last['gen_gap']}")
    return rows


def analyze_and_plot(rows):
    import pandas as pd
    b1 = pd.DataFrame(rows)
    for c in ("leakage", "gen_gap"):
        b1[c] = pd.to_numeric(b1[c], errors="coerce")
    a1 = b1.groupby(["dataset", "balance_slack"]).agg(
        leakage=("leakage", "mean"), gen_gap=("gen_gap", "mean")).reset_index()

    have_h = os.path.exists(HARDNESS_CSV)
    if have_h:
        h = pd.read_csv(HARDNESS_CSV)
        for c in ("leakage", "gen_gap"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
        ah = h.groupby(["dataset", "hardness"]).agg(
            leakage=("leakage", "mean"), gen_gap=("gen_gap", "mean")).reset_index()

    datasets = sorted(a1["dataset"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, ds in enumerate(datasets):
        c = cmap(i)
        g1 = a1[a1.dataset == ds].sort_values("leakage")
        ax.plot(g1["leakage"], g1["gen_gap"], "s--", color=c, ms=6,
                label=f"{ds} · balance_slack")
        if have_h:
            gh = ah[ah.dataset == ds].sort_values("leakage")
            ax.plot(gh["leakage"], gh["gen_gap"], "o-", color=c, ms=5, alpha=0.7,
                    label=f"{ds} · hardness")
    ax.set_xlabel(r"leakage $L(\pi)$", fontsize=12)
    ax.set_ylabel("realized generalization gap", fontsize=12)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=300); plt.close(fig)

    # numeric test: at matched leakage, is the balance-gap ≈ hardness-gap?
    print("\n== leakage→gap: balance_slack sweep vs hardness sweep ==")
    for ds in datasets:
        g1 = a1[a1.dataset == ds].sort_values("balance_slack")
        print(f"  {ds}:")
        print("    balance_slack: " + "  ".join(
            f"L={r.leakage:.3f}/gap={r.gen_gap:.3f}" for r in g1.itertuples()))
        if have_h:
            gh = ah[ah.dataset == ds].sort_values("hardness")
            print("    hardness:      " + "  ".join(
                f"L={r.leakage:.3f}/gap={r.gen_gap:.3f}" for r in gh.itertuples()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["moleculenet_esol", "moleculenet_bace", "moleculenet_freesolv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--route", action="store_true", default=True)
    args = ap.parse_args(argv)
    rows = run(args.datasets, args.seeds, args.limit, args.route)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    analyze_and_plot(rows)
    print(f"\n== {len(rows)} runs -> {OUT_CSV}\n== figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
