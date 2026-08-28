"""Direction #3 experiment: multi-way (train/val/test) splits.

Two questions, mirroring Steps 1 and 3 but for a 3-way [8,1,1] split:
  (2a) leakage <-> balance: does opening ``balance_slack`` lower 3-way leakage while
       keeping every block inside the corridor (needs the new k>2 corridor_assign)?
  (2b) hardness -> gap: does the ``hardness`` dial still control the realized
       train->test generalization gap on a 3-way split (Spearman(alpha, gap) > 0)?

One single-panel PNG per dataset per question (no titles/subplots).

    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    python -m PALM.lowrank.experiments.kway_split --seeds 0 1 2
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
OUT_BAL = os.path.join(HERE, "kway_balance.csv")
OUT_HARD = os.path.join(HERE, "kway_hardness.csv")

SPLITS = [8, 1, 1]
NAMES = ["train", "val", "test"]
SLACKS = [0.0, 0.05, 0.10, 0.20, 0.30]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------- #
# 2a: leakage <-> balance on a 3-way split
# --------------------------------------------------------------------------- #
def run_balance(datasets, seeds, limit, route):
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data:
            print(f"[{name}] skip")
            continue
        data = b.feature_data
        ids = list(data)
        n = len(ids)
        print(f"[balance {name}] n={n} feat={b.meta.get('feature_set','default')}")
        for slack in SLACKS:
            for seed in seeds:
                res = split("lowrank", data, SplitSpec(SPLITS, NAMES, seed=seed),
                            balance_slack=slack)
                lab = np.array([NAMES.index(res.assignment[i]) for i in ids])
                rows.append(dict(dataset=name, n=n, balance_slack=slack, seed=seed,
                                 leakage=res.diagnostics.get("leakage"),
                                 imbalance=round(realized_imbalance(lab, SPLITS), 4),
                                 f_train=round(float((lab == 0).mean()), 4),
                                 f_val=round(float((lab == 1).mean()), 4),
                                 f_test=round(float((lab == 2).mean()), 4)))
            last = rows[-1]
            print(f"    slack={slack:.2f} leakage={last['leakage']} "
                  f"imb={last['imbalance']} fracs=({last['f_train']},{last['f_val']},{last['f_test']})")
    return rows


# --------------------------------------------------------------------------- #
# 2b: hardness -> gap on a 3-way split (train vs test blocks)
# --------------------------------------------------------------------------- #
def run_hardness(datasets, seeds, limit, route):
    from PALM.benchmarks.master.model_eval import evaluate_gap
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data or not b.targets or not b.task_type:
            print(f"[hardness {name}] skip (no target)")
            continue
        ids = list(b.feature_data)
        X = np.stack([b.feature_data[i] for i in ids])
        y = np.array([b.targets.get(i, np.nan) for i in ids], dtype=float)
        n = len(ids)
        print(f"[hardness {name}] n={n} task={b.task_type}")
        for alpha in ALPHAS:
            for seed in seeds:
                res = split("lowrank", b.feature_data, SplitSpec(SPLITS, NAMES, seed=seed),
                            hardness=alpha)
                tr = [j for j, i in enumerate(ids) if res.assignment[i] == "train"]
                te = [j for j, i in enumerate(ids) if res.assignment[i] == "test"]
                g = evaluate_gap(X, y, b.task_type, tr, te, seed=seed)
                rows.append(dict(dataset=name, task=b.task_type, hardness=alpha, seed=seed,
                                 leakage=res.diagnostics.get("leakage"),
                                 gen_gap=g.get("gen_gap"), metric=g.get("metric_name")))
            last = rows[-1]
            print(f"    hardness={alpha:.2f} leakage={last['leakage']} gap={last['gen_gap']}")
    return rows


# --------------------------------------------------------------------------- #
# plotting: one single-panel PNG per dataset
# --------------------------------------------------------------------------- #
def _agg(rows, key, val):
    import pandas as pd
    df = pd.DataFrame(rows)
    df[val] = pd.to_numeric(df[val], errors="coerce")
    return df.groupby(["dataset", key]).agg(m=(val, "mean"), s=(val, "std")).reset_index()


def plot_balance(rows):
    agg = _agg(rows, "balance_slack", "leakage")
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("balance_slack")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.errorbar(g["balance_slack"], g["m"], yerr=g["s"], fmt="o-", ms=6, capsize=3)
        ax.set_xlabel("balance_slack (0 = exact 8:1:1)", fontsize=12)
        ax.set_ylabel(r"3-way leakage $L(\pi)$ (lower is better)", fontsize=12)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(HERE, f"kway_balance_{ds}.png")
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"  -> {out}")


def plot_hardness(rows):
    agg = _agg(rows, "hardness", "gen_gap")
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("hardness")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.errorbar(g["hardness"], g["m"], yerr=g["s"], fmt="o-", ms=6, capsize=3)
        ax.set_xlabel("hardness dial α  (0 = random, 1 = leakage-minimized)", fontsize=12)
        ax.set_ylabel("realized train→test generalization gap", fontsize=12)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(HERE, f"kway_hardness_{ds}.png")
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"  -> {out}")


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}")


def _spearman_report(rows, key, val):
    import pandas as pd
    from scipy.stats import spearmanr
    df = pd.DataFrame(rows)
    df[val] = pd.to_numeric(df[val], errors="coerce")
    agg = df.groupby(["dataset", key]).agg(m=(val, "mean")).reset_index()
    print(f"\n== Spearman({key}, {val}) per dataset ==")
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values(key)
        rho = spearmanr(g[key], g["m"]).correlation
        print(f"  {ds:26s} rho={rho:+.2f}  [{val} {g['m'].min():.3f}->{g['m'].max():.3f}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bal-datasets", nargs="+",
                    default=["moleculenet_bace", "moleculenet_esol",
                             "moleculenet_freesolv", "qmof"])
    ap.add_argument("--hard-datasets", nargs="+",
                    default=["moleculenet_esol", "moleculenet_bace", "moleculenet_freesolv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--route", action="store_true", default=True)
    ap.add_argument("--skip-balance", action="store_true")
    ap.add_argument("--skip-hardness", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_balance:
        print("== 2a: 3-way leakage <-> balance_slack ==")
        bal = run_balance(args.bal_datasets, args.seeds, args.limit, args.route)
        _write(OUT_BAL, bal)
        if bal:
            plot_balance(bal)
            _spearman_report(bal, "balance_slack", "leakage")
    if not args.skip_hardness:
        print("\n== 2b: 3-way hardness -> gap ==")
        hard = run_hardness(args.hard_datasets, args.seeds, args.limit, args.route)
        _write(OUT_HARD, hard)
        if hard:
            plot_hardness(hard)
            _spearman_report(hard, "hardness", "gen_gap")


if __name__ == "__main__":
    main()
