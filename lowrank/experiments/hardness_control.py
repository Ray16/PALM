"""Step 3 experiment: does the hardness dial control the realized OOD gap?

For each dataset with a target, sweep the low-rank splitter's ``hardness`` knob
(1 = hardest / leakage-minimized, 0 = random) and measure both the split leakage
and the *realized generalization gap* of a fixed RandomForest (train-minus-test
score, via the benchmark's ``evaluate_gap``). Validates that the gap rises
monotonically as hardness rises, and fits a per-dataset calibration gap = f(alpha)
so a target difficulty can be requested.

    CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.hardness_control \
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

HERE = os.path.dirname(__file__)
OUT_CSV = os.path.join(HERE, "hardness_control.csv")
OUT_FIG = os.path.join(HERE, "hardness_control.png")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


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
        for alpha in ALPHAS:
            for seed in seeds:
                res = split("lowrank", b.feature_data,
                            SplitSpec([8, 2], ["train", "test"], seed=seed), hardness=alpha)
                tr = [j for j, i in enumerate(ids) if res.assignment[i] == "train"]
                te = [j for j, i in enumerate(ids) if res.assignment[i] == "test"]
                g = evaluate_gap(X, y, b.task_type, tr, te, seed=seed)
                rows.append(dict(dataset=name, task=b.task_type, hardness=alpha, seed=seed,
                                 leakage=res.diagnostics.get("leakage"),
                                 gen_gap=g.get("gen_gap"), metric=g.get("metric_name")))
            last = rows[-1]
            print(f"    hardness={alpha:.2f}  leakage={last['leakage']}  gen_gap={last['gen_gap']}")
    return rows


def analyze_and_plot(rows):
    import pandas as pd
    from scipy.stats import spearmanr
    df = pd.DataFrame(rows)
    for c in ("leakage", "gen_gap"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby(["dataset", "hardness"]).agg(
        leakage=("leakage", "mean"), gen_gap=("gen_gap", "mean")).reset_index()

    print("\n== hardness -> gap monotonicity (Spearman) + calibration ==")
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("hardness")
        rho = spearmanr(g["hardness"], g["gen_gap"]).correlation
        a, bcoef = np.polyfit(g["hardness"], g["gen_gap"], 1)   # gap ~ a*hardness + b
        print(f"  {ds:26s} spearman(hardness,gap)={rho:+.2f}  gap≈{bcoef:.3f}+{a:.3f}·α  "
              f"[gap {g['gen_gap'].min():.3f}→{g['gen_gap'].max():.3f}]")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("hardness")
        ax.plot(g["hardness"], g["gen_gap"], "o-", ms=5, label=ds)
    ax.set_xlabel("hardness dial α  (0 = random, 1 = leakage-minimized)", fontsize=12)
    ax.set_ylabel("realized generalization gap", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=300); plt.close(fig)


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
