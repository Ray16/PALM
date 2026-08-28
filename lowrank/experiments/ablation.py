"""Ablation study: component contributions and hyperparameter neutrality.

Two parts, both reporting **leakage AND realized generalization gap side by side**
(the point of the study — per FINDINGS.md the two do NOT move together):

Part A (consolidation). For each dataset, four configs of the low-rank splitter:
  1. random          — hardness=0.0 (fully randomized; easy / high-leakage)
  2. lloyd           — fm=False, balance_slack=0.0 (leakage-min, no polish)
  3. lloyd_fm        — fm=True,  balance_slack=0.0  <- THE ANCHOR (default splitter)
  4. lloyd_fm_slack  — fm=True,  balance_slack=0.30 (spend balance for lower leakage)
The claim to see: the anchor (3) is simultaneously MIN leakage and MAX gap; moving
either way (randomness in 1, imbalance in 4) makes the eval easier.

Part B (targeted grid). On the anchor config, sweep rank in {64,256,1024} and
n_restarts in {1,4,16} (one axis at a time). Question: does the tuning-neutrality
found on leakage (Steps 2 & 4) also hold for the gap?

    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.ablation \
        --seeds 0 1 2
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
OUT_COMP = os.path.join(HERE, "ablation_components.csv")
OUT_GRID = os.path.join(HERE, "ablation_grid.csv")

# Part A: (config label, splitter params). Anchor = lloyd_fm.
COMPONENT_CONFIGS = [
    ("random",         dict(hardness=0.0)),
    ("lloyd",          dict(fm=False, balance_slack=0.0)),
    ("lloyd_fm",       dict(fm=True,  balance_slack=0.0)),      # anchor
    ("lloyd_fm_slack", dict(fm=True,  balance_slack=0.30)),
]
GAP_DATASETS = ["moleculenet_esol", "moleculenet_bace", "moleculenet_freesolv"]
LEAKAGE_ONLY = ["qmof"]

RANKS = [64, 256, 1024]
N_RESTARTS = [1, 4, 16]


# --------------------------------------------------------------------------- #
# shared: one split -> (leakage, factor_leakage, gap, metric)
# --------------------------------------------------------------------------- #
def _load(name, limit, route):
    b = load_dataset(name, limit=limit, route=route)
    if not b.available or not b.feature_data:
        return None
    ids = list(b.feature_data)
    X = np.stack([b.feature_data[i] for i in ids])
    has_target = bool(b.targets) and bool(b.task_type)
    y = (np.array([b.targets.get(i, np.nan) for i in ids], dtype=float)
         if has_target else None)
    return b, ids, X, y, has_target


def _run_one(b, ids, X, y, has_target, params, seed):
    spec = SplitSpec([8, 2], ["train", "test"], seed=seed)
    res = split("lowrank", b.feature_data, spec, **params)
    leak = res.diagnostics.get("leakage")
    floss = res.diagnostics.get("factor_leakage")
    gap, metric = "", ""
    if has_target:
        tr = [j for j, i in enumerate(ids) if res.assignment[i] == "train"]
        te = [j for j, i in enumerate(ids) if res.assignment[i] == "test"]
        g = evaluate_gap(X, y, b.task_type, tr, te, seed=seed)
        gap, metric = g.get("gen_gap"), g.get("metric_name")
    return leak, floss, gap, metric


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #
def run_components(datasets, seeds, limit, route):
    rows = []
    for name in datasets:
        loaded = _load(name, limit, route)
        if loaded is None:
            print(f"[{name}] skip (unavailable)")
            continue
        b, ids, X, y, has_target = loaded
        print(f"[{name}] n={len(ids)} task={b.task_type} "
              f"feature_set={b.meta.get('feature_set', 'default')} gap={'yes' if has_target else 'NO'}")
        for label, params in COMPONENT_CONFIGS:
            for seed in seeds:
                leak, floss, gap, metric = _run_one(b, ids, X, y, has_target, params, seed)
                rows.append(dict(dataset=name, config=label, seed=seed, task=b.task_type or "",
                                 leakage=leak, factor_leakage=floss, gen_gap=gap, metric=metric))
            last = rows[-1]
            print(f"    {label:16s} leakage={last['leakage']}  gen_gap={last['gen_gap']}")
    return rows


# --------------------------------------------------------------------------- #
# Part B
# --------------------------------------------------------------------------- #
def run_grid(datasets, seeds, limit, route):
    rows = []
    for name in datasets:
        loaded = _load(name, limit, route)
        if loaded is None:
            print(f"[{name}] skip (unavailable)")
            continue
        b, ids, X, y, has_target = loaded
        print(f"[grid {name}] n={len(ids)} task={b.task_type}")
        sweeps = ([("rank", r, dict(fm=True, balance_slack=0.0, rank=r, n_restarts=4)) for r in RANKS] +
                  [("n_restarts", nr, dict(fm=True, balance_slack=0.0, rank=256, n_restarts=nr))
                   for nr in N_RESTARTS])
        for axis, value, params in sweeps:
            for seed in seeds:
                leak, floss, gap, metric = _run_one(b, ids, X, y, has_target, params, seed)
                rows.append(dict(dataset=name, axis=axis, value=value, seed=seed,
                                 leakage=leak, factor_leakage=floss, gen_gap=gap, metric=metric))
            last = rows[-1]
            print(f"    {axis}={value:<5} leakage={last['leakage']}  gen_gap={last['gen_gap']}")
    return rows


# --------------------------------------------------------------------------- #
# plots + analysis
# --------------------------------------------------------------------------- #
def _agg(rows, keys, values=("leakage", "gen_gap")):
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in values:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    m = df.groupby(list(keys)).agg(**{f"{v}_mean": (v, "mean") for v in values},
                                   **{f"{v}_std": (v, "std") for v in values}).reset_index()
    return df, m


def analyze_components(rows):
    import pandas as pd
    _, m = _agg(rows, ["dataset", "config"])
    order = [c for c, _ in COMPONENT_CONFIGS]
    print("\n== Part A: components (leakage & gap, mean±std) ==")
    for ds, g in m.groupby("dataset"):
        g = g.set_index("config").reindex(order)
        print(f"\n  {ds}")
        for cfg in order:
            r = g.loc[cfg]
            gp = "" if pd.isna(r["gen_gap_mean"]) else f"{r['gen_gap_mean']:.3f}±{r['gen_gap_std']:.3f}"
            print(f"    {cfg:16s} leakage={r['leakage_mean']:.3f}±{r['leakage_std']:.3f}  gap={gp}")
        # anchor check
        if not pd.isna(g.loc['lloyd_fm', 'gen_gap_mean']):
            anchor_leak = g.loc['lloyd_fm', 'leakage_mean']
            anchor_gap = g.loc['lloyd_fm', 'gen_gap_mean']
            min_leak = g['leakage_mean'].min() == anchor_leak
            max_gap = g['gen_gap_mean'].max() == anchor_gap
            print(f"    -> anchor(lloyd_fm) min-leakage={min_leak}  max-gap={max_gap}")

    # one PNG per gap-dataset: configs as points in (leakage, gap) space
    for ds, g in m.groupby("dataset"):
        if g["gen_gap_mean"].isna().all():
            continue
        g = g.set_index("config").reindex(order)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for cfg in order:
            r = g.loc[cfg]
            if pd.isna(r["gen_gap_mean"]):
                continue
            ax.errorbar(r["leakage_mean"], r["gen_gap_mean"],
                        xerr=r["leakage_std"], yerr=r["gen_gap_std"],
                        fmt="o", ms=9, capsize=3, label=cfg)
        ax.set_xlabel("leakage L(π)  (lower = less similar train/test)", fontsize=12)
        ax.set_ylabel("realized generalization gap  (higher = harder)", fontsize=12)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"ablation_components_{ds}.png"), dpi=300)
        plt.close(fig)


def analyze_grid(rows):
    import pandas as pd
    from scipy.stats import spearmanr
    _, m = _agg(rows, ["dataset", "axis", "value"])
    print("\n== Part B: hyperparameter grid (does rank / n_restarts move leakage or gap?) ==")
    for axis in ("rank", "n_restarts"):
        sub = m[m["axis"] == axis]
        print(f"\n  axis = {axis}")
        for ds, g in sub.groupby("dataset"):
            g = g.sort_values("value")
            lk = g["leakage_mean"].values
            gp = g["gen_gap_mean"].values
            lk_rng = f"{lk.min():.3f}→{lk.max():.3f} (span {lk.max()-lk.min():.3f})"
            gp_rng = f"{gp.min():.3f}→{gp.max():.3f} (span {gp.max()-gp.min():.3f})"
            print(f"    {ds:24s} leakage {lk_rng}   gap {gp_rng}")
        # plots: leakage vs axis, gap vs axis (separate single-metric files)
        for metric, col in (("leakage", "leakage_mean"), ("gap", "gen_gap_mean")):
            fig, ax = plt.subplots(figsize=(7.5, 6))
            any_line = False
            for ds, g in sub.groupby("dataset"):
                g = g.sort_values("value")
                if g[col].isna().all():
                    continue
                ax.plot(g["value"], g[col], "o-", ms=6, label=ds)
                any_line = True
            ax.set_xlabel(axis, fontsize=12)
            ax.set_ylabel(f"{'leakage L(π)' if metric=='leakage' else 'realized generalization gap'}",
                          fontsize=12)
            if axis == "rank":
                ax.set_xscale("log", base=2)
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
            fig.tight_layout()
            if any_line:
                fig.savefig(os.path.join(HERE, f"ablation_{axis}_{metric}.png"), dpi=300)
            plt.close(fig)


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--route", action="store_true", default=True)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_a:
        print("== Part A: component consolidation ==")
        comp = run_components(GAP_DATASETS + LEAKAGE_ONLY, args.seeds, args.limit, args.route)
        _write(OUT_COMP, comp)
        analyze_components(comp)
        print(f"-> {OUT_COMP}")
    if not args.skip_b:
        print("\n== Part B: targeted hyperparameter grid ==")
        grid = run_grid(GAP_DATASETS, args.seeds, args.limit, args.route)
        _write(OUT_GRID, grid)
        analyze_grid(grid)
        print(f"-> {OUT_GRID}")


if __name__ == "__main__":
    main()
