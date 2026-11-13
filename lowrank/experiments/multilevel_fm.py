"""Direction #1 experiment: multilevel FM vs flat FM, small n and at scale.

Two questions:
  1. Small real datasets (triplicate): does the multilevel V-cycle lower leakage
     below the flat best-of-4 Lloyd + single-move FM?  -> NO: flat FM is already at
     the global optimum wherever it runs (best-of-30 == best-of-4), so multilevel
     ties it (guaranteed never worse via the seed candidate) with ~0% gain.
  2. At scale: the current splitter DISABLES FM above ``fm_max_n=200k`` (splitter.py),
     so >200k it ships an un-polished Lloyd split. The synthetic scale sweep shows the
     leakage that leaves on the table -- and that simply REMOVING THE CAP and running
     the existing flat FM recovers it (e.g. +4% at 300k in ~2s, at 1M in ~12s), while
     the multilevel machinery recovers the *same* leakage ~47x slower.

Conclusion: the actionable win is raising ``fm_max_n``, not multilevel. See report.

    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.multilevel_fm \
        --datasets moleculenet_bace moleculenet_esol moleculenet_freesolv qmof --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PALM.data.sources import load_dataset
from PALM.lowrank import (balanced_lloyd, factor_leakage, fm_polish,
                          nystrom_features)
from PALM.lowrank.multilevel import multilevel_split
from PALM.lowrank.objective import realized_imbalance

HERE = os.path.dirname(__file__)
OUT_SMALL = os.path.join(HERE, "multilevel_small.csv")
OUT_SCALE = os.path.join(HERE, "multilevel_scale.csv")
OUT_FIG = os.path.join(HERE, "multilevel_scale.png")
FM_CAP = 200_000                       # the current splitter's fm_max_n


def _best_lloyd(B, splits, seed, n_restarts=4):
    best, best_obj = None, np.inf
    for r in range(n_restarts):
        lab = balanced_lloyd(B, splits, seed=seed + r)
        o = factor_leakage(B, lab, len(splits))
        if o < best_obj:
            best_obj, best = o, lab
    return best, best_obj


def run_small(datasets, seeds, limit, route, rank=256, eps=0.05):
    """Flat (Lloyd+FM) vs multilevel on real datasets, triplicate. Expect a tie."""
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data:
            print(f"[{name}] skip")
            continue
        data = b.feature_data
        ids = list(data)
        X = np.stack([data[i] for i in ids])
        B, metric = nystrom_features(X, rank=rank, seed=0)
        n = len(ids)
        print(f"[{name}] n={n} feat={b.meta.get('feature_set')} metric={metric}")
        for seed in seeds:
            lloyd, _ = _best_lloyd(B, [8, 2], seed)
            flat, _ = fm_polish(B, lloyd.copy(), [8, 2], epsilon=eps)
            ml = multilevel_split(B, [8, 2], epsilon=eps, seed=seed, seed_labels=lloyd)
            Lf = factor_leakage(B, flat, 2)
            Lm = factor_leakage(B, ml, 2)
            rows.append(dict(dataset=name, n=n, seed=seed,
                             flat_leakage=round(Lf, 1), ml_leakage=round(Lm, 1),
                             pct_gain=round(100 * (Lf - Lm) / Lf, 2) if Lf else 0.0,
                             flat_imb=round(realized_imbalance(flat, [8, 2]), 4),
                             ml_imb=round(realized_imbalance(ml, [8, 2]), 4)))
        g = np.mean([r["pct_gain"] for r in rows if r["dataset"] == name])
        print(f"    multilevel gain over flat: {g:+.2f}% (mean over seeds)")
    return rows


def _synth(n, d=64, k=80, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 2.5
    return np.vstack([centers[c] + rng.normal(size=(n // k, d))
                      for c in range(k)]).astype("float32")


def run_scale(sizes, rank=128, eps=0.05):
    """lloyd-only (current, capped) vs flat FM cap-removed vs multilevel, with timing."""
    rows = []
    for n in sizes:
        X = _synth(n)
        B, _ = nystrom_features(X, rank=rank, seed=0)
        lloyd, Ll = _best_lloyd(B, [8, 2], 0)
        current = Ll if n > FM_CAP else None            # what the splitter ships today

        t = time.time()
        flat, mv = fm_polish(B, lloyd.copy(), [8, 2], epsilon=eps)
        t_flat = time.time() - t
        Lflat = factor_leakage(B, flat, 2)

        t = time.time()
        ml = multilevel_split(B, [8, 2], epsilon=eps, seed=0, seed_labels=lloyd)
        t_ml = time.time() - t
        Lml = factor_leakage(B, ml, 2)

        rows.append(dict(n=n, capped=(n > FM_CAP), lloyd_leakage=round(Ll, 0),
                         flat_leakage=round(Lflat, 0), flat_s=round(t_flat, 2),
                         ml_leakage=round(Lml, 0), ml_s=round(t_ml, 2),
                         flat_gain_pct=round(100 * (Ll - Lflat) / Ll, 2),
                         ml_vs_flat_pct=round(100 * (Lflat - Lml) / Lflat, 3)))
        print(f"  n={n:>7} capped={n>FM_CAP}  lloyd={Ll:.0f}  "
              f"flat={Lflat:.0f}({t_flat:.1f}s)  ml={Lml:.0f}({t_ml:.1f}s)")
    return rows


def plot_scale(rows, out=OUT_FIG):
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("n")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    # leakage relative to un-polished Lloyd (=1.0): the gap the 200k cap leaves,
    # closed identically by flat-FM-cap-removed and by multilevel (they overlap).
    ax.plot(df["n"], df["lloyd_leakage"] / df["lloyd_leakage"], "o-",
            label="Lloyd only (current, >200k)", ms=6)
    ax.plot(df["n"], df["flat_leakage"] / df["lloyd_leakage"], "s-",
            label="flat FM, cap removed", ms=6)
    ax.plot(df["n"], df["ml_leakage"] / df["lloyd_leakage"], "x--",
            label="multilevel FM", ms=8)
    ax.axvline(FM_CAP, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("n (rows)", fontsize=12)
    ax.set_ylabel("factor leakage / Lloyd-only leakage", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["moleculenet_bace", "moleculenet_esol",
                             "moleculenet_freesolv", "qmof"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--route", action="store_true", default=True)
    ap.add_argument("--scale-sizes", nargs="+", type=int, default=[20000, 100000, 300000])
    ap.add_argument("--skip-small", action="store_true")
    ap.add_argument("--skip-scale", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_small:
        print("== small real datasets (flat vs multilevel, triplicate) ==")
        small = run_small(args.datasets, args.seeds, args.limit, args.route)
        if small:
            _write(OUT_SMALL, small)
            print(f"-> {OUT_SMALL}")
    if not args.skip_scale:
        print("== scale sweep (lloyd-only vs flat-FM-cap-removed vs multilevel) ==")
        scale = run_scale(args.scale_sizes)
        _write(OUT_SCALE, scale)
        plot_scale(scale)
        print(f"-> {OUT_SCALE}\n-> {OUT_FIG}")


if __name__ == "__main__":
    main()
