"""Target-gap validation: does a *requested* gap match the *realized* gap?

Step 3 showed the ``hardness`` dial controls the realized generalization gap.
This experiment closes the loop with the inversion workflow (``target_gap.py``):

  1. calibrate ``gap ≈ b + a·alpha`` from probe splits (triplicate seeds),
  2. request several target gaps spanning the achievable range,
  3. invert each to an alpha, produce the split, and *measure* the realized gap,
  4. report requested-vs-realized error (MAE) and correlation.

If the flagship's within-dataset control is real, requested and realized gaps
line up on ``y = x``. Uses the same estimator/datasets as hardness_control.py.

    CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.target_gap \
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

from PALM.data.sources import load_dataset
from PALM.lowrank.target_gap import calibrate_gap, measure_gap

HERE = os.path.dirname(__file__)
OUT_CSV = os.path.join(HERE, "target_gap.csv")


def _fig_path(dataset: str) -> str:
    short = dataset.replace("moleculenet_", "")
    return os.path.join(HERE, f"target_gap_{short}.png")


def run(datasets, seeds, limit, route, n_targets):
    rows = []
    for name in datasets:
        b = load_dataset(name, limit=limit, route=route)
        if not b.available or not b.feature_data or not b.targets or not b.task_type:
            print(f"[{name}] skip (no target/features)")
            continue
        ids = list(b.feature_data)
        X = np.stack([b.feature_data[i] for i in ids])
        y = np.array([b.targets.get(i, np.nan) for i in ids], dtype=float)
        print(f"[{name}] n={len(ids)} task={b.task_type} "
              f"feature_set={b.meta.get('feature_set','default')}")

        # (1) calibrate
        cal = calibrate_gap(b.feature_data, X, y, b.task_type,
                            seeds=seeds, dataset=name, verbose=True)
        lo, hi = cal.achievable_range
        print(f"  calibration: gap ≈ {cal.b:.3f} + {cal.a:.3f}·α  "
              f"R²={cal.r2:.3f} spearman={cal.spearman:+.3f}  range=[{lo:.3f},{hi:.3f}]")

        # (2,3) request targets spanning the achievable range, invert, re-measure.
        # Interior targets only (skip exact endpoints, which are the probe points).
        targets = np.linspace(lo, hi, n_targets + 2)[1:-1] if hi > lo else [lo]
        for tgt in targets:
            inv = cal.invert(tgt)
            m = measure_gap(b.feature_data, X, y, b.task_type, inv.alpha, seeds=seeds)
            rows.append(dict(
                dataset=name, task=b.task_type,
                target_gap=round(float(tgt), 4), alpha=round(inv.alpha, 4),
                predicted_gap=round(inv.predicted_gap, 4),
                realized_gap=round(m["gap_mean"], 4),
                realized_gap_std=round(m["gap_std"], 4),
                abs_err=round(abs(m["gap_mean"] - float(tgt)), 4),
                in_range=inv.in_range,
                cal_b=round(cal.b, 4), cal_a=round(cal.a, 4),
                cal_r2=round(cal.r2, 4), cal_spearman=round(cal.spearman, 4),
                range_lo=round(lo, 4), range_hi=round(hi, 4)))
            print(f"    target={tgt:.3f} -> alpha={inv.alpha:.3f} "
                  f"predicted={inv.predicted_gap:.3f} realized={m['gap_mean']:.3f} "
                  f"|err|={rows[-1]['abs_err']:.3f}")
    return rows


def analyze_and_plot(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n== requested vs realized gap (per dataset) ==")
    for ds, g in df.groupby("dataset"):
        mae = float(g["abs_err"].mean())
        if len(g) >= 2 and g["target_gap"].std() > 1e-9:
            corr = float(np.corrcoef(g["target_gap"], g["realized_gap"])[0, 1])
        else:
            corr = float("nan")
        print(f"  {ds:26s} MAE(requested,realized)={mae:.3f}  "
              f"corr={corr:+.3f}  (n={len(g)})")

        # one single-panel PNG per dataset: requested (x) vs realized (y) + y=x.
        fig, ax = plt.subplots(figsize=(6, 6))
        gg = g.sort_values("target_gap")
        ax.errorbar(gg["target_gap"], gg["realized_gap"],
                    yerr=gg["realized_gap_std"], fmt="o-", ms=6, capsize=3)
        lims = [min(gg["target_gap"].min(), gg["realized_gap"].min()),
                max(gg["target_gap"].max(), gg["realized_gap"].max())]
        pad = 0.05 * (lims[1] - lims[0] + 1e-9)
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, "--", color="gray", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("requested generalization gap", fontsize=12)
        ax.set_ylabel("realized generalization gap", fontsize=12)
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(_fig_path(ds), dpi=300); plt.close(fig)
        print(f"    figure -> {_fig_path(ds)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["moleculenet_esol", "moleculenet_bace", "moleculenet_freesolv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--route", action="store_true", default=True)
    ap.add_argument("--n-targets", type=int, default=4,
                    help="number of interior target gaps to request per dataset")
    args = ap.parse_args(argv)
    rows = run(args.datasets, args.seeds, args.limit, args.route, args.n_targets)
    if not rows:
        print("no rows produced"); return
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    analyze_and_plot(rows)
    print(f"\n== {len(rows)} target requests -> {OUT_CSV}")


if __name__ == "__main__":
    main()
