"""Publication-ready figures for the low-rank findings (Steps 1, 3, and mechanism).

Reads the experiment CSVs and emits clean single-panel PNGs (no titles/annotations;
explanation belongs in the caption). Regenerate after re-running the experiments.

    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.lowrank.experiments.plot_findings
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300})

# colour-blind-safe, consistent per dataset
COLORS = {"moleculenet_esol": "#0072B2", "moleculenet_bace": "#D55E00",
          "moleculenet_freesolv": "#009E73", "qmof": "#CC79A7"}
LABEL = {"moleculenet_esol": "ESOL", "moleculenet_bace": "BACE",
         "moleculenet_freesolv": "FreeSolv", "qmof": "QMOF"}


def _num(df, *cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fig_balance_tradeoff():
    """Step 1: leakage falls as the balance corridor opens."""
    df = _num(pd.read_csv(os.path.join(HERE, "balance_pareto.csv")), "leakage")
    agg = df.groupby(["dataset", "balance_slack"])["leakage"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("balance_slack")
        ax.errorbar(g["balance_slack"], g["mean"], yerr=g["std"], marker="o", ms=6,
                    capsize=3, color=COLORS.get(ds, "#333"), label=LABEL.get(ds, ds))
    ax.set_xlabel("balance slack  (max fractional size deviation)")
    ax.set_ylabel(r"leakage  $L(\pi)$")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_balance_tradeoff.png")); plt.close(fig)


def fig_hardness_control():
    """Step 3: realized generalization gap rises monotonically with the hardness dial."""
    df = _num(pd.read_csv(os.path.join(HERE, "hardness_control.csv")), "gen_gap")
    agg = df.groupby(["dataset", "hardness"])["gen_gap"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for ds, g in agg.groupby("dataset"):
        g = g.sort_values("hardness")
        ax.errorbar(g["hardness"], g["mean"], yerr=g["std"], marker="o", ms=6, capsize=3,
                    color=COLORS.get(ds, "#333"), label=LABEL.get(ds, ds))
    ax.set_xlabel(r"hardness dial  $\alpha$   (0 = random,  1 = leakage-minimized)")
    ax.set_ylabel("realized generalization gap")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_hardness_control.png")); plt.close(fig)


def fig_mechanism_divergence():
    """Steps 1 vs 3: same leakage axis, opposite gap effect (the key mechanism figure)."""
    h = _num(pd.read_csv(os.path.join(HERE, "hardness_control.csv")), "leakage", "gen_gap")
    b = _num(pd.read_csv(os.path.join(HERE, "balance_gap.csv")), "leakage", "gen_gap")
    ah = h.groupby(["dataset", "hardness"]).agg(leakage=("leakage", "mean"),
                                                gap=("gen_gap", "mean")).reset_index()
    ab = b.groupby(["dataset", "balance_slack"]).agg(leakage=("leakage", "mean"),
                                                     gap=("gen_gap", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for ds in sorted(set(ah["dataset"]) & set(ab["dataset"])):
        c = COLORS.get(ds, "#333")
        gh = ah[ah.dataset == ds].sort_values("leakage")
        gb = ab[ab.dataset == ds].sort_values("leakage")
        ax.plot(gh["leakage"], gh["gap"], "o-", color=c, ms=6, label=f"{LABEL.get(ds, ds)} · hardness")
        ax.plot(gb["leakage"], gb["gap"], "s--", color=c, ms=6, mfc="white",
                label=f"{LABEL.get(ds, ds)} · balance slack")
    ax.set_xlabel(r"leakage  $L(\pi)$")
    ax.set_ylabel("realized generalization gap")
    ax.legend(frameon=False, fontsize=8.5, ncol=1); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_mechanism_divergence.png")); plt.close(fig)


def main():
    fig_balance_tradeoff()
    fig_hardness_control()
    fig_mechanism_divergence()
    print("wrote: fig_balance_tradeoff.png, fig_hardness_control.png, fig_mechanism_divergence.png")


if __name__ == "__main__":
    main()
