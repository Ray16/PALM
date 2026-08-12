"""OMol25 quality-preservation figure: low-rank's L(pi) is stable across scale.

A low-rank-ONLY claim (no DataSAIL): as n grows from 1k to 9.55M, the Nystrom-
factorized split keeps L(pi) flat at ~0.27, consistently below the random
baseline (~0.32) at every scale. This shows the rank-r approximation does not
degrade the split as n grows -- the honest OMol25 quality point, with no
competitor to misrepresent (at a matched 20% ratio, DataSAIL's L(pi) is a tie;
the real difference is time, shown in the scaling figure).

    python poster/make_omol25_quality.py   (palm env)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ORES = os.path.join(ROOT, "lowrank_split", "omol25", "results")
OUT = os.path.join(HERE, "figures")

LOWRANK = "#2563EB"
RANDOM = "#C7CDD6"
INK = "#14243B"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 23,
    "axes.labelsize": 25, "xtick.labelsize": 20, "ytick.labelsize": 21,
    "legend.fontsize": 22, "axes.linewidth": 1.5,
    "lines.linewidth": 4.0, "lines.markersize": 15,
    "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
})


def main():
    df = pd.read_csv(os.path.join(ORES, "omol25_scaling.csv"))
    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.semilogx(df["n"], df["random_lpi"], "s--", color="#8A93A3",
                label="Random", markeredgecolor="white", markeredgewidth=1.4,
                zorder=3)
    ax.semilogx(df["n"], df["lowrank_lpi"], "o-", color=LOWRANK,
                label="Low-rank  (exact 20% test)", markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)
    # highlight that the low-rank line is flat all the way to 9.55M
    n_full = df["n"].max()
    ax.annotate("flat to 9.55M", (n_full, df["lowrank_lpi"].iloc[-1]),
                xytext=(-12, 34), textcoords="offset points", ha="right",
                va="bottom", fontsize=20, color=LOWRANK,
                arrowprops=dict(arrowstyle="-|>", color=LOWRANK, lw=2.2))
    ax.set_xlabel("dataset size  (n structures)")
    ax.set_ylabel(r"$L(\pi)$")
    ax.set_ylim(0.20, 0.36)
    ax.legend(loc="center left", frameon=True, framealpha=0.96)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_omol25_quality_vs_scale.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print("saved", p)


if __name__ == "__main__":
    main()
