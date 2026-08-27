"""Figure: multi-axis hypergraph split on USPTO MCR -- macro scaled L(pi) for
random vs identity/cluster vs k-NN constructions. Saves to the manuscript Figs/
directory.

Pure plotter: all numbers come from ``results/mcr_results.csv`` (produced by
``python -m PALM.benchmarks.reactions.benchmark_mcr``). Re-run that benchmark to
refresh them.

    python -m PALM.benchmarks.charts.make_mcr_chart
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "results", "mcr_results.csv")
FIGDIR = os.path.join(HERE, "..", "..", "palm_manuscript", "Figs")

# palette shared with make_chart.py: random=grey, PALM identity=orange, PALM k-NN=blue
STYLE = {
    "random": ("Random", "#9aa0a6"),
    "hypergraph_identity": ("PALM (identity/cluster edges)", "#f4a36c"),
    "hypergraph_knn_k25": ("PALM (k-NN edges, k=25)", "#3b6fb6"),
}
ORDER = ["random", "hypergraph_identity", "hypergraph_knn_k25"]


def main():
    df = pd.read_csv(CSV).set_index("method")
    methods = [m for m in ORDER if m in df.index]
    vals = [float(df.loc[m, "macro_lpi"]) for m in methods]
    labels = [STYLE[m][0] for m in methods]
    colors = [STYLE[m][1] for m in methods]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(x, vals, 0.6, color=colors, zorder=3)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace(" (", "\n(") for lbl in labels], fontsize=8)
    ax.set_ylabel(r"macro $L(\pi)$  (lower is better)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "mcr_benchmark.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", os.path.normpath(out))
    print("macro L(pi): " + "  ".join(f"{m}={v:.4f}" for m, v in zip(methods, vals)))


if __name__ == "__main__":
    main()
