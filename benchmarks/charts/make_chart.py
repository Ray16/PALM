"""Grouped bar chart from moleculenet1d_results.csv: hypergraph vs DataSAIL vs random
vs paper DataSAIL-S1, all scored with the GPU scaled L(pi) (== eval_split).

Pure plotter: all numbers come from ``moleculenet1d_results.csv`` (produced by
``python -m PALM.benchmarks.moleculenet.benchmark_moleculenet1d``). Re-run the benchmark to refresh them.

    python -m PALM.benchmarks.charts.make_chart        # -> benchmark_chart.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "results", "moleculenet1d_results.csv")
FIGDIR = os.path.join(HERE, "..", "..", "palm_manuscript", "Figs")


def plot(df):
    labels = [f"{r.dataset}\n(n={int(r.n):,})" for r in df.itertuples()]
    x = np.arange(len(df)); w = 0.2
    fig, ax = plt.subplots(figsize=(15, 6))
    # palette shared with make_mcr_chart.py (Fig. 2): random=grey, PALM=blue,
    # DataSAIL family=orange (fresh = light, paper-S1 = dark).
    series = [("hypergraph", "#3b6fb6"), ("datasail", "#f4a36c"),
              ("paper_s1", "#b5651d"), ("random", "#9aa0a6")]
    names = {"hypergraph": "PALM", "datasail": "DataSAIL (fresh)",
             "random": "Random", "paper_s1": "Paper DataSAIL-S1"}

    def _fmt(t):
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return None
        return f"{t:.1f}s" if t < 100 else f"{t:.0f}s"

    for j, (col, c) in enumerate(series):
        vals = [v if v is not None and not pd.isna(v) else 0 for v in df[col]]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=names[col], color=c)
        # runtime annotations above the bars that actually run (hypergraph, DataSAIL)
        time_col = {"hypergraph": "hg_time", "datasail": "ds_time"}.get(col)
        if time_col:
            for i, b in enumerate(bars):
                t = _fmt(df[time_col].iloc[i])
                if t is not None:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, t,
                            rotation=90, ha="center", va="bottom", fontsize=7,
                            color=c, fontweight="bold")
                elif col == "datasail":
                    ax.text(x[i] + (j - 1.5) * w, 0.012, "needs >220 GB", rotation=90,
                            ha="center", va="bottom", fontsize=7, color="#b5651d", fontweight="bold")
    ax.set_ylabel(r"$L(\pi)$  (lower is better)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    # legend outside the axes (upper right) so it never overlaps bars/annotations
    ax.legend(fontsize=10, framealpha=0.95, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.12)            # headroom so the tallest bars + rotated time labels clear the top
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "benchmark_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", os.path.normpath(out))


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    plot(df)
