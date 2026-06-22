"""Grouped bar chart from final_results.csv: hypergraph vs DataSAIL vs random
vs paper DataSAIL-S1, all scored with the GPU scaled L(pi) (== eval_split).

Pure plotter: all numbers come from ``final_results.csv`` (produced by
``python -m PALM.benchmark.benchmark``). Re-run the benchmark to refresh them.

    python -m PALM.benchmark.make_chart        # -> benchmark_chart.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "final_results.csv")


def plot(df):
    labels = [f"{r.dataset}\n(n={int(r.n):,})" for r in df.itertuples()]
    x = np.arange(len(df)); w = 0.2
    fig, ax = plt.subplots(figsize=(15, 6))
    series = [("hypergraph", "#2563eb"), ("datasail", "#dc2626"),
              ("paper_s1", "#d97706"), ("random", "#94a3b8")]
    names = {"hypergraph": "Hypergraph (ours)", "datasail": "DataSAIL (fresh)",
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
                    ax.text(x[i] + (j - 1.5) * w, 0.012, "timeout", rotation=90,
                            ha="center", va="bottom", fontsize=7, color="#dc2626", fontweight="bold")
    ax.set_ylabel("scaled L(π)  (lower = less leakage)", fontsize=12)
    ax.set_title("Train/test leakage: Hypergraph vs DataSAIL  (80/20, MoleculeNet)", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, framealpha=0.9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "benchmark_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    plot(df)
