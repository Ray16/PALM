"""Grouped bar charts comparing the low-rank splitter to DataSAIL and hypergraph.

Two figures, all numbers straight from the CSVs (re-run the benchmark to refresh):
  1. leakage L(pi) per dataset  ->  results/lowrank_benchmark_chart.png
  2. runtime vs DataSAIL         ->  results/lowrank_timing_chart.png

Leakage bars: DataSAIL (reference), hypergraph (mean+/-std over seeds), low-rank
(mean+/-std) -- all scored with the SAME factorized L(pi), lower is better.

    python -m PALM.lowrank_split.make_benchmark_chart
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

# palette consistent with benchmark/make_chart.py: DataSAIL=orange, PALM/hypergraph
# =blue; low-rank gets its own green so the new method reads at a glance.
C_DATASAIL = "#f4a36c"
C_HYPER = "#3b6fb6"
C_LOWRANK = "#5aa469"


def plot_leakage():
    df = pd.read_csv(os.path.join(RES, "lowrank_benchmark.csv"))
    labels = [f"{r.dataset}\n(n={int(r.n):,})" for r in df.itertuples()]
    x = np.arange(len(df)); w = 0.26
    fig, ax = plt.subplots(figsize=(14, 6))

    # DataSAIL reference (no std); hiv/muv have no DataSAIL value -> annotate n/a
    ds = df["datasail_ref"].to_numpy(dtype=float)
    ax.bar(x - w, np.nan_to_num(ds), w, label="DataSAIL", color=C_DATASAIL)
    for i, v in enumerate(ds):
        if np.isnan(v):
            ax.text(x[i] - w, 0.006, "n/a", rotation=90, ha="center", va="bottom",
                    fontsize=7, color="#b5651d", fontweight="bold")

    # hypergraph and low-rank: mean +/- std over seeds
    ax.bar(x, df["hyperedge_mean"], w, yerr=df["hyperedge_std"], capsize=2,
           label="Hypergraph", color=C_HYPER, error_kw=dict(lw=0.8))
    ax.bar(x + w, df["lowrank_mean"], w, yerr=df["lowrank_std"], capsize=2,
           label="Low-rank", color=C_LOWRANK, error_kw=dict(lw=0.8))

    # mark where low-rank loses to hypergraph (higher L = more leakage)
    for i, r in enumerate(df.itertuples()):
        if r.lowrank_mean > r.hyperedge_mean:
            ax.text(x[i] + w, r.lowrank_mean + 0.004, "▲", ha="center",
                    va="bottom", fontsize=8, color="#c0392b")

    ax.set_ylabel(r"$L(\pi)$  (lower is better)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, framealpha=0.95, loc="upper left", bbox_to_anchor=(1.005, 1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.10)
    fig.tight_layout()
    out = os.path.join(RES, "lowrank_benchmark_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", os.path.normpath(out))


def plot_timing():
    df = pd.read_csv(os.path.join(RES, "lowrank_timing.csv"))
    labels = [f"{r.dataset}\n(n={int(r.n):,})" for r in df.itertuples()]
    x = np.arange(len(df)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))

    lr = df["lowrank_s"].to_numpy(dtype=float)
    ds = pd.to_numeric(df["datasail_s"], errors="coerce").to_numpy(dtype=float)
    ax.bar(x - w / 2, lr, w, label="Low-rank", color=C_LOWRANK)
    ax.bar(x + w / 2, np.nan_to_num(ds), w, label="DataSAIL", color=C_DATASAIL)
    ax.set_yscale("log")

    for i, r in enumerate(df.itertuples()):
        ax.text(x[i] - w / 2, lr[i] * 1.15, f"{lr[i]:.2f}s", ha="center", va="bottom",
                fontsize=7, color=C_LOWRANK, fontweight="bold")
        if np.isnan(ds[i]):
            ax.text(x[i] + w / 2, 1.2, "timeout", rotation=90, ha="center", va="bottom",
                    fontsize=7, color="#b5651d", fontweight="bold")
        else:
            ax.text(x[i] + w / 2, ds[i] * 1.15, f"{ds[i]:.0f}s\n({r.speedup}×)",
                    ha="center", va="bottom", fontsize=7, color="#b5651d", fontweight="bold")

    ax.set_ylabel("split time (s, log scale)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.margins(y=0.15)
    fig.tight_layout()
    out = os.path.join(RES, "lowrank_timing_chart.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", os.path.normpath(out))


if __name__ == "__main__":
    plot_leakage()
    plot_timing()
