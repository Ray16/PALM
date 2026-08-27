"""Publication-ready method-comparison figures from ``split_results/summary.csv``.

Single-panel PNGs (no titles; caption-only, per project convention):
  - leakage_by_method.png   scaled L(pi) per dataset x method (lower = better)
  - runtime_by_method.png   wall-clock seconds per dataset x method (log y)

Colorblind-safe (Okabe-Ito) palette; ``random`` drawn as a hatched grey
reference every method should beat. 300 DPI.

    python -m PALM.data.plot_results
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "split_results")

# stable method order + Okabe-Ito colorblind-safe colors
METHOD_ORDER = ["random", "hypergraph", "graph", "lowrank", "datasail",
                "hypergraph_nd", "hypergraph_nd_knn"]
COLORS = {
    "random": "#999999",
    "hypergraph": "#0072B2", "graph": "#009E73", "lowrank": "#D55E00",
    "datasail": "#CC79A7", "hypergraph_nd": "#E69F00", "hypergraph_nd_knn": "#56B4E9",
}
LABELS = {
    "hypergraph_nd": "hypergraph-nd", "hypergraph_nd_knn": "hypergraph-nd-knn",
}
DATASET_LABELS = {
    "moleculenet_bace": "BACE", "moleculenet_bbbp": "BBBP", "moleculenet_esol": "ESOL",
    "qmof": "QMOF", "omol25": "OMol25", "materials_project": "Materials\nProject",
    "uspto_mcr": "USPTO-MCR", "openpolymer26": "OPoly26",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12,
    "axes.linewidth": 0.9, "axes.edgecolor": "#444444",
    "figure.dpi": 300, "savefig.dpi": 300,
})


def _grouped_bar(df, value_col, ylabel, path, logy=False):
    datasets = list(df["dataset"].unique())
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    x = np.arange(len(datasets))
    width = 0.82 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(12, 5.4))
    for i, m in enumerate(methods):
        vals = []
        for d in datasets:
            row = df[(df["dataset"] == d) & (df["method"] == m)]
            vals.append(float(row[value_col].iloc[0]) if len(row) and pd.notna(row[value_col].iloc[0]) else np.nan)
        pos = x + (i - (len(methods) - 1) / 2) * width
        ax.bar(pos, vals, width, label=LABELS.get(m, m), color=COLORS.get(m),
               edgecolor="white", linewidth=0.6,
               hatch="///" if m == "random" else None, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)
    ax.grid(axis="y", color="#cccccc", linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=len(methods), frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=1.3, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    df = pd.read_csv(os.path.join(OUT, "summary.csv"))
    df = df[df["status"] == "ok"].copy()

    leak = df[pd.to_numeric(df["leakage"], errors="coerce").notna()].copy()
    leak["leakage"] = leak["leakage"].astype(float)
    _grouped_bar(leak, "leakage", "scaled  L(π)  leakage   (lower is better)",
                 os.path.join(OUT, "leakage_by_method.png"))

    rt = df[pd.to_numeric(df["runtime_s"], errors="coerce").notna()].copy()
    rt["runtime_s"] = rt["runtime_s"].astype(float)
    _grouped_bar(rt, "runtime_s", "wall-clock runtime (s)",
                 os.path.join(OUT, "runtime_by_method.png"), logy=True)


if __name__ == "__main__":
    main()
