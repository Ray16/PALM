"""Figure: n-D hypergraph split on USPTO MCR -- per-axis & macro scaled L(pi)
for random vs identity/cluster vs k-NN constructions. Saves to the manuscript
Figs/ directory. Run in boltz-2 env from the PALM parent dir."""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import reactions as R
from ..hypergraph import run_hypergraph_split_nd, run_hypergraph_split_nd_knn
from .benchmark_reactions import evaluate, random_split

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "palm_manuscript", "Figs")
K = 25


def _per_axis(records, afm, labels):
    macro, _, per = evaluate(records, afm, labels)
    return [per[a]["lpi"] for a in afm], macro


def main():
    records, afm, _ = R.load_uspto_mcr()
    n = len(records)
    axes = list(afm.keys())

    rnd_ax, rnd_m = _per_axis(records, afm, random_split(n))
    a_id, _ = run_hypergraph_split_nd(records, afm, [8, 2], ["train", "test"], sim_threshold=1.0)
    id_ax, id_m = _per_axis(records, afm, a_id)
    a_knn, _ = run_hypergraph_split_nd_knn(records, afm, [8, 2], ["train", "test"], k=K)
    kn_ax, kn_m = _per_axis(records, afm, a_knn)

    groups = axes + ["macro"]
    rnd = rnd_ax + [rnd_m]
    idn = id_ax + [id_m]
    knn = kn_ax + [kn_m]

    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - w, rnd, w, label="Random", color="#9aa0a6")
    b2 = ax.bar(x,     idn, w, label="Hypergraph (identity/cluster edges)", color="#f4a36c")
    b3 = ax.bar(x + w, knn, w, label=f"Hypergraph (k-NN edges, k={K})", color="#3b6fb6")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)

    ax.set_xticks(x); ax.set_xticklabels(["reactant A", "reactant B", "reactant C", "macro avg"])
    ax.set_ylabel(r"scaled $L(\pi)$ leakage  (lower is better)")
    ax.set_ylim(0, max(rnd) * 1.18)
    ax.set_title(f"USPTO multicomponent reactions (n={n}, 3 reactant axes)\n"
                 "n-D hypergraph split: k-NN edges beat random and identity grouping")
    # legend outside the axes (upper right) so it never overlaps the bars/labels
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "mcr_benchmark.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", os.path.normpath(out))
    print(f"macro L(pi): random={rnd_m:.4f}  identity={id_m:.4f}  knn={kn_m:.4f}")


if __name__ == "__main__":
    main()
