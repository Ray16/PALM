"""n-D comparison: lowrank_nd vs hypergraph_nd / hypergraph_nd_knn / random.

On the one ``kind=nd`` dataset (uspto_mcr, 3-reactant reactions), does the new
``lowrank_nd`` match the hypergraph n-D splitters on macro-axis leakage? The metric
(``macro_axis_lpi``) is recomputed identically from every method's assignment so the
comparison is fair regardless of what each splitter reports.

    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    python -m PALM.lowrank.experiments.nd_compare
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PALM.data.sources import load_dataset
from PALM.splitters import SplitSpec, split
from PALM.splitters.common.leakage_metrics import macro_axis_lpi
from PALM.lowrank.objective import realized_imbalance

HERE = os.path.dirname(__file__)
OUT_CSV = os.path.join(HERE, "nd_compare.csv")
OUT_FIG = os.path.join(HERE, "nd_compare.png")

METHODS = [
    ("lowrank_nd",        dict()),
    ("hypergraph_nd",     dict(preset="deterministic", sim_threshold=0.6)),
    ("hypergraph_nd_knn", dict(preset="deterministic", k=25)),
    ("random",            dict()),
]


def run(seeds=(0, 1, 2), limit=10000):
    b = load_dataset("uspto_mcr", limit=limit, route=True)
    assert b.available, b.reason
    records, afm = b.records, b.axis_feature_maps
    axes = list(afm.keys())
    n = len(records)
    print(f"uspto_mcr: n={n} axes={axes}")
    rows = []
    for method, params in METHODS:
        for seed in seeds:
            t = time.time()
            res = split(method, (records, afm),
                        SplitSpec([8, 2], ["train", "test"], seed=seed), **params)
            dt = time.time() - t
            labels = np.array([0 if res.assignment[i] == "train" else 1 for i in range(n)])
            macro, per_axis = macro_axis_lpi(records, afm, labels)          # fair, uniform metric
            row = dict(method=method, seed=seed, macro_leakage=round(macro, 6),
                       imbalance=round(realized_imbalance(labels, [8, 2]), 4),
                       runtime_s=round(dt, 2))
            row.update({f"axis_{a}": round(per_axis[a], 6) for a in axes})
            rows.append(row)
            print(f"  {method:18s} s{seed} macro={macro:.4f} imb={row['imbalance']} t={dt:.1f}s")
    return rows, axes


def plot(rows, out=OUT_FIG):
    import pandas as pd
    df = pd.DataFrame(rows)
    order = [m for m, _ in METHODS]
    means = df.groupby("method")["macro_leakage"].mean().reindex(order)
    stds = df.groupby("method")["macro_leakage"].std().reindex(order).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(order)), means.values, yerr=stds.values, capsize=4,
           color=["#0072B2", "#E69F00", "#009E73", "#999999"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("macro-axis leakage L(π)  (lower = better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    rows, axes = run()
    fields = ["method", "seed", "macro_leakage", "imbalance", "runtime_s"] + [f"axis_{a}" for a in axes]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    plot(rows)
    print(f"-> {OUT_CSV}\n-> {OUT_FIG}")


if __name__ == "__main__":
    main()
