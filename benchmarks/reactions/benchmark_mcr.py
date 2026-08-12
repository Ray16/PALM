"""n-D hypergraph split on the USPTO MCR dataset: k-NN vs cluster vs random.

This is the high-cardinality, near-unique multi-component case (3 reactant axes,
~91% of reactants unique). It contrasts two n-D constructions against random:
  - k-NN  : per-axis record-level k nearest-neighbour hyperedges (the 1D
            construction applied per axis). Tracks scaled L(pi) directly.
  - ident : identity/cluster hyperedges (run_hypergraph_split_nd). With near-
            unique values these form almost no non-trivial groups, so the split
            is ~ random -- the informative negative control.

Run (boltz-2 env, from PALM parent):  python -m PALM.benchmarks.reactions.benchmark_mcr
Writes benchmark/mcr_results.csv.
"""

import csv
import os
import time

from PALM import reactions as R
from PALM.splitters import SplitSpec, split
from .benchmark_reactions import evaluate, random_split

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "mcr_results.csv")
K = 25


def main():
    records, afm, _ = R.load_uspto_mcr()
    n = len(records)
    axes = ";".join(afm.keys())

    r_lpi, r_id, _ = evaluate(records, afm, random_split(n))

    t0 = time.time()
    r_ident = split("hypergraph_nd", (records, afm),
                    SplitSpec([8, 2], ["train", "test"], seed=42), sim_threshold=1.0)
    a_ident = [r_ident.assignment[i] for i in range(n)]
    t_ident = round(time.time() - t0, 1)
    i_lpi, i_id, _ = evaluate(records, afm, a_ident)

    t0 = time.time()
    r_knn = split("hypergraph_nd_knn", (records, afm),
                  SplitSpec([8, 2], ["train", "test"], seed=42), k=K)
    a_knn = [r_knn.assignment[i] for i in range(n)]
    info = r_knn.diagnostics
    t_knn = round(time.time() - t0, 1)
    k_lpi, k_id, k_axis = evaluate(records, afm, a_knn)

    rows = [
        ["random", n, len(afm), axes, r_lpi, r_id, "-"],
        ["hypergraph_identity", n, len(afm), axes, i_lpi, i_id, t_ident],
        [f"hypergraph_knn_k{K}", n, len(afm), axes, k_lpi, k_id, t_knn],
    ]
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "n_axes", "axes", "macro_lpi", "macro_id_leak", "time_s"])
        w.writerows(rows)

    print(f"USPTO MCR  n={n}, axes=({axes})\n")
    print(f"  random:               macro L(pi)={r_lpi:.4f}")
    print(f"  hypergraph identity:  macro L(pi)={i_lpi:.4f}   (cluster/identity edges; ~random on near-unique axes)")
    print(f"  hypergraph k-NN k={K}: macro L(pi)={k_lpi:.4f}   t={t_knn}s  KM1={info['km1']}")
    for a, m in k_axis.items():
        print(f"      {a}: L(pi)={m['lpi']:.4f}")
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
