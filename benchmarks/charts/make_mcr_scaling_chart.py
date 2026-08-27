"""Figure: multi-axis split scaling on USPTO-MCR. Two panels from mcr_scaling_results.csv:
(left) split wall-clock vs number of records (log-log, with a linear-scaling
reference); (right) macro scaled L(pi) of the k-NN n-D split vs random, as n
grows. Saves to the manuscript Figs/ directory. Run from the PALM parent dir."""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "..", "results", "mcr_scaling_results.csv")
FIGDIR = os.path.join(HERE, "..", "..", "palm_manuscript", "Figs")
BLUE, GREY = "#3b6fb6", "#9aa0a6"


def main():
    d = pd.read_csv(CSV).sort_values("n")
    n = d["n"].values
    t = d["split_time_s"].values

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    # ── left: split time vs n (log-log) with a linear-scaling guide ──────────
    axL.loglog(n, t, "o-", color=BLUE, lw=2, ms=6, label="PALM (k-NN, k=25)")
    ref = t[0] * (n / n[0])                      # slope-1 (perfectly linear) reference
    axL.loglog(n, ref, "--", color=GREY, lw=1.3, label="linear reference (slope 1)")
    axL.set_xlabel("number of records")
    axL.set_ylabel("split wall-clock (s)")
    axL.legend(fontsize=8, loc="upper left")
    axL.grid(True, which="both", ls=":", alpha=0.4)
    axL.spines[["top", "right"]].set_visible(False)

    # ── right: leakage vs n, k-NN split vs random ────────────────────────────
    axR.plot(n, d["random_macro_lpi"], "s--", color=GREY, lw=2, ms=6, label="Random split")
    axR.plot(n, d["knn_macro_lpi"], "o-", color=BLUE, lw=2, ms=6,
             label="PALM (k-NN, k=25)")
    axR.fill_between(n, d["knn_macro_lpi"], d["random_macro_lpi"], color=BLUE, alpha=0.10)
    axR.set_xscale("log")
    axR.set_xlabel("number of records")
    axR.set_ylabel(r"$L(\pi)$")
    axR.legend(fontsize=8, loc="center left")
    axR.grid(True, which="both", ls=":", alpha=0.4)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "mcr_scaling.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
