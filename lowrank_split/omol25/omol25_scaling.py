"""Scaling study on merged OMol25: split TIME and resulting L(pi) vs dataset size,
for low-rank vs hypergraph. Produces two plots.

At each log-spaced size m (nested subsample of the merged set) we run an 80/20
split with each backend, timing it and scoring its L(pi) with the SAME factorized
cosine metric. Hypergraph is skipped above HG_MAX_N (its O(n^2) k-NN becomes
infeasible) — the point where its curve stops is the headline scaling result.

Run (palm env):  python -m PALM.lowrank_split.omol25.omol25_scaling
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PALM.lowrank_split.lowrank_split import balanced_lloyd
from PALM.hypergraph import run_hypergraph_split
from PALM.lowrank_split.omol25 import omol25_leakage as LK

RESULTS = os.path.join(os.path.dirname(__file__), "results")
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, None]  # None = full
HG_MAX_N = 200_000        # hypergraph O(n^2) k-NN feasibility cap
RANK = 256


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, meta = LK.load_features()
    n_full = X.shape[0]
    scale = LK.fit_nonneg_scale(X)
    Xs = np.asarray(X, dtype=np.float32) * scale
    order = np.random.default_rng(0).permutation(n_full)   # fixed nested subsampling

    # warm up CUDA + Mt-KaHyPar init so the first timed point isn't inflated
    _w = np.ascontiguousarray(Xs[order[:500]])
    _B = LK.build_factor(_w, rank=64, seed=0)
    balanced_lloyd(_B, [8, 2], n_iter=3, seed=0)
    run_hypergraph_split({i: _w[i] for i in range(500)}, [8, 2], ["train", "test"],
                         metric="cosine", preset="quality")

    rows = []
    for size in SIZES:
        m = n_full if size is None else min(size, n_full)
        idx = order[:m]
        Xm = np.ascontiguousarray(Xs[idx])

        # low-rank split: time = factor build + balanced assignment
        t = time.time()
        B = LK.build_factor(Xm, rank=min(RANK, m), seed=0)
        lab_lr = balanced_lloyd(B, [8, 2], epsilon=0.0, n_iter=20, seed=0)
        t_lr = time.time() - t
        L_lr = LK.lpi_from_factor(B, lab_lr, 2)

        rng = np.random.default_rng(1)
        rand = np.zeros(m, dtype=int); rand[rng.choice(m, size=m // 5, replace=False)] = 1
        L_rand = LK.lpi_from_factor(B, rand, 2)

        # hypergraph split (feasible only up to the cap)
        t_hg, L_hg = None, None
        if m <= HG_MAX_N:
            fd = {i: Xm[i] for i in range(m)}
            t = time.time()
            hg = run_hypergraph_split(fd, [8, 2], ["train", "test"], k=15,
                                      metric="cosine", preset="quality")
            t_hg = time.time() - t
            lab_hg = np.array([0 if hg[i] == "train" else 1 for i in range(m)])
            L_hg = LK.lpi_from_factor(B, lab_hg, 2)

        rows.append({"n": m, "lowrank_time_s": round(t_lr, 3),
                     "lowrank_lpi": round(L_lr, 4),
                     "hypergraph_time_s": None if t_hg is None else round(t_hg, 3),
                     "hypergraph_lpi": None if L_hg is None else round(L_hg, 4),
                     "random_lpi": round(L_rand, 4)})
        print(f"  n={m:>9,}  lowrank {t_lr:6.2f}s L={L_lr:.4f} | "
              f"hypergraph {'--' if t_hg is None else f'{t_hg:6.2f}s L={L_hg:.4f}'} | "
              f"random L={L_rand:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "omol25_scaling.csv"), index=False)

    # ---- plot 1: split time vs n (log-log) ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(df["n"], df["lowrank_time_s"], "o-", color="#2563eb", label="low-rank")
    hg = df.dropna(subset=["hypergraph_time_s"])
    ax.loglog(hg["n"], hg["hypergraph_time_s"], "s--", color="#dc2626", label="hypergraph")
    if len(hg):
        ax.axvline(hg["n"].max(), color="#dc2626", ls=":", alpha=0.5)
        ax.text(hg["n"].max(), ax.get_ylim()[0] * 1.5,
                "  hypergraph\n  infeasible →", color="#dc2626", fontsize=8, va="bottom")
    ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("split time (s)")
    ax.set_title("OMol25 split time vs dataset size"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "omol25_scaling_time.png"), dpi=150)

    # ---- plot 2: L(pi) vs n ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(df["n"], df["lowrank_lpi"], "o-", color="#2563eb", label="low-rank")
    ax.semilogx(hg["n"], hg["hypergraph_lpi"], "s--", color="#dc2626", label="hypergraph")
    ax.semilogx(df["n"], df["random_lpi"], "^:", color="#6b7280", label="random")
    ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("L(π)  (lower = less leakage)")
    ax.set_title("OMol25 leakage L(π) vs dataset size"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "omol25_scaling_lpi.png"), dpi=150)

    print(f"\nsaved: {RESULTS}/omol25_scaling.csv"
          f"\n       {RESULTS}/omol25_scaling_time.png"
          f"\n       {RESULTS}/omol25_scaling_lpi.png", flush=True)


if __name__ == "__main__":
    main()
