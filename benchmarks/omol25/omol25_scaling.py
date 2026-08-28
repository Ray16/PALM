"""Scaling study on merged OMol25: split TIME and resulting L(pi) vs dataset size,
for low-rank vs hypergraph. Produces two plots.

At each log-spaced size m (nested subsample of the merged set) we run an 80/20
split with each backend, timing it and scoring its L(pi) with the SAME factorized
cosine metric. Hypergraph is skipped above HG_MAX_N (its O(n^2) k-NN becomes
infeasible) — the point where its curve stops is the headline scaling result.

Methodology notes (see also the plot captions):
  * Timing is the MEDIAN of REPS_SMALL CUDA-synced runs for n <= SMALL_N (where
    per-call launch/init overhead dominates and single-shot times are noisy);
    a single run above that (there the O(n) work dominates the noise).
  * The warm-up uses the SAME rank/n_iter as the real runs so the first timed
    point isn't inflated by first-shape CUDA kernel init.
  * The low-rank time is the core primitive (Nystrom factor + one balanced
    assignment). The production `run_lowrank_split` adds k-means++ landmarks,
    4 restarts and an FM polish — a constant factor higher, still O(n).
  * Hypergraph is run REPS_SMALL times (varying seed) at each small size; we
    report the median L(pi) with its spread, plus the realized test fraction,
    so Mt-KaHyPar's run-to-run variance and balance drift are visible rather
    than hidden in a single point.

Run (palm/boltz-2 env):  python -m PALM.benchmarks.omol25.omol25_scaling
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PALM.splitters import SplitSpec, split
from PALM.lowrank import balanced_lloyd
from PALM.benchmarks.common.timing import _sync, _time
from PALM.benchmarks.omol25 import omol25_leakage as LK

RESULTS = os.path.join(os.path.dirname(__file__), "results")
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, None]  # None = full
HG_MAX_N = 200_000        # hypergraph O(n^2) k-NN feasibility cap
RANK = 256
REPS_SMALL = 5            # timing / hypergraph repeats where launch overhead & run-to-run variance matter
SMALL_N = 100_000         # sizes <= this are repeated; larger sizes run once (O(n) work >> noise)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, meta = LK.load_features()
    n_full = X.shape[0]
    scale = LK.fit_nonneg_scale(X)
    Xs = np.asarray(X, dtype=np.float32) * scale
    order = np.random.default_rng(0).permutation(n_full)   # fixed nested subsampling

    # Warm up CUDA + Mt-KaHyPar with the SAME rank/n_iter as the real runs, so the
    # first timed point isn't inflated by first-shape kernel autotuning. 2000 pts
    # so the rank-256 eigh/GEMM shapes match the real path.
    _w = np.ascontiguousarray(Xs[order[:2000]])
    _B = LK.build_factor(_w, rank=RANK, seed=0)
    balanced_lloyd(_B, [8, 2], epsilon=0.0, n_iter=20, seed=0)
    split("hypergraph", {i: _w[i] for i in range(2000)},
          SplitSpec([8, 2], ["train", "test"], seed=42),
          k=15, metric="cosine", preset="quality")

    rows = []
    for size in SIZES:
        m = n_full if size is None else min(size, n_full)
        idx = order[:m]
        Xm = np.ascontiguousarray(Xs[idx])
        reps = REPS_SMALL if m <= SMALL_N else 1

        # low-rank split (deterministic): time = factor build + balanced assignment,
        # median over `reps` runs. Realized test fraction is exact (epsilon=0).
        def lr_run():
            B_ = LK.build_factor(Xm, rank=min(RANK, m), seed=0)
            lab_ = balanced_lloyd(B_, [8, 2], epsilon=0.0, n_iter=20, seed=0)
            return B_, lab_
        t_lr, (B, lab_lr) = _time(lr_run, reps)
        L_lr = LK.lpi_from_factor(B, lab_lr, 2)
        lr_frac = float(np.mean(lab_lr == 1))

        rng = np.random.default_rng(1)
        rand = np.zeros(m, dtype=int); rand[rng.choice(m, size=m // 5, replace=False)] = 1
        L_rand = LK.lpi_from_factor(B, rand, 2)

        # hypergraph split (feasible only up to the cap): run `reps` times with
        # different seeds -> median time, median L(pi) + spread, realized test
        # fraction (Mt-KaHyPar balance can drift within its +/-epsilon corridor).
        t_hg = L_hg = L_hg_std = hg_frac = None
        if m <= HG_MAX_N:
            fd = {i: Xm[i] for i in range(m)}
            ts, Ls, fracs = [], [], []
            for r in range(reps):
                _sync(); t0 = time.time()
                hg = split("hypergraph", fd,
                           SplitSpec([8, 2], ["train", "test"], seed=42 + r),
                           k=15, metric="cosine", preset="quality").assignment
                _sync(); ts.append(time.time() - t0)
                lab_hg = np.fromiter((0 if hg[i] == "train" else 1 for i in range(m)), int, m)
                Ls.append(LK.lpi_from_factor(B, lab_hg, 2))
                fracs.append(float(np.mean(lab_hg == 1)))
            t_hg = float(np.median(ts)); L_hg = float(np.median(Ls))
            L_hg_std = float(np.std(Ls)); hg_frac = float(np.median(fracs))

        rows.append({"n": m, "reps": reps,
                     "lowrank_time_s": round(t_lr, 3), "lowrank_lpi": round(L_lr, 4),
                     "lowrank_test_frac": round(lr_frac, 4),
                     "hypergraph_time_s": None if t_hg is None else round(t_hg, 3),
                     "hypergraph_lpi": None if L_hg is None else round(L_hg, 4),
                     "hypergraph_lpi_std": None if L_hg_std is None else round(L_hg_std, 4),
                     "hypergraph_test_frac": None if hg_frac is None else round(hg_frac, 4),
                     "random_lpi": round(L_rand, 4)})
        hg_str = "--" if t_hg is None else \
            f"{t_hg:6.2f}s L={L_hg:.4f}±{L_hg_std:.4f} frac={hg_frac:.3f}"
        print(f"  n={m:>9,} x{reps}  lowrank {t_lr:7.3f}s L={L_lr:.4f} frac={lr_frac:.3f} | "
              f"hypergraph {hg_str} | random L={L_rand:.4f}", flush=True)

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
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    cap = (f"low-rank time = Nystrom factor + one balanced assignment "
           f"(median of {REPS_SMALL} runs for n<={SMALL_N:,}, single run above).\n"
           "Full run_lowrank_split (k-means++ landmarks, 4 restarts, FM polish) is a "
           "constant factor higher, still O(n);\nuniform landmarks used here vs k-means++ "
           "in the library (minor fidelity gap).")
    fig.text(0.01, -0.02, cap, fontsize=6.5, ha="left", va="top", color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "omol25_scaling_time.png"), dpi=300, bbox_inches="tight")

    # ---- plot 2: L(pi) vs n ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(df["n"], df["lowrank_lpi"], "o-", color="#2563eb", label="low-rank")
    hgl = df.dropna(subset=["hypergraph_lpi"])
    ax.errorbar(hgl["n"], hgl["hypergraph_lpi"], yerr=hgl["hypergraph_lpi_std"],
                fmt="s--", color="#dc2626", label="hypergraph (median±std)", capsize=3)
    ax.semilogx(df["n"], df["random_lpi"], "^:", color="#6b7280", label="random")
    ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("L(π)  (lower = less leakage)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "omol25_scaling_lpi.png"), dpi=300, bbox_inches="tight")

    print(f"\nsaved: {RESULTS}/omol25_scaling.csv"
          f"\n       {RESULTS}/omol25_scaling_time.png"
          f"\n       {RESULTS}/omol25_scaling_lpi.png", flush=True)


if __name__ == "__main__":
    main()
