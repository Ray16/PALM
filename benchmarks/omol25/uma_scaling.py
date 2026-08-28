"""UMA-embedding scaling study: split TIME and resulting L(pi) vs dataset size,
low-rank vs hypergraph, on the LEARNED UMA-RBF similarity.

Unlike omol25_scaling.py (hand descriptor, cosine, n up to 9.55M), this runs on
the mean-pooled UMA-small embeddings, so it is CAPPED at the embedded 100k
subsample (embedding the full 9.55M with UMA is a multi-day job). Similarity for
scoring L(pi) is the RBF kernel over the whitened UMA embedding (sigma = median
pairwise distance), the same metric as uma_leakage.py. Overwrites the two scaling
PNGs with these UMA versions; the full-scale hand-descriptor data is preserved in
omol25_scaling.csv.

Run (palm env):  python -m PALM.benchmarks.omol25.uma_scaling
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PALM.splitters import SplitSpec, split
from PALM.lowrank import balanced_lloyd
from PALM.benchmarks.common.timing import _sync, _time
from PALM.benchmarks.omol25 import uma_leakage as UL

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache_uma")
RESULTS = os.path.join(HERE, "results")
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]   # capped at the embedded subsample
RANK = 256
REPS_SMALL = 5
SMALL_N = 100_000     # all sizes here are repeated (<=100k); larger would run once


def main():
    os.makedirs(RESULTS, exist_ok=True)
    Z = np.load(os.path.join(CACHE, "uma_emb.npy"))
    Z = ((Z - Z.mean(0)) / (Z.std(0) + 1e-8)).astype(np.float32)
    n_full = len(Z)
    sigma = UL.median_sigma(Z)                       # fixed similarity across all n
    print(f"UMA emb {Z.shape}; RBF sigma={sigma:.3f}", flush=True)
    order = np.random.default_rng(0).permutation(n_full)

    # warm up CUDA + Mt-KaHyPar with the real rank/n_iter so the first point isn't inflated
    _w = np.ascontiguousarray(Z[order[:2000]])
    _B = UL.rbf_factor(_w, rank=RANK, sigma=sigma, seed=0)
    balanced_lloyd(_B, [8, 2], epsilon=0.0, n_iter=20, seed=0)
    split("hypergraph", {i: _w[i] for i in range(2000)},
          SplitSpec([8, 2], ["train", "test"], seed=42),
          k=15, metric="cosine", preset="quality")

    rows = []
    for m in SIZES:
        idx = order[:m]
        Zm = np.ascontiguousarray(Z[idx])
        reps = REPS_SMALL if m <= SMALL_N else 1

        def lr_run():
            B_ = UL.rbf_factor(Zm, rank=min(RANK, m), sigma=sigma, seed=0)
            lab_ = balanced_lloyd(B_, [8, 2], epsilon=0.0, n_iter=20, seed=0)
            return B_, lab_
        t_lr, (B, lab_lr) = _time(lr_run, reps)
        L_lr = UL.lpi_from_factor(B, lab_lr, 2)
        lr_frac = float(np.mean(lab_lr == 1))

        rng = np.random.default_rng(1)
        rand = np.zeros(m, dtype=int); rand[rng.choice(m, size=m // 5, replace=False)] = 1
        L_rand = UL.lpi_from_factor(B, rand, 2)

        # hypergraph (cosine kNN cut), median over `reps` seeds.
        # k scales with n: in a fixed embedding volume at fixed sigma, the number of
        # genuinely-similar neighbors grows ~proportional to density (~n), so a fixed
        # k captures a shrinking similarity fraction as n grows (=> its L(pi) rises
        # spuriously). Anchor k=15 where it sufficed (n=10k) and grow k ∝ n.
        k_n = max(15, int(round(m * 0.0015)))
        ts, Ls, fracs = [], [], []
        fd = {i: Zm[i] for i in range(m)}
        for r in range(reps):
            _sync(); t0 = time.time()
            hg = split("hypergraph", fd,
                       SplitSpec([8, 2], ["train", "test"], seed=42 + r),
                       k=k_n, metric="cosine", preset="quality").assignment
            _sync(); ts.append(time.time() - t0)
            lab_hg = np.fromiter((0 if hg[i] == "train" else 1 for i in range(m)), int, m)
            Ls.append(UL.lpi_from_factor(B, lab_hg, 2))
            fracs.append(float(np.mean(lab_hg == 1)))
        t_hg, L_hg = float(np.median(ts)), float(np.median(Ls))
        L_hg_std, hg_frac = float(np.std(Ls)), float(np.median(fracs))

        rows.append({"n": m, "reps": reps,
                     "lowrank_time_s": round(t_lr, 3), "lowrank_lpi": round(L_lr, 4),
                     "lowrank_test_frac": round(lr_frac, 4),
                     "hypergraph_k": k_n,
                     "hypergraph_time_s": round(t_hg, 3), "hypergraph_lpi": round(L_hg, 4),
                     "hypergraph_lpi_std": round(L_hg_std, 4),
                     "hypergraph_test_frac": round(hg_frac, 4),
                     "random_lpi": round(L_rand, 4)})
        print(f"  n={m:>7,} x{reps}  lowrank {t_lr:7.3f}s L={L_lr:.4f} frac={lr_frac:.3f} | "
              f"hypergraph k={k_n:>3} {t_hg:6.2f}s L={L_hg:.4f}±{L_hg_std:.4f} frac={hg_frac:.3f} | "
              f"random L={L_rand:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "uma_scaling.csv"), index=False)

    SUB = "learned UMA-RBF similarity (100k subsample; UMA embeddings unavailable beyond this)"

    # ---- plot 1: split time vs n (log-log) ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(df["n"], df["lowrank_time_s"], "o-", color="#2563eb", label="low-rank")
    ax.loglog(df["n"], df["hypergraph_time_s"], "s--", color="#dc2626", label="hypergraph")
    ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("split time (s)")
    ax.set_title(f"OMol25 split time vs n\n{SUB}", fontsize=9.5)
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    cap = (f"low-rank time = Nystrom RBF factor + one balanced assignment "
           f"(median of {REPS_SMALL} runs). Splitter time is embedding-agnostic\n"
           "(depends on n x rank), so it mirrors the hand-descriptor scaling; only the "
           "range differs (capped at the 100k embedded set).")
    fig.text(0.01, -0.02, cap, fontsize=6.5, ha="left", va="top", color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "omol25_scaling_time_uma.png"), dpi=300, bbox_inches="tight")

    # ---- plot 2: L(pi) vs n ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(df["n"], df["lowrank_lpi"], "o-", color="#2563eb", label="low-rank")
    ax.errorbar(df["n"], df["hypergraph_lpi"], yerr=df["hypergraph_lpi_std"],
                fmt="s--", color="#dc2626", label="hypergraph (k∝n, median±std)", capsize=3)
    ax.semilogx(df["n"], df["random_lpi"], "^:", color="#6b7280", label="random")
    ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("L(π)  (lower = less leakage)")
    ax.set_title(f"OMol25 leakage vs n\n{SUB}", fontsize=9.5)
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    cap = ("L(π) scored on the RBF kernel over whitened UMA-small embeddings "
           "(sigma = median pairwise dist), 80/20 split.\nHypergraph k scales with n "
           "(k=max(15, 0.0015·n)): a FIXED k captures a shrinking similarity fraction as "
           "density grows, spuriously\nraising its L(π); low-rank needs no such tuning "
           "(it factorizes the full matrix). Estimator validated exact vs O(n²).")
    fig.text(0.01, -0.02, cap, fontsize=6.5, ha="left", va="top", color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "omol25_scaling_lpi_uma.png"), dpi=300, bbox_inches="tight")

    print(f"\nsaved NEW UMA plots (300 dpi; originals kept):"
          f"\n  {RESULTS}/omol25_scaling_time_uma.png"
          f"\n  {RESULTS}/omol25_scaling_lpi_uma.png"
          f"\n  data -> {RESULTS}/uma_scaling.csv", flush=True)


if __name__ == "__main__":
    main()
