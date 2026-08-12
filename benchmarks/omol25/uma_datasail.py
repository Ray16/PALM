"""Run DataSAIL on a subsample of OMol25, using the SAME UMA embeddings as the
low-rank splitter -- no SMILES needed. OMol25 has no valid SMILES for many
structures (metal complexes etc.), which is why DataSAIL was never run there
before; but DataSAIL accepts a precomputed distance matrix via e_dist=(names,
matrix) with e_type="O" (generic/"Other" data), so we can feed it the UMA
embedding distances directly.

Subsamples the already-embedded 100k (_cache_uma/uma_emb.npy) down to N_SUB
structures (DataSAIL's ILP does not scale like the low-rank splitter), builds
a Euclidean distance matrix on the whitened UMA embedding (monotonic with the
RBF similarity uma_leakage.py scores with, so clustering on it is equivalent
neighbor information), runs DataSAIL C1e, and scores BOTH DataSAIL's split and
a low-rank split on the SAME RBF-factor L(pi) for a fair comparison.

Run (palm env):  python -m PALM.benchmarks.omol25.uma_datasail
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging; logging.disable(logging.CRITICAL)
from PALM.splitters.methods.lowrank import balanced_lloyd, lowrank_leakage
from PALM.benchmarks.common.datasail import datasail_distance
from PALM.benchmarks.omol25 import uma_leakage as UL

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache_uma")
RESULTS = os.path.join(HERE, "results")
N_SUB = 2000
MAX_SEC = 1800
SEED = 0


def main():
    os.makedirs(RESULTS, exist_ok=True)
    Z = np.load(os.path.join(CACHE, "uma_emb.npy"))
    n_full = len(Z)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_full, size=N_SUB, replace=False)
    Zs = Z[idx]
    Zs = ((Zs - Zs.mean(0)) / (Zs.std(0) + 1e-8)).astype(np.float32)
    print(f"subsampled {N_SUB} of {n_full} OMol25 UMA-embedded structures", flush=True)

    names = [f"s{i}" for i in range(N_SUB)]
    e_data = {n: n for n in names}

    # Euclidean distance matrix on the whitened UMA embedding: monotonic with
    # the RBF similarity used for scoring, so DataSAIL clusters on equivalent
    # neighbor structure without needing the kernel itself.
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(Zs, metric="euclidean")).astype(np.float64)
    print(f"distance matrix {dist.shape}, running DataSAIL C1e (max_sec={MAX_SEC}) ...", flush=True)

    t0 = time.time()
    # default epsilon=0.05 was INFEASIBLE at this n (SCIP: "Problem status: infeasible") --
    # the automatic clustering (default 50 clusters) produces cluster sizes too lopsided
    # to hit an 80/20 split within 5% slack. Relaxing epsilon and using more/finer
    # clusters (verified in a debug run: n=2000 solves in ~10s at epsilon=0.2/e_clusters=100,
    # landing at 84/16) makes the ILP feasible.
    ds_split = datasail_distance(names, dist, [8, 2], ["train", "test"], max_sec=MAX_SEC,
                                 epsilon=0.2, e_clusters=100)
    dt_ds = time.time() - t0
    ds_lab = np.array([0 if ds_split.get(n, "train") == "train" else 1 for n in names])
    ds_frac = float(ds_lab.mean())
    print(f"DataSAIL done in {dt_ds:.1f}s, test_frac={ds_frac:.3f}", flush=True)

    # low-rank split on the SAME subsample, SAME RBF factor used to score both
    sigma = UL.median_sigma(Zs)
    B = UL.rbf_factor(Zs, rank=256, sigma=sigma, seed=0)
    t0 = time.time()
    best_lab, best_obj = None, np.inf
    for r in range(4):
        lab = balanced_lloyd(B, [8, 2], epsilon=0.0, n_iter=20, seed=r)
        obj = lowrank_leakage(B, lab, 2)
        if obj < best_obj:
            best_obj, best_lab = obj, lab
    dt_lr = time.time() - t0
    lr_frac = float(best_lab.mean())
    print(f"low-rank done in {dt_lr:.2f}s, test_frac={lr_frac:.3f}", flush=True)

    lpi_ds = UL.lpi_from_factor(B, ds_lab, 2)
    lpi_lr = UL.lpi_from_factor(B, best_lab, 2)
    rnd = np.random.default_rng(1).permutation(np.array([0]*int(0.8*N_SUB) + [1]*(N_SUB-int(0.8*N_SUB))))
    lpi_rand = UL.lpi_from_factor(B, rnd, 2)

    print(f"\n=== OMol25 (n={N_SUB}, UMA-RBF similarity) ===")
    print(f"  random    L(pi)={lpi_rand:.4f}")
    print(f"  DataSAIL  L(pi)={lpi_ds:.4f}  time={dt_ds:.1f}s  test_frac={ds_frac:.3f}")
    print(f"  low-rank  L(pi)={lpi_lr:.4f}  time={dt_lr:.2f}s  test_frac={lr_frac:.3f}")

    out = {"n": N_SUB, "sigma": float(sigma),
           "lpi_random": float(lpi_rand),
           "datasail": {"lpi": float(lpi_ds), "time_s": round(dt_ds, 1), "test_frac": ds_frac},
           "lowrank": {"lpi": float(lpi_lr), "time_s": round(dt_lr, 2), "test_frac": lr_frac}}
    outp = os.path.join(RESULTS, "omol25_uma_datasail.json")
    json.dump(out, open(outp, "w"), indent=2)
    print("\nsaved", outp)


if __name__ == "__main__":
    main()
