"""DataSAIL vs low-rank scaling on OMol25 itself -- same nested subsamples as
omol25_scaling.py's low-rank curve (identical seed/order) and the same 115-d
structural descriptor + cosine similarity (omol25_leakage.load_features), so
points are directly comparable to the blue line in
poster/figures/fig_omol25_scaling_to_9-55M.png. This replaces that figure's
current DataSAIL markers, which are actually MoleculeNet/ECFP borrowed in only
for a rough scale reference.

DataSAIL's C1e technique needs SMILES; OMol25 has none (metal complexes RDKit
can't parse), so -- same trick as uma_datasail.py -- we feed it a precomputed
distance matrix via e_type="O"/e_dist. Rows are L2-normalized before the
Euclidean distance is built, so distance is monotonic with the cosine
similarity the low-rank curve and L(pi) metric already use.

The distance matrix is O(n^2) memory (float32): ~3.6GB at n=30k, ~40GB at
n=100k. Sizes are attempted in increasing order, each capped at MAX_SEC; we
stop after the first timeout/infeasible/OOM since difficulty only grows with
n from there (mirrors how muv is already shown as "timed out" rather than
guessed at on the MoleculeNet chart).

Run (palm env):  python -m PALM.lowrank_split.omol25.omol25_datasail_scaling
"""
import os
import sys
import csv
import time

import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
from PALM.lowrank_split.omol25 import omol25_leakage as LK

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RESULTS, "omol25_datasail_scaling.csv")
FIELDS = ["n", "datasail_s", "test_frac", "lpi", "status"]

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
RANK = 256
MAX_SEC = 7200          # 2 hr cap per size
SEED = 0


def _write(rows):
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def datasail_split(Xm):
    """Run DataSAIL C1e on a precomputed distance matrix. Returns (dt, labels|None)."""
    n = len(Xm)
    names = [f"s{i}" for i in range(n)]
    e_data = {nm: nm for nm in names}
    Xn = Xm / (np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-12)
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(Xn.astype(np.float32), metric="euclidean")).astype(np.float32)

    from datasail.sail import datasail
    t0 = time.time()
    e_s, _, _ = datasail(techniques=["C1e"], splits=[8, 2], names=["train", "test"],
                         e_type="O", e_data=e_data, e_dist=(names, dist),
                         max_sec=MAX_SEC, epsilon=0.2, e_clusters=100)
    dt = time.time() - t0
    if e_s.get("C1e") is None:
        return dt, None
    split = e_s["C1e"][0]
    lab = np.array([0 if split.get(nm, "train") == "train" else 1 for nm in names])
    return dt, lab


def main():
    os.makedirs(RESULTS, exist_ok=True)
    X, meta = LK.load_features()
    n_full = X.shape[0]
    scale = LK.fit_nonneg_scale(X)
    Xs = np.asarray(X, dtype=np.float32) * scale
    order = np.random.default_rng(SEED).permutation(n_full)   # SAME order as omol25_scaling.py

    rows = []
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as fh:
            rows = list(csv.DictReader(fh))
    have = {int(r["n"]) for r in rows}

    for size in SIZES:
        if size in have:
            print(f"n={size}: already have a result, skipping", flush=True)
            continue
        idx = np.sort(order[:size])
        Xm = np.ascontiguousarray(Xs[idx])
        print(f"n={size}: distance matrix + DataSAIL C1e (max_sec={MAX_SEC}) ...",
              flush=True)
        try:
            dt, lab = datasail_split(Xm)
        except MemoryError as e:
            print(f"  OOM building/solving at n={size}: {e}", flush=True)
            rows.append({"n": size, "datasail_s": "", "test_frac": "", "lpi": "",
                        "status": "oom"})
            _write(rows)
            break
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"n": size, "datasail_s": "", "test_frac": "", "lpi": "",
                        "status": "error"})
            _write(rows)
            break
        if lab is None:
            print(f"  infeasible/timed out after {dt:.0f}s", flush=True)
            rows.append({"n": size, "datasail_s": round(dt, 1), "test_frac": "",
                        "lpi": "", "status": "timeout"})
            _write(rows)
            break
        B = LK.build_factor(Xm, rank=min(RANK, size), seed=0)   # SAME as the low-rank curve
        lpi = LK.lpi_from_factor(B, lab, 2)
        frac = float(lab.mean())
        print(f"  done in {dt:.1f}s  test_frac={frac:.3f}  lpi={lpi:.4f}", flush=True)
        rows.append({"n": size, "datasail_s": round(dt, 1), "test_frac": round(frac, 4),
                    "lpi": round(lpi, 4), "status": "ok"})
        _write(rows)

    print("done ->", OUT_CSV)


if __name__ == "__main__":
    main()
