"""DataSAIL in FINE mode (tuned to hold ~20% test) on OMol25, at the small n
where it is feasible -- the FAIR matched-ratio comparison to low-rank.

The scaling sweep (omol25_datasail_scaling.py) ran DataSAIL fast/coarse
(e_clusters=100), which collapses the test fraction to ~5%; that makes L(pi)
look artificially low and isn't the same 20% split low-rank does. Here we use
fine clustering + tight epsilon so DataSAIL actually lands near 20%, recording
its TRUE time and a comparable L(pi). Same structural-descriptor cosine
similarity and same nested subsample (seed 0) as the low-rank scaling curve.

Run (palm env, DataSAIL importable there):
    python -m PALM.benchmarks.omol25.omol25_datasail_finemode
Writes results/omol25_datasail_finemode.csv
"""
import os, sys, csv, time
import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
from PALM.benchmarks.common.datasail import datasail_distance
from PALM.benchmarks.omol25 import omol25_leakage as LK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "omol25_datasail_finemode.csv")
SIZES = [1000, 3000, 10000]
MAX_SEC = 21600           # 6 h cap per n (overnight; n=10k fine-ILP is heavy)
EPSILON = 0.02            # tight -> forces close to the 20% target
SEED = 0


def datasail_split(Xm, e_clusters, epsilon, max_sec):
    n = len(Xm)
    names = [f"s{i}" for i in range(n)]
    e_data = {nm: nm for nm in names}
    Xn = Xm / (np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-12)
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(Xn.astype(np.float32), metric="euclidean")).astype(np.float32)
    t0 = time.time()
    try:
        assign = datasail_distance(names, dist, [8, 2], ["train", "test"],
                                   max_sec=max_sec, epsilon=epsilon, e_clusters=e_clusters)
    except RuntimeError:
        return time.time() - t0, None
    dt = time.time() - t0
    lab = np.array([0 if assign.get(nm, "train") == "train" else 1 for nm in names])
    return dt, lab


def main():
    X, meta = LK.load_features()
    n_full = X.shape[0]
    scale = LK.fit_nonneg_scale(X)
    Xs = np.asarray(X, dtype=np.float32) * scale
    order = np.random.default_rng(SEED).permutation(n_full)   # SAME as low-rank curve

    rows = []
    if os.path.exists(OUT):
        rows = list(csv.DictReader(open(OUT)))
    have = {int(r["n"]) for r in rows}

    for n in SIZES:
        if n in have:
            print(f"n={n}: already done, skipping", flush=True)
            continue
        idx = np.sort(order[:n])
        Xm = np.ascontiguousarray(Xs[idx])
        B = LK.build_factor(Xm, rank=min(256, n), seed=0)   # SAME cosine factor as low-rank
        # fine clustering: ~4 points/cluster, capped so the ILP stays tractable
        e_clusters = min(n // 4, 2000)
        print(f"n={n}: DataSAIL fine (e_clusters={e_clusters}, eps={EPSILON}, "
              f"max_sec={MAX_SEC}) ...", flush=True)
        try:
            dt, lab = datasail_split(Xm, e_clusters, EPSILON, MAX_SEC)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"n": n, "e_clusters": e_clusters, "datasail_s": "",
                         "test_frac": "", "lpi": "", "status": f"error:{type(e).__name__}"})
            _write(rows); continue
        if lab is None:
            print(f"  infeasible/timeout after {dt:.0f}s", flush=True)
            rows.append({"n": n, "e_clusters": e_clusters, "datasail_s": round(dt, 1),
                         "test_frac": "", "lpi": "", "status": "timeout"})
            _write(rows); continue
        lpi = LK.lpi_from_factor(B, lab, 2)
        frac = float(lab.mean())
        print(f"  done {dt:.1f}s  test_frac={frac:.3f}  lpi={lpi:.4f}", flush=True)
        rows.append({"n": n, "e_clusters": e_clusters, "datasail_s": round(dt, 1),
                     "test_frac": round(frac, 4), "lpi": round(lpi, 4), "status": "ok"})
        _write(rows)
    print("done ->", OUT, flush=True)


def _write(rows):
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "e_clusters", "datasail_s",
                                           "test_frac", "lpi", "status"])
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
