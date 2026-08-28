"""Fair L(pi) comparison of low-rank vs DataSAIL on OMol25 (structural descriptor,
cosine), on the SAME nested subsamples as omol25_datasail_scaling.py.

DataSAIL cannot hold 80/20 (its test fraction collapses as n grows), and a
smaller test block mechanically lowers L(pi). So for each n we score low-rank
BOTH at exact 80/20 (its real operating point) and at DataSAIL's own test
fraction (matched balance -> the apples-to-apples leakage comparison).

Outputs results/lowrank_vs_datasail_lpi.csv
"""
import os, sys, csv, time
import numpy as np
sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.benchmarks.omol25 import omol25_leakage as LK
from PALM.lowrank import balanced_lloyd, fm_polish

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RANK = 512
SEED = 0

# DataSAIL results (from omol25_datasail_scaling.csv): n -> (datasail_lpi, datasail_test_frac)
DS = {1000: (0.2197, 0.16), 3000: (0.1624, 0.1183), 10000: (0.1001, 0.0745),
      30000: (0.0607, 0.0482), 100000: (0.058, 0.0457)}


def lowrank_lpi(B, test_frac, exact=True):
    """Optimize a 2-way low-rank split at the given test fraction; return (lpi, frac).

    epsilon=0 keeps the realized fraction exactly at the target (no FM drift), so
    the matched-balance comparison to DataSAIL is airtight.
    """
    splits = [1.0 - test_frac, test_frac]
    lab = balanced_lloyd(B, splits, epsilon=0.0, n_iter=25, seed=SEED)
    if len(B) <= 200_000:
        lab, _ = fm_polish(B, lab, splits, epsilon=0.0)
    return LK.lpi_from_factor(B, lab, 2), float(np.mean(lab))


# sizes past DataSAIL's OOM wall get the 80/20 line only (DataSAIL cannot run)
EXTRA = [300_000, 1_000_000, 3_000_000]


def main():
    X, _ = LK.load_features()
    scale = LK.fit_nonneg_scale(X)
    n_full = X.shape[0]
    order = np.random.default_rng(SEED).permutation(n_full)   # SAME order as the scaling scripts

    rows = []
    for n in sorted(DS) + EXTRA:
        idx = np.sort(order[:n])
        Xm = np.ascontiguousarray(np.asarray(X[idx], dtype=np.float32)) * scale
        B = LK.build_factor(Xm, rank=min(RANK, n), seed=SEED)
        lpi_82, f82 = lowrank_lpi(B, 0.20)               # low-rank real operating point
        if n in DS:
            ds_lpi, ds_frac = DS[n]
            lpi_m, fm = lowrank_lpi(B, ds_frac)          # matched to DataSAIL's balance
        else:
            ds_lpi, ds_frac, lpi_m, fm = "", "", "", ""
        print(f"n={n:>7}  DataSAIL={ds_lpi if ds_lpi=='' else f'{ds_lpi:.4f}@{ds_frac:.3f}'} | "
              f"low-rank 80/20={lpi_82:.4f}@{f82:.3f} | "
              f"matched={lpi_m if lpi_m=='' else f'{lpi_m:.4f}@{fm:.3f}'}", flush=True)
        rows.append({"n": n, "datasail_lpi": ds_lpi, "datasail_frac": ds_frac,
                     "lowrank_lpi_8020": round(lpi_82, 4), "lowrank_frac_8020": round(f82, 3),
                     "lowrank_lpi_matched": (lpi_m if lpi_m == "" else round(lpi_m, 4)),
                     "lowrank_frac_matched": (fm if fm == "" else round(fm, 3))})

    with open(os.path.join(RES, "lowrank_vs_datasail_lpi.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote results/lowrank_vs_datasail_lpi.csv")


if __name__ == "__main__":
    main()
