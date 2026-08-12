"""astartes (Burns et al., JOSS 2023) samplers vs low-rank on MoleculeNet.

astartes is the field-standard dataset-splitting toolbox for chemistry ML.  Its
samplers are heuristic clustering / diversity picks with NO leakage objective,
so the expected story mirrors our other baselines: cheap-ish and leaky, on a
different part of the frontier than low-rank.  This script measures exactly two
axes -- split wall-clock and leakage L(pi) -- for the four samplers astartes
*uniquely* contributes over what we already benchmark (DeepChem / baselines):

    kennard_stone     interpolative, deterministic, O(n^2) distance matrix
    spxy              interpolative in joint X-Y space (needs targets y)
    sphere_exclusion  extrapolative, radius-based, random_state seeded
    optisim           extrapolative, K-dissimilarity, random_state seeded

Scaffold / k-means / Butina are deliberately omitted: astartes wraps the same
underlying algorithms we already score under DeepChem, so including them would
double-count.

FAIRNESS.  astartes clustering cannot hit an exact 20% test block (it emits an
ImperfectSplittingWarning).  Leakage L(pi) is monotone in the test fraction, so
comparing a 13%-test astartes split against an exact-20% low-rank split would be
a confound.  For every astartes method we therefore ALSO run low-rank at that
method's *realized* fraction (`lowrank@matched`) and report both side by side.
The split clock excludes scoring and excludes ECFP featurization (shared
preprocessing for every method, low-rank included).

    conda activate palm
    python -m PALM.benchmarks.moleculenet.benchmark_astartes --datasets freesolv esol ...

Writes results/astartes_benchmark.csv
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# tractable-by-default order (small -> large); the O(n^2) samplers skip past cap
ALL_DATASETS = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
                "lipophilicity", "tox21", "qm8", "hiv", "muv"]

N_SEEDS = 4                 # for the randomized samplers + low-rank restarts
NYSTROM_RANK = 256
FM_MAX_N = 50_000
TEST_FRAC = 0.2
EPSILON = 0.0               # low-rank hits an exact target block (no frac drift)

# astartes samplers we uniquely add.  All four are O(n^2): Kennard-Stone/SPXY
# materialize a full n x n distance matrix, and sphere_exclusion/optisim are
# greedy pairwise diversity picks.  Empirically none is tractable past ~15k on
# these 1024-bit ECFP inputs -- sphere_exclusion/optisim were left churning for
# 30+ min single-threaded on qm8 (22k) and hiv (41k) with GPU idle -- so a
# uniform 15k cap is the honest "astartes does not scale here" boundary.  Above
# it every sampler is reported "not run", which is itself the scaling result.
SAMPLERS = {
    "kennard_stone":    dict(needs_y=False, deterministic=True,  max_n=15_000),
    "spxy":             dict(needs_y=True,  deterministic=True,  max_n=15_000),
    "sphere_exclusion": dict(needs_y=False, deterministic=False, max_n=15_000),
    "optisim":          dict(needs_y=False, deterministic=False, max_n=15_000),
}

CSV_COLS = ["dataset", "n", "method", "lpi_mean", "lpi_std", "time_s",
            "test_frac", "deterministic", "note"]


def load_smiles_y(dataset):
    """SMILES + a scalar target y, aligned under the SAME dedup as load_smiles.

    y is the row-wise mean of the numeric non-SMILES columns (NaNs ignored,
    then filled with the column mean).  SPXY only needs a 1-D target to define
    the joint X-Y distance; the mean-of-tasks is a reasonable stand-in for the
    multi-task sets and exactly the label for the single-task ones.
    """
    from PALM.benchmarks.common.datasets import DATA, SMILES_COL
    col = SMILES_COL.get(dataset, "smiles")
    df = pd.read_csv(os.path.join(DATA, f"{dataset}.csv"))
    df = df.dropna(subset=[col]).drop_duplicates(subset=col).reset_index(drop=True)
    mask = df[col].astype(str).map(lambda s: bool(s) and s != "nan")
    df = df[mask].reset_index(drop=True)
    smiles = df[col].astype(str).tolist()

    num = df.drop(columns=[col]).select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        y = np.zeros(len(smiles), dtype=np.float64)
    else:
        num = num.fillna(num.mean())
        y = num.mean(axis=1).to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=float(np.nanmean(y)))
    return smiles, y


def _run_one_dataset(dataset, gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import logging
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
    from astartes import train_test_split
    from PALM.splitters import SplitSpec, split
    from PALM.benchmarks.common.featurize import ecfp1024
    from PALM.benchmarks.moleculenet.leakage import scaled_lpi

    smiles, y = load_smiles_y(dataset)
    n = len(smiles)
    X = ecfp1024(smiles)                       # shared preprocessing (untimed)
    feature_data = {smiles[i]: X[i] for i in range(n)}

    def score_idx(idx_test):
        test = set(int(i) for i in idx_test)
        d = {smiles[i]: ("test" if i in test else "train") for i in range(n)}
        return scaled_lpi(list(smiles), d)[0]

    def astartes_split(sampler, seed, needs_y):
        """Return (idx_test, split_seconds).  Clock covers only the sampler."""
        kw = dict(sampler=sampler, test_size=TEST_FRAC, train_size=1 - TEST_FRAC,
                  return_indices=True, random_state=seed)
        t0 = time.time()
        if needs_y:
            out = train_test_split(X, y, **kw)
        else:
            out = train_test_split(X, **kw)
        dt = time.time() - t0
        idx_test = np.asarray(out[-1]).ravel()
        return idx_test, dt

    def lowrank_at(frac, seed):
        """Low-rank split targeting `frac` test, exact block (epsilon=0)."""
        te = max(1, int(round(frac * 100)))
        tr = max(1, 100 - te)
        t0 = time.time()
        sp = split("lowrank", feature_data,
                   SplitSpec([tr, te], ["train", "test"], seed=seed, epsilon=EPSILON),
                   rank=NYSTROM_RANK, n_restarts=4, fm=True, fm_max_n=FM_MAX_N).assignment
        dt = time.time() - t0
        realized = sum(1 for v in sp.values() if v == "test") / n
        lpi = scaled_lpi(list(smiles), sp)[0]
        return lpi, dt, realized

    rows = []

    def add(method, lpis, secs, tfrac, deterministic, note=""):
        rows.append({"dataset": dataset, "n": n, "method": method,
                     "lpi_mean": round(float(np.mean(lpis)), 4),
                     "lpi_std": round(float(np.std(lpis)), 4),
                     "time_s": round(float(np.mean(secs)), 4),
                     "test_frac": round(float(tfrac), 4),
                     "deterministic": deterministic, "note": note})

    # -- low-rank anchor at exact 20% (deterministic reference) --------------
    lpi0, dt0, fr0 = lowrank_at(TEST_FRAC, seed=0)
    add("lowrank@0.20", [lpi0], [dt0], fr0, True)

    # -- each astartes sampler + a matched-fraction low-rank point -----------
    for sampler, cfg in SAMPLERS.items():
        if n > cfg["max_n"]:
            add(sampler, [np.nan], [np.nan], np.nan, cfg["deterministic"],
                note=f"skipped n>{cfg['max_n']}")
            continue
        seeds = [0] if cfg["deterministic"] else list(range(N_SEEDS))
        lpis, secs, fracs = [], [], []
        try:
            for s in seeds:
                idx_test, dt = astartes_split(sampler, s, cfg["needs_y"])
                lpis.append(score_idx(idx_test))
                secs.append(dt)
                fracs.append(len(idx_test) / n)
        except Exception as e:                 # degenerate split, memory, etc.
            add(sampler, [np.nan], [np.nan], np.nan, cfg["deterministic"],
                note=f"error: {type(e).__name__}")
            continue
        f_mean = float(np.mean(fracs))
        add(sampler, lpis, secs, f_mean, cfg["deterministic"])

        # low-rank matched to this sampler's realized fraction
        lpi_m, dt_m, fr_m = lowrank_at(f_mean, seed=0)
        add(f"lowrank@{sampler}", [lpi_m], [dt_m], fr_m, True,
            note=f"matched to {sampler} frac")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    print(f"astartes vs low-rank on {len(args.datasets)} datasets "
          f"({N_SEEDS} seeds for randomized samplers)\n")
    t0 = time.time()

    if args.workers == 1:
        out = [_run_one_dataset(d, 0) for d in args.datasets]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        jobs = [(d, i % args.gpus) for i, d in enumerate(args.datasets)]
        with ctx.Pool(args.workers) as pool:
            out = pool.starmap(_run_one_dataset, jobs)

    rows = [r for sub in out for r in sub]
    df = pd.DataFrame(rows, columns=CSV_COLS).sort_values(
        ["n", "method"]).reset_index(drop=True)
    p = os.path.join(RESULTS, "astartes_benchmark.csv")
    df.to_csv(p, index=False)
    print(df.to_string(index=False))
    print(f"\nWall time: {time.time()-t0:.0f}s -> {p}")


if __name__ == "__main__":
    main()
