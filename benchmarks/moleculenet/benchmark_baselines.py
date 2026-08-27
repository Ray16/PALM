"""Baseline splitters vs low-rank on MoleculeNet, one scorer, exact 80/20.

Adds the splitters practitioners actually use -- scaffold (Bemis-Murcko), Butina
cluster split, and a k-means cluster split -- alongside random and low-rank, all
scored with the *same* ``scaled_lpi`` (ECFP-1024 / Tanimoto) at the same target
ratio.  Purpose is the leakage-vs-time frontier, not a single winner: the fast
heuristics are expected to be cheap and leaky, DataSAIL slow and clean, and
low-rank the only point that is both.

Every method reports its *realized* test fraction, because a smaller test block
mechanically lowers L(pi) and that confound has bitten this comparison before.

    python -m PALM.benchmarks.moleculenet.benchmark_baselines --workers 8 --gpus 8

Writes results/baseline_benchmark.csv
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

ALL_DATASETS = ["freesolv", "esol", "clintox", "sider", "bace", "bbbp",
                "lipophilicity", "tox21", "qm8", "hiv", "muv"]

N_SEEDS = 4
NYSTROM_RANK = 256
FM_MAX_N = 50_000
TEST_FRAC = 0.2
EPSILON = 0.0               # exact 80/20 for every method -- no balance confound
BUTINA_CUTOFF = 0.65        # Tanimoto distance cutoff (0.35 similarity)
BUTINA_MAX_N = 25_000       # O(n^2) neighbour lists + per-row loop; skip above this
KMEANS_K = 100              # clusters, bin-packed into blocks

CSV_COLS = ["dataset", "n", "method", "lpi_mean", "lpi_std",
            "time_s", "test_frac", "deterministic"]


# ── split helpers ───────────────────────────────────────────────────────────

def _bin_pack(groups, n, test_frac=TEST_FRAC):
    """Greedily fill the test block with whole groups until its quota is met.

    ``groups`` is a list of index-arrays.  Largest-first so the realized ratio
    lands close to target; the realized fraction is returned for auditing.
    """
    quota = int(round(n * test_frac))
    order = sorted(groups, key=len, reverse=True)
    labels = np.zeros(n, dtype=np.int64)      # 0 = train
    filled = 0
    # walk smallest-first for the tail so we can land near the quota exactly
    for g in sorted(order, key=len):
        if filled >= quota:
            break
        if filled + len(g) <= quota:
            labels[g] = 1
            filled += len(g)
    # top up from the largest untouched group if still short (keeps exactness)
    if filled < quota:
        for g in order:
            if labels[g].any():
                continue
            need = quota - filled
            take = g[:need]
            labels[take] = 1
            filled += len(take)
            if filled >= quota:
                break
    return labels, filled / n


def scaffold_groups(smiles):
    """Bemis-Murcko scaffold groups (SMILES only)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    buckets = {}
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        key = "" if m is None else MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        buckets.setdefault(key, []).append(i)
    return [np.asarray(v) for v in buckets.values()]


def butina_groups(X, cutoff=BUTINA_CUTOFF, block=2048):
    """Taylor-Butina clustering on Tanimoto distance, GPU-chunked.

    Works on any non-negative feature matrix; for binary ECFP this is the
    standard Tanimoto Butina clustering.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    card = Xt.sum(1)
    n = Xt.shape[0]
    thresh = 1.0 - cutoff        # similarity threshold

    neigh = []
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = Xt[s:e] @ Xt.T
        union = card[s:e][:, None] + card[None, :] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        for r in range(e - s):
            idx = torch.nonzero(sim[r] >= thresh, as_tuple=False).flatten()
            neigh.append(idx.cpu().numpy())
    order = np.argsort([-len(a) for a in neigh])
    assigned = np.zeros(n, dtype=bool)
    groups = []
    for c in order:
        if assigned[c]:
            continue
        members = np.asarray([i for i in neigh[c] if not assigned[i]])
        if members.size == 0:
            continue
        assigned[members] = True
        groups.append(members)
    leftover = np.nonzero(~assigned)[0]
    groups.extend([np.asarray([i]) for i in leftover])
    return groups


def kmeans_groups(X, k=KMEANS_K, seed=0, iters=25):
    """k-means in feature space; each cluster is one indivisible group."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    n = Xt.shape[0]
    k = min(k, n)
    g = torch.Generator(device=dev).manual_seed(seed)
    cent = Xt[torch.randperm(n, generator=g, device=dev)[:k]].clone()
    lab = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(iters):
        d = torch.cdist(Xt, cent)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                cent[j] = Xt[m].mean(0)
    lab = lab.cpu().numpy()
    return [np.nonzero(lab == j)[0] for j in range(k) if (lab == j).any()]


# ── per-dataset worker ──────────────────────────────────────────────────────

def _run_one_dataset(dataset: str, gpu_id: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import logging
    logging.disable(logging.CRITICAL)

    sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
    from PALM.splitters import SplitSpec, split as run_split
    from PALM.benchmarks.common.datasets import load_smiles
    from PALM.benchmarks.common.featurize import ecfp1024
    from PALM.benchmarks.common.random_split import random_labels
    from PALM.benchmarks.moleculenet.leakage import scaled_lpi

    smiles = load_smiles(dataset)
    n = len(smiles)
    X = ecfp1024(smiles)

    def score_labels(labels):
        d = {smiles[i]: ("test" if labels[i] else "train") for i in range(n)}
        return scaled_lpi(list(smiles), d)[0]

    rows = []

    def add(method, lpis, secs, tfrac, deterministic):
        rows.append({"dataset": dataset, "n": n, "method": method,
                     "lpi_mean": round(float(np.mean(lpis)), 4),
                     "lpi_std": round(float(np.std(lpis)), 4),
                     "time_s": round(secs, 3), "test_frac": round(tfrac, 4),
                     "deterministic": deterministic})

    # NOTE ON TIMING: the clock must cover ONLY the split.  scaled_lpi is an
    # O(n^2) Tanimoto pass that dwarfs every fast splitter (24 s on muv), so
    # including it makes all cheap methods look identically slow.
    def timed(make_split, seeds):
        """Return (mean split seconds, [labels/split per seed]) -- scoring excluded."""
        outs, t0 = [], time.time()
        for s in seeds:
            outs.append(make_split(s))
        return (time.time() - t0) / len(seeds), outs

    # -- random (seeded) ----------------------------------------------------
    dt, outs = timed(lambda s: random_labels(n, seed=s), range(N_SEEDS))
    add("random", [score_labels(l) for l, _ in outs], dt, outs[-1][1], False)

    # -- scaffold (deterministic, SMILES only) ------------------------------
    dt, outs = timed(lambda s: _bin_pack(scaffold_groups(smiles), n), [0])
    add("scaffold", [score_labels(outs[0][0])], dt, outs[0][1], True)

    # -- Butina cluster split (deterministic) -------------------------------
    if n <= BUTINA_MAX_N:
        dt, outs = timed(lambda s: _bin_pack(butina_groups(X), n), [0])
        add("butina", [score_labels(outs[0][0])], dt, outs[0][1], True)
    else:
        rows.append({"dataset": dataset, "n": n, "method": "butina",
                     "lpi_mean": np.nan, "lpi_std": np.nan, "time_s": np.nan,
                     "test_frac": np.nan, "deterministic": True})

    # -- k-means cluster split ----------------------------------------------
    dt, outs = timed(lambda s: _bin_pack(kmeans_groups(X, seed=s), n),
                     range(N_SEEDS))
    add("kmeans-cluster", [score_labels(l) for l, _ in outs], dt,
        float(np.mean([tf for _, tf in outs])), False)

    # -- graph / hypergraph / low-rank, all at EPSILON=0 --------------------
    # epsilon=0 forces an exact 80/20 block so no method can gain from a
    # smaller test set.  The library default (0.05) lets low-rank drift to
    # ~18.9%, which mechanically lowers L(pi) -- that is the confound this
    # whole comparison exists to avoid.
    feature_data = {smiles[i]: X[i] for i in range(n)}

    def _frac(sp):
        v = list(sp.values())
        return sum(1 for x in v if x == "test") / len(v)

    for label, method, kw in [
        ("hypergraph", "hypergraph",
         dict(k=15, preset="quality", threads=4)),
        ("graph+thr+fm", "graph",
         dict(k=15, threshold=0.3, preset="quality", threads=4, fm=True)),
        ("lowrank", "lowrank",
         dict(rank=NYSTROM_RANK, n_restarts=4, fm=True, fm_max_n=FM_MAX_N)),
    ]:
        dt, sps = timed(
            lambda s: run_split(method, feature_data,
                                SplitSpec([8, 2], ["train", "test"], seed=s,
                                          epsilon=EPSILON), **kw).assignment,
            range(N_SEEDS))
        add(label, [scaled_lpi(list(smiles), sp)[0] for sp in sps], dt,
            float(np.mean([_frac(sp) for sp in sps])), label == "lowrank")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    args = ap.parse_args()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    jobs = [(d, i % args.gpus) for i, d in enumerate(args.datasets)]
    print(f"Running {len(jobs)} datasets on {args.workers} workers "
          f"across {args.gpus} GPUs, {N_SEEDS} seeds ...\n")
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        out = pool.starmap(_run_one_dataset, jobs)

    rows = [r for sub in out for r in sub]
    df = pd.DataFrame(rows, columns=CSV_COLS)
    df = df.sort_values(["n", "method"]).reset_index(drop=True)
    p = os.path.join(RESULTS, "baseline_benchmark.csv")
    df.to_csv(p, index=False)
    print(df.to_string(index=False))
    print(f"\nWall time: {time.time()-t0:.0f}s -> {p}")


if __name__ == "__main__":
    main()
