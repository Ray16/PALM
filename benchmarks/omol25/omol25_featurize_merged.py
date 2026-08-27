"""Featurize the merged OMol25 (train_4M + val + test) set and cache to disk.

Reads every structure from the *.aselmdb shards (via ase.db — no fairchem
needed), computes a NON-NEGATIVE structural descriptor, and writes:

  _cache/features.npy      float32 [N, 115]  (memmap-friendly)
  _cache/meta.parquet      per-structure: split, shard, db_id, formula,
                                           charge, spin, natoms, data_id

The descriptor is non-negative on purpose so cosine similarity lies in [0, 1] —
matching the bounded-similarity form of DataSAIL's L(pi):
    composition histogram (83)  |  elemental mass/radius stats (5)  |
    3D radial-distance histogram (24)  |  charge+ , charge- , spin (3)   = 115

Parallelized across the 240 shards. Run (palm env):
    python -m PALM.benchmarks.omol25.omol25_featurize_merged --workers 64
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from ase.data import atomic_masses, covalent_radii
from ase.db import connect

DATA_DIR = "/nfs/lambda_stor_01/homes/rzhu/PALM/data/DataSAIL_data/1D/omol25"
CACHE_DIR = os.path.join(DATA_DIR, "_cache")
SPLITS = ["train_4M", "val", "test"]
SPLIT_CODE = {"train_4M": 0, "val": 1, "test": 2}

Z_MAX = 83
RDF_BINS = 24
RDF_MAX_A = 6.0
FEAT_DIM = Z_MAX + 5 + RDF_BINS + 3      # 115


def _featurize(numbers: np.ndarray, positions: np.ndarray,
               charge: float, spin: float) -> np.ndarray:
    """Non-negative structural descriptor for one system (see module docstring)."""
    Z = np.asarray(numbers)
    n = len(Z)
    comp = np.bincount(Z, minlength=Z_MAX + 1)[1:Z_MAX + 1].astype(np.float64)
    comp /= max(comp.sum(), 1.0)

    masses = atomic_masses[Z]
    radii = covalent_radii[Z]
    elemental = np.array([masses.mean(), masses.std(), radii.mean(), radii.std(), float(n)])

    if n > 1:
        R = np.asarray(positions)
        d = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
        d = d[np.triu_indices(n, k=1)]
        rdf, _ = np.histogram(d, bins=RDF_BINS, range=(0.0, RDF_MAX_A))
        rdf = rdf.astype(np.float64)
        rdf /= max(rdf.sum(), 1.0)
    else:
        rdf = np.zeros(RDF_BINS)

    state = np.array([max(charge, 0.0), max(-charge, 0.0), float(spin)])
    return np.concatenate([comp, elemental, rdf, state]).astype(np.float32)


def _process_shard(args):
    """Featurize one shard -> write feat/meta cache files; return (key, count)."""
    split, shard_path = args
    shard_idx = int(os.path.basename(shard_path).replace("data", "").replace(".aselmdb", ""))
    key = f"{SPLIT_CODE[split]}_{shard_idx:04d}"
    feat_path = os.path.join(CACHE_DIR, f"feat_{key}.npy")
    meta_path = os.path.join(CACHE_DIR, f"meta_{key}.parquet")
    if os.path.exists(feat_path) and os.path.exists(meta_path):     # resume: skip done shards
        return key, int(np.load(feat_path, mmap_mode="r").shape[0])
    db = connect(shard_path)
    feats, ids, formulas, charges, spins, natoms, data_ids = [], [], [], [], [], [], []
    for row in db.select():
        ch = float(row.data.get("charge", getattr(row, "charge", 0)))
        sp = float(row.data.get("spin", 1))
        feats.append(_featurize(row.numbers, row.positions, ch, sp))
        ids.append(int(row.id))
        formulas.append(row.formula)
        charges.append(ch); spins.append(sp); natoms.append(int(row.natoms))
        data_ids.append(str(row.data.get("data_id", "")))
    X = np.vstack(feats).astype(np.float32)
    key = f"{SPLIT_CODE[split]}_{shard_idx:04d}"
    np.save(os.path.join(CACHE_DIR, f"feat_{key}.npy"), X)
    pd.DataFrame({
        "split": SPLIT_CODE[split], "shard": shard_idx, "db_id": ids,
        "formula": formulas, "charge": charges, "spin": spins,
        "natoms": natoms, "data_id": data_ids,
    }).to_parquet(os.path.join(CACHE_DIR, f"meta_{key}.parquet"))
    return key, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)

    tasks = []
    for split in SPLITS:
        for shard in sorted(glob.glob(os.path.join(DATA_DIR, split, "*.aselmdb"))):
            tasks.append((split, shard))
    print(f"{len(tasks)} shards across {SPLITS}; featurizing on {args.workers} workers ...", flush=True)

    t0 = time.time()
    done, total = 0, 0
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_process_shard, t): t for t in tasks}
        for fut in as_completed(futs):
            key, cnt = fut.result()
            done += 1; total += cnt
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} shards, {total:,} structures, "
                      f"{time.time()-t0:.0f}s", flush=True)

    # merge per-shard caches into one features.npy + meta.parquet (canonical order)
    print("merging shard caches ...", flush=True)
    keys = sorted(f.replace("feat_", "").replace(".npy", "")
                  for f in os.listdir(CACHE_DIR) if f.startswith("feat_"))
    counts = [np.load(os.path.join(CACHE_DIR, f"feat_{k}.npy"), mmap_mode="r").shape[0] for k in keys]
    N = int(sum(counts))
    feat_mm = np.lib.format.open_memmap(os.path.join(CACHE_DIR, "features.npy"),
                                        mode="w+", dtype=np.float32, shape=(N, FEAT_DIM))
    metas, off = [], 0
    for k, c in zip(keys, counts):
        feat_mm[off:off + c] = np.load(os.path.join(CACHE_DIR, f"feat_{k}.npy"))
        metas.append(pd.read_parquet(os.path.join(CACHE_DIR, f"meta_{k}.parquet")))
        off += c
    feat_mm.flush()
    meta = pd.concat(metas, ignore_index=True)
    meta.to_parquet(os.path.join(CACHE_DIR, "meta.parquet"))
    assert N == len(meta) == feat_mm.shape[0], f"merge mismatch: N={N} meta={len(meta)}"
    print(f"DONE: {N:,} structures from {len(keys)} shards -> {CACHE_DIR}/features.npy "
          f"[{N},{FEAT_DIM}] + meta.parquet in {time.time()-t0:.0f}s", flush=True)
    # per-shard temporaries are left in place (feat_*/meta_*); delete manually after
    # verifying features.npy — leaving them avoids any resume/merge race.


if __name__ == "__main__":
    main()
