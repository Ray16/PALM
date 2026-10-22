"""Exact tiled GPU neighbours versus packed-bit ANN: recall, time and split quality.

GPU tiling bounds temporary similarity storage but retains quadratic compute.
The ANN path is approximate and is assessed against exact sampled query rows.
Both use the same group optimizer and exact verification of retained edge weights.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from scaling import load
from lowrank.grouped import grouped_split, must_link_groups
from lowrank.neighbours import approximate_tanimoto_neighbours
from lowrank.nystrom import nystrom_features
from lowrank.objective import factor_leakage
from lowrank.optimize import balanced_lloyd, fm_polish
from splitters.common.balanced_assignment import target_sizes
from splitters.common.pairwise_similarity import pairwise_similarity


def tiled_exact_neighbours(X, k=32, queries=None, row_block=256, ref_block=8192):
    """Exact top k using two-dimensional tiling; O(n²d) time, bounded work tiles."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = torch.as_tensor(X, device=device)
    n = len(X)
    k = min(k, n - 1)
    queries = np.arange(n) if queries is None else np.asarray(queries)
    indices = np.empty((len(queries), k), dtype=np.int64)
    scores = np.empty((len(queries), k), dtype=np.float32)
    for start in range(0, len(queries), row_block):
        q = queries[start:start + row_block]
        best = torch.full((len(q), k), -torch.inf, device=device)
        ids = torch.zeros((len(q), k), dtype=torch.long, device=device)
        for r in range(0, n, ref_block):
            S = pairwise_similarity(ref[q], ref[r:r + ref_block], "tanimoto")
            self_rows = np.flatnonzero((q >= r) & (q < r + S.shape[1]))
            if len(self_rows):
                S[torch.as_tensor(self_rows, device=device), torch.as_tensor(q[self_rows] - r, device=device)] = -torch.inf
            val, col = S.topk(min(k, S.shape[1]), dim=1)
            combined = torch.cat([best, val], dim=1)
            combined_ids = torch.cat([ids, col + r], dim=1)
            best, which = combined.topk(k, dim=1)
            ids = combined_ids.gather(1, which)
        indices[start:start + len(q)] = ids.cpu().numpy()
        scores[start:start + len(q)] = best.cpu().numpy()
    return indices, scores


def optimize(X, neighbours):
    B, _ = nystrom_features(X, rank=256, metric="tanimoto", kernel_power=3, seed=0)
    candidates = [balanced_lloyd(B, [8, 1, 1], seed=r, balance_slack=0.01) for r in range(4)]
    initial = min(candidates, key=lambda lab: factor_leakage(B, lab, 3))
    initial, _ = fm_polish(B, initial, [8, 1, 1], epsilon=0.01)
    maximum = max(1, int(0.5 * target_sizes(len(X), [8, 1, 1]).min()))
    groups = must_link_groups(X, "tanimoto", 0.4, maximum, nbrs=neighbours)
    labels, info = grouped_split(B, groups, [8, 1, 1], 0.01, seed=0, init_labels=initial)
    return study.NAMES[labels], info


def run(sizes):
    directory = study.OUT / "results/neighbours"
    directory.mkdir(parents=True, exist_ok=True)
    # Compilation/startup is explicitly separated from warm graph timing.
    sample = load(100)
    t = time.perf_counter()
    approximate_tanimoto_neighbours(sample, k=8, candidates=24, seed=0, n_jobs=2)
    warmup_s = time.perf_counter() - t
    rows = []
    for n in sizes:
        X = load(n)
        query = np.sort(np.random.default_rng(42).choice(n, min(n, 256), replace=False))
        truth_i, truth_s = tiled_exact_neighbours(X, queries=query)
        for backend in ("gpu_tiled_exact", "nndescent"):
            out_path = directory / f"{backend}__n{n}.json"
            if out_path.exists():
                rows.append(json.loads(out_path.read_text()))
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            nn = (tiled_exact_neighbours(X) if backend == "gpu_tiled_exact"
                  else approximate_tanimoto_neighbours(X, candidates=64, seed=0, n_jobs=2))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            graph_s = time.perf_counter() - t
            matches = np.array([np.isin(truth_i[j], nn[0][q]) for j, q in enumerate(query)])
            strong = truth_s >= 0.4
            recall = float(matches.mean())
            strong_recall = float(matches[strong].mean()) if strong.any() else None
            t = time.perf_counter()
            lab, info = optimize(X, nn)
            optimization_s = time.perf_counter() - t
            np.save(directory / f"{backend}__n{n}__assignment.npy", lab)
            quality = study.score(X, lab, "tanimoto", 0)
            row = dict(study="neighbours", backend=backend, n=n, graph_s=graph_s,
                       optimizer_s=optimization_s, total_split_s=graph_s + optimization_s,
                       recall_at32=recall, strong_edge_recall_at32=strong_recall,
                       recall_query_n=len(query), ann_warmup_s=warmup_s,
                       feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                       source_sha256=study.source_identity(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                       group_diagnostics=info, **quality)
            study.atomic_json(out_path, study.clean(row))
            rows.append(row)
            print(json.dumps(study.clean(row)), flush=True)
            del nn
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del X
    study.atomic_json(directory / "summary.json", study.clean(dict(study="neighbours_summary", rows=rows)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 30000, 100000])
    run(p.parse_args().sizes)
