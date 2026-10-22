"""Row-resumable full-size benchmark without hypergraph comparators.

CPU protocol, same scoring for every method. Saved DataSAIL assignments are
identity-checked and rescored; their original runtimes are never mixed into
same-hardware timing claims. External adapters with different retention/count
contracts require their own explicitly matched comparison, not this table.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from revision_baselines import graph_cut
from splitters import SplitSpec, split
from splitters.common.pairwise_similarity import pairwise_similarity, standardize_for_metric
from splitters.common.balanced_assignment import capacity_corridor

DEST = study.OUT / "revision_20261021/benchmark"
METHODS = ("random", "plain", "combined", "combined_tau06", "knn_graph", "scaffold", "butina", "datasail")


def score(X, labels, metric, seed):
    """Global sample <=4000, test and pooled-held queries <=2048, full train.

    All quantities use original similarities regardless of optimization power.
    The same fixed global row sample is used for all methods at a given seed.
    Primary local scores are test-only; pooled validation+test is secondary.
    """
    labels = np.asarray(labels)
    ref = standardize_for_metric(torch.as_tensor(X), metric)
    n = len(labels)
    rng = np.random.default_rng(173 + seed)
    sub = np.sort(rng.choice(n, min(n, 4000), replace=False))
    total = cross = 0.
    for start in range(0, len(sub), 256):
        rows = sub[start:start + 256]
        S = pairwise_similarity(ref[rows], ref[sub], metric).double().numpy()
        S[np.arange(len(rows)), np.arange(start, start + len(rows))] = 0.
        total += S.sum()
        cross += S[labels[rows, None] != labels[sub][None, :]].sum()
    counts = np.array([np.sum(labels == name) for name in study.NAMES])
    expected = 1 - sum(c * (c - 1) for c in counts) / (n * (n - 1))
    if metric == "tanimoto":
        ramps = {"default": (.4, .8), "03_07": (.3, .7), "05_09": (.5, .9)}
    else:
        from splitters.common.leakage_metrics import _pair_sim_quantiles
        ramps = {"default": _pair_sim_quantiles(ref, metric, (.99, .999), seed=seed)}

    def local(lab, test_only):
        train = np.flatnonzero(lab == "train")
        query = np.flatnonzero(lab == "test" if test_only else lab != "train")
        query = np.sort(np.random.default_rng(911 + seed).choice(query, min(len(query), 2048), replace=False))
        top = []
        kk = min(5, len(train))
        for s in range(0, len(query), 128):
            best = torch.full((len(query[s:s + 128]), kk), -torch.inf)
            for t in range(0, len(train), 8192):
                sim = pairwise_similarity(ref[query[s:s + 128]], ref[train[t:t + 8192]], metric)
                best = torch.cat([best, sim], 1).topk(kk, dim=1).values
            top.append(best.numpy())
        vals = np.vstack(top)
        result = {"mean_top1": float(vals[:, 0].mean()), "mean_top5": float(vals.mean()),
                  "frac_gt06": float(np.mean(vals[:, 0] > .6)), "frac_gt08": float(np.mean(vals[:, 0] > .8))}
        for name, (lo, hi) in ramps.items():
            result["ramp_" + name] = float(np.clip((vals[:, 0] - lo) / max(hi - lo, 1e-12), 0, 1).mean())
        return result, len(query), len(train)

    out = dict(leak_ratio=float(cross / total / expected), n=n, global_query_n=len(sub),
               global_scoring="exact" if n <= 4000 else "uniform 4000-row sample", metric=metric,
               n_train=int(counts[0]), n_val=int(counts[1]), n_test=int(counts[2]))
    permutations = [np.random.default_rng(seed + 5000 + j).permutation(labels) for j in range(3)]
    for name, test_only in (("test", True), ("held", False)):
        observed, nq, nt = local(labels, test_only)
        randoms = [local(lab, test_only)[0] for lab in permutations]
        out[name + "_query_n"], out[name + "_reference_n"] = nq, nt
        for key, value in observed.items():
            base = float(np.mean([r[key] for r in randoms]))
            out[name + "_" + key] = value
            out[name + "_" + key + "_rel"] = value / base if base > 0 else None
    return out


def assign(method, b, ids, X, metric, eps, seed):
    if method == "datasail":
        directory = study.ROOT / "comparisons/results/paper/full/assign"
        path = directory / f"{b.name}__datasail__eps{eps}__s0.npy"
        if not path.exists():
            raise FileNotFoundError("no saved DataSAIL assignment at this configuration")
        old_ids = np.load(directory / f"{b.name}__ids.npy", allow_pickle=True).tolist()
        old_labels = np.load(path)
        if len(old_ids) != len(set(old_ids)) or set(old_ids) != set(ids):
            raise ValueError("saved DataSAIL entity identities differ")
        mapping = dict(zip(old_ids, old_labels))
        return np.array([mapping[i] for i in ids]), dict(
            assignment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            timing_kind="reused assignment; no new split timing"), None
    start = time.perf_counter()
    if method == "knn_graph":
        codes, diagnostics = graph_cut(X, metric, epsilon=eps, seed=seed, k=32, threads=2)
        labels = study.NAMES[codes]
    elif method in ("scaffold", "butina"):
        labels, diagnostics = study.grouped_baseline(method, b, ids, seed)
    else:
        import splitters.methods.random as randommod
        randommod._LEAKAGE_MAX_N = 0
        kw = {} if method == "random" else dict(allowed_size_imbalance=eps, score_exact_leakage=False)
        if method.startswith("combined"):
            kw.update(kernel_power=3, must_link=(.6 if method.endswith("06") else .4)
                      if metric == "tanimoto" else "auto")
        result = split("random" if method == "random" else "lowrank", b.feature_data,
                       SplitSpec([8, 1, 1], study.NAMES.tolist(), seed=seed, epsilon=eps), **kw)
        labels = np.array([result.assignment[i] for i in ids])
        diagnostics = result.diagnostics
    return labels, diagnostics, time.perf_counter() - start


def run(dataset, methods, tolerances, seeds):
    b, ids, X, y, metric = study.prepare(dataset)
    dest = DEST / dataset
    dest.mkdir(parents=True, exist_ok=True)
    sources = source_files()
    for name in ("revision_benchmark.py", "revision_baselines.py"):
        p = Path(__file__).with_name(name)
        sources[str(p.relative_to(study.ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    study.atomic_json(dest / "inputs.json", dict(ids=ids, source_files=sources,
        feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(), host=socket.gethostname(), threads=2))
    for method in methods:
        for eps in tolerances:
            for seed in ([0] if method == "datasail" else seeds):
                stem = f"{method}__eps{eps}__s{seed}"
                final = dest / (stem + ".json")
                if final.exists():
                    if json.loads(final.read_text())["source_files"] != sources:
                        raise RuntimeError("existing row source differs; do not overwrite")
                    continue
                row = dict(dataset=dataset, method=method, epsilon=eps, seed=seed,
                           source_files=sources, n=len(ids), host=socket.gethostname(), threads=2)
                try:
                    labels, diagnostics, seconds = assign(method, b, ids, X, metric, eps, seed)
                    if len(labels) != len(ids) or set(labels) != set(study.NAMES):
                        raise ValueError("incomplete assignment or empty fold")
                    counts = np.array([np.sum(labels == name) for name in study.NAMES])
                    caps, floors = capacity_corridor(len(ids), [8, 1, 1], eps)
                    corridor_ok = bool(np.all(counts >= floors) and np.all(counts <= caps))
                    if method in ("plain", "combined", "combined_tau06", "knn_graph", "random") and not corridor_ok:
                        raise ValueError("capacity postcondition failed")
                    np.save(dest / (stem + ".npy"), labels)
                    row.update(status="ok", split_s=seconds, diagnostics=diagnostics,
                               corridor_ok=corridor_ok, **score(X, labels, metric, seed))
                except (FileNotFoundError, ValueError) as exc:
                    row.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
                if source_files() != {key: value for key, value in sources.items()
                        if not key.endswith(("revision_benchmark.py", "revision_baselines.py"))}:
                    raise RuntimeError("method source changed during inference")
                study.atomic_json(final, study.clean(row))
                print(dataset, method, eps, seed, row["status"], row.get("split_s"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=study.DATASETS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--eps", nargs="+", type=float, default=[.01, .05])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU protocol requires CUDA_VISIBLE_DEVICES=''")
    for dataset in args.datasets:
        run(dataset, args.methods, args.eps, args.seeds)
