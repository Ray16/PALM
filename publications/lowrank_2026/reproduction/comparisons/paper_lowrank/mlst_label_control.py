"""Sensitivity to label-distribution balance at fixed split cardinalities.

Minimal-count point reassignments match class/quantile-bin proportions. This
explicitly relaxes grouping; it is a sensitivity intervention, not a label-blind
splitter or a causal isolation of structure. Both leakage axes are remeasured.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study

DATASETS = ["moleculenet_" + s for s in
            ("freesolv", "esol", "lipophilicity", "qm8", "bace", "bbbp", "sider", "tox21")]


def balanced_labels(y, labels, task, seed, names=None):
    names = study.NAMES if names is None else np.asarray(names)
    valid = np.isfinite(y)
    if task == "classification":
        bins = np.where(valid, y, -1)
    else:
        edges = np.unique(np.quantile(y[valid], np.linspace(0, 1, 6)))
        bins = np.where(valid, np.searchsorted(edges[1:-1], y, side="right"), -1)
    _, bins = np.unique(bins, return_inverse=True)
    bin_counts = np.bincount(bins)
    sizes = np.array([(labels == s).sum() for s in names])
    raw = bin_counts[:, None] * sizes[None, :] / len(y)
    desired = np.floor(raw).astype(int)
    rows, cols = desired.shape
    A = np.zeros((rows + cols, rows * cols))
    for i in range(rows):
        A[i, i * cols:(i + 1) * cols] = 1
    for j in range(cols):
        A[rows + j, j::cols] = 1
    rhs = np.r_[bin_counts - desired.sum(1), sizes - desired.sum(0)]
    result = linprog(-(raw - desired).ravel(), A_eq=A, b_eq=rhs, bounds=(0, 1), method="highs")
    if not result.success:
        raise RuntimeError("bin count rounding failed: " + result.message)
    desired += np.rint(result.x.reshape(raw.shape)).astype(int)
    rng = np.random.default_rng(seed)
    new = labels.copy()
    for b in range(rows):
        pool, deficits = [], []
        for c, name in enumerate(names):
            members = np.flatnonzero((bins == b) & (labels == name))
            rng.shuffle(members)
            keep = min(len(members), desired[b, c])
            pool.extend(members[keep:].tolist())
            deficits.extend([name] * (desired[b, c] - keep))
        rng.shuffle(pool)
        new[pool] = deficits
    assert all((new == name).sum() == size for name, size in zip(names, sizes))
    for b in range(rows):
        assert all(((bins == b) & (new == name)).sum() == desired[b, c] for c, name in enumerate(names))
    return new, float((new != labels).mean())


def run(dataset):
    source = study.OUT / "results" / f"benchmark__{dataset}.json"
    directory = study.OUT / "results/label_control" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        # Completed method chunks can be consumed before unrelated methods finish.
        # Require every relevant split and all its fitted-model outputs, then
        # snapshot that dependency evidence; never read an in-progress split.
        progress = study.OUT / "results/benchmark" / dataset / "progress.json"
        if not progress.exists():
            print("WAITING for", source.name, flush=True)
            return False
        p = json.loads(progress.read_text())
        selected_rows = [r for r in p["rows"] if r["method"] in ("random", "plain", "tail")]
        selected_models = [r for r in p["models"] if r["method"] in ("random", "plain", "tail")]
        if len(selected_rows) != 9 or len(selected_models) != 36 or any(r.get("status") != "ok" for r in selected_rows):
            print("WAITING for completed dependency chunks", dataset, flush=True)
            return False
        source = directory / "source_completed_chunks.json"
        study.atomic_json(source, dict(rows=selected_rows, models=selected_models,
                          interpretation="completed method chunks; unrelated benchmark methods may still be running"))
    b, ids, X, y, metric = study.prepare(dataset)
    rows, models = [], []
    for method in ("random", "plain", "tail"):
        for seed in (0, 1, 2):
            path = study.OUT / "results/benchmark" / dataset / f"{method}__s{seed}.npy"
            original = np.load(path)
            lab, moved = balanced_labels(y, original, b.task_type, seed)
            np.save(directory / f"{method}__s{seed}.npy", lab)
            rows.append(dict(method=method, seed=seed, moved_fraction=moved,
                             original_assignment_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             **study.score(X, lab, metric, seed)))
            for model in ("rf", "xgb"):
                m = study.model_scores(X, y, lab, b.task_type, 0, model)
                models.append(dict(method=method, split_seed=seed, model=model, model_seed=0, task=b.task_type, **m))
            print(dataset, method, seed, "moved", round(moved, 3), flush=True)
    result = dict(dataset=dataset, study="label_control", n=len(ids), rows=rows, models=models,
                  original_artifact_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  original_artifact=str(source.relative_to(study.OUT)),
                  feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                  source_sha256=study.source_identity(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation="label-aware point repair; grouping relaxed; sensitivity analysis, not causal identification")
    study.atomic_json(study.OUT / "results" / f"label_control__{dataset}.json", study.clean(result))
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    for d in p.parse_args().datasets:
        dest = study.OUT / "results" / f"label_control__{d}.json"
        if not dest.exists():
            run(d)
