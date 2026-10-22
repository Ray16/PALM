"""Release-code external assays with absolute scores and label-balance controls.

Run on CPU with CUDA hidden. Each assay is an independent resumable record.
Earlier results are never overwritten. Every control starts from the same new
partition and preserves its train/test counts. Target-aware repair is a
sensitivity intervention, not the label-blind splitting method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from mlst_label_control import balanced_labels
from datasets.featurize import ecfp_matrix
from splitters import SplitSpec, split
from splitters.common.leakage_metrics import nn_leakage, scaled_lpi

DEST = study.OUT / "revision_20261021"
NAMES = ["train", "test"]
METHODS = {"random": {}, "plain": {}, "combined": {"kernel_power": 3, "must_link": .4}}


def source_files():
    files = []
    for module in ("lowrank", "splitters", "datasets", "features", "eval"):
        files.extend(sorted((study.ROOT / "low_rank" / module).rglob("*.py")))
    files += [Path(__file__), Path(study.__file__), Path(__file__).with_name("mlst_label_control.py")]
    return {str(p.relative_to(study.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def neighbours(X, y, labels):
    """Save all test queries and a simple label-support diagnostic, with no fit."""
    from sklearn.metrics import roc_auc_score
    train = np.flatnonzero(labels == "train")
    test = np.flatnonzero(labels == "test")
    a, b = X[test].astype(np.float64), X[train].astype(np.float64)
    dot = a @ b.T
    S = dot / np.maximum(a.sum(1)[:, None] + b.sum(1)[None, :] - dot, 1e-12)
    nearest = np.argmax(S, axis=1)
    idx = np.argsort(-S, axis=1, kind="stable")[:, :min(5, len(train))]
    support = y[train[idx]].mean(1)
    per_point = np.column_stack([test, y[test], S[np.arange(len(test)), nearest],
                                 y[train[nearest]], support])
    diagnostics = dict(
        nearest_label_agreement=float(np.mean(y[test] == y[train[nearest]])),
        top5_label_auc=float(roc_auc_score(y[test], support)) if len(np.unique(y[test])) == 2 else None,
        test_n=int(len(test)), train_positive_rate=float(y[train].mean()),
        test_positive_rate=float(y[test].mean()),
        positive_rate_difference=float(y[test].mean() - y[train].mean()),
        test_nn_gt06=float(np.mean(S.max(1) > .6)),
        test_nn_gt08=float(np.mean(S.max(1) > .8)),
    )
    return per_point, diagnostics


def run(filename):
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("This CPU protocol requires CUDA_VISIBLE_DEVICES='' before launch")
    start_sources = source_files()
    stem = Path(filename).stem
    dest = DEST / "external" / stem
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "result.json"
    if final.exists():
        old = json.loads(final.read_text())
        if old["source_files"] != start_sources:
            raise RuntimeError("existing result used different sources; use a new revision directory")
        print(stem, "verified existing record", flush=True)
        return
    data_path = study.OUT / "external_simpd" / filename
    df = pd.read_csv(data_path)
    kept, X = ecfp_matrix(df.canonical_smiles.astype(str).tolist())
    df = df.iloc[kept].reset_index(drop=True)
    X = X.astype(np.float32)
    y = df.active.to_numpy(float)
    published = df.split.to_numpy(str)
    if set(published) != set(NAMES) or not np.isfinite(y).all():
        raise ValueError("invalid published labels or activity labels")
    counts = [int(np.sum(published == name)) for name in NAMES]
    fd = dict(enumerate(X))
    rows, models = [], []
    jobs = [("published_simpd", 0)] + [(m, s) for m in METHODS for s in (0, 1, 2)]
    for method, seed in jobs:
        if method == "published_simpd":
            original, diag, seconds = published.copy(), {}, None
        else:
            params = ({} if method == "random" else dict(METHODS[method],
                       allowed_size_imbalance=0., score_exact_leakage=False))
            # The random adapter has its own optional quadratic diagnostic.
            import splitters.methods.random as rmod
            rmod._LEAKAGE_MAX_N = 0
            t = time.perf_counter()
            result = split("random" if method == "random" else "lowrank", fd,
                           SplitSpec(counts, NAMES, seed=seed, epsilon=0.), **params)
            seconds = time.perf_counter() - t
            original = np.array([result.assignment[i] for i in range(len(X))])
            diag = result.diagnostics
        for condition in ("original", "label_balanced"):
            labels, moved = ((original, 0.) if condition == "original" else
                balanced_labels(y, original, "classification", seed, names=NAMES))
            if [int(np.sum(labels == name)) for name in NAMES] != counts:
                raise AssertionError("control changed fold counts")
            basename = f"{method}__s{seed}__{condition}"
            np.save(dest / (basename + ".npy"), labels)
            per_point, diagnostic = neighbours(X, y, labels)
            np.save(dest / (basename + "__neighbours.npy"), per_point)
            nn = nn_leakage(X, labels, metric="tanimoto", reference="train", seed=seed,
                            n_perm=3, device="cpu")
            leak = scaled_lpi(X, labels, metric="tanimoto")
            expected = 1 - sum(c * (c - 1) for c in counts) / (len(X) * (len(X) - 1))
            key = dict(dataset=stem, method=method, split_seed=seed, condition=condition)
            rows.append(dict(key, n=len(X), split_s=seconds, moved_fraction=moved,
                leak_ratio=leak / expected, nnl_rel=nn["nnl_rel"], nnl=nn["nnl"],
                broken_groups=diag.get("broken_groups"), **diagnostic))
            for model in ("rf", "xgb"):
                for model_seed in (0, 1):
                    score = study.model_scores(X, y, labels, "classification", model_seed, model)
                    models.append(dict(key, model=model, model_seed=model_seed, **score))
            study.atomic_json(dest / "progress.json", study.clean(dict(rows=rows, models=models)))
            print(stem, method, seed, condition,
                  "rf_auc", models[-4].get("test_auc"), "moved", round(moved, 4), flush=True)
    if source_files() != start_sources:
        raise RuntimeError("source changed during inference; result is not finalized")
    study.atomic_json(final, study.clean(dict(
        dataset=stem, source_files=start_sources, input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(), kept_source_rows=kept,
        host=socket.gethostname(), platform=platform.platform(), threads=2,
        pid=os.getpid(), n=len(X), target_counts=counts, rows=rows, models=models,
        neighbours_columns=["test_index", "activity", "nearest_train_similarity", "nearest_train_activity", "top5_train_activity_mean"],
        interpretation="label-aware repair changes structural composition; no causal isolation or prospective validation")))


def summarize():
    records = [json.loads(p.read_text()) for p in sorted((DEST / "external").glob("*/result.json"))]
    if not records:
        return
    for key in ("rows", "models"):
        df = pd.DataFrame([r for record in records for r in record[key]])
        df.to_csv(DEST / ("external_" + key + ".csv"), index=False)
    models = pd.read_csv(DEST / "external_models.csv")
    cell = models.groupby(["dataset", "method", "condition", "model"]).agg(
        auc=("test_auc", "mean"), min_auc=("test_auc", "min"), ap=("test_ap", "mean"),
        train_positive_rate=("train_positive_rate", "mean"), test_positive_rate=("test_positive_rate", "mean"),
        y_shift=("y_shift", "mean"), n_scores=("test_auc", "count")).reset_index()
    cell.to_csv(DEST / "external_per_assay.csv", index=False)
    # Equal assay weighting: first aggregate split/model seeds within each assay.
    cell.groupby(["method", "condition", "model"]).agg(
        auc=("auc", "mean"), ap=("ap", "mean"), y_shift=("y_shift", "mean"),
        assays=("dataset", "nunique")).to_csv(DEST / "external_summary.csv")
    models[models.test_auc < .5].to_csv(DEST / "external_below_chance.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assays", nargs="*")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if not args.summarize_only:
        manifest = json.loads((study.OUT / "external_simpd/manifest.json").read_text())
        for entry in manifest["files"]:
            if args.assays is None or Path(entry["path"]).stem in args.assays:
                run(entry["path"])
                summarize()
    summarize()
