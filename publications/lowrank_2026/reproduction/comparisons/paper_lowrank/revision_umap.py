"""Published UMAP-cluster recipe transferred with its native LOCO protocol.

Verified against Sahet11/nci60_clustering at commit 8c55880...: 1024-bit
radius-2 Morgan, UMAP(100 neighbours, min_dist=0, 2D, Jaccard, seed42), then
Ward agglomeration into seven clusters. Evaluate every cluster as test; compare
low-rank settings at its exact counts. Retain all input rows (no outlier filter).
This transfers the published recipe, not its NCI-60 dataset or fitted models.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from revision_benchmark import score
from splitters import SplitSpec, split

DEST = study.OUT / "revision_20261021/umap"
DATASETS = ["moleculenet_" + s for s in ("freesolv", "esol", "bace", "bbbp", "sider",
            "clintox", "lipophilicity", "tox21", "qm8", "hiv", "muv")]
UPSTREAM = "8c55880f6e4f9f0bcc6fd8369e18745c2a166038"


def run(dataset):
    directory = DEST / dataset
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "result.json"
    sources = source_files()
    for name in ("revision_umap.py", "revision_benchmark.py"):
        path = Path(__file__).with_name(name)
        sources[str(path.relative_to(study.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if final.exists():
        if json.loads(final.read_text())["source_files"] != sources:
            raise RuntimeError("existing UMAP result uses different sources")
        return
    b, ids, X, y, metric = study.prepare(dataset)
    n = len(X)
    record = dict(dataset=dataset, n=n, ids=ids, source_files=sources,
        feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        upstream_commit=UPSTREAM, upstream_repository="https://github.com/Sahet11/nci60_clustering",
        recipe=dict(n_neighbors=100, min_dist=0., n_components=2, metric="jaccard",
                    random_state=42, n_clusters=7, linkage="ward", n_jobs=1),
        adaptation="native leave-one-cluster-out on new datasets; no outlier removal; no supervised tuning",
        memory_policy="skip when condensed float64 agglomeration distances alone exceed 16 GiB")
    required_bytes = n * (n - 1) // 2 * 8
    if required_bytes > 16 * 1024 ** 3 or metric != "tanimoto":
        record.update(status="unavailable", reason="prespecified memory/input eligibility", estimated_distance_bytes=required_bytes)
        study.atomic_json(final, study.clean(record))
        return
    try:
        import umap
        from sklearn.cluster import AgglomerativeClustering
        record["versions"] = {name: importlib.metadata.version(name) for name in ("umap-learn", "scikit-learn", "numpy")}
        start = time.perf_counter()
        embedding = umap.UMAP(n_neighbors=100, min_dist=0., n_components=2,
            metric="jaccard", random_state=42, n_jobs=1).fit_transform(X.astype(bool))
        cluster = AgglomerativeClustering(linkage="ward", n_clusters=7).fit_predict(embedding)
        record["cluster_s"] = time.perf_counter() - start
        np.save(directory / "embedding.npy", embedding)
        np.save(directory / "clusters.npy", cluster)
        rows = []
        for fold in range(7):
            original = np.where(cluster == fold, "test", "train")
            counts = [int(np.sum(original == name)) for name in ("train", "test")]
            if min(counts) == 0:
                raise ValueError("empty UMAP cluster fold")
            for method in ("umap_cluster", "plain", "combined"):
                if method == "umap_cluster":
                    labels, diagnostics, seconds = original.copy(), {}, None
                else:
                    kwargs = dict(allowed_size_imbalance=0., score_exact_leakage=False)
                    if method == "combined":
                        kwargs.update(kernel_power=3, must_link=.4)
                    start = time.perf_counter()
                    res = split("lowrank", b.feature_data,
                        SplitSpec(counts, ["train", "test"], seed=0, epsilon=0.), **kwargs)
                    seconds = time.perf_counter() - start
                    labels = np.array([res.assignment[i] for i in ids])
                    diagnostics = res.diagnostics
                if [int(np.sum(labels == name)) for name in ("train", "test")] != counts:
                    raise AssertionError("LOCO comparison changed exact counts")
                np.save(directory / f"{method}__fold{fold}.npy", labels)
                rows.append(dict(method=method, fold=fold, split_s=seconds,
                    diagnostics=diagnostics, **score(X, labels, metric, seed=0)))
                study.atomic_json(directory / "progress.json", study.clean(dict(rows=rows)))
                print(dataset, fold, method, "test count", counts[1], flush=True)
        record.update(status="ok", rows=rows, cluster_sizes=np.bincount(cluster).tolist())
    except Exception as exc:
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    study.atomic_json(final, study.clean(record))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU protocol requires CUDA_VISIBLE_DEVICES=''")
    for dataset in args.datasets:
        run(dataset)
