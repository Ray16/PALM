"""Same-host split-call-only DataSAIL timing and exact-count low-rank controls.

Feature loading is outside the clock; DataSAIL's distance construction,
clustering and integer program are inside. Same C1e/SCIP settings as the saved
benchmark, but explicitly two solver threads and a 7200-second solver budget.
Solver budget is not an end-to-end wall-time bound. Native assignments and
matching low-rank partitions are preserved; missing IDs never default to test.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from revision_benchmark import score
from splitters import SplitSpec, split

DEST = study.OUT / "revision_20261021/runtime"
DATASETS = ["moleculenet_" + s for s in ("clintox", "bace", "bbbp", "lipophilicity")]


def run(dataset):
    if not hasattr(np, "ComplexWarning"):
        np.ComplexWarning = np.exceptions.ComplexWarning
    from datasail.sail import datasail
    directory = DEST / dataset
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "result.json"
    sources = source_files()
    sources[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if final.exists():
        if json.loads(final.read_text())["source_files"] != sources:
            raise RuntimeError("existing timing uses another source")
        return
    b, ids, X, y, metric = study.prepare(dataset)
    names = [str(i) for i in ids]
    if len(set(names)) != len(names):
        raise ValueError("string conversion makes entity identifiers ambiguous")
    record = dict(dataset=dataset, n=len(ids), host=socket.gethostname(), threads=2,
        source_files=sources, feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        datasail_version=importlib.metadata.version("datasail"),
        configuration=dict(technique="C1e", solver="SCIP", max_sec=7200, epsilon=.01,
            delta=.1, e_clusters=min(max(9, len(ids) // 20), len(ids))),
        timed_scope="split call: distance/representation construction, clustering and optimization; feature loading and scoring excluded")
    try:
        start = time.perf_counter()
        distance = np.nan_to_num(squareform(pdist(X, metric="jaccard")), nan=1.)
        e_splits, _, _ = datasail(techniques=["C1e"], splits=[8, 1, 1], names=study.NAMES.tolist(),
            runs=1, solver="SCIP", max_sec=7200, threads=2, e_type="O",
            e_data={name: name for name in names}, e_dist=(names, distance),
            e_clusters=min(max(9, len(ids) // 20), len(ids)), delta=.1, epsilon=.01, verbose="I")
        seconds = time.perf_counter() - start
        raw = e_splits["C1e"][0]
        if set(raw) != set(names) or not set(raw.values()).issubset(set(study.NAMES)):
            raise ValueError("DataSAIL returned missing/unknown IDs or fold names")
        labels = np.array([raw[name] for name in names])
        counts = [int(np.sum(labels == name)) for name in study.NAMES]
        if min(counts) == 0:
            raise ValueError("DataSAIL returned an empty fold")
        np.save(directory / "datasail.npy", labels)
        rows = [dict(method="datasail", seed=0, split_s=seconds, **score(X, labels, metric, 0))]
        for method in ("plain", "combined"):
            # Small warm-up precedes warm split-call timings and is reported.
            warm_kwargs = dict(allowed_size_imbalance=0., score_exact_leakage=False, rank=32)
            if method == "combined":
                warm_kwargs.update(kernel_power=3, must_link=.4)
            split("lowrank", {i: X[i] for i in range(min(100, len(X)))},
                  SplitSpec([8, 1, 1], study.NAMES.tolist(), epsilon=0.), **warm_kwargs)
            for seed in (0, 1, 2):
                kwargs = dict(allowed_size_imbalance=0., score_exact_leakage=False)
                if method == "combined":
                    kwargs.update(kernel_power=3, must_link=.4)
                start = time.perf_counter()
                result = split("lowrank", b.feature_data,
                    SplitSpec(counts, study.NAMES.tolist(), seed=seed, epsilon=0.), **kwargs)
                split_s = time.perf_counter() - start
                lab = np.array([result.assignment[i] for i in ids])
                if [int(np.sum(lab == name)) for name in study.NAMES] != counts:
                    raise AssertionError("timing control did not match DataSAIL counts")
                np.save(directory / f"{method}__s{seed}.npy", lab)
                rows.append(dict(method=method, seed=seed, split_s=split_s,
                                 **score(X, lab, metric, seed)))
        record.update(status="ok", rows=rows,
            solver_status="inspect retained native solver log; assignment alone is not an optimality certificate",
            timing_initialization="DataSAIL imports excluded, native first solve included; low-rank small warm-up excluded")
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    study.atomic_json(final, study.clean(record))
    print(dataset, record["status"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU timing protocol requires CUDA_VISIBLE_DEVICES=''")
    for dataset in args.datasets:
        run(dataset)
