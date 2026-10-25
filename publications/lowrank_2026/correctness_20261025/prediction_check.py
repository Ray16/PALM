"""Descriptive predictive validation on the three additional molecular datasets.

All split settings are frozen before fitting. Models use training labels only;
no hyperparameter search, test-outcome selection, simulated chronology or claim
of prospective validation. Undefined discrimination metrics remain explicit.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import benchmark

DATASETS = ["moleculenet_clintox", "moleculenet_sider", "moleculenet_qm8"]
SETTINGS = [
    "previous_plain",
    "previous_combined",
    "new_default_plain",
    "new_default_cut",
    "new_default_weighted",
]


def case(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    source = (
        args.inputs
        / "rows"
        / f"{job['dataset']}__s{job['seed']}__{job['setting']}.json"
    )
    reference = json.loads(source.read_text())
    if reference["status"] != "ok":
        raise RuntimeError("split unavailable: " + str(source))
    array = source.with_suffix(".npy")
    if benchmark.sha(array) != reference["assignment_sha256"]:
        raise RuntimeError("assignment drift")
    meta = json.loads((args.inputs / "inputs" / f"{job['dataset']}.json").read_text())
    path = args.inputs / "cache" / f"{job['dataset']}.npz"
    if benchmark.sha(path) != meta["cache_sha256"]:
        raise RuntimeError("input drift")
    bundle = np.load(path)
    X, y, labels = bundle["X"], bundle["y"], np.load(array)
    valid = np.isfinite(y)
    train = (labels == 0) & valid
    test = (labels == 2) & valid
    row = dict(
        job,
        assignment_sha256=reference["assignment_sha256"],
        split_source_sha256=reference["source_sha256"],
        task=meta["task"],
        n_train_labels=int(train.sum()),
        n_test_labels=int(test.sum()),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        model_selection="none; fixed defaults; training labels only",
    )
    started = time.perf_counter()
    if not train.any() or not test.any():
        row.update(status="no_labeled_training_or_test_data", metrics=None)
    elif meta["task"] == "classification":
        classes = np.unique(y[valid])
        if len(classes) != 2 or len(np.unique(y[train])) != 2:
            row.update(
                status="unsupported_or_single_class_training",
                metrics=None,
                classes=classes.tolist(),
            )
        else:
            target = (y == classes[-1]).astype(int)
            model = (
                LogisticRegression(max_iter=1000, random_state=job["seed"])
                if job["model"] == "linear"
                else RandomForestClassifier(
                    n_estimators=128, n_jobs=1, random_state=job["seed"]
                )
            )
            model.fit(X[train], target[train])
            probability = model.predict_proba(X[test])[:, 1]
            both = len(np.unique(target[test])) == 2
            row.update(
                status="ok",
                train_positive_fraction=float(target[train].mean()),
                test_positive_fraction=float(target[test].mean()),
                metrics=dict(
                    auroc=float(roc_auc_score(target[test], probability))
                    if both
                    else None,
                    average_precision=float(
                        average_precision_score(target[test], probability)
                    )
                    if both
                    else None,
                    brier=float(brier_score_loss(target[test], probability)),
                ),
            )
    else:
        model = (
            Ridge(alpha=1.0)
            if job["model"] == "linear"
            else RandomForestRegressor(
                n_estimators=128, n_jobs=1, random_state=job["seed"]
            )
        )
        model.fit(X[train], y[train])
        predicted = model.predict(X[test])
        row.update(
            status="ok",
            metrics=dict(
                mae=float(mean_absolute_error(y[test], predicted)),
                rmse=float(np.sqrt(mean_squared_error(y[test], predicted))),
                r2=float(r2_score(y[test], predicted))
                if test.sum() > 1 and np.var(y[test]) > 0
                else None,
            ),
        )
    row["fit_predict_seconds"] = time.perf_counter() - started
    benchmark.atomic(args.output / "rows" / (job["name"] + ".json"), row)
    print(job["name"], row["status"], flush=True)


def launch(args):
    import psutil

    completion = json.loads((args.inputs / "complete.json").read_text())
    if completion["completed"] != completion["expected"]:
        raise RuntimeError("complete the split grid first")
    jobs = [
        dict(
            name=f"{d}__s{s}__{setting}__{model}",
            dataset=d,
            seed=s,
            setting=setting,
            model=model,
        )
        for d in DATASETS
        for s in range(3)
        for setting in SETTINGS
        for model in ["linear", "forest"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    cpus = ["6,7", "8,9", "10,11", "12,13", "14,15", "16,17"]
    allocated = set(range(6, 18))
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if (
                proc.info["cmdline"]
                and len(proc.cpu_affinity()) <= 8
                and set(proc.cpu_affinity()) & allocated
            ):
                raise RuntimeError("CPU slots still occupied")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if os.getloadavg()[0] + 12 > os.cpu_count():
        raise RuntimeError("insufficient CPU capacity")
    manifest = dict(
        jobs=jobs,
        split_manifest_sha256=benchmark.sha(args.inputs / "manifest.json"),
        runner_sha256=benchmark.sha(__file__),
        cpus=cpus,
        protocol="Three additional datasets, five frozen split settings, three seeds, two fixed models. "
        "No fitting on validation/test labels, parameter search or prospective claim.",
    )
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("manifest drift")
    benchmark.atomic(path, manifest)
    import threading

    pending = list(enumerate(jobs))
    lock = threading.Lock()

    def worker(cpus):
        while True:
            with lock:
                if not pending:
                    return
                index, job = pending.pop(0)
            final = args.output / "rows" / (job["name"] + ".json")
            if final.exists():
                continue
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2",
                OMP_THREAD_LIMIT="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                NUMEXPR_NUM_THREADS="2",
            )
            cmd = [
                "taskset",
                "-c",
                cpus,
                sys.executable,
                str(Path(__file__).resolve()),
                "case",
                "--output",
                str(args.output),
                "--inputs",
                str(args.inputs),
                "--index",
                str(index),
            ]
            with (args.output / "logs" / (job["name"] + ".log")).open("w") as log:
                result = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600
                )
            if result.returncode:
                raise RuntimeError("prediction check failed: " + job["name"])
            print("done", job["name"], flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(worker, c) for c in cpus]
        for future in futures:
            future.result()
    rows = [json.loads(p.read_text()) for p in (args.output / "rows").glob("*.json")]
    benchmark.atomic(
        args.output / "complete.json",
        dict(
            expected=len(jobs),
            completed=len(rows),
            ok=sum(r["status"] == "ok" for r in rows),
            other_statuses=[r["name"] for r in rows if r["status"] != "ok"],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["launch", "case"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--index", type=int)
    args = p.parse_args()
    dict(case=case, launch=launch)[args.action](args)
