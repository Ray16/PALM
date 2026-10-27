"""Fixed-bin descriptive calibration cross-check of the saved classification fits.

Models and assignments remain fixed. Reliability-bin summaries and calibration
in the large complement Brier scores; they do not establish prospective reliability.
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


def case(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import brier_score_loss

    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    prior = json.loads(
        (args.predictions / "rows" / (job["name"] + ".json")).read_text()
    )
    name = f"{job['dataset']}__s{job['seed']}__{job['setting']}"
    split_row = json.loads((args.inputs / "rows" / (name + ".json")).read_text())
    if split_row["assignment_sha256"] != prior["assignment_sha256"]:
        raise RuntimeError("assignment mismatch")
    cache = args.inputs / "cache" / f"{job['dataset']}.npz"
    meta = json.loads((args.inputs / "inputs" / f"{job['dataset']}.json").read_text())
    if benchmark.sha(cache) != meta["cache_sha256"]:
        raise RuntimeError("input drift")
    data = np.load(cache)
    X, y = data["X"], data["y"]
    labels = np.load(args.inputs / "rows" / (name + ".npy"))
    valid = np.isfinite(y)
    train = (labels == 0) & valid
    test = (labels == 2) & valid
    classes = np.unique(y[valid])
    target = (y == classes[-1]).astype(int)
    model = (
        LogisticRegression(max_iter=1000, random_state=job["seed"])
        if job["model"] == "linear"
        else RandomForestClassifier(
            n_estimators=128, n_jobs=1, random_state=job["seed"]
        )
    )
    started = time.perf_counter()
    model.fit(X[train], target[train])
    prob = model.predict_proba(X[test])[:, 1]
    brier = float(brier_score_loss(target[test], prob))
    if abs(brier - prior["metrics"]["brier"]) > 1e-8:
        raise RuntimeError("fixed-model refit changed probability score")
    bins = np.minimum((prob * 10).astype(int), 9)
    records = []
    ece = 0.0
    for b in range(10):
        mask = bins == b
        count = int(mask.sum())
        confidence = float(prob[mask].mean()) if count else None
        frequency = float(target[test][mask].mean()) if count else None
        if count:
            ece += count / len(prob) * abs(confidence - frequency)
        records.append(
            dict(
                bin=b, n=count, mean_probability=confidence, positive_fraction=frequency
            )
        )
    row = dict(
        job,
        status="ok",
        assignment_sha256=prior["assignment_sha256"],
        cache_sha256=meta["cache_sha256"],
        n_test=int(test.sum()),
        brier=brier,
        prior_brier_difference=brier - prior["metrics"]["brier"],
        calibration_in_large=float(prob.mean() - target[test].mean()),
        ece10=float(ece),
        bins=records,
        fit_predict_seconds=time.perf_counter() - started,
        interpretation="descriptive; fixed ten equal-width bins; sample-size/distribution dependent",
    )
    benchmark.atomic(args.output / "rows" / (job["name"] + ".json"), row)


def launch(args):
    import psutil

    prior = json.loads((args.predictions / "manifest.json").read_text())
    jobs = [
        job
        for job in prior["jobs"]
        if json.loads(
            (args.predictions / "rows" / (job["name"] + ".json")).read_text()
        )["task"]
        == "classification"
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    cpus = ["12,13", "14,15", "16,17", "18,19", "21,22", "23,24"]
    allocated = {int(c) for pair in cpus for c in pair.split(",")}
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if (
                proc.info["cmdline"]
                and len(proc.cpu_affinity()) <= 8
                and set(proc.cpu_affinity()) & allocated
            ):
                raise RuntimeError("CPU slots occupied")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if os.getloadavg()[0] + len(allocated) > os.cpu_count():
        raise RuntimeError("insufficient CPU capacity")
    benchmark.atomic(
        args.output / "manifest.json",
        dict(
            jobs=jobs,
            runner_sha256=benchmark.sha(__file__),
            split_manifest_sha256=benchmark.sha(args.inputs / "manifest.json"),
            prediction_manifest_sha256=benchmark.sha(
                args.predictions / "manifest.json"
            ),
            protocol="Refit exactly the same fixed classifiers; cross-check Brier; compute fixed ten-bin descriptive reliability summaries.",
        ),
    )
    import threading

    pending = list(enumerate(jobs))
    lock = threading.Lock()

    def worker(cpu):
        while True:
            with lock:
                if not pending:
                    return
                index, job = pending.pop(0)
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
                cpu,
                sys.executable,
                str(Path(__file__).resolve()),
                "case",
                "--inputs",
                str(args.inputs),
                "--predictions",
                str(args.predictions),
                "--output",
                str(args.output),
                "--index",
                str(index),
            ]
            with (args.output / "logs" / (job["name"] + ".log")).open("w") as log:
                result = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300
                )
            if result.returncode:
                raise RuntimeError("calibration check failed: " + job["name"])

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(worker, c) for c in cpus]
        for future in futures:
            future.result()
    benchmark.atomic(
        args.output / "complete.json",
        dict(
            expected=len(jobs),
            completed=len(list((args.output / "rows").glob("*.json"))),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["launch", "case"])
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int)
    args = p.parse_args()
    dict(case=case, launch=launch)[args.action](args)
