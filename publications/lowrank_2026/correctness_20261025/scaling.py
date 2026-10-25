"""Prespecified structured-factor, rank-budget and calibration checks.

Factor timings exclude optimization. Identity storage is measured as array/CSR
bytes, not misreported as process RSS. Rank-budget choices use no target labels.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import benchmark


def case(args):
    import numpy as np

    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    package = Path(manifest["implementations"][job["implementation"]])
    if benchmark.source_identity(package) != manifest["sources"][job["implementation"]]:
        raise RuntimeError("source drift")
    sys.path.insert(0, str(package))
    import lowrank
    import torch
    from scipy import sparse

    if (
        torch.cuda.is_available()
        or torch.get_num_threads() != 2
        or len(os.sched_getaffinity(0)) != 2
    ):
        raise RuntimeError("two CPU slots/threads required")
    row = dict(job, cpu_affinity=sorted(os.sched_getaffinity(0)))
    if job["kind"] == "cosine":
        X = np.random.default_rng(job["seed"]).random((job["n"], 16)).astype(np.float32)
        lowrank.nystrom_features(X[:20], rank=8, metric="cosine")
        options = dict(factor_backend="auto") if job["implementation"] == "new" else {}
        start = time.perf_counter()
        B, _, info = lowrank.nystrom_features(
            X, rank=256, metric="cosine", return_info=True, **options
        )
        elapsed = time.perf_counter() - start
        rng = np.random.default_rng(42)
        i = rng.integers(len(X), size=2048)
        j = (i + rng.integers(1, len(X), size=len(i))) % len(X)
        a, b = X[i].astype(float), X[j].astype(float)
        exact = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        approximate = (B[i].astype(float) * B[j].astype(float)).sum(1)
        error = float(np.abs(exact - approximate).max())
        if error > 1e-5:
            raise RuntimeError("cosine factor mismatch")
        row.update(
            seconds=elapsed,
            factor_shape=list(B.shape),
            factor_bytes=B.nbytes,
            maximum_checked_error=error,
            backend=info.get("factor_backend", "nystrom"),
        )
    elif job["kind"] == "identity":
        from lowrank.nd import _identity_factor

        values = np.repeat(np.arange(job["n"] // 2), 2)
        start = time.perf_counter()
        B = _identity_factor(values)
        elapsed = time.perf_counter() - start
        size = (
            B.data.nbytes + B.indices.nbytes + B.indptr.nbytes
            if sparse.issparse(B)
            else B.nbytes
        )
        labels = np.arange(len(values)) % 2
        if lowrank.factor_leakage(B, labels, 2) != job["n"] / 2:
            raise RuntimeError("identity factor cut mismatch")
        row.update(
            seconds=elapsed,
            storage_bytes=int(size),
            storage="CSR" if sparse.issparse(B) else "dense",
            factor_shape=list(B.shape),
            dense_equivalent_bytes=2 * job["n"] ** 2,
        )
    elif job["kind"] == "rank_budget":
        X = np.load(args.inputs / "cache" / f"{job['dataset']}.npz")["X"]
        options = dict(
            rank="auto",
            rank_min=32,
            rank_max=512,
            rank_tolerance=0.05,
            rank_check="cut_error",
            rank_probe_pairs=job["pairs"],
            refine_restarts=1,
            on_rank_failure="return",
            score_exact_leakage=False,
        )
        start = time.perf_counter()
        result = lowrank.split(
            "lowrank",
            dict(enumerate(X)),
            lowrank.SplitSpec([8, 1, 1], ["train", "val", "test"], seed=job["seed"]),
            **options,
        )
        row.update(seconds=time.perf_counter() - start, diagnostics=result.diagnostics)
    elif job["kind"] == "calibration":
        from lowrank.neighbours import audited_neighbours

        X = np.load(args.inputs / "cache" / "moleculenet_tox21.npz")["X"]
        records = []
        for _ in range(2):
            start = time.perf_counter()
            (indices, scores), info = audited_neighbours(
                X,
                "tanimoto",
                32,
                backend="auto",
                candidates=64,
                seed=0,
                n_jobs=1,
                audit_queries=64,
            )
            if not np.isfinite(scores).all() or np.any(
                indices == np.arange(len(X))[:, None]
            ):
                raise RuntimeError("invalid neighbours")
            records.append(dict(seconds=time.perf_counter() - start, diagnostics=info))
        if (
            records[0]["diagnostics"]["selected_backend"] == "tiled"
            and not records[1]["diagnostics"]["calibration_cache_hit"]
        ):
            raise RuntimeError("exact calibration was not reused")
        row.update(records=records)
    if benchmark.source_identity(package) != manifest["sources"][job["implementation"]]:
        raise RuntimeError("source changed")
    row.update(
        status="ok",
        process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    benchmark.atomic(args.output / "rows" / (job["name"] + ".json"), row)
    print(job["name"], "ok", flush=True)


def launch(args):
    import psutil

    jobs = []
    for n in [1000, 10000, 100000]:
        for seed in range(3):
            for implementation in ["previous", "new"]:
                jobs.append(
                    dict(
                        name=f"cosine_{n}_{seed}_{implementation}",
                        kind="cosine",
                        n=n,
                        seed=seed,
                        implementation=implementation,
                    )
                )
    for n in [1000, 5000, 100000]:
        for implementation in ["previous", "new"] if n <= 5000 else ["new"]:
            jobs.append(
                dict(
                    name=f"identity_{n}_{implementation}",
                    kind="identity",
                    n=n,
                    implementation=implementation,
                )
            )
    for dataset in ["moleculenet_freesolv", "moleculenet_bace", "moleculenet_qm8"]:
        for seed in range(3):
            for pairs in [2048, 8192, 32768]:
                jobs.append(
                    dict(
                        name=f"cut_budget_{dataset}_{seed}_{pairs}",
                        kind="rank_budget",
                        dataset=dataset,
                        seed=seed,
                        pairs=pairs,
                        implementation="new",
                    )
                )
    jobs.append(
        dict(name="calibration_reuse", kind="calibration", implementation="new")
    )
    implementations = dict(
        previous=str(args.previous.resolve()), new=str(args.new.resolve())
    )
    manifest = dict(
        jobs=jobs,
        implementations=implementations,
        sources={
            k: benchmark.source_identity(Path(v)) for k, v in implementations.items()
        },
        runner_sha256=benchmark.sha(__file__),
        inputs=str(args.inputs.resolve()),
        protocol="18 cosine factor builds; five identity builds, with no 20-GB dense allocation; "
        "27 label-blind cut-audit budget cases; one repeated calibration case. No model-outcome tuning.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("manifest drift")
    benchmark.atomic(path, manifest)
    cpus = ["40,41", "42,43", "44,45"]
    occupied = set()
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if proc.info["cmdline"] and len(proc.cpu_affinity()) <= 8:
                occupied.update(proc.cpu_affinity())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if occupied & set(range(40, 46)) or os.getloadavg()[0] + 6 > os.cpu_count():
        raise RuntimeError("CPU capacity unavailable")
    pending = list(enumerate(jobs))
    import threading

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
            command = [
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
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600
                )
            if result.returncode:
                raise RuntimeError("scaling case failed: " + job["name"])
            print("done", job["name"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--new", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--index", type=int)
    args = p.parse_args()
    dict(case=case, launch=launch)[args.action](args)
