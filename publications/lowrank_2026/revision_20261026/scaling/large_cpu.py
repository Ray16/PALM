"""Staged largest-scale CPU audit on lambda13; no GPU allocation is used.

The numerical package/parameters are unchanged. Cache preparation, timed split
and independent audit have separate durable outputs and budgets. A completed
split is never repeated merely because a later audit is interrupted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

import run as base

N = 2_000_000


def sources():
    identity = base.sources()
    identity[str(Path(__file__).relative_to(base.ROOT.parent))] = base.sha(__file__)
    return identity


def prepared(args):
    meta = json.loads((args.cache / "prepared.json").read_text())
    for name in ["features.npy", "indices.npy"]:
        p = args.cache / name
        if (
            p.stat().st_size != meta[name]["size"]
            or p.stat().st_mtime_ns != meta[name]["mtime_ns"]
        ):
            raise RuntimeError("prepared input changed")
    return meta


def prepare(args):
    import numpy as np
    import shutil

    args.cache.mkdir(parents=True, exist_ok=True)
    if (args.cache / "prepared.json").exists():
        meta = prepared(args)
        if meta["packed_sha256"] != base.sha(args.packed):
            raise RuntimeError("packed input changed")
        return
    if shutil.disk_usage(args.cache).free < 24 * 1024**3:
        raise RuntimeError("insufficient local feature-cache disk")
    print("preparing node-local features", flush=True)
    packed = np.load(args.packed, mmap_mode="r")
    indices = np.sort(np.random.default_rng(0).permutation(len(packed))[:N])
    if len(indices) != N:
        raise ValueError("input corpus too small")
    X = np.unpackbits(np.asarray(packed[indices]), axis=1).astype(np.float32)
    if X.shape != (N, 2048):
        raise ValueError("expected 2048-bit input")
    for name, array in [("features.npy", X), ("indices.npy", indices)]:
        tmp = args.cache / (name + ".partial.npy")
        np.save(tmp, array)
        os.replace(tmp, args.cache / name)
    meta = dict(
        host=socket.gethostname(),
        n=N,
        packed_sha256=base.sha(args.packed),
        recipe="sorted prefix of rng(0) permutation; unpackbits to float32",
        files_are_read_only=True,
    )
    for name in ["features.npy", "indices.npy"]:
        path = args.cache / name
        meta[name] = dict(
            sha256=base.sha(path),
            size=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,
        )
        path.chmod(0o444)
    base.atomic(args.cache / "prepared.json", meta)
    print("prepared", X.shape, flush=True)


def context(args):
    import numpy as np
    import torch

    manifest = json.loads((args.output / "manifest.json").read_text())
    if sources() != manifest["sources"] or torch.cuda.is_available():
        raise RuntimeError("source changed or GPU exposed")
    if (
        len(os.sched_getaffinity(0)) != manifest["threads"]
        or torch.get_num_threads() != manifest["threads"]
    ):
        raise RuntimeError("CPU affinity/thread mismatch")
    if prepared(args) != manifest["prepared_input"]:
        raise RuntimeError("prepared input identity mismatch")
    sys.path.insert(0, str(base.PACKAGE))
    job = manifest["jobs"][args.index]
    directory = args.output / job["name"]
    directory.mkdir(exist_ok=True)
    X = np.load(args.cache / "features.npy", mmap_mode="r")
    return manifest, job, directory, X


def split_case(args):
    import numpy as np

    manifest, job, directory, X = context(args)
    if (directory / "native.json").exists():
        return
    from lowrank import split, SplitSpec
    from lowrank.splitters.common.balanced_assignment import capacity_corridor

    params = dict(
        rank=256,
        metric="tanimoto",
        refine_restarts=2,
        score_exact_leakage=False,
        allowed_size_imbalance=0.01,
    )
    names = ["train", "val", "test"]
    data = dict(enumerate(X))
    started = time.perf_counter()
    split(
        "lowrank",
        dict(enumerate(X[:100])),
        SplitSpec([8, 1, 1], names, epsilon=0.01),
        **dict(params, rank=32),
    )
    warmup = time.perf_counter() - started
    print("starting timed largest split", job["name"], flush=True)
    started = time.perf_counter()
    result = split(
        "lowrank",
        data,
        SplitSpec([8, 1, 1], names, seed=job["seed"], epsilon=0.01),
        **params,
    )
    seconds = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    labels = np.fromiter(
        (names.index(result.assignment[i]) for i in range(N)), dtype=np.int8, count=N
    )
    counts = np.bincount(labels, minlength=3)
    caps, floors = capacity_corridor(N, [8, 1, 1], 0.01)
    if not ((counts >= floors) & (counts <= caps)).all():
        raise AssertionError("capacity corridor violated")
    np.save(directory / "labels.npy", labels)
    base.atomic(
        directory / "native.json",
        dict(
            job,
            status="split_complete_audit_pending",
            seconds=seconds,
            warmup_seconds=warmup,
            peak_split_rss_gib=rss,
            counts=counts.tolist(),
            params=params,
            diagnostics=result.diagnostics,
            host=socket.gethostname(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            threads=manifest["threads"],
            manifest_sha256=base.sha(args.output / "manifest.json"),
            assignment_sha256=base.sha(directory / "labels.npy"),
            timing_scope="warm public CPU split call; node-local preparation, feature mapping, imports and auditing excluded",
        ),
    )
    print("split complete", job["name"], seconds, flush=True)


def audit(args):
    import numpy as np
    import torch

    manifest, job, directory, X = context(args)
    if (directory / "result.json").exists():
        return
    native = json.loads((directory / "native.json").read_text())
    if native["manifest_sha256"] != base.sha(args.output / "manifest.json") or native[
        "assignment_sha256"
    ] != base.sha(directory / "labels.npy"):
        raise RuntimeError("native split identity mismatch")
    from lowrank import nystrom_features
    from lowrank.objective import factor_leakage
    from lowrank.diagnostics import factor_pair_error, sampled_leakage
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

    labels = np.load(directory / "labels.npy")
    started = time.perf_counter()
    B, _, info = nystrom_features(
        X, rank=256, metric="tanimoto", seed=job["seed"], return_info=True
    )
    if (
        info["landmark_indices"]
        != native["diagnostics"]["factorization"]["landmark_indices"]
    ):
        raise AssertionError("reconstructed landmarks mismatch")
    if (
        abs(factor_leakage(B, labels, 3) - native["diagnostics"]["factor_leakage"])
        > 1e-6
    ):
        raise AssertionError("reconstructed factor objective mismatch")
    ref = torch.as_tensor(X)
    fidelity = factor_pair_error(
        B, ref, "tanimoto", pairs=262144, seed=8000, labels=labels
    )
    global_cut = sampled_leakage(
        X, labels, "tanimoto", sample_pairs_count=262144, seed=9000
    )
    factor_audit_seconds = time.perf_counter() - started
    del B
    started = time.perf_counter()
    test = np.flatnonzero(labels == 2)
    train = np.flatnonzero(labels == 0)
    query = np.sort(
        np.random.default_rng(7000).choice(test, min(128, len(test)), replace=False)
    )
    nearest = np.empty(len(query))
    for s in range(0, len(query), 16):
        q = ref[query[s : s + 16]]
        best = torch.zeros(len(q))
        for t in range(0, len(train), 8192):
            best = torch.maximum(
                best,
                pairwise_similarity(q, ref[train[t : t + 8192]], "tanimoto")
                .max(1)
                .values,
            )
        nearest[s : s + len(q)] = best.numpy()
    np.save(directory / "query_indices.npy", query)
    np.save(directory / "nearest.npy", nearest)
    radius = float(np.sqrt(np.log(40) / (2 * len(query))))
    if sources() != manifest["sources"]:
        raise RuntimeError("source changed during audit")
    base.atomic(
        directory / "result.json",
        dict(
            native,
            status="ok",
            fidelity=fidelity,
            global_cut=global_cut,
            factor_audit_seconds=factor_audit_seconds,
            neighbour_audit_seconds=time.perf_counter() - started,
            local=dict(
                queries=len(query),
                training_references=len(train),
                exact_over_all_training=True,
                nearest_mean=float(nearest.mean()),
                nearest_mean_ci95=[
                    max(0.0, float(nearest.mean()) - radius),
                    min(1.0, float(nearest.mean()) + radius),
                ],
                fraction_above_06=float((nearest > 0.6).mean()),
                uncertainty="bounded-mean Hoeffding interval over uniform held-out queries",
            ),
            artifacts={p.name: base.sha(p) for p in directory.glob("*.npy")},
        ),
    )
    print("audit complete", job["name"], flush=True)


def complete(output, job):
    p = output / job["name"] / "result.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    if r["manifest_sha256"] != base.sha(output / "manifest.json") or any(
        base.sha(p.parent / f) != h for f, h in r["artifacts"].items()
    ):
        raise RuntimeError("large-result provenance mismatch")
    return r["status"] == "ok"


def launch(args):
    import psutil

    sys.path.insert(0, str(base.PACKAGE))
    import lowrank
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("large CPU protocol requires hidden CUDA")

    sets = [set(map(int, c.split(","))) for c in args.cpus]
    union = set.union(*sets)
    if (
        any(len(c) != 32 for c in sets)
        or len(union) != 32 * len(sets)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("32-core CPU capacity unavailable")
    if psutil.virtual_memory().available < 128 * len(sets) * 1024**3:
        raise RuntimeError("insufficient large-job memory")
    for p in psutil.process_iter(["cmdline"]):
        try:
            if (
                p.info["cmdline"]
                and len(p.cpu_affinity()) <= 64
                and set(p.cpu_affinity()) & union
            ):
                raise RuntimeError(f"CPU overlap with {p.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for k in [
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        env[k] = "32"
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ["logs", "claims", "failures"]:
        (args.output / folder).mkdir(exist_ok=True)
    prep_cmd = [
        "taskset",
        "-c",
        ",".join(map(str, sorted(sets[0]))),
        sys.executable,
        str(Path(__file__).resolve()),
        "prepare",
        "--output",
        str(args.output),
        "--cache",
        str(args.cache),
        "--packed",
        str(args.packed),
    ]
    with (args.output / "logs/preparation.txt").open("a") as log:
        prepared_process = subprocess.Popen(
            prep_cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        base.atomic(
            args.output / "preparation_owner.json",
            dict(
                host=socket.gethostname(),
                pid=prepared_process.pid,
                start_ticks=base.ticks(prepared_process.pid),
                command=prep_cmd,
            ),
        )
        try:
            preparation_code = prepared_process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(prepared_process.pid, signal.SIGTERM)
            try:
                prepared_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(prepared_process.pid, signal.SIGKILL)
                prepared_process.wait()
            raise RuntimeError("node-local preparation exceeded budget")
        if preparation_code:
            raise RuntimeError("node-local preparation failed")
    if os.getloadavg()[0] + len(union) > os.cpu_count():
        raise RuntimeError("CPU capacity changed during preparation")
    for process in psutil.process_iter(["cmdline"]):
        try:
            if (
                process.info["cmdline"]
                and len(process.cpu_affinity()) <= 64
                and set(process.cpu_affinity()) & union
            ):
                raise RuntimeError(f"CPU overlap after preparation with {process.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    jobs = [
        dict(
            name=f"plain__n{N}__eps0.01__s{s}",
            n=N,
            epsilon=0.01,
            seed=s,
            method="plain",
        )
        for s in range(3)
    ]
    manifest = dict(
        jobs=jobs,
        sources=sources(),
        prepared_input=prepared(args),
        host=socket.gethostname(),
        python=sys.executable,
        cpu_sets=[sorted(c) for c in sets],
        threads=32,
        phase_budget_seconds=3600,
        policy="Same default numerical package; 32-core CPU calls and independent audits, staged to preserve completed splits. No GPU is used. These timings are separate from two-core measurements.",
    )
    p = args.output / "manifest.json"
    if p.exists() and json.loads(p.read_text()) != manifest:
        raise RuntimeError("large-CPU manifest drift")
    base.atomic(p, manifest)

    def worker(worker_id, cores):
        failures = 0
        for index in range(worker_id, len(jobs), len(sets)):
            job = jobs[index]
            if complete(args.output, job):
                continue
            claim = args.output / "claims" / job["name"]
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            try:
                owner = dict(
                    pid=os.getpid(),
                    start_ticks=base.ticks(os.getpid()),
                    host=socket.gethostname(),
                    cpus=sorted(cores),
                )
                base.atomic(claim / "owner.json", owner)
                for phase in ["split", "audit"]:
                    if (
                        phase == "split"
                        and (args.output / job["name"] / "native.json").exists()
                    ):
                        continue
                    command = [
                        "taskset",
                        "-c",
                        ",".join(map(str, sorted(cores))),
                        sys.executable,
                        str(Path(__file__).resolve()),
                        phase,
                        "--output",
                        str(args.output),
                        "--cache",
                        str(args.cache),
                        "--index",
                        str(index),
                    ]
                    with (args.output / "logs" / f"{job['name']}__{phase}.txt").open(
                        "a"
                    ) as log:
                        child = subprocess.Popen(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        base.atomic(
                            claim / "owner.json",
                            dict(
                                owner,
                                child_pid=child.pid,
                                child_start_ticks=base.ticks(child.pid),
                                phase=phase,
                                command=command,
                            ),
                        )
                        try:
                            code = child.wait(timeout=3600)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            code = -999
                    if code:
                        failures += 1
                        base.atomic(
                            args.output / "failures" / f"{job['name']}__{phase}.json",
                            dict(job, phase=phase, exit_code=code),
                        )
                        break
                else:
                    if not complete(args.output, job):
                        raise RuntimeError("missing large audit output")
                    failures = 0
                    print("done", job["name"], flush=True)
                if failures >= 3:
                    return
            finally:
                (claim / "owner.json").unlink()
                claim.rmdir()

    with ThreadPoolExecutor(max_workers=len(sets)) as pool:
        for future in [pool.submit(worker, i, c) for i, c in enumerate(sets)]:
            future.result()
    base.atomic(
        args.output / "progress.json",
        dict(expected=len(jobs), completed=sum(complete(args.output, j) for j in jobs)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "split", "audit", "launch"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cache", type=Path, default=Path("/tmp/palm_lowrank_large_20261026")
    )
    p.add_argument("--packed", type=Path, default=base.INPUT)
    p.add_argument("--index", type=int)
    p.add_argument("--cpus", nargs="+")
    args = p.parse_args()
    args.output = args.output.resolve()
    dict(prepare=prepare, split=split_case, audit=audit, launch=launch)[args.action](
        args
    )
