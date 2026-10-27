"""Portable CPU reproduction of the pinned primary numerical study.

Numerical case functions remain unchanged. This adapter supplies cache paths,
structured native infeasibility, source/input verification, and NFS-safe claim
retirement. Use a fresh output directory for changed settings or sources.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import study


def filtered_jobs(args):
    return [
        j
        for j in study.all_jobs()
        if (not args.studies or j["study"] in args.studies)
        and (not args.datasets or j.get("dataset") in args.datasets)
        and (not args.seeds or j["seed"] in args.seeds)
    ]


def prepare(args):
    os.environ["LOWRANK_DATA"] = str(args.data.resolve())
    path = ROOT.parent / "correctness_20261025/benchmark.py"
    spec = importlib.util.spec_from_file_location("pinned_input_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATASETS = sorted(
        {j["dataset"] for j in filtered_jobs(args) if "dataset" in j}
    )
    args.new, args.output = study.PACKAGE, args.cache_root
    (args.output / "cache").mkdir(parents=True, exist_ok=True)
    module.prepare(args)


def plan(args):
    jobs = filtered_jobs(args)
    if not jobs:
        raise ValueError("no jobs match the requested filters")
    datasets = sorted({j["dataset"] for j in jobs if "dataset" in j})
    manifest = dict(
        jobs=jobs,
        sources=study.sources(),
        cache_root=str(args.cache_root),
        inputs={
            d: dict(
                cache_sha256=study.sha(args.cache_root / "cache" / f"{d}.npz"),
                metadata=json.loads(
                    (args.cache_root / "inputs" / f"{d}.json").read_text()
                ),
            )
            for d in datasets
        },
        cpu_sets=args.cpus,
        threads=2,
        timeout_seconds=1800,
        max_attempts=1,
        host=socket.gethostname(),
        python=sys.executable,
        portable_adapter_sha256=study.sha(__file__),
        policy="Pinned numerical functions, configurable prepared-cache root, explicit infeasible outcomes and one bounded attempt per case; atomic claims retire without deleting open NFS metadata.",
    )
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("manifest changed; choose a fresh output directory")
    study.atomic(path, manifest)
    print("planned", len(jobs), flush=True)


def case(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    if study.sha(__file__) != manifest["portable_adapter_sha256"]:
        raise RuntimeError("portable adapter changed")
    study.CACHE_ROOT = Path(manifest["cache_root"])
    job = manifest["jobs"][args.index]
    statuses = []

    class Capture(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "SCIP status:" in message or "No assignments could be made" in message:
                statuses.append(message)

    logger = None
    handler = Capture()
    if job.get("method") == "datasail":
        from datasail.settings import LOGGER

        logger = LOGGER
        logger.addHandler(handler)
    try:
        try:
            study.case(args)
        except TypeError:
            if job.get("method") != "datasail" or not any(
                "SCIP status: infeasible" in s for s in statuses
            ):
                raise
            study.atomic(
                args.output / "rows" / f"{job['name']}.json",
                dict(
                    job,
                    status="native_infeasible",
                    solver_evidence=statuses,
                    artifacts={},
                    manifest_sha256=study.sha(args.output / "manifest.json"),
                    protocol="SCIP reported infeasibility and the native API returned no assignment; no score is imputed.",
                ),
            )
    finally:
        if logger is not None:
            logger.removeHandler(handler)


def complete(output, job):
    path = output / "rows" / f"{job['name']}.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    if row["manifest_sha256"] != study.sha(output / "manifest.json"):
        raise RuntimeError("result provenance mismatch")
    if row["status"] == "native_infeasible":
        if job.get("method") != "datasail" or not any(
            "SCIP status: infeasible" in s for s in row["solver_evidence"]
        ):
            raise RuntimeError("unverified native exclusion")
        return True
    if row["status"] != "ok" or not row["artifacts"]:
        raise RuntimeError("invalid result status/artifacts")
    if any(
        not (output / p).exists() or study.sha(output / p) != h
        for p, h in row["artifacts"].items()
    ):
        raise RuntimeError("missing or changed result artifact")
    return True


def retire_claim(claim, token, retired):
    """Release only our claim, preserving metadata even if an NFS reader is open."""
    owner = json.loads((claim / "owner.json").read_text())
    if owner.get("token") != token:
        raise RuntimeError("claim owner changed; refusing to release")
    retired.mkdir(parents=True, exist_ok=True)
    destination = retired / (claim.name + "__" + token)
    os.rename(claim, destination)
    return destination


def launch(args):
    import psutil

    manifest = json.loads((args.output / "manifest.json").read_text())
    if (
        study.sources() != manifest["sources"]
        or study.sha(__file__) != manifest["portable_adapter_sha256"]
    ):
        raise RuntimeError("producer/adapter source drift")
    cpus = [set(map(int, c.split(","))) for c in manifest["cpu_sets"]]
    union = set.union(*cpus)
    if (
        any(len(c) != 2 for c in cpus)
        or len(union) != 2 * len(cpus)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("invalid/disallowed/overloaded CPU allocation")
    if psutil.virtual_memory().available < 4 * len(cpus) * 1024**3:
        raise RuntimeError("insufficient memory headroom")
    for p in psutil.process_iter(["cmdline"]):
        try:
            if (
                p.info["cmdline"]
                and len(p.cpu_affinity()) <= 8
                and set(p.cpu_affinity()) & union
            ):
                raise RuntimeError(f"CPU overlap with narrow process {p.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    for name in ["rows", "claims", "retired_claims", "logs", "failures"]:
        (args.output / name).mkdir(exist_ok=True)
    work = queue.Queue()
    for index, job in enumerate(manifest["jobs"]):
        if (
            not complete(args.output, job)
            and not (args.output / "failures" / f"{job['name']}.json").exists()
        ):
            work.put((index, job))

    def worker(cores):
        failures = 0
        while failures < 3:
            try:
                index, job = work.get_nowait()
            except queue.Empty:
                return
            claim = args.output / "claims" / job["name"]
            try:
                claim.mkdir()
            except FileExistsError:
                print("held claim requires owner verification", job["name"], flush=True)
                continue
            token = uuid.uuid4().hex
            owner = dict(
                token=token,
                pid=os.getpid(),
                start_ticks=study.ticks(os.getpid()),
                host=socket.gethostname(),
                cpus=sorted(cores),
            )
            study.atomic(claim / "owner.json", owner)
            try:
                if complete(args.output, job):
                    continue
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                for key in [
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]:
                    env[key] = "2"
                command = [
                    "taskset",
                    "-c",
                    ",".join(map(str, sorted(cores))),
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "case",
                    "--output",
                    str(args.output),
                    "--index",
                    str(index),
                ]
                with (args.output / "logs" / f"{job['name']}.txt").open("w") as log:
                    child = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    study.atomic(
                        claim / "owner.json",
                        dict(
                            owner,
                            child_pid=child.pid,
                            child_start_ticks=study.ticks(child.pid),
                            command=command,
                        ),
                    )
                    try:
                        code = child.wait(timeout=manifest["timeout_seconds"])
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        code = -999
                if code or not complete(args.output, job):
                    failures += 1
                    study.atomic(
                        args.output / "failures" / f"{job['name']}.json",
                        dict(job, exit_code=code, terminal=True),
                    )
                else:
                    failures = 0
                    print("done", job["name"], flush=True)
            finally:
                retire_claim(claim, token, args.output / "retired_claims")

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        for future in [pool.submit(worker, c) for c in cpus]:
            future.result()
    outcomes = [
        dict(name=j["name"], complete=complete(args.output, j))
        for j in manifest["jobs"]
    ]
    study.atomic(
        args.output / "progress.json",
        dict(
            expected=len(outcomes),
            completed=sum(r["complete"] for r in outcomes),
            outcomes=outcomes,
        ),
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "plan", "case", "launch"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    p.add_argument("--data", type=Path, default=ROOT.parent / "inputs")
    p.add_argument("--index", type=int)
    p.add_argument("--cpus", nargs="+", default=["0,1"])
    p.add_argument("--studies", nargs="+")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    dict(prepare=prepare, plan=plan, case=case, launch=launch)[args.action](args)
