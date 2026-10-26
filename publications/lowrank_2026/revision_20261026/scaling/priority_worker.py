"""Additional disjoint CPU workers for pending relaxed cases in the same queue.

Producer code, inputs, job identities and timeout remain unchanged. Resource
allocation is recorded as an addendum to the immutable run manifest. Atomic
claims prevent duplicate execution by the original and additional workers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import psutil

import run as producer


def main(args):
    out = args.output.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    if producer.sources() != manifest["sources"]:
        raise RuntimeError("producer source drift")
    cpus = [set(map(int, c.split(","))) for c in args.cpus]
    union = set.union(*cpus)
    if (
        any(len(c) != 2 for c in cpus)
        or len(union) != 2 * len(cpus)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("CPU capacity unavailable")
    if psutil.virtual_memory().available < 64 * len(cpus) * 1024**3:
        raise RuntimeError("insufficient large-worker memory headroom")
    for p in psutil.process_iter(["cmdline"]):
        try:
            if (
                p.info["cmdline"]
                and len(p.cpu_affinity()) <= 8
                and set(p.cpu_affinity()) & union
            ):
                raise RuntimeError(f"CPU overlap with {p.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    allocation = dict(
        pid=os.getpid(),
        host=socket.gethostname(),
        start_ticks=producer.ticks(os.getpid()),
        cpus=[sorted(c) for c in cpus],
        manifest_sha256=producer.sha(out / "manifest.json"),
        worker_sha256=producer.sha(__file__),
        worker_path=str(Path(__file__).resolve()),
        policy="Prioritize pending relaxed plain cases; unchanged producer and timeout, atomic shared claims; no duplicate work.",
    )
    producer.atomic(out / "allocations" / f"priority_{os.getpid()}.json", allocation)
    jobs = sorted(
        [
            (index, j)
            for index, j in enumerate(manifest["jobs"])
            if j["method"] == "plain" and j["epsilon"] == 0.01
        ],
        key=lambda item: (-item[1]["n"], item[1]["seed"]),
    )

    def worker(cores):
        failures = 0
        for index, job in jobs:
            if failures >= 3:
                return
            if (
                producer.complete(out, job)
                or (out / "failures" / f"{job['name']}.json").exists()
            ):
                continue
            claim = out / "claims" / job["name"]
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            try:
                owner = dict(allocation, cpu_affinity=sorted(cores))
                producer.atomic(claim / "owner.json", owner)
                if producer.complete(out, job):
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
                    str(Path(producer.__file__).resolve()),
                    "case",
                    "--output",
                    str(out),
                    "--index",
                    str(index),
                ]
                with (out / "logs" / f"{job['name']}.log").open("w") as log:
                    child = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    producer.atomic(
                        claim / "owner.json",
                        dict(
                            owner,
                            child_pid=child.pid,
                            child_start_ticks=producer.ticks(child.pid),
                            command=command,
                        ),
                    )
                    print("started", job["name"], child.pid, sorted(cores), flush=True)
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
                if code or not producer.complete(out, job):
                    failures += 1
                    producer.atomic(
                        out / "failures" / f"{job['name']}.json",
                        dict(
                            job,
                            exit_code=code,
                            terminal=True,
                            allocation_pid=os.getpid(),
                        ),
                    )
                    print("failed", job["name"], code, flush=True)
                else:
                    failures = 0
                    print("done", job["name"], flush=True)
            finally:
                (claim / "owner.json").unlink()
                claim.rmdir()

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        for future in [pool.submit(worker, c) for c in cpus]:
            future.result()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpus", nargs="+", default=["22,23", "24,25"])
    main(p.parse_args())
