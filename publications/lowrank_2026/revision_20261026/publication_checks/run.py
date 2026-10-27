"""Equal-confidence audit controls and native-method memory profiles.

The numerical package remains frozen. A fixed pair-count grid is obtained using
the public joint-valid policy with initial=max probes, so its coverage matches
the adaptive grid. Native memory is captured before the first scoring call.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import resource
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import study


def sources():
    result = study.sources()
    result[str(Path(__file__).relative_to(ROOT.parent))] = study.sha(__file__)
    return result


def jobs():
    items = []
    # Start the slow controls first; free workers can complete other checks.
    for d in ["moleculenet_qm8", "moleculenet_freesolv", "moleculenet_bace"]:
        for pairs in [2048, 32768]:
            for seed in range(3):
                items.append(
                    dict(
                        name=f"fixed_joint__{d}__q{pairs}__s{seed}",
                        study="equal_confidence",
                        dataset=d,
                        seed=seed,
                        pairs=pairs,
                    )
                )
    for job in study.all_jobs():
        if job["study"] == "frontier" and job["seed"] == 0:
            items.append(dict(job, name="memory__" + job["name"], study="memory"))
    return items


def case(args):
    import numpy as np
    import torch

    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    if (
        sources() != manifest["sources"]
        or torch.cuda.is_available()
        or torch.get_num_threads() != 2
        or len(os.sched_getaffinity(0)) != 2
    ):
        raise RuntimeError("publication-check source/resource drift")
    sys.path.insert(0, str(study.PACKAGE))
    from lowrank import split, SplitSpec

    split(
        "lowrank",
        dict(enumerate(np.eye(20, dtype=np.float32))),
        SplitSpec([8, 1, 1], ["train", "val", "test"]),
        rank=8,
        score_exact_leakage=False,
    )
    artifacts = {}
    started = time.perf_counter()
    if job["study"] == "equal_confidence":
        data = study.public_data(job["dataset"])
        X = data["X"]
        params = dict(
            rank="auto",
            rank_min=32,
            rank_max=512,
            rank_tolerance=0.05,
            rank_check="cut_error",
            rank_probe_policy="adaptive",
            rank_probe_pairs=job["pairs"],
            rank_probe_max_pairs=job["pairs"],
            refine_restarts=2,
            metric="tanimoto",
            on_rank_failure="return",
            score_exact_leakage=False,
        )
        t = time.perf_counter()
        result = split(
            "lowrank",
            dict(enumerate(X)),
            SplitSpec([8, 1, 1], ["train", "val", "test"], seed=job["seed"]),
            **params,
        )
        seconds = time.perf_counter() - t
        labels = np.array(
            [
                ["train", "val", "test"].index(result.assignment[i])
                for i in range(len(X))
            ]
        )
        from lowrank.splitters.common.balanced_assignment import target_sizes

        study.check_assignment(labels, target_sizes(len(X), [8, 1, 1]))
        study.record_array(args.output, job, "labels", labels, artifacts)
        details = dict(
            status="ok",
            seconds=seconds,
            params=params,
            diagnostics=result.diagnostics,
            scores=study.common_score(X, labels, data),
            protocol="Fixed-size probe grid under the same 95% joint selection/final-audit coverage as joint allocation.",
        )
    else:
        from frontier import frontier_case

        peak = []
        statuses = []
        original_score = study.common_score

        def measured_score(*a, **kw):
            if not peak:
                peak.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            return original_score(*a, **kw)

        class Capture(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                if "SCIP status:" in msg or "No assignments could be made" in msg:
                    statuses.append(msg)

        handler = Capture()
        logger = None
        if job["method"] == "datasail":
            from datasail.settings import LOGGER

            logger = LOGGER
            logger.addHandler(handler)
        study.common_score = measured_score
        try:
            try:
                details = frontier_case(job, args.output, artifacts)
            except TypeError:
                if job["method"] != "datasail" or not any(
                    "SCIP status: infeasible" in s for s in statuses
                ):
                    raise
                details = dict(
                    status="native_infeasible",
                    solver_evidence=statuses,
                    protocol="Native SCIP reported infeasibility and DataSAIL returned no assignment; no score or native timing is imputed.",
                )
            else:
                if not peak:
                    raise AssertionError("native pre-scoring memory was not observed")
                details.update(
                    status="ok",
                    native_peak_rss_before_scoring_mb=peak[0],
                    solver_evidence=statuses,
                    memory_scope="Fresh-process native method peak before first scoring call or matched controls; includes imports, input/features and common small low-rank warm-up.",
                )
        finally:
            study.common_score = original_score
            if logger is not None:
                logger.removeHandler(handler)
    if sources() != manifest["sources"]:
        raise RuntimeError("publication-check source changed during run")
    study.atomic(
        args.output / "rows" / f"{job['name']}.json",
        dict(
            job,
            **details,
            artifacts=artifacts,
            manifest_sha256=study.sha(args.output / "manifest.json"),
            case_seconds=time.perf_counter() - started,
            host=socket.gethostname(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
        ),
    )
    print(job["name"], details["status"], flush=True)


def complete(output, job):
    p = output / "rows" / f"{job['name']}.json"
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    if r["manifest_sha256"] != study.sha(output / "manifest.json"):
        raise RuntimeError("publication-check row provenance mismatch")
    if r["status"] == "native_infeasible":
        if job.get("method") != "datasail" or not any(
            "SCIP status: infeasible" in x for x in r["solver_evidence"]
        ):
            raise RuntimeError("unproved native exclusion")
        return True
    if (
        r["status"] != "ok"
        or not r["artifacts"]
        or any(
            not (output / p).exists() or study.sha(output / p) != h
            for p, h in r["artifacts"].items()
        )
    ):
        raise RuntimeError("invalid publication-check artifacts")
    return True


def launch(args):
    import psutil

    items = jobs()
    datasets = sorted({j["dataset"] for j in items})
    manifest = dict(
        jobs=items,
        sources=sources(),
        inputs={
            d: study.sha(study.CACHE_ROOT / "cache" / f"{d}.npz") for d in datasets
        },
        host=socket.gethostname(),
        python=sys.executable,
        cpu_sets=args.cpus,
        threads=2,
        timeout_seconds=1800,
        protocol="Equal-confidence fixed probe grids; independent native-memory profiles at seed 0 for all graph degrees, low-rank settings and DataSAIL cluster/time budgets. All solver outcomes retained.",
    )
    p = args.output / "manifest.json"
    if p.exists() and json.loads(p.read_text()) != manifest:
        raise RuntimeError("publication-check manifest drift")
    cpus = [set(map(int, c.split(","))) for c in args.cpus]
    union = set.union(*cpus)
    if (
        any(len(c) != 2 for c in cpus)
        or len(union) != 2 * len(cpus)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("CPU capacity unavailable")
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
    if psutil.virtual_memory().available < 32 * 1024**3:
        raise RuntimeError("insufficient memory")
    study.atomic(args.output / "manifest.json", manifest)
    for name in ["rows", "logs", "claims", "failures"]:
        (args.output / name).mkdir(exist_ok=True)
    work = queue.Queue()
    for index, job in enumerate(items):
        if not complete(args.output, job):
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
                continue
            try:
                owner = dict(
                    pid=os.getpid(),
                    start_ticks=study.ticks(os.getpid()),
                    host=socket.gethostname(),
                    cpus=sorted(cores),
                )
                study.atomic(claim / "owner.json", owner)
                if complete(args.output, job):
                    continue
                if (args.output / "failures" / f"{job['name']}.json").exists():
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
                cmd = [
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
                with (args.output / "logs" / f"{job['name']}.log").open("w") as log:
                    child = subprocess.Popen(
                        cmd,
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
                            command=cmd,
                        ),
                    )
                    try:
                        code = child.wait(timeout=1800)
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
    study.atomic(
        args.output / "progress.json",
        dict(
            expected=len(items), completed=sum(complete(args.output, j) for j in items)
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["case", "launch"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--cpus", nargs="+", default=["3,4", "6,7", "8,9", "10,11"])
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    dict(case=case, launch=launch)[args.action](args)
