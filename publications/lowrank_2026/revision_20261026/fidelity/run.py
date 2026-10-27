"""Current-version full-kernel fidelity at frozen returned/perturbed partitions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / "package"
CACHE = Path(
    "/nfs/lambda_stor_01/homes/rzhu/data_splitting_lowrank_reviewed/publications/lowrank_2026/correctness_20261025/verified_run/cache"
)
DATASETS = [
    "moleculenet_" + s
    for s in ["freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21"]
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, ProcessLookupError):
        return None


def sources():
    return {
        str(p.relative_to(ROOT.parent)): sha(p)
        for p in [
            *sorted((PACKAGE / "lowrank").rglob("*.py")),
            PACKAGE / "pyproject.toml",
            Path(__file__),
        ]
    }


def assignment(dataset, seed, method):
    return (
        ROOT
        / "run/assignments"
        / f"frontier__{dataset}__{method}__s{seed}"
        / f"{method}.npy"
    )


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
        raise RuntimeError("fidelity source/resource drift")
    sys.path.insert(0, str(PACKAGE))
    from lowrank import nystrom_features

    path = CACHE / f"{job['dataset']}.npz"
    if sha(path) != manifest["inputs"][job["dataset"]]:
        raise RuntimeError("cache drift")
    X = np.load(path)["X"]
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("reference assumes binary fingerprints")
    # Independent NumPy Tanimoto, not the implementation's torch kernel.
    intersection = X @ X.T
    weights = X.sum(1)
    union = weights[:, None] + weights[None, :] - intersection
    S = np.divide(
        intersection, union, out=np.zeros_like(union, dtype=np.float64), where=union > 0
    )
    del intersection, union
    np.fill_diagonal(S, 0)
    rows = []
    for power, method in [(1, "plain"), (3, "combined")]:
        label_path = assignment(job["dataset"], job["seed"], method)
        if sha(label_path) != manifest["assignments"][str(label_path)]:
            raise RuntimeError("returned assignment drift")
        base = np.load(label_path)
        partitions = [base]
        rng = np.random.default_rng(60000 + job["seed"])
        for fraction in [0.05, 0.1, 0.2, 0.4]:
            lab = base.copy()
            changed = rng.choice(
                len(lab), max(2, int(fraction * len(lab))), replace=False
            )
            lab[changed] = rng.permutation(lab[changed])
            if not np.array_equal(
                np.bincount(lab, minlength=3), np.bincount(base, minlength=3)
            ):
                raise AssertionError("partition perturbation changed counts")
            partitions.append(lab)
        kernel = S if power == 1 else S**power
        total = float(kernel.sum(dtype=np.float64)) / 2
        truths = []
        for lab in partitions:
            within = (
                sum(
                    kernel[np.ix_(lab == c, lab == c)].sum(dtype=np.float64)
                    for c in range(3)
                )
                / 2
            )
            truths.append(total - float(within))
        for rank in [32, 64, 128, 256, 512]:
            started = time.perf_counter()
            B, _, info = nystrom_features(
                X,
                rank=rank,
                metric="tanimoto",
                kernel_power=power,
                seed=job["seed"],
                return_info=True,
            )
            seconds = time.perf_counter() - started
            B = B.astype(np.float64)
            for index, (lab, truth) in enumerate(zip(partitions, truths)):
                sums = np.stack([B[lab == c].sum(0) for c in range(3)])
                approx = 0.5 * (
                    float(np.square(sums.sum(0)).sum()) - float(np.square(sums).sum())
                )
                rows.append(
                    dict(
                        power=power,
                        rank=rank,
                        effective_rank=B.shape[1],
                        partition=index,
                        partition_kind="returned"
                        if index == 0
                        else "count_preserving_perturbation",
                        true_cut=truth,
                        approximate_cut=approx,
                        total_mass=total,
                        abs_cut_error_over_mass=abs(truth - approx) / total,
                        factor_seconds=seconds,
                        factorization=info if index == 0 else None,
                    )
                )
    if sources() != manifest["sources"]:
        raise RuntimeError("fidelity source changed during run")
    atomic(
        args.output / "rows" / f"{job['name']}.json",
        dict(
            job,
            status="ok",
            rows=rows,
            manifest_sha256=sha(args.output / "manifest.json"),
            host=socket.gethostname(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            n=len(X),
        ),
    )
    print(job["name"], "ok", flush=True)


def complete(output, job):
    path = output / "rows" / f"{job['name']}.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    if (
        row["manifest_sha256"] != sha(output / "manifest.json")
        or len(row["rows"]) != 50
    ):
        raise RuntimeError("invalid fidelity output")
    return row["status"] == "ok"


def launch(args):
    import psutil

    jobs = [
        dict(name=f"{d}__s{s}", dataset=d, seed=s) for d in DATASETS for s in range(3)
    ]
    inputs = {d: sha(CACHE / f"{d}.npz") for d in DATASETS}
    assignments = {
        str(assignment(j["dataset"], j["seed"], method)): sha(
            assignment(j["dataset"], j["seed"], method)
        )
        for j in jobs
        for method in ["plain", "combined"]
    }
    manifest = dict(
        jobs=jobs,
        inputs=inputs,
        assignments=assignments,
        sources=sources(),
        cpu_sets=args.cpus,
        host=socket.gethostname(),
        python=sys.executable,
        threads=2,
        timeout_seconds=900,
        policy="Five ranks, two powers, three seeds and six full datasets. Independent dense NumPy kernel reference; one current returned partition plus four count-preserving perturbations per setting. Descriptive fidelity, not independent statistical replication or optimizer certification.",
    )
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("fidelity manifest drift")
    cpus = [set(map(int, c.split(","))) for c in args.cpus]
    union = set.union(*cpus)
    if (
        any(len(c) != 2 for c in cpus)
        or len(union) != 2 * len(cpus)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("fidelity CPU capacity unavailable")
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
    if psutil.virtual_memory().available < 8 * len(cpus) * 1024**3:
        raise RuntimeError("insufficient fidelity memory")
    atomic(path, manifest)
    for name in ["rows", "logs", "claims", "failures"]:
        (args.output / name).mkdir(exist_ok=True)
    work = queue.Queue()
    for index, job in enumerate(jobs):
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
                    start_ticks=ticks(os.getpid()),
                    host=socket.gethostname(),
                    cpus=sorted(cores),
                )
                atomic(claim / "owner.json", owner)
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
                    atomic(
                        claim / "owner.json",
                        dict(
                            owner,
                            child_pid=child.pid,
                            child_start_ticks=ticks(child.pid),
                            command=cmd,
                        ),
                    )
                    try:
                        code = child.wait(timeout=900)
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
                    atomic(
                        args.output / "failures" / f"{job['name']}.json",
                        dict(job, exit_code=code, terminal=True),
                    )
                else:
                    failures = 0
                    print("done", job["name"], flush=True)
            finally:
                (claim / "owner.json").unlink()
                claim.rmdir()

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        for future in [pool.submit(worker, c) for c in cpus]:
            future.result()
    atomic(
        args.output / "progress.json",
        dict(expected=len(jobs), completed=sum(complete(args.output, j) for j in jobs)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["case", "launch"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--cpus", nargs="+", default=["26,28", "29,30"])
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    dict(case=case, launch=launch)[args.action](args)
