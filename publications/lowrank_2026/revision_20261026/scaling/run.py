"""Current-code CPU scaling with independent cut and full-training NN audits.

GPU results are a separate protocol requiring a reservation; these measurements
must never be labelled H100 timings. Plain through 2M, combined through 10k on
CPU. The requested 100k exact-neighbour combined GPU study remains separate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
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
PACKAGE = ROOT.parent / "package"
INPUT = Path(
    "/nfs/lambda_stor_01/homes/rzhu/data_splitting/low_rank/data/chembl/ecfp2048_packed.npy"
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sources():
    paths = [
        *sorted((PACKAGE / "lowrank").rglob("*.py")),
        PACKAGE / "pyproject.toml",
        Path(__file__),
    ]
    return {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}


def ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, ProcessLookupError):
        return None


def jobs():
    items = []
    for n in [10000, 100000, 1000000, 2000000]:
        for epsilon in [0.0, 0.01]:
            for seed in [0, 1, 2] if n in [100000, 2000000] else [0]:
                items.append(
                    dict(
                        name=f"plain__n{n}__eps{epsilon}__s{seed}",
                        n=n,
                        epsilon=epsilon,
                        seed=seed,
                        method="plain",
                    )
                )
    for epsilon in [0.0, 0.01]:
        items.append(
            dict(
                name=f"combined__n10000__eps{epsilon}__s0",
                n=10000,
                epsilon=epsilon,
                seed=0,
                method="combined",
            )
        )
    return items


def case(args):
    import numpy as np
    import torch

    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    if sources() != manifest["sources"]:
        raise RuntimeError("scaling source drift")
    if (
        torch.cuda.is_available()
        or len(os.sched_getaffinity(0)) != 2
        or torch.get_num_threads() != 2
    ):
        raise RuntimeError("requires two CPU slots/threads, with hidden CUDA")
    sys.path.insert(0, str(PACKAGE))
    from lowrank import split, SplitSpec, nystrom_features
    from lowrank.objective import factor_leakage
    from lowrank.diagnostics import factor_pair_error, sampled_leakage
    from lowrank.splitters.common.balanced_assignment import capacity_corridor
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

    directory = args.output / job["name"]
    directory.mkdir(exist_ok=True)
    packed = np.load(INPUT, mmap_mode="r")
    indices = np.sort(np.random.default_rng(0).permutation(len(packed))[: job["n"]])
    if len(indices) != job["n"]:
        raise ValueError("input corpus is smaller than requested sample")
    X = np.unpackbits(np.asarray(packed[indices]), axis=1).astype(np.float32)
    print("loaded", job["name"], X.shape, flush=True)
    names = ["train", "val", "test"]
    data = dict(enumerate(X))
    params = dict(
        rank=256,
        allowed_size_imbalance=job["epsilon"],
        score_exact_leakage=False,
        metric="tanimoto",
        refine_restarts=2,
    )
    if job["method"] == "combined":
        params.update(
            kernel_power=3,
            must_link=0.4,
            must_link_backend="tiled",
            must_link_row_block=256,
            must_link_ref_block=8192,
        )
    warm = dict(params, rank=32)
    started = time.perf_counter()
    split(
        "lowrank",
        dict(enumerate(X[:100])),
        SplitSpec([8, 1, 1], names, epsilon=job["epsilon"]),
        **warm,
    )
    warmup = time.perf_counter() - started
    print("starting timed public split", job["name"], flush=True)
    started = time.perf_counter()
    result = split(
        "lowrank",
        data,
        SplitSpec([8, 1, 1], names, seed=job["seed"], epsilon=job["epsilon"]),
        **params,
    )
    seconds = time.perf_counter() - started
    print("split finished; auditing", job["name"], seconds, flush=True)
    split_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    labels = np.fromiter(
        (names.index(result.assignment[i]) for i in range(len(X))),
        dtype=np.int8,
        count=len(X),
    )
    counts = np.bincount(labels, minlength=3)
    caps, floors = capacity_corridor(len(X), [8, 1, 1], job["epsilon"])
    if not ((counts >= floors) & (counts <= caps)).all():
        raise AssertionError("capacity corridor violated")
    np.save(directory / "labels.npy", labels)
    np.save(directory / "indices.npy", indices)
    # Reconstruct the deterministic actual factor outside the timed split. Verify
    # its landmarks and objective against the public split's saved diagnostics.
    started = time.perf_counter()
    power = params.get("kernel_power", 1)
    B, _, factor_info = nystrom_features(
        X,
        rank=256,
        metric="tanimoto",
        seed=job["seed"],
        kernel_power=power,
        return_info=True,
    )
    actual = result.diagnostics["factorization"]
    if (
        factor_info["landmark_indices"] != actual["landmark_indices"]
        or B.shape[1] != actual["effective_rank"]
    ):
        raise AssertionError("reconstructed factor differs from split factor")
    if abs(factor_leakage(B, labels, 3) - result.diagnostics["factor_leakage"]) > 1e-6:
        raise AssertionError("reconstructed factor objective mismatch")
    ref = torch.as_tensor(X)
    fidelity = factor_pair_error(
        B, ref, "tanimoto", pairs=262144, seed=8000, kernel_power=power, labels=labels
    )
    global_cut = sampled_leakage(
        X, labels, "tanimoto", sample_pairs_count=262144, seed=9000
    )
    factor_audit_seconds = time.perf_counter() - started
    del B
    # Uniform held-out queries, exact maximum over every training molecule.
    # Sampling only queries does not approximate or subsample the train reference.
    started = time.perf_counter()
    test = np.flatnonzero(labels == 2)
    query = np.sort(
        np.random.default_rng(7000).choice(test, min(128, len(test)), replace=False)
    )
    train = np.flatnonzero(labels == 0)
    nearest = np.zeros(len(query), dtype=np.float64)
    for q in range(0, len(query), 16):
        query_rows = ref[query[q : q + 16]]
        best = torch.zeros(len(query_rows))
        for r in range(0, len(train), 8192):
            best = torch.maximum(
                best,
                pairwise_similarity(query_rows, ref[train[r : r + 8192]], "tanimoto")
                .max(1)
                .values,
            )
        nearest[q : q + len(best)] = best.numpy()
    np.save(directory / "query_indices.npy", query)
    np.save(directory / "nearest.npy", nearest)
    radius = float(np.sqrt(np.log(40) / (2 * len(query))))
    local = dict(
        queries=len(query),
        training_references=len(train),
        exact_over_all_training=True,
        nearest_mean=float(nearest.mean()),
        nearest_mean_ci95=[
            max(0.0, float(nearest.mean()) - radius),
            min(1.0, float(nearest.mean()) + radius),
        ],
        fraction_above_06=float((nearest > 0.6).mean()),
        ramp_04_08=float(np.clip((nearest - 0.4) / 0.4, 0, 1).mean()),
        uncertainty="conservative bounded-mean Hoeffding interval over uniformly sampled held-out queries",
    )
    if sources() != manifest["sources"]:
        raise RuntimeError("source changed during scaling")
    atomic(
        directory / "result.json",
        dict(
            job,
            status="ok",
            manifest_sha256=sha(args.output / "manifest.json"),
            host=socket.gethostname(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            params=params,
            seconds=seconds,
            warmup_seconds=warmup,
            peak_split_rss_gib=split_rss,
            process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            factor_audit_seconds=factor_audit_seconds,
            neighbour_audit_seconds=time.perf_counter() - started,
            counts=counts.tolist(),
            fidelity=fidelity,
            global_cut=global_cut,
            local=local,
            diagnostics=result.diagnostics,
            timing_scope="warm public CPU split call; loading, factor reconstruction and independent scoring excluded",
            memory_scope="fresh-process peak RSS; split peak precedes scoring; host input/features included",
            artifacts={p.name: sha(p) for p in directory.glob("*.npy")},
        ),
    )
    print(job["name"], seconds, "ok", flush=True)


def complete(output, job):
    directory = output / job["name"]
    p = directory / "result.json"
    if not p.exists():
        return False
    row = json.loads(p.read_text())
    if (
        row["manifest_sha256"] != sha(output / "manifest.json")
        or not row["artifacts"]
        or any(
            not (directory / f).exists() or sha(directory / f) != h
            for f, h in row["artifacts"].items()
        )
    ):
        raise RuntimeError("scaling output provenance/artifact mismatch")
    return row["status"] == "ok"


def launch(args):
    import psutil

    manifest_path = args.output / "manifest.json"
    proposed = dict(
        jobs=jobs(),
        sources=sources(),
        packed_sha256=sha(INPUT),
        python=sys.executable,
        cpu_sets=args.cpus,
        host=socket.gethostname(),
        timeout_seconds=1800,
        quality_pairs=262144,
        neighbour_queries=128,
        neighbour_reference="entire training fold",
        device="CPU",
        threads=2,
        memory_estimate_per_large_worker_gib=64,
    )
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != proposed:
        raise RuntimeError("scaling manifest drift")
    cpus = [set(map(int, c.split(","))) for c in args.cpus]
    union = set.union(*cpus)
    if (
        any(len(c) != 2 for c in cpus)
        or len(union) != 2 * len(cpus)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("invalid scaling CPU capacity")
    if psutil.virtual_memory().available < 64 * len(cpus) * 1024**3:
        raise RuntimeError("insufficient scaling memory headroom")
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
    atomic(manifest_path, proposed)
    for folder in ["claims", "logs", "failures"]:
        (args.output / folder).mkdir(exist_ok=True)
    work = queue.Queue()
    for index, job in enumerate(proposed["jobs"]):
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
                print("claim requires owner verification", job["name"], flush=True)
                continue
            try:
                owner = dict(
                    pid=os.getpid(),
                    host=socket.gethostname(),
                    start_ticks=ticks(os.getpid()),
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
                    atomic(
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
    done = [j["name"] for j in proposed["jobs"] if complete(args.output, j)]
    atomic(
        args.output / "progress.json",
        dict(
            expected=len(proposed["jobs"]),
            completed=len(done),
            missing=[j["name"] for j in proposed["jobs"] if j["name"] not in done],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["launch", "case"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--cpus", nargs="+", default=["16,17", "18,19", "20,21"])
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    dict(case=case, launch=launch)[args.action](args)
