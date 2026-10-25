"""Prespecified, resumable comparison against the preceding development version.

Nine datasets, three seeds, nine fixed configurations. Six fresh CPU subprocesses
can run concurrently with disjoint two-core affinities. No hypergraph baseline,
test-outcome tuning, or claim of observed chronological generalization.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DATASETS = [
    "moleculenet_" + s
    for s in (
        "freesolv",
        "esol",
        "bace",
        "bbbp",
        "lipophilicity",
        "tox21",
        "clintox",
        "sider",
        "qm8",
    )
]
WEIGHTS = [[0.0, 1.0, 2.0], [1.0, 0.0, 0.25], [2.0, 0.25, 0.0]]
SETTINGS = {
    "previous_plain": ("previous", {}),
    "new_plain": ("new", {}),
    "new_budget1": ("new", {"refine_restarts": 1}),
    "previous_combined": ("previous", {"kernel_power": 3, "must_link": 0.4}),
    "new_combined": ("new", {"kernel_power": 3, "must_link": 0.4}),
    "new_strict": (
        "new",
        {"kernel_power": 3, "must_link": 0.4, "group_policy": "strict"},
    ),
    "new_auto_pair": (
        "new",
        {
            "rank": "auto",
            "rank_min": 32,
            "rank_max": 512,
            "rank_tolerance": 0.05,
            "on_rank_failure": "return",
            "refine_restarts": 1,
        },
    ),
    "new_auto_cut": (
        "new",
        {
            "rank": "auto",
            "rank_min": 32,
            "rank_max": 512,
            "rank_tolerance": 0.05,
            "on_rank_failure": "return",
            "refine_restarts": 1,
            "rank_check": "cut_error",
        },
    ),
    "new_weighted": ("new", {"fold_weights": WEIGHTS}),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def source_identity(package):
    files = {
        str(p.relative_to(package)): sha(p)
        for p in sorted((package / "lowrank").rglob("*.py"))
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(files=files, sha256=digest)


def prepare(args):
    import numpy as np

    sys.path.insert(0, str(args.new))
    from lowrank.datasets.sources import load_dataset
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys

    for name in DATASETS:
        started = time.perf_counter()
        bundle = load_dataset(name, limit=None)
        if not bundle.available:
            raise RuntimeError(bundle.reason)
        ids = np.array(sorted(bundle.feature_data))
        X = np.stack([bundle.feature_data[int(i)] for i in ids]).astype(np.float32)
        y = np.array([(bundle.targets or {}).get(int(i), np.nan) for i in ids])
        maccs = []
        for i in ids:
            row = np.zeros(167, dtype=np.uint8)
            fp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(bundle.smiles[int(i)]))
            DataStructs.ConvertToNumpyArray(fp, row)
            maccs.append(row)
        path = args.output / "cache" / f"{name}.npz"
        np.savez_compressed(path, X=X, ids=ids, y=y, maccs=np.stack(maccs))
        raw = (
            Path(os.environ["LOWRANK_DATA"])
            / "DataSAIL_data/1D/moleculenet"
            / f"{name.removeprefix('moleculenet_')}.csv"
        )
        atomic(
            args.output / "inputs" / f"{name}.json",
            dict(
                dataset=name,
                n=len(ids),
                task=bundle.task_type,
                feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                cache_sha256=sha(path),
                raw_csv_sha256=sha(raw),
                preparation_seconds=time.perf_counter() - started,
                recipe="ECFP-1024 radius 2; MACCS-167 only for evaluation",
                additional_validation=name in DATASETS[-3:],
            ),
        )
        print("prepared", name, len(ids), flush=True)


def scorer():
    # Reuse the previously independently checked common molecular scorer.
    path = HERE.parent / "development/benchmark.py"
    spec = importlib.util.spec_from_file_location("previous_common_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score


def weighted_score(X, labels):
    import torch
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

    ref = torch.as_tensor(X)
    lab = torch.as_tensor(labels, dtype=torch.long)
    W = torch.as_tensor(WEIGHTS, dtype=torch.float64)
    total, cross = 0.0, 0.0
    for start in range(0, len(X), 256):
        S = pairwise_similarity(ref[start : start + 256], ref, "tanimoto")
        rows = torch.arange(len(S))
        S[rows, rows + start] = 0
        total += float(S.sum(dtype=torch.float64))
        cross += float((S * W[lab[start : start + len(S), None], lab[None, :]]).sum())
    return cross / total if total > 0 else None


def case(args):
    import numpy as np
    import torch

    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text())
    job = manifest["jobs"][args.index]
    package = Path(manifest["implementations"][job["implementation"]])
    identity = manifest["sources"][job["implementation"]]
    if source_identity(package) != identity:
        raise RuntimeError("implementation changed")
    if sha(__file__) != manifest["runner_sha256"]:
        raise RuntimeError("runner changed")
    sys.path.insert(0, str(package))
    from lowrank import split, SplitSpec
    from lowrank.splitters.common.balanced_assignment import target_sizes

    if (
        torch.cuda.is_available()
        or torch.get_num_threads() != 2
        or len(os.sched_getaffinity(0)) != 2
    ):
        raise RuntimeError("case requires two CPU slots/threads with hidden CUDA")
    meta = json.loads((args.output / "inputs" / f"{job['dataset']}.json").read_text())
    cache = args.output / "cache" / f"{job['dataset']}.npz"
    if sha(cache) != meta["cache_sha256"]:
        raise RuntimeError("cache drift")
    started = time.perf_counter()
    bundle = np.load(cache)
    X, ids, y, maccs = [bundle[k] for k in ["X", "ids", "y", "maccs"]]
    data = {int(i): row for i, row in zip(ids, X)}
    loading = time.perf_counter() - started
    warmup = time.perf_counter()
    split(
        "lowrank",
        dict(enumerate(np.eye(20, dtype=np.float32))),
        SplitSpec([8, 1, 1], ["train", "val", "test"]),
        rank=8,
        score_exact_leakage=False,
    )
    warmup = time.perf_counter() - warmup
    spec = SplitSpec([8, 1, 1], ["train", "val", "test"], seed=job["seed"])
    row = dict(
        job,
        manifest_sha256=sha(path),
        source_sha256=identity["sha256"],
        feature_sha256=meta["feature_sha256"],
        preparation_seconds=meta["preparation_seconds"],
        loading_seconds=loading,
        warmup_seconds=warmup,
        host=socket.gethostname(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
    )
    started = time.perf_counter()
    try:
        result = split(
            "lowrank", data, spec, score_exact_leakage=False, **job["params"]
        )
    except (ValueError, RuntimeError) as exc:
        if type(exc).__name__ not in (
            "GroupInfeasibleError",
            "GroupFeasibilityUnresolved",
        ):
            raise
        row.update(
            status=type(exc).__name__,
            reason=str(exc),
            split_seconds=time.perf_counter() - started,
        )
    else:
        elapsed = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        names = ["train", "val", "test"]
        labels = np.array(
            [names.index(result.assignment[int(i)]) for i in ids], dtype=np.uint8
        )
        if len(result.assignment) != len(ids) or not np.array_equal(
            np.bincount(labels, minlength=3), target_sizes(len(ids), [8, 1, 1])
        ):
            raise RuntimeError("retention/count contract failed")
        target = args.output / "rows" / (job["name"] + ".npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, labels)
        started = time.perf_counter()
        scores = scorer()(X, labels, maccs, y, meta["task"])
        scores["weighted_global_ratio"] = weighted_score(X, labels)
        row.update(
            status="ok",
            split_seconds=elapsed,
            scoring_seconds=time.perf_counter() - started,
            peak_rss_after_split_mb=rss,
            process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            diagnostics=result.diagnostics,
            assignment_sha256=sha(target),
            scores=scores,
        )
    if source_identity(package) != identity:
        raise RuntimeError("source changed during case")
    atomic(args.output / "rows" / (job["name"] + ".json"), row)
    print(job["name"], row["status"], round(row["split_seconds"], 3), flush=True)


def launch(args):
    import psutil

    for folder in ["cache", "rows", "logs", "claims"]:
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    cpus = [set(map(int, c.split(","))) for c in args.cpus]
    combined = set.union(*cpus)
    if any(len(c) != 2 for c in cpus) or len(combined) != 2 * len(cpus):
        raise RuntimeError("CPU sets must be disjoint pairs")
    if (
        not combined <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(combined) > os.cpu_count()
    ):
        raise RuntimeError("insufficient verified CPU capacity")
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if (
                proc.info["cmdline"]
                and len(proc.cpu_affinity()) <= 8
                and set(proc.cpu_affinity()) & combined
            ):
                raise RuntimeError(f"CPU overlap with narrow process {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    jobs = [
        dict(
            name=f"{d}__s{s}__{name}",
            dataset=d,
            seed=s,
            setting=name,
            implementation=implementation,
            params=params,
        )
        for d in DATASETS
        for s in range(3)
        for name, (implementation, params) in SETTINGS.items()
    ]
    implementations = dict(
        previous=str(args.previous.resolve()), new=str(args.new.resolve())
    )
    manifest = dict(
        jobs=jobs,
        implementations=implementations,
        sources={
            key: source_identity(Path(value)) for key, value in implementations.items()
        },
        runner_sha256=sha(__file__),
        scorer_sha256=sha(HERE.parent / "development/benchmark.py"),
        cpu_sets=[sorted(c) for c in cpus],
        host=socket.gethostname(),
        threads=2,
        protocol="Nine fixed settings, nine full datasets, three seeds; previous six plus three additional validation datasets. "
        "One refined restart per rank trial in both rank ablations. Weighted setting is a declared policy example, "
        "not a tuned winner. No hypergraph baseline, model-outcome tuning or chronological claim.",
    )
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("manifest drift; use a new run directory")
    for name in DATASETS:
        if not (args.output / "inputs" / f"{name}.json").exists():
            raise RuntimeError("prepare all inputs before launch")
    atomic(path, manifest)
    if args.plan_only:
        print(f"planned {len(jobs)} cases", flush=True)
        return
    work = queue.Queue()
    for index, job in enumerate(jobs):
        final = args.output / "rows" / (job["name"] + ".json")
        if final.exists():
            row = json.loads(final.read_text())
            if row["manifest_sha256"] != sha(path):
                raise RuntimeError("completed manifest mismatch")
            if (
                row["status"] == "ok"
                and sha(final.with_suffix(".npy")) != row["assignment_sha256"]
            ):
                raise RuntimeError("assignment drift")
        else:
            work.put((index, job))

    def worker(cpus):
        while True:
            try:
                index, job = work.get_nowait()
            except queue.Empty:
                return
            claim = args.output / "claims" / job["name"]
            claim.mkdir()
            owner = dict(
                host=socket.gethostname(),
                pid=os.getpid(),
                process_start=psutil.Process().create_time(),
                cpus=sorted(cpus),
            )
            atomic(claim / "owner.json", owner)
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
                ",".join(map(str, sorted(cpus))),
                sys.executable,
                str(Path(__file__).resolve()),
                "case",
                "--output",
                str(args.output),
                "--index",
                str(index),
            ]
            with (args.output / "logs" / (job["name"] + ".log")).open("w") as log:
                started = time.perf_counter()
                child = subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                atomic(
                    claim / "owner.json",
                    dict(
                        owner,
                        child_pid=child.pid,
                        child_start=psutil.Process(child.pid).create_time(),
                    ),
                )
                try:
                    code = child.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise RuntimeError("case timeout: " + job["name"])
            if code:
                raise RuntimeError("case failed: " + job["name"])
            final = args.output / "rows" / (job["name"] + ".json")
            row = json.loads(final.read_text())
            row["fresh_process_wall_seconds"] = time.perf_counter() - started
            atomic(final, row)
            (claim / "owner.json").unlink()
            claim.rmdir()
            print("done", job["name"], flush=True)

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [pool.submit(worker, c) for c in cpus]
        for future in futures:
            future.result()
    rows = [
        json.loads((args.output / "rows" / (job["name"] + ".json")).read_text())
        for job in jobs
    ]
    atomic(
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
    p.add_argument("action", choices=["prepare", "case", "launch"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--new", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--cpus",
        nargs="+",
        default=["64,65", "66,67", "68,69", "70,71", "72,73", "74,75"],
    )
    args = p.parse_args()
    (args.output / "cache").mkdir(parents=True, exist_ok=True)
    dict(prepare=prepare, case=case, launch=launch)[args.action](args)
