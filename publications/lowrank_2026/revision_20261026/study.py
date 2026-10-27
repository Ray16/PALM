"""Prespecified CPU studies; atomic claims and immutable producer/input manifests.

Use plan, launch, and case. No target-dependent policy selection or hypergraph
baseline. All case computations run in fresh, two-core, CPU-only subprocesses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import queue
import resource
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "package"
CACHE_ROOT = Path(
    "/nfs/lambda_stor_01/homes/rzhu/data_splitting_lowrank_reviewed/publications/lowrank_2026/correctness_20261025/verified_run"
)
DATASETS = [
    "moleculenet_" + x
    for x in ("freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21")
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (FileNotFoundError, ProcessLookupError):
        return None


def sources():
    paths = [
        *sorted((PACKAGE / "lowrank").rglob("*.py")),
        PACKAGE / "pyproject.toml",
        *sorted(HERE.glob("*.py")),
        HERE.parent / "development/benchmark.py",
    ]
    return {str(p.relative_to(HERE.parent)): sha(p) for p in paths}


def all_jobs():
    jobs = []
    for d in DATASETS:
        for seed in range(3):
            jobs.append(
                dict(name=f"groups__{d}__s{seed}", study="groups", dataset=d, seed=seed)
            )
    for d in ["moleculenet_freesolv", "moleculenet_bace", "moleculenet_qm8"]:
        for seed in range(3):
            for policy, pairs in [
                ("fixed", 2048),
                ("fixed", 32768),
                ("adaptive", 2048),
            ]:
                jobs.append(
                    dict(
                        name=f"budget__{d}__s{seed}__{policy}{pairs}",
                        study="budget",
                        dataset=d,
                        seed=seed,
                        policy=policy,
                        pairs=pairs,
                    )
                )
    for seed in range(20):
        jobs.append(dict(name=f"exact__s{seed}", study="exact", seed=seed))
        jobs.append(dict(name=f"deployment__s{seed}", study="deployment", seed=seed))
    for d in DATASETS:
        for seed in range(3):
            for degree in [16, 32, 128]:
                jobs.append(
                    dict(
                        name=f"graph__{d}__k{degree}__s{seed}",
                        study="frontier",
                        method="graph",
                        dataset=d,
                        seed=seed,
                        degree=degree,
                    )
                )
            for method in ["plain", "combined"]:
                jobs.append(
                    dict(
                        name=f"frontier__{d}__{method}__s{seed}",
                        study="frontier",
                        method=method,
                        dataset=d,
                        seed=seed,
                    )
                )
    for d in DATASETS[:4]:
        for cluster_policy in ["coarse", "medium", "fine"]:
            for seconds in [60, 300]:
                jobs.append(
                    dict(
                        name=f"datasail__{d}__{cluster_policy}__t{seconds}",
                        study="frontier",
                        method="datasail",
                        dataset=d,
                        seed=0,
                        cluster_policy=cluster_policy,
                        solver_seconds=seconds,
                    )
                )
    return jobs


def record_array(output, job, name, a, artifacts):
    path = output / "assignments" / job["name"] / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(path, a)
    artifacts[str(path.relative_to(output))] = sha(path)


def public_data(dataset):
    import numpy as np

    path = CACHE_ROOT / "cache" / f"{dataset}.npz"
    arrays = np.load(path)
    return {key: arrays[key] for key in arrays.files}


def common_score(X, labels, data):
    path = HERE.parent / "development/benchmark.py"
    spec = importlib.util.spec_from_file_location("common_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score(X, labels, data["maccs"], data["y"], "unspecified")


def feasible_groups(gid, counts, seed):
    """Random-preference capacity packing, independent of feature/factor values."""
    import numpy as np
    from lowrank.grouped import _weighted_peel, _whole_group_assignment

    weights = np.bincount(gid)
    lab = _weighted_peel(
        np.random.default_rng(seed).random((len(weights), len(counts))),
        weights,
        counts,
        counts,
    )
    if not np.array_equal(
        np.bincount(lab, weights=weights, minlength=len(counts)), counts
    ):
        lab = _whole_group_assignment(weights, counts, counts, lab, time_limit=30)
    labels = lab[gid]
    check_assignment(labels, counts, gid)
    return labels


def check_assignment(labels, counts, gid=None):
    import numpy as np

    if not np.array_equal(np.bincount(labels, minlength=len(counts)), counts):
        raise AssertionError("exact counts failed")
    if gid is not None:
        lo, hi = np.full(gid.max() + 1, len(counts)), np.full(gid.max() + 1, -1)
        np.minimum.at(lo, gid, labels)
        np.maximum.at(hi, gid, labels)
        if not np.array_equal(lo, hi):
            raise AssertionError("strict group preservation failed")


def groups_case(job, output, artifacts):
    import numpy as np
    from lowrank import nystrom_features
    from lowrank.grouped import grouped_split, must_link_groups
    from lowrank.objective import factor_leakage
    from lowrank.splitters.common.balanced_assignment import target_sizes

    data = public_data(job["dataset"])
    X = data["X"]
    counts = target_sizes(len(X), [8, 1, 1])
    start = time.perf_counter()
    gid = must_link_groups(
        X, "tanimoto", tau=0.4, max_group=max(1, int(0.5 * min(counts))), k=32
    )
    construction = time.perf_counter() - start
    record_array(output, job, "groups", gid, artifacts)
    start = time.perf_counter()
    base = feasible_groups(gid, counts, job["seed"])
    baseline_seconds = time.perf_counter() - start
    record_array(output, job, "feasible", base, artifacts)
    rows = [
        dict(
            method="feasible_groups",
            seconds=construction + baseline_seconds,
            scores=common_score(X, base, data),
        )
    ]
    for power in [1, 3]:
        start = time.perf_counter()
        B, _, info = nystrom_features(
            X,
            metric="tanimoto",
            rank=256,
            kernel_power=power,
            seed=job["seed"],
            return_info=True,
        )
        factor_seconds = time.perf_counter() - start
        start = time.perf_counter()
        lab, opt = grouped_split(
            B,
            gid,
            counts,
            epsilon=0,
            seed=job["seed"],
            init_labels=base,
            n_restarts=4,
            n_iter=25,
            group_policy="strict",
            refine_restarts=2,
        )
        optimize_seconds = time.perf_counter() - start
        check_assignment(lab, counts, gid)
        before, after = factor_leakage(B, base, 3), factor_leakage(B, lab, 3)
        if after > before + 1e-7:
            raise AssertionError("optimization worsened matched feasible incumbent")
        record_array(output, job, f"optimized_p{power}", lab, artifacts)
        rows.append(
            dict(
                method=f"optimized_p{power}",
                seconds=construction
                + baseline_seconds
                + factor_seconds
                + optimize_seconds,
                factor_seconds=factor_seconds,
                optimize_seconds=optimize_seconds,
                factorization=info,
                optimization=opt,
                feasible_factor_ratio=before,
                optimized_factor_ratio=after,
                scores=common_score(X, lab, data),
            )
        )
    return dict(
        rows=rows,
        counts=counts.tolist(),
        groups=int(gid.max() + 1),
        group_sha256=digest_array(gid),
        grouping_seconds=construction,
        feasible_seconds=baseline_seconds,
        protocol="All methods retain identical rows, exact counts and every constructed group; optimization starts from the same feasible incumbent.",
    )


def budget_case(job, output, artifacts):
    import numpy as np
    from lowrank import split, SplitSpec
    from lowrank.splitters.common.balanced_assignment import target_sizes

    data = public_data(job["dataset"])
    X = data["X"]
    params = dict(
        rank="auto",
        rank_min=32,
        rank_max=512,
        rank_tolerance=0.05,
        rank_check="cut_error",
        rank_probe_policy=job["policy"],
        rank_probe_pairs=job["pairs"],
        on_rank_failure="return",
        refine_restarts=2,
        score_exact_leakage=False,
        metric="tanimoto",
    )
    if job["policy"] == "adaptive":
        params["rank_probe_max_pairs"] = 32768
    start = time.perf_counter()
    result = split(
        "lowrank",
        dict(enumerate(X)),
        SplitSpec([8, 1, 1], ["train", "val", "test"], seed=job["seed"]),
        **params,
    )
    seconds = time.perf_counter() - start
    lab = np.array(
        [["train", "val", "test"].index(result.assignment[i]) for i in range(len(X))]
    )
    check_assignment(lab, target_sizes(len(X), [8, 1, 1]))
    record_array(output, job, "labels", lab, artifacts)
    return dict(
        seconds=seconds,
        params=params,
        diagnostics=result.diagnostics,
        scores=common_score(X, lab, data),
    )


def exact_case(job, output, artifacts):
    import numpy as np
    from lowrank import nystrom_features
    from lowrank.diagnostics import prepare_reference
    from lowrank.grouped import grouped_split
    from lowrank.optimize import optimize_restarts
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

    rng = np.random.default_rng(job["seed"])
    X = (rng.random((12, 24)) < 0.3).astype(np.float32)
    counts = np.array([8, 2, 2])
    i, j = np.triu_indices(12, 1)
    rows = []
    for grouped in [False, True]:
        gid = np.arange(12) // (2 if grouped else 1)
        feasible = []
        for test in itertools.combinations(
            range(int(gid.max()) + 1), 1 if grouped else 2
        ):
            remaining = [g for g in range(int(gid.max()) + 1) if g not in test]
            for val in itertools.combinations(remaining, 1 if grouped else 2):
                lab = np.zeros(int(gid.max()) + 1, dtype=np.int64)
                lab[list(test)], lab[list(val)] = 2, 1
                feasible.append(lab[gid])
        feasible = np.stack(feasible)
        mask = feasible[:, i] != feasible[:, j]
        base = feasible[np.random.default_rng(job["seed"]).integers(len(feasible))]
        for power in [1, 3]:
            ref = prepare_reference(X, "tanimoto").double()
            S = pairwise_similarity(ref, ref, "tanimoto").pow(power).numpy()
            exact_costs = mask @ S[i, j]
            mass = float(S[i, j].sum())
            for rank in [2, 4, 8, 12]:
                B, _ = nystrom_features(
                    X,
                    rank=rank,
                    metric="tanimoto",
                    kernel_power=power,
                    seed=job["seed"],
                )
                A = B.astype(np.float64) @ B.astype(np.float64).T
                factor_costs = mask @ A[i, j]
                if grouped:
                    lab, _ = grouped_split(
                        B,
                        gid,
                        counts,
                        epsilon=0,
                        seed=job["seed"],
                        init_labels=base,
                        group_policy="strict",
                        refine_restarts=2,
                    )
                else:
                    lab, _, _ = optimize_restarts(
                        B, counts, seed=job["seed"], refine_restarts=2
                    )
                check_assignment(lab, counts, gid if grouped else None)
                crossing = lab[i] != lab[j]
                found_factor = float(A[i[crossing], j[crossing]].sum())
                found_true = float(S[i[crossing], j[crossing]].sum())
                eta = found_factor - float(factor_costs.min())
                delta = float(np.max(np.abs(factor_costs - exact_costs)))
                gap = found_true - float(exact_costs.min())
                if eta < -1e-6 or gap < -1e-6 or gap > 2 * delta + eta + 1e-6:
                    raise AssertionError("exact-reference decomposition failed")
                rows.append(
                    dict(
                        grouped=grouped,
                        power=power,
                        rank=rank,
                        feasible_partitions=len(feasible),
                        exact_optimum=float(exact_costs.min()),
                        factor_optimum=float(factor_costs.min()),
                        approximation_uniform_error=delta,
                        optimization_gap=eta,
                        true_gap=gap,
                        mass=mass,
                        normalized_true_gap=gap / mass,
                        normalized_optimization_gap=eta / mass,
                    )
                )
                record_array(
                    output, job, f"g{int(grouped)}_p{power}_r{rank}", lab, artifacts
                )
    return dict(rows=rows, feature_sha256=digest_array(X), counts=counts.tolist(), n=12)


def deployment_case(job, output, artifacts):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from lowrank import split, SplitSpec

    seed = job["seed"]
    parameters = np.random.default_rng(10000 + seed)
    effects = parameters.normal(0, 2, 100)

    def observations(families, per_family, draw_seed):
        rng = np.random.default_rng(draw_seed)
        fam = np.repeat(families, per_family)
        z = rng.normal(size=(len(fam), 4))
        y = (
            2 * z[:, 0]
            - z[:, 1]
            + 0.5 * z[:, 2]
            + effects[fam]
            + rng.normal(0, 0.25, len(fam))
        )
        # Nonnegative cosine features encode known family identity and a weaker
        # continuous signal. Generative features are declared, never selected.
        X = np.column_stack([np.eye(100)[fam], 0.05 * (np.clip(z, -4, 4) + 4)]).astype(
            np.float32
        )
        return X, z, y, fam

    X, z, y, fam = observations(np.arange(50), 40, 20000 + seed)
    known = observations(np.arange(50), 40, 30000 + seed)
    novel = observations(np.arange(50, 100), 40, 40000 + seed)
    counts = np.array([1600, 200, 200])
    rng = np.random.default_rng(50000 + seed)
    assignments = {"random": rng.permutation(np.repeat(np.arange(3), counts))}
    family_codes = rng.permutation(np.repeat(np.arange(3), [40, 5, 5]))
    assignments["oracle_family_holdout"] = family_codes[fam]
    times = dict(random=0.0, oracle_family_holdout=0.0)
    for method, params in [
        ("plain", {}),
        ("combined", dict(kernel_power=3, must_link=0.4, group_policy="strict")),
    ]:
        started = time.perf_counter()
        result = split(
            "lowrank",
            dict(enumerate(X)),
            SplitSpec([8, 1, 1], ["train", "val", "test"], seed=seed),
            rank=128,
            metric="cosine",
            factor_backend="auto",
            score_exact_leakage=False,
            **params,
        )
        times[method] = time.perf_counter() - started
        assignments[method] = np.array(
            [
                ["train", "val", "test"].index(result.assignment[i])
                for i in range(len(X))
            ]
        )
    rows = []
    for method, labels in assignments.items():
        check_assignment(labels, counts)
        record_array(output, job, method, labels, artifacts)
        train, test = labels == 0, labels == 2
        seen = np.unique(fam[train])
        known_mask = np.isin(known[3], seen)
        for model_name in ["invariant_ridge", "family_knn"]:
            if model_name == "invariant_ridge":
                model = Ridge(alpha=1).fit(z[train], y[train])
                views = [z[test], known[1][known_mask], novel[1]]
            else:
                model = KNeighborsRegressor(n_neighbors=5, n_jobs=1).fit(
                    X[train], y[train]
                )
                views = [X[test], known[0][known_mask], novel[0]]
            targets = [y[test], known[2][known_mask], novel[2]]
            rmse = [
                float(np.sqrt(np.mean((model.predict(v) - t) ** 2)))
                for v, t in zip(views, targets)
            ]
            rows.append(
                dict(
                    method=method,
                    model=model_name,
                    estimated_rmse=rmse[0],
                    interpolation_rmse=rmse[1],
                    new_family_rmse=rmse[2],
                    interpolation_bias=rmse[0] - rmse[1],
                    new_family_bias=rmse[0] - rmse[2],
                    split_seconds=times[method],
                    train_families=len(seen),
                    test_seen_family_fraction=float(np.isin(fam[test], seen).mean()),
                )
            )
    return dict(
        rows=rows,
        feature_sha256=digest_array(X),
        target_sha256=digest_array(y),
        independent_known_sha256=digest_array(known[2]),
        independent_novel_sha256=digest_array(novel[2]),
        n=len(X),
        counts=counts.tolist(),
        protocol="Synthetic mechanism test. Fixed generative causal features for invariant model; independent cohort observations, shared known-family effects and independently drawn novel-family effects. Interpolation risk is conditional on families present in each training set. No chronological or prospective application claim.",
    )


def case(args):
    import numpy as np
    import torch

    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    if sources() != manifest["sources"]:
        raise RuntimeError("producer source drift")
    if (
        torch.cuda.is_available()
        or len(os.sched_getaffinity(0)) != 2
        or torch.get_num_threads() != 2
    ):
        raise RuntimeError("requires two pinned CPU slots/threads and hidden CUDA")
    if "dataset" in job:
        path = CACHE_ROOT / "cache" / f"{job['dataset']}.npz"
        if sha(path) != manifest["inputs"][job["dataset"]]["cache_sha256"]:
            raise RuntimeError("input drift")
    sys.path.insert(0, str(PACKAGE))
    from lowrank import split, SplitSpec

    start = time.perf_counter()
    split(
        "lowrank",
        dict(enumerate(np.eye(20, dtype=np.float32))),
        SplitSpec([8, 1, 1], ["train", "val", "test"]),
        rank=8,
        score_exact_leakage=False,
    )
    warmup = time.perf_counter() - start
    artifacts = {}
    started = time.perf_counter()
    from frontier import frontier_case

    result = dict(
        groups=groups_case,
        budget=budget_case,
        exact=exact_case,
        deployment=deployment_case,
        frontier=frontier_case,
    )[job["study"]](job, args.output, artifacts)
    if sources() != manifest["sources"]:
        raise RuntimeError("producer changed during run")
    atomic(
        args.output / "rows" / f"{job['name']}.json",
        dict(
            job,
            status="ok",
            manifest_sha256=sha(args.output / "manifest.json"),
            artifacts=artifacts,
            warmup_seconds=warmup,
            case_seconds=time.perf_counter() - started,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            host=socket.gethostname(),
            **result,
        ),
    )
    print(job["name"], "ok", flush=True)


def verify_row(output, job):
    path = output / "rows" / f"{job['name']}.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    if row["manifest_sha256"] != sha(output / "manifest.json") or row["status"] != "ok":
        raise RuntimeError("result provenance/status mismatch")
    if not row["artifacts"] or any(
        not (output / p).exists() or sha(output / p) != h
        for p, h in row["artifacts"].items()
    ):
        raise RuntimeError("missing or changed output artifacts")
    return True


def plan(args):
    datasets = sorted({j["dataset"] for j in all_jobs() if "dataset" in j})
    inputs = {
        d: dict(
            cache_sha256=sha(CACHE_ROOT / "cache" / f"{d}.npz"),
            metadata=json.loads((CACHE_ROOT / "inputs" / f"{d}.json").read_text()),
        )
        for d in datasets
    }
    manifest = dict(
        jobs=all_jobs(),
        sources=sources(),
        inputs=inputs,
        host=socket.gethostname(),
        python=sys.executable,
        cpu_sets=args.cpus,
        threads=2,
        timeout_seconds=1800,
        max_attempts=2,
        policy="All outcomes retained, no target-based tuning. Same-group mechanism controls, exact feasible-set references, joint-budget study and synthetic independent-cohort validation.",
    )
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("manifest drift; choose new output directory")
    atomic(path, manifest)
    print("planned", len(manifest["jobs"]), flush=True)


def launch(args):
    import psutil

    manifest = json.loads((args.output / "manifest.json").read_text())
    if sources() != manifest["sources"]:
        raise RuntimeError("source changed since planning")
    cpus = [set(map(int, c.split(","))) for c in manifest["cpu_sets"]]
    union = set.union(*cpus)
    if (
        any(len(c) != 2 for c in cpus)
        or len(union) != 2 * len(cpus)
        or not union <= os.sched_getaffinity(0)
        or os.getloadavg()[0] + len(union) > os.cpu_count()
    ):
        raise RuntimeError("invalid/disallowed/overloaded CPU allocation")
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
    if psutil.virtual_memory().available < 32 * 1024**3:
        raise RuntimeError("insufficient memory for concurrent studies")
    for folder in ["rows", "logs", "claims", "failures", "assignments"]:
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    todo = queue.Queue()
    for index, job in enumerate(manifest["jobs"]):
        if not verify_row(args.output, job):
            todo.put((index, job))

    def worker(cores):
        consecutive = 0
        while consecutive < 3:
            try:
                index, job = todo.get_nowait()
            except queue.Empty:
                return
            claim = args.output / "claims" / job["name"]
            try:
                claim.mkdir()
            except FileExistsError:
                print("held claim requires owner verification", job["name"], flush=True)
                continue
            owner = dict(
                host=socket.gethostname(),
                pid=os.getpid(),
                start_ticks=ticks(os.getpid()),
                cpus=sorted(cores),
            )
            atomic(claim / "owner.json", owner)
            try:
                if verify_row(args.output, job):
                    continue
                attempts = list(
                    (args.output / "failures").glob(job["name"] + "__attempt*.json")
                )
                if len(attempts) >= manifest["max_attempts"]:
                    continue
                attempt = len(attempts) + 1
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES="",
                    **{
                        k: "2"
                        for k in [
                            "OMP_NUM_THREADS",
                            "OMP_THREAD_LIMIT",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        ]
                    },
                )
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
                log_path = args.output / "logs" / f"{job['name']}__attempt{attempt}.log"
                with log_path.open("w") as log:
                    child = subprocess.Popen(
                        command,
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
                            command=command,
                        ),
                    )
                    try:
                        code = child.wait(timeout=manifest["timeout_seconds"])
                    except subprocess.TimeoutExpired:
                        import signal

                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        code = -999
                if code or not verify_row(args.output, job):
                    atomic(
                        args.output
                        / "failures"
                        / f"{job['name']}__attempt{attempt}.json",
                        dict(job, code=code, log=str(log_path), terminal=True),
                    )
                    consecutive += 1
                    print("failed", job["name"], code, flush=True)
                else:
                    consecutive = 0
                    print("done", job["name"], flush=True)
            finally:
                (claim / "owner.json").unlink()
                claim.rmdir()

    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        for future in [pool.submit(worker, c) for c in cpus]:
            future.result()
    complete = [j["name"] for j in manifest["jobs"] if verify_row(args.output, j)]
    atomic(
        args.output / "progress.json",
        dict(
            expected=len(manifest["jobs"]),
            completed=len(complete),
            missing=[j["name"] for j in manifest["jobs"] if j["name"] not in complete],
        ),
    )
    print("complete", len(complete), "/", len(manifest["jobs"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "launch", "case"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--cpus", nargs="+", default=["3,4", "6,7", "8,9", "10,11", "12,13", "14,15"]
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        dict(plan=plan, launch=launch, case=case)[args.action](args)
    except Exception:
        traceback.print_exc()
        raise
