"""Paired release/development benchmark, no models or hypergraph comparisons.

Prepare public features once, then run fresh subprocesses with fixed affinity.
Each completed row retains input/code hashes, assignments and exact common scores.
The fixed grid must be written before launching; runtime includes adaptive checks
and local graph construction. Shared preparation and scoring costs are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import resource
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
DATASETS = ["moleculenet_" + s for s in ("freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21")]
SETTINGS = {
    "release_plain": ("release", {}),
    "new_plain": ("development", {}),
    "new_auto": ("development", {"rank": "auto", "rank_min": 32, "rank_max": 512, "rank_tolerance": .05}),
    "release_combined": ("release", {"kernel_power": 3, "must_link": .4}),
    "new_combined": ("development", {"kernel_power": 3, "must_link": .4}),
    "new_auto_combined": ("development", {"rank": "auto", "rank_min": 32, "rank_max": 512,
                                          "rank_tolerance": .05, "kernel_power": 3, "must_link": .4}),
    "new_local_025": ("development", {"local_weight": .25}),
    "new_local_1": ("development", {"local_weight": 1.}),
    "new_combined_strict": ("development", {"kernel_power": 3, "must_link": .4, "group_policy": "strict"}),
    "new_auto_sampled": ("development", {"rank": "auto", "rank_min": 32, "rank_max": 512,
                                         "rank_tolerance": .05, "leakage_diagnostic": "sampled"}),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(package):
    return {str(p.relative_to(package)): sha(p) for p in sorted((package / "lowrank").rglob("*.py"))}


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def prepare(args):
    import numpy as np
    sys.path.insert(0, str(args.development))
    from lowrank.datasets.sources import load_dataset
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys
    for dataset in DATASETS:
        start = time.perf_counter()
        b = load_dataset(dataset, limit=None)
        if not b.available: raise RuntimeError(b.reason)
        ids = sorted(b.feature_data)
        X = np.stack([b.feature_data[i] for i in ids]).astype(np.float32)
        y = np.array([(b.targets or {}).get(i, np.nan) for i in ids])
        maccs = []
        for i in ids:
            row = np.zeros(167, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(b.smiles[i])), row)
            maccs.append(row)
        np.savez_compressed(args.output / "cache" / f"{dataset}.npz", X=X, ids=ids, y=y, maccs=np.stack(maccs))
        raw = Path(os.environ["LOWRANK_DATA"]) / "DataSAIL_data/1D/moleculenet" / (dataset.removeprefix("moleculenet_") + ".csv")
        atomic(args.output / "cache" / f"{dataset}.json", dict(dataset=dataset, n=len(ids), task=b.task_type,
            feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(), ids_sha256=hashlib.sha256(np.array(ids).tobytes()).hexdigest(),
            raw_csv_sha256=sha(raw), preparation_seconds=time.perf_counter() - start,
            feature_recipe="ECFP-1024 radius 2; MACCS-167 used only for independent evaluation",
            rdkit_version=__import__("rdkit").__version__))
        print("prepared", dataset, len(ids), flush=True)


def score(X, labels, maccs, y, task):
    import numpy as np
    import torch
    from scipy.stats import wasserstein_distance
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
    ref = torch.as_tensor(X)
    lab = torch.as_tensor(labels)
    total, cross = 0., 0.
    nearest = []
    top5 = []
    for s in range(0, len(X), 256):
        S = pairwise_similarity(ref[s:s + 256], ref, "tanimoto")
        rows = torch.arange(S.shape[0]); S[rows, rows + s] = 0
        total += float(S.sum(dtype=torch.float64))
        mask = lab[s:s + len(S), None] != lab[None, :]
        cross += float(S[mask].sum(dtype=torch.float64))
        local = S[lab[s:s + len(S)] == 2][:, lab == 0]
        if len(local):
            nearest.extend(local.max(1).values.tolist())
            top5.extend(local.topk(min(5, local.shape[1]), dim=1).values.mean(1).tolist())
    nearest = np.asarray(nearest)
    independent = []
    reference = torch.as_tensor(maccs[labels == 0], dtype=torch.float32)
    test = torch.as_tensor(maccs[labels == 2], dtype=torch.float32)
    for s in range(0, len(test), 256):
        independent.extend(pairwise_similarity(test[s:s + 256], reference, "tanimoto").max(1).values.tolist())
    composition = {}
    for c, name in enumerate(["train", "val", "test"]):
        v = y[(labels == c) & np.isfinite(y)]
        composition[name] = dict(n_labels=len(v), mean=float(v.mean()) if len(v) else None,
                                 std=float(v.std()) if len(v) else None)
    a = y[(labels == 0) & np.isfinite(y)]; b = y[(labels == 2) & np.isfinite(y)]
    return dict(global_ratio=cross / total if total > 0 else None,
        mean_nearest_training=float(nearest.mean()), top5_training_mean=float(np.mean(top5)),
        local_ramp_04=float(np.maximum(0, (nearest - .4) / .6).mean()),
        test_fraction_above_04=float((nearest > .4).mean()),
        test_fraction_above_06=float((nearest > .6).mean()),
        test_fraction_above_08=float((nearest > .8).mean()),
        maccs_mean_nearest_training=float(np.mean(independent)),
        composition=composition, task=task,
        target_wasserstein=float(wasserstein_distance(a, b)) if len(a) and len(b) else None)


def case(args):
    import numpy as np
    import torch
    manifest = json.loads((args.output / "manifest.json").read_text())
    job = manifest["jobs"][args.index]
    package = Path(manifest["implementations"][job["implementation"]])
    if sources(package) != manifest["source_files"][job["implementation"]]: raise RuntimeError("source drift")
    sys.path.insert(0, str(package))
    from lowrank import split, SplitSpec
    if torch.cuda.is_available() or torch.get_num_threads() != 2 or len(os.sched_getaffinity(0)) != 2:
        raise RuntimeError("case must run on CPU with two native threads and two CPU slots")
    start = time.perf_counter()
    bundle = np.load(args.output / "cache" / f"{job['dataset']}.npz")
    X, ids, y, maccs = [bundle[k] for k in ["X", "ids", "y", "maccs"]]
    meta = json.loads((args.output / "cache" / f"{job['dataset']}.json").read_text())
    if hashlib.sha256(X.tobytes()).hexdigest() != meta["feature_sha256"]: raise RuntimeError("input drift")
    data = {int(i): row for i, row in zip(ids, X)}
    loading = time.perf_counter() - start
    start = time.perf_counter()
    split("lowrank", dict(enumerate(np.eye(20, dtype=np.float32))),
          SplitSpec([8, 1, 1], ["train", "val", "test"]), rank=8, score_exact_leakage=False)
    warmup = time.perf_counter() - start
    params = dict(job["params"], score_exact_leakage=False)
    spec = SplitSpec([8, 1, 1], ["train", "val", "test"], seed=job["seed"])
    row = dict(job, feature_sha256=meta["feature_sha256"], loading_seconds=loading,
               preparation_seconds=meta["preparation_seconds"], warmup_seconds=warmup,
               source_files=manifest["source_files"][job["implementation"]],
               cpu_affinity=sorted(os.sched_getaffinity(0)), host=socket.gethostname())
    start = time.perf_counter()
    try:
        result = split("lowrank", data, spec, **params)
    except ValueError as exc:
        if exc.__class__.__name__ != "GroupInfeasibleError": raise
        row.update(status="proven_infeasible", reason=str(exc), split_seconds=time.perf_counter() - start)
    else:
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        names = ["train", "val", "test"]
        labels = np.array([names.index(result.assignment[int(i)]) for i in ids], dtype=np.uint8)
        from lowrank.splitters.common.balanced_assignment import target_sizes
        if not np.array_equal(np.bincount(labels, minlength=3), target_sizes(len(X), [8, 1, 1])):
            raise RuntimeError("exact-count/retention contract failed")
        assignment = args.output / "rows" / (job["name"] + ".npy")
        np.save(assignment, labels)
        start = time.perf_counter()
        scores = score(X, labels, maccs, y, meta["task"])
        scoring = time.perf_counter() - start
        row.update(status="ok", split_seconds=elapsed, scoring_seconds=scoring,
                   preparation_plus_split_plus_scoring_seconds=meta["preparation_seconds"] + elapsed + scoring,
                   peak_rss_after_split_mb=rss, process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   diagnostics=result.diagnostics, assignment_sha256=sha(assignment), scores=scores)
    if sources(package) != manifest["source_files"][job["implementation"]]: raise RuntimeError("source drift during case")
    atomic(args.output / "rows" / (job["name"] + ".json"), row)
    print(job["name"], row["status"], round(row["split_seconds"], 3), flush=True)


def launch(args):
    import psutil
    output = args.output
    for folder in ("rows", "logs", "cache", "claims"): (output / folder).mkdir(parents=True, exist_ok=True)
    implementations = dict(release=str(args.release.resolve()), development=str(args.development.resolve()))
    jobs = []
    for dataset in DATASETS:
        for seed in range(3):
            for setting, (implementation, params) in SETTINGS.items():
                jobs.append(dict(name=f"{dataset}__s{seed}__{setting}", dataset=dataset, seed=seed,
                                 setting=setting, implementation=implementation, params=params))
    cpus = [set(map(int, group.split(','))) for group in args.cpus]
    if any(len(c) != 2 for c in cpus) or len(set.union(*cpus)) != 2 * len(cpus):
        raise RuntimeError("each worker requires two disjoint CPU slots")
    allowed = os.sched_getaffinity(0)
    if not set.union(*cpus) <= allowed or os.getloadavg()[0] + 2 * len(cpus) > os.cpu_count():
        raise RuntimeError("insufficient verified CPU capacity")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if not proc.info["cmdline"]: continue
            affinity = proc.cpu_affinity()
            if len(affinity) <= 8 and set(affinity) & set.union(*cpus):
                raise RuntimeError(f"worker CPU slots overlap narrow process {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    manifest = dict(implementations=implementations, jobs=jobs, cpu_sets=[sorted(c) for c in cpus],
        source_files={k: sources(Path(v)) for k,v in implementations.items()}, benchmark_sha256=sha(__file__),
        policy="six full datasets; three seeds; settings fixed before execution; exact counts and common exact scorer; "
               "no model fitting, hypergraph methods or test-outcome tuning; all failed/infeasible cases retained",
        host=socket.gethostname(), threads_per_worker=2)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest: raise RuntimeError("manifest drift; use a new output directory")
    atomic(path, manifest)
    work = queue.Queue()
    for index, job in enumerate(jobs):
        final = output / "rows" / (job["name"] + ".json")
        if final.exists():
            row = json.loads(final.read_text())
            if row["source_files"] != manifest["source_files"][job["implementation"]]: raise RuntimeError("completed source drift")
            if row["status"] == "ok" and sha(final.with_suffix('.npy')) != row["assignment_sha256"]: raise RuntimeError("assignment drift")
            continue
        work.put((index, job))
    def worker(cpu_set):
        while True:
            try: index, job = work.get_nowait()
            except queue.Empty: return
            claim = output / "claims" / job["name"]
            try: claim.mkdir()
            except FileExistsError: raise RuntimeError("existing claim requires live-owner inspection: " + str(claim))
            atomic(claim / "owner.json", dict(host=socket.gethostname(), pid=os.getpid(),
                                             process_start=psutil.Process().create_time(), cpus=sorted(cpu_set)))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OMP_THREAD_LIMIT="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
            cmd = ["taskset", "-c", ','.join(map(str, sorted(cpu_set))), sys.executable, str(Path(__file__).resolve()),
                   "case", "--output", str(output), "--index", str(index)]
            with (output / "logs" / (job["name"] + ".log")).open('w') as log:
                completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            (claim / "owner.json").unlink(); claim.rmdir()
            if completed.returncode: raise RuntimeError("case failed; inspect " + job["name"])
            print("done", job["name"], flush=True)
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [pool.submit(worker, c) for c in cpus]
        for future in futures: future.result()
    rows = [json.loads((output / "rows" / (job["name"] + ".json")).read_text()) for job in jobs]
    atomic(output / "complete.json", dict(expected=len(jobs), completed=len(rows),
        ok=sum(r["status"] == "ok" for r in rows), infeasible=sum(r["status"] == "proven_infeasible" for r in rows)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "launch", "case"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--development", type=Path)
    p.add_argument("--release", type=Path)
    p.add_argument("--cpus", nargs='+', default=["50,51", "52,53", "54,55"])
    p.add_argument("--index", type=int)
    args = p.parse_args(); (args.output / "cache").mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "launch": launch, "case": case}[args.action](args)
