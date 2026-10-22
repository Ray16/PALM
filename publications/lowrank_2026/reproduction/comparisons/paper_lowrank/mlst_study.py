"""Versioned MLST follow-up studies; one resumable job per dataset/study.

Run --jobs to inspect the fixed grid, --job STUDY DATASET for one job, or
--worker for an atomically claimed shared queue. All data, assignments, failures,
source identities and model settings are retained under low_rank/paper/mlst_2026.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _env  # noqa: E402,F401
from datasets.sources import load_dataset
from lowrank.grouped import topk_neighbours
from lowrank.neighbours import approximate_tanimoto_neighbours
from lowrank.objective import factor_leakage
from lowrank.optimize import interpolate_to_random
from lowrank.target_gap import GapCalibrator
from splitters import SplitSpec, split
from splitters.common.balanced_assignment import target_sizes
from splitters.common.feature_preparation import choose_metric
from splitters.common.pairwise_similarity import pairwise_similarity, standardize_for_metric

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "low_rank/paper/mlst_2026"
DATASETS = ["moleculenet_" + x for x in
            ("freesolv", "esol", "bace", "bbbp", "sider", "clintox", "lipophilicity",
             "tox21", "qm8", "hiv", "muv")] + ["qmof", "materials_project"]
ABLATION_DATASETS = ["moleculenet_" + x for x in
                     ("freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21")]
CAL_DATASETS = ["moleculenet_" + x for x in ("freesolv", "esol", "bace", "bbbp", "lipophilicity", "sider")]
NAMES = np.array(["train", "val", "test"])
SEEDS = (0, 1, 2)
MODEL_SEEDS = (0, 1)
MODEL_THREADS = 2
CORE = {"random": None, "scaffold": None, "butina": None,
        "plain": {}, "power3": {"kernel_power": 3},
        "groups": {"must_link": 0.4},
        "tail": {"kernel_power": 3, "must_link": 0.4},
        "tail_ann": {"kernel_power": 3, "must_link": 0.4, "must_link_backend": "nndescent"}}
ABLATIONS = {
    "plain": {}, "rank32": {"rank": 32}, "rank64": {"rank": 64}, "rank128": {"rank": 128},
    "uniform": {"landmark": "uniform"}, "one_restart": {"n_restarts": 1}, "no_fm": {"fm": False},
    "tolerance05": {"allowed_size_imbalance": 0.05},
    "power2": {"kernel_power": 2}, "power3": {"kernel_power": 3},
    "groups": {"must_link": 0.4}, "tail": {"kernel_power": 3, "must_link": 0.4},
    "tau05": {"kernel_power": 3, "must_link": 0.5},
    "tau06": {"kernel_power": 3, "must_link": 0.6},
    "cap25": {"kernel_power": 3, "must_link": 0.4, "max_group_frac": 0.25},
    "cap100": {"kernel_power": 3, "must_link": 0.4, "max_group_frac": 1.0},
    "cap_skip": {"kernel_power": 3, "must_link": 0.4, "must_link_cap": "skip"},
    "tail_ann": {"kernel_power": 3, "must_link": 0.4, "must_link_backend": "nndescent"},
}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def source_identity():
    files = sorted((ROOT / "low_rank/lowrank").glob("*.py"))
    files += sorted((ROOT / "low_rank/splitters/common").glob("*.py"))
    files += [Path(__file__)]
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def prepare(dataset, limit=None):
    b = load_dataset(dataset, limit=limit)
    if not b.available or not b.feature_data:
        raise RuntimeError(b.reason or "dataset unavailable")
    ids = sorted(b.feature_data)
    X = np.stack([b.feature_data[i] for i in ids]).astype(np.float32)
    y = np.array([(b.targets or {}).get(i, np.nan) for i in ids])
    return b, ids, X, y, choose_metric(X)


def grouped_baseline(method, b, ids, seed):
    """Whole-scaffold/Butina baselines retain every row; sizes may deviate."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
    if not b.smiles:
        raise ValueError("no molecular SMILES for group baseline")
    missing = sum(i not in b.smiles for i in ids)
    if missing:
        raise ValueError(f"SMILES missing for {missing} retained entities; a whole-group baseline cannot cover this dataset")
    if method == "scaffold":
        groups = {}
        for j, i in enumerate(ids):
            mol = Chem.MolFromSmiles(b.smiles[i])
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            groups.setdefault(key, []).append(j)
        members = list(groups.values())
    else:
        if len(ids) > 10_000:
            raise ValueError("Butina all-pairs baseline limited to n<=10000")
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(b.smiles[i])) for i in ids]
        distances = []
        for j in range(1, len(fps)):
            distances.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[j], fps[:j]))
        members = list(Butina.ClusterData(distances, len(ids), 0.6, isDistData=True))
    rng = np.random.default_rng(seed)
    rng.shuffle(members)
    members.sort(key=len, reverse=True)
    remaining = target_sizes(len(ids), [8, 1, 1]).astype(float)
    lab = np.empty(len(ids), dtype=int)
    for group in members:
        c = int(np.argmax(remaining))
        lab[list(group)] = c
        remaining[c] -= len(group)
    return NAMES[lab], {"n_groups": len(members), "whole_groups": True}


def assign(method, b, ids, metric, seed, params):
    if method in ("scaffold", "butina"):
        return grouped_baseline(method, b, ids, seed)
    kw = dict(params or {})
    if metric != "tanimoto" and kw.get("must_link") is not None:
        kw["must_link"] = "auto"
        if kw.get("must_link_backend") == "nndescent":
            raise ValueError("ANN backend supports binary fingerprints only")
    kw.setdefault("allowed_size_imbalance", 0.0)
    if method == "random":
        kw = {}
    res = split("random" if method == "random" else "lowrank", b.feature_data,
                SplitSpec([8, 1, 1], NAMES.tolist(), seed=seed, epsilon=0.0), **kw)
    return np.array([res.assignment[i] for i in ids]), res.diagnostics


def score(X, labels, metric, seed):
    """Full-data standardization, exact kernels, bounded sampling above 4000 rows.

    Global mass uses at most 4000 uniformly sampled rows. NN uses at most 2048
    uniformly sampled held-out rows against all train rows; random normalization
    averages three count-preserving label permutations. Counts always use all n.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = standardize_for_metric(torch.as_tensor(X, device=dev), metric)
    rng = np.random.default_rng(seed + 173)
    n = len(labels)
    sub = np.arange(n) if n <= 4000 else np.sort(rng.choice(n, 4000, replace=False))
    total = cross = 0.0
    for s in range(0, len(sub), 512):
        rows = sub[s:s + 512]
        S = pairwise_similarity(ref[rows], ref[sub], metric)
        S[torch.arange(len(rows), device=dev), torch.arange(s, s + len(rows), device=dev)] = 0
        mask = torch.as_tensor(labels[rows, None] != labels[sub][None, :], device=dev)
        total += S.double().sum().item()
        cross += (S * mask).double().sum().item()
    counts = np.array([(labels == name).sum() for name in NAMES])
    expected = 1 - np.sum(counts * (counts - 1)) / (n * (n - 1))
    if metric == "tanimoto":
        lo, hi = 0.4, 0.8
    else:
        from splitters.common.leakage_metrics import _pair_sim_quantiles
        lo, hi = _pair_sim_quantiles(ref, metric, (0.99, 0.999), seed=seed)

    def nn(lab):
        train = np.flatnonzero(lab == "train")
        held = np.flatnonzero(lab != "train")
        if len(held) > 2048:
            held = np.sort(rng.choice(held, 2048, replace=False))
        best = []
        for s in range(0, len(held), 128):
            v = pairwise_similarity(ref[held[s:s + 128]], ref[train], metric).max(1).values
            best.append(v.cpu().numpy())
        values = np.concatenate(best)
        return held, values, np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1)

    held, values, ramp = nn(labels)
    random_base = np.mean([nn(rng.permutation(labels))[2].mean() for _ in range(3)])
    tests = labels[held] == "test"
    sizes = target_sizes(n, [8, 1, 1])
    return dict(leakage=cross / total if total > 0 else None,
                leak_ratio=cross / total / expected if total > 0 else None,
                nnl=float(ramp.mean()), nnl_rel=float(ramp.mean() / random_base) if random_base > 0 else None,
                test_nn_gt06=float((values[tests] > 0.6).mean()) if tests.any() else None,
                n_train=int(counts[0]), n_val=int(counts[1]), n_test=int(counts[2]),
                imbalance=float(np.max(np.abs(counts - sizes) / np.maximum(sizes, 1))),
                global_score_n=len(sub), nn_score_n=len(held), metric=metric,
                exact_global=n <= 4000, nn_reference_n=int(counts[0]), ramp=[lo, hi])


def model_scores(X, y, labels, task, seed, model_name):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
    train = (labels == "train") & np.isfinite(y)
    val = (labels == "val") & np.isfinite(y)
    test = (labels == "test") & np.isfinite(y)
    if train.sum() < 10 or test.sum() < 5:
        return {"status": "insufficient_labels"}
    kw = dict(n_estimators=128, random_state=seed, n_jobs=MODEL_THREADS)
    if model_name == "rf":
        model = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(**kw)
    else:
        from xgboost import XGBClassifier, XGBRegressor
        model = (XGBClassifier if task == "classification" else XGBRegressor)(
            **kw, max_depth=4, learning_rate=0.05, tree_method="hist", verbosity=0)
    if task == "classification" and len(np.unique(y[train])) < 2:
        return {"status": "single_class_train"}
    t = time.perf_counter()
    model.fit(X[train], y[train])
    out = dict(status="ok", train_n=int(train.sum()), test_n=int(test.sum()), fit_s=time.perf_counter() - t,
               y_shift=float(abs(y[test].mean() - y[train].mean()) / max(y[train].std(), 1e-12)))
    if task == "classification":
        for name, mask in (("train", train), ("val", val), ("test", test)):
            if mask.sum() == 0:
                continue
            p = model.predict_proba(X[mask])[:, 1]
            out[name + "_auc"] = float(roc_auc_score(y[mask], p)) if len(np.unique(y[mask])) == 2 else None
            out[name + "_ap"] = float(average_precision_score(y[mask], p)) if y[mask].sum() > 0 else None
            out[name + "_positive_rate"] = float(y[mask].mean())
    else:
        for name, mask in (("train", train), ("val", val), ("test", test)):
            if mask.sum() < 2:
                continue
            pred = model.predict(X[mask])
            out[name + "_rmse"] = float(np.sqrt(mean_squared_error(y[mask], pred)))
            out[name + "_r2"] = float(r2_score(y[mask], pred))
    return out


def benchmark(dataset, ablation=False):
    b, ids, X, y, metric = prepare(dataset)
    variants = ABLATIONS if ablation else CORE
    rows, models, paths = [], [], []
    directory = OUT / "results" / ("ablation" if ablation else "benchmark") / dataset
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "ids.json", ids)
    for method, params in variants.items():
        for seed in SEEDS:
            t = time.perf_counter()
            reason = None
            if method in ("scaffold", "butina") and (not b.smiles or any(i not in b.smiles for i in ids)):
                reason = "complete molecular SMILES coverage is required for this all-row group baseline"
            elif method == "butina" and len(ids) > 10_000:
                reason = "prespecified n>10000 all-pairs baseline limit"
            elif method == "tail_ann" and metric != "tanimoto":
                reason = "ANN backend applies to binary fingerprints only"
            if reason:
                rows.append(dict(method=method, seed=seed, status="skipped", reason=reason))
                continue
            try:
                lab, diag = assign(method, b, ids, metric, seed, params)
                secs = time.perf_counter() - t
                path = directory / f"{method}__s{seed}.npy"
                np.save(path, lab)
                paths.append(str(path.relative_to(OUT)))
                row = dict(method=method, seed=seed, status="ok", split_s=secs,
                           diagnostics=diag, params=params, **score(X, lab, metric, seed))
                if (not ablation and len(ids) <= 25_000 and b.task_type in ("regression", "classification")):
                    for model_name in ("rf", "xgb"):
                        for model_seed in MODEL_SEEDS:
                            try:
                                m = model_scores(X, y, lab, b.task_type, model_seed, model_name)
                            except Exception as exc:
                                m = {"status": "failed", "error": str(exc)}
                            models.append(dict(method=method, split_seed=seed, model=model_name,
                                               model_seed=model_seed, task=b.task_type, **m))
                print(f"{dataset} {method} seed={seed} split={secs:.2f}s nn={row['nnl_rel']}", flush=True)
            except Exception as exc:
                row = dict(method=method, seed=seed, status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"{dataset} {method} seed={seed} FAILED: {exc}", flush=True)
            rows.append(row)
            atomic_json(directory / "progress.json", clean(dict(rows=rows, models=models)))
    return dict(dataset=dataset, study="ablation" if ablation else "benchmark", n=len(ids),
                target=b.target_name, feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                rows=rows, models=models, assignment_files=paths)


def calibration(dataset):
    """Calibrate on split seeds 0-2; validate on seeds 10-14 and another model."""
    b, ids, X, y, metric = prepare(dataset)
    probes, evals = [], []
    knots = [0., 0.25, 0.5, 0.75, 1.]
    for seed in [0, 1, 2, 10, 11, 12, 13, 14]:
        hard, _ = assign("plain", b, ids, metric, seed, {})
        for alpha in knots:
            lab = interpolate_to_random(hard, alpha, seed=seed)
            for model_name in (["rf"] if seed < 10 else ["rf", "xgb"]):
                m = model_scores(X, y, lab, b.task_type, 0, model_name)
                key = "auc" if b.task_type == "classification" else "r2"
                gap = (m.get("train_" + key) - m.get("test_" + key)
                       if m.get("train_" + key) is not None and m.get("test_" + key) is not None else None)
                (probes if seed < 10 else evals).append(dict(alpha=alpha, seed=seed, model=model_name,
                                                           gap=gap, **m))
    valid = [r for r in probes if r["gap"] is not None]
    aa = np.array([a for a in knots if any(r["alpha"] == a for r in valid)])
    gg = np.array([np.mean([r["gap"] for r in valid if r["alpha"] == a]) for a in aa])
    slope, intercept = np.polyfit(aa, gg, 1)
    cal = GapCalibrator(b=float(intercept), a=float(slope), probe_alphas=aa.tolist(), probe_gaps=gg.tolist())
    # Validation at the same fixed alpha knots tests curve transfer without tuning
    # to validation scores. Interpolate/invert separately reported interior targets.
    for r in evals:
        r["predicted_gap"] = cal.predict(r["alpha"])
    return dict(dataset=dataset, study="calibration", calibration_seeds=[0, 1, 2],
                validation_seeds=[10, 11, 12, 13, 14], alphas=knots,
                probes=probes, validation=evals, calibration_model="rf",
                intercept=float(intercept), slope=float(slope), knots=gg.tolist())


def jobs():
    return ([{"study": "benchmark", "dataset": d} for d in DATASETS]
            + [{"study": "ablation", "dataset": d} for d in ABLATION_DATASETS]
            + [{"study": "calibration", "dataset": d} for d in CAL_DATASETS])


def run(study, dataset):
    import torch
    import lowrank.splitter as splitter_module
    import splitters.methods.random as random_module
    splitter_module._LEAKAGE_MAX_N = 0  # Diagnostics scored separately, outside split timing.
    random_module._LEAKAGE_MAX_N = 0
    t = time.perf_counter()
    initial_hash = source_identity()
    result = (benchmark(dataset, ablation=study == "ablation") if study != "calibration"
              else calibration(dataset))
    result.update(source_sha256=initial_hash, source_unchanged=initial_hash == source_identity(),
                  host=socket.gethostname(), python=sys.executable, torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  elapsed_s=time.perf_counter() - t, numpy_version=np.__version__,
                  model_spec={"rf_trees": 128, "xgb_trees": 128, "xgb_depth": 4,
                              "xgb_learning_rate": 0.05, "threads": MODEL_THREADS,
                              "model_seeds": list(MODEL_SEEDS)})
    path = OUT / "results" / f"{study}__{dataset}.json"
    atomic_json(path, clean(result))
    return path


def complete(path):
    if not path.exists():
        return False
    try:
        r = json.loads(path.read_text())
        if not r.get("source_unchanged"):
            return False
        if r.get("source_sha256") != source_identity():
            return False
        if r.get("study") == "calibration":
            return len(r.get("validation", [])) == 50 and len(r.get("probes", [])) == 15
        return (len(r.get("rows", [])) == len(ABLATIONS if r.get("study") == "ablation" else CORE) * len(SEEDS)
                and all((OUT / p).exists() for p in r.get("assignment_files", [])))
    except (ValueError, KeyError):
        return False


def worker(queue_path):
    queue = Path(queue_path)
    owner = {"host": socket.gethostname(), "pid": os.getpid(),
             "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
             "affinity": sorted(os.sched_getaffinity(0)), "source_sha256": source_identity()}
    consecutive = 0
    for item in json.loads(queue.read_text())["jobs"]:
        study, ds = item["study"], item["dataset"]
        name = study + "__" + ds
        path = OUT / "results" / (name + ".json")
        if complete(path):
            continue
        claim = OUT / "queue" / (name + ".claim")
        claim.parent.mkdir(exist_ok=True)
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        atomic_json(claim / "owner.json", owner)
        try:
            if not complete(path):
                print("START " + name, flush=True)
                run(study, ds)
            if not complete(path):
                raise RuntimeError("output completeness verification failed")
            print("DONE " + name, flush=True)
            consecutive = 0
        except Exception:
            atomic_json(OUT / "queue" / (name + ".failure.json"),
                        dict(**owner, traceback=traceback.format_exc(), time=time.time()))
            print(traceback.format_exc(), flush=True)
            consecutive += 1
        finally:
            (claim / "owner.json").unlink(missing_ok=True)
            claim.rmdir()
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if consecutive >= 3:
            raise RuntimeError("three consecutive dataset failures; worker circuit breaker")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", action="store_true")
    parser.add_argument("--job", nargs=2, metavar=("STUDY", "DATASET"))
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.jobs:
        print(json.dumps({"jobs": jobs(), "source_sha256": source_identity()}, indent=2))
    elif args.worker:
        worker(args.worker)
    else:
        print(run(*args.job))


if __name__ == "__main__":
    main()
