"""Lo-Hi Hi with official objective and explicit solver/retention accounting.

Only resource controls differ from the installed solve function: two threads,
seed zero and a 7200-second solver limit. Preserve both native survivor indices
and matched-subset low-rank comparisons at exact integer counts. No hypergraphs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_baselines import lohi_hi
from revision_external_control import source_files
from splitters import SplitSpec, split
from splitters.common.leakage_metrics import nn_leakage, scaled_lpi
from splitters.common.pairwise_similarity import pairwise_similarity

DEST = study.OUT / "revision_20261021/lohi"
DATASETS = ["moleculenet_" + s for s in
            ("freesolv", "esol", "bace", "bbbp", "sider", "clintox", "lipophilicity", "tox21", "qm8", "hiv", "muv")]


def score(X, labels, seed):
    counts = [np.sum(labels == s) for s in ("train", "test")]
    expected = 1 - sum(c * (c - 1) for c in counts) / (len(X) * (len(X) - 1))
    nn = nn_leakage(X, labels, metric="tanimoto", seed=seed, n_perm=3, device="cpu")
    return dict(leak_ratio=scaled_lpi(X, labels, metric="tanimoto") / expected,
                nnl_rel=nn["nnl_rel"], n_train=int(counts[0]), n_test=int(counts[1]))


def run(dataset):
    import lohi_splitter.min_vertex_k_cut as implementation
    directory = DEST / dataset
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "result.json"
    sources = source_files()
    for path in (Path(__file__), Path(__file__).with_name("revision_baselines.py")):
        sources[str(path.relative_to(study.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources["installed_lohi_min_vertex_k_cut.py"] = hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    if final.exists():
        if json.loads(final.read_text())["source_files"] != sources:
            raise RuntimeError("existing Lo-Hi result uses different sources")
        return
    b, ids, X, y, metric = study.prepare(dataset)
    record = dict(dataset=dataset, n=len(ids), source_files=sources, ids=ids,
        feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        resource_policy="full molecular datasets up to 5000 entities; 2 CPU threads; 7200 solver seconds")
    if len(ids) > 5000 or not b.smiles or any(i not in b.smiles for i in ids):
        record.update(status="unavailable", reason="prespecified input/size eligibility; not solver infeasibility")
        study.atomic_json(final, study.clean(record))
        return
    solver_record = {}
    original_solve = implementation.solve_linear_problem

    def bounded_solve(model, max_mip_gap=.1, verbose=True):
        model.max_mip_gap = max_mip_gap
        model.threads = 2
        model.emphasis = 2  # unchanged official parameter
        model.verbose = False
        model.seed = 0
        status = model.optimize(max_seconds=7200)
        solver_record.update(status=str(status), num_solutions=model.num_solutions,
            objective_value=model.objective_value, objective_bound=model.objective_bound,
            mip_gap=model.gap, threads=2, seed=0, max_seconds=7200)
        return model

    start = time.perf_counter()
    try:
        implementation.solve_linear_problem = bounded_solve
        full_labels, diag = lohi_hi([b.smiles[i] for i in ids])
        np.save(directory / "lohi_full_assignment.npy", full_labels)
        retained = np.flatnonzero(full_labels != "discard")
        np.save(directory / "retained_indices.npy", retained)
        Z, native = X[retained], full_labels[retained]
        counts = [int(np.sum(native == name)) for name in ("train", "test")]
        if min(counts) < 1:
            raise ValueError("Lo-Hi returned an empty fold")
        max_cross = 0.
        train, test = torch.as_tensor(Z[native == "train"]), torch.as_tensor(Z[native == "test"])
        for s in range(0, len(test), 128):
            max_cross = max(max_cross, float(pairwise_similarity(test[s:s + 128], train, "tanimoto").max()))
        if max_cross > .400001:
            raise ValueError(f"Lo-Hi threshold postcondition failed under shared fingerprint: {max_cross}")
        rows = [dict(method="lohi", seed=0, scope="matched_retained_subset", retained_n=len(retained),
                     **score(Z, native, 0))]
        for method in ("random", "plain", "combined"):
            for seed in (0, 1, 2):
                kw = {} if method == "random" else dict(allowed_size_imbalance=0., score_exact_leakage=False)
                if method == "combined":
                    kw.update(kernel_power=3, must_link=.4)
                import splitters.methods.random as rmod
                rmod._LEAKAGE_MAX_N = 0
                result = split("random" if method == "random" else "lowrank", dict(enumerate(Z)),
                    SplitSpec(counts, ["train", "test"], seed=seed, epsilon=0.), **kw)
                labels = np.array([result.assignment[i] for i in range(len(Z))])
                if [np.sum(labels == name) for name in ("train", "test")] != counts:
                    raise AssertionError("retained-subset comparison counts differ")
                np.save(directory / f"{method}__s{seed}.npy", labels)
                rows.append(dict(method=method, seed=seed, scope="matched_retained_subset",
                                 retained_n=len(retained), **score(Z, labels, seed)))
        record.update(status="ok", retained_n=len(retained), rows=rows, diagnostics=diag,
                      solver=solver_record, maximum_cross_similarity=max_cross)
    except Exception as exc:
        record.update(status="failed", reason=f"{type(exc).__name__}: {exc}",
                      solver=solver_record, traceback=traceback.format_exc())
    finally:
        implementation.solve_linear_problem = original_solve
    record["wall_s"] = time.perf_counter() - start
    study.atomic_json(final, study.clean(record))
    print(dataset, record["status"], record.get("retained_n"), record.get("reason"), flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU protocol requires CUDA_VISIBLE_DEVICES=''")
    for dataset in args.datasets:
        run(dataset)
