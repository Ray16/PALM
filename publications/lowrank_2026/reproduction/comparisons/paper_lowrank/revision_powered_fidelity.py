"""Full-dataset powered-kernel fidelity, numerical bound and combined rank study.

Exact S is retained only for the six prespecified development datasets (n<=5000).
The residual spectral norm uses a symmetric eigensolver with a convergence
residual recorded; it is an empirical numerical estimate, not a certified bound.
The analytical bound is n*||E||_2. Each reported empirical cut uses exact S.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from lowrank.nystrom import nystrom_features
from lowrank.objective import factor_leakage
from lowrank.optimize import interpolate_to_random
from splitters import SplitSpec, split
from splitters.common.leakage_metrics import nn_leakage
from splitters.common.pairwise_similarity import pairwise_similarity, standardize_for_metric

DEST = study.OUT / "revision_20261021/powered_fidelity"
RANKS = (32, 64, 128, 256, 512)
DATASETS = ["moleculenet_" + s for s in ("freesolv", "esol", "bace", "bbbp", "clintox", "lipophilicity")]


def exact_cut(S, labels):
    return float(S[labels[:, None] != labels[None, :]].sum() / 2)


def run(dataset):
    if torch.cuda.is_available():
        raise RuntimeError("CPU protocol: hide CUDA before launch")
    dest = DEST / dataset
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "result.json"
    sources = source_files()
    sources[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if final.exists():
        old = json.loads(final.read_text())
        if old["source_files"] != sources:
            raise RuntimeError("source differs from retained result")
        print(dataset, "verified existing result", flush=True)
        return
    b, ids, X, y, metric = study.prepare(dataset)
    n = len(ids)
    if n > 5000:
        raise ValueError("exact-matrix fidelity protocol is restricted to n<=5000")
    ref = standardize_for_metric(torch.as_tensor(X), metric)
    S = pairwise_similarity(ref, ref, metric).double().numpy()
    original_mass = (S.sum() - np.trace(S)) / 2
    rows, quality, assignments = [], [], []
    for power in (1, 3):
        K = S ** power
        mass = float((K.sum() - np.trace(K)) / 2)
        for rank in RANKS:
            for seed in (0, 1, 2):
                params = dict(rank=rank, kernel_power=power, allowed_size_imbalance=0.,
                              score_exact_leakage=False)
                if power == 3:
                    params["must_link"] = .4 if metric == "tanimoto" else "auto"
                result = split("lowrank", b.feature_data,
                               SplitSpec([8, 1, 1], study.NAMES.tolist(), seed=seed, epsilon=0.), **params)
                labels = np.array([result.assignment[i] for i in ids])
                codes = np.array([list(study.NAMES).index(s) for s in labels])
                path = dest / f"p{power}__r{rank}__s{seed}.npy"
                np.save(path, labels)
                assignments.append(str(path.relative_to(study.OUT)))
                counts = np.bincount(codes, minlength=3)
                expected = 1 - sum(c * (c - 1) for c in counts) / (n * (n - 1))
                nn = nn_leakage(X, labels, metric=metric, seed=seed, n_perm=3, device="cpu")
                quality.append(dict(dataset=dataset, power=power, rank=rank, seed=seed,
                    leak_ratio=exact_cut(S, codes) / original_mass / expected,
                    nnl_rel=nn["nnl_rel"], broken_groups=result.diagnostics.get("broken_groups", 0),
                    n_train=int(counts[0]), n_val=int(counts[1]), n_test=int(counts[2])))
                B, _ = nystrom_features(X, rank=rank, metric=metric, seed=seed, kernel_power=power)
                B = B.astype(np.float64)
                E = K - B @ B.T
                # Fixed starting vector makes the diagnostic reproducible.
                eigenvalues, vectors = eigsh(E, k=1, which="LM", tol=1e-8,
                    v0=np.random.default_rng(1000 + seed).normal(size=n))
                norm = float(abs(eigenvalues[0]))
                eigen_residual = float(np.linalg.norm(E @ vectors[:, 0] - eigenvalues[0] * vectors[:, 0]))
                # Evaluate optimized, random and intermediate partitions against
                # the SAME factor and exact powered kernel.
                for alpha in (0., .25, .5, .75, 1.):
                    lab = interpolate_to_random(codes, alpha, seed=seed)
                    exact = exact_cut(K, lab)
                    approximate = factor_leakage(B, lab, 3)
                    approx_mass = float(((B.sum(0) ** 2).sum() - (B * B).sum()) / 2)
                    rows.append(dict(dataset=dataset, n=n, power=power, rank=rank, seed=seed,
                        alpha=alpha, exact_cut=exact, factor_cut=approximate,
                        offdiag_mass=mass, exact_ratio=exact / mass,
                        factor_ratio_true_denominator=approximate / mass,
                        factor_ratio_own_denominator=approximate / approx_mass,
                        abs_cut_error_over_mass=abs(exact - approximate) / mass,
                        spectral_norm_estimate=norm, eigen_residual=eigen_residual,
                        numerical_delta_over_mass=n * norm / mass))
                study.atomic_json(dest / "progress.json", study.clean(dict(rows=rows, quality=quality)))
                print(dataset, "power", power, "rank", rank, "seed", seed,
                      "delta/mass", round(n * norm / mass, 4), flush=True)
    check = source_files()
    check[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if check != sources:
        raise RuntimeError("source changed during study")
    study.atomic_json(final, study.clean(dict(dataset=dataset, n=n, metric=metric,
        source_files=sources, feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        ids=ids, rows=rows, quality=quality, assignment_files=assignments,
        spectral_norm_method="scipy eigsh LM, tol=1e-8, deterministic start; numerical estimate, not certified")))


def summarize():
    records = [json.loads(p.read_text()) for p in sorted(DEST.glob("*/result.json"))]
    for field in ("rows", "quality"):
        pd.DataFrame([r for record in records for r in record[field]]).to_csv(
            DEST / (field + ".csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        run(dataset)
        summarize()
