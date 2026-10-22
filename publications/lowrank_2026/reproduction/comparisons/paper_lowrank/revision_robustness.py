"""Prespecified full-dataset power/threshold frontier and independent metrics.

Six development datasets, three split seeds, p in {2,3,4}, tau in {.3,.4,.5,.6}.
Plain, power-only and group-only controls are included. All evaluation uses the
original Morgan kernel plus MACCS, which the optimizer never sees. No response
labels or prediction scores select parameters. Exact test-only scoring is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from splitters import SplitSpec, split
from splitters.common.balanced_assignment import target_sizes
from splitters.common.pairwise_similarity import pairwise_similarity

DEST = study.OUT / "revision_20261021/robustness"
DATASETS = ["moleculenet_" + s for s in ("freesolv", "esol", "bace", "bbbp", "clintox", "lipophilicity")]
SETTINGS = [("plain", 1, None), ("groups", 1, .4)] + [
    (f"power{p}", p, None) for p in (2, 3, 4)] + [
    (f"p{p}_tau{tau}", p, tau) for p in (2, 3, 4) for tau in (.3, .4, .5, .6)]


def dense_metrics(S, labels):
    """Exact whole-matrix cut and test-to-train top-k metrics at fixed counts."""
    n = len(labels)
    counts = np.bincount(labels, minlength=3)
    expected = 1 - sum(c * (c - 1) for c in counts) / (n * (n - 1))
    mass = S.sum(dtype=np.float64) - np.trace(S)
    crossing = S[labels[:, None] != labels[None, :]].sum(dtype=np.float64)
    train, test = np.flatnonzero(labels == 0), np.flatnonzero(labels == 2)
    sim = S[np.ix_(test, train)]
    top = np.partition(sim, -5, axis=1)[:, -5:]
    nn, top5 = top.max(1), top.mean(1)
    return dict(global_ratio=float(crossing / mass / expected), nn_mean=float(nn.mean()),
        top5_mean=float(top5.mean()), frac_gt06=float(np.mean(nn > .6)),
        frac_gt08=float(np.mean(nn > .8)),
        nnl=float(np.clip((nn - .4) / .4, 0, 1).mean()),
        nnl_03_07=float(np.clip((nn - .3) / .4, 0, 1).mean()),
        nnl_05_09=float(np.clip((nn - .5) / .4, 0, 1).mean()),
        top5_ramp=float(np.clip((top5 - .4) / .4, 0, 1).mean()))


def maccs(smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys
    X = np.zeros((len(smiles), 167), dtype=np.float32)
    for i, sm in enumerate(smiles):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            raise ValueError("MACCS input contains an invalid SMILES")
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), X[i])
    return X


def run(dataset):
    sources = source_files()
    sources[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest = DEST / dataset
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "result.json"
    if final.exists():
        if json.loads(final.read_text())["source_files"] != sources:
            raise RuntimeError("existing robustness result has different source")
        return
    b, ids, X, y, metric = study.prepare(dataset)
    if metric != "tanimoto" or len(X) > 5000:
        raise ValueError("this exact robustness protocol uses six molecular datasets with n<=5000")
    alt = maccs([b.smiles[i] for i in ids])
    matrices = {"morgan": pairwise_similarity(torch.as_tensor(X), torch.as_tensor(X), "tanimoto").numpy(),
                "maccs": pairwise_similarity(torch.as_tensor(alt), torch.as_tensor(alt), "tanimoto").numpy()}
    counts = target_sizes(len(X), [8, 1, 1])
    ordered_labels = np.repeat(np.arange(3), counts)
    base = {}
    for seed in (0, 1, 2):
        for representation, S in matrices.items():
            permutations = [np.random.default_rng(seed + 5000 + j).permutation(ordered_labels) for j in range(3)]
            values = [dense_metrics(S, lab) for lab in permutations]
            base[seed, representation] = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
    rows = []
    for name, power, tau in SETTINGS:
        for seed in (0, 1, 2):
            assignment_path = dest / f"{name}__s{seed}.npy"
            kwargs = dict(rank=256, kernel_power=power, score_exact_leakage=False, allowed_size_imbalance=0.)
            if tau is not None:
                kwargs["must_link"] = tau
            start = time.perf_counter()
            res = split("lowrank", b.feature_data,
                SplitSpec([8, 1, 1], study.NAMES.tolist(), seed=seed, epsilon=0.), **kwargs)
            seconds = time.perf_counter() - start
            labels = np.array([res.assignment[i] for i in ids])
            codes = np.array([list(study.NAMES).index(s) for s in labels])
            if not np.array_equal(np.bincount(codes, minlength=3), counts):
                raise AssertionError("robustness case violates exact fold counts")
            np.save(assignment_path, labels)
            row = dict(dataset=dataset, setting=name, power=power, tau=tau, seed=seed,
                split_s=seconds, broken_groups=res.diagnostics.get("broken_groups", 0),
                assignment_sha256=hashlib.sha256(assignment_path.read_bytes()).hexdigest())
            for representation, S in matrices.items():
                metrics = dense_metrics(S, codes)
                for key, value in metrics.items():
                    row[representation + "_" + key] = value
                    denominator = base[seed, representation][key]
                    row[representation + "_" + key + "_rel"] = value / denominator if denominator > 0 else None
            rows.append(row)
            study.atomic_json(dest / "progress.json", study.clean(dict(rows=rows)))
            print(dataset, name, seed, "Morgan local", row["morgan_nnl_rel"],
                  "MACCS raw NN", row["maccs_nn_mean"], flush=True)
    current = source_files()
    current[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != sources:
        raise RuntimeError("sources changed during robustness study")
    study.atomic_json(final, study.clean(dict(dataset=dataset, n=len(X), ids=ids,
        source_files=sources, feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        maccs_sha256=hashlib.sha256(alt.tobytes()).hexdigest(), rows=rows,
        primary_local_queries="test only; complete training reference; exact scoring",
        normalization="3 fixed-count random assignments shared by all settings for each seed",
        alternative_representation="RDKit MACCS 167-bit keys, evaluation only; no parameter selection",
        threshold_interpretation="absolute Morgan ramp cutoffs do not define universal MACCS relatedness; raw MACCS nearest similarity is retained")))


def summarize():
    records = [json.loads(p.read_text()) for p in sorted(DEST.glob("*/result.json"))]
    if records:
        pd.DataFrame([row for r in records for row in r["rows"]]).to_csv(DEST / "rows.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CPU protocol requires CUDA_VISIBLE_DEVICES=''")
    for dataset in args.datasets:
        run(dataset)
        summarize()
