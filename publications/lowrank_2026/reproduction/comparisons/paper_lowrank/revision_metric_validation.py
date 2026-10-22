"""Reanalyze existing synthetic and per-molecule evidence, excluding hypergraphs.

These are validations of metrics using retained observations, not reruns of
the partition methods. Pairwise metric comparisons use identical finite rows.
Per-molecule reference features use training only; train+validation is a labelled
secondary analysis. Ensemble predictions must match test SMILES in row order.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from datasets.featurize import ecfp_matrix

DEST = study.OUT / "revision_20261021/metric_validation"
INPUTS = {}


def read(path):
    INPUTS[str(path.relative_to(study.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def synthetic():
    frames = [read(study.ROOT / f"comparisons/results/regimes_{name}.csv") for name in
              ("near_dup", "continuum", "hierarchy", "long_tail", "tightness")]
    raw = pd.concat(frames, ignore_index=True)
    # Nonrandom methods only, with the author's explicit comparator exclusion.
    d = raw[(raw.status == "ok") & raw.method.isin(["lowrank", "datasail"])].copy()
    rows = []
    for regime, frame in [("pooled", d)] + list(d.groupby("regime")):
        common = frame[["family_leak", "leak_ratio", "nnl_rel"]].replace([np.inf, -np.inf], np.nan).dropna()
        for proxy in ("leak_ratio", "nnl_rel"):
            corr = spearmanr(common[proxy], common.family_leak)
            rows.append(dict(regime=regime, proxy=proxy, n=len(common), rho=float(corr.statistic)))
    d.to_csv(DEST / "synthetic_observations.csv", index=False)
    pd.DataFrame(rows).to_csv(DEST / "synthetic_correlations.csv", index=False)


def fingerprints(smiles):
    kept, X = ecfp_matrix(list(smiles))
    if len(kept) != len(smiles):
        raise ValueError("invalid SMILES would change retained prediction identities")
    return X.astype(np.float64)


def molecule_errors():
    root = study.ROOT / "literature/repro/experiments/chemprop_moleculenet"
    records = []
    for dataset in ("esol", "freesolv", "lipophilicity", "bbbp"):
        for method in ("random", "lowrank"):
            directory = root / "split_data" / dataset / method
            train, valid, test = [read(directory / (part + ".csv")) for part in ("train", "val", "test")]
            target = test.columns[1]
            predictions = []
            for path in sorted((root / "split_ckpt" / dataset / method).glob("seed_*/test_preds_permol.csv")):
                frame = read(path)
                if frame.smiles.tolist() != test.smiles.tolist():
                    raise ValueError("prediction/test identities differ")
                predictions.append(frame[target].to_numpy(float))
            if len(predictions) != 5:
                raise ValueError(f"expected five model seeds: {dataset}/{method}")
            error = np.abs(np.mean(predictions, axis=0) - test[target].to_numpy(float))
            T = fingerprints(test.smiles)
            dot = T @ T.T
            TT = dot / np.maximum(T.sum(1)[:, None] + T.sum(1)[None, :] - dot, 1e-12)
            np.fill_diagonal(TT, 0.)
            for reference_name, reference_frame in [("train", train), ("train_plus_val", pd.concat([train, valid]))]:
                R = fingerprints(reference_frame.smiles)
                dot = T @ R.T
                S = dot / np.maximum(T.sum(1)[:, None] + R.sum(1)[None, :] - dot, 1e-12)
                nearest = np.sort(S, axis=1)[:, -min(5, len(R)):]
                denominator = S.sum(1) + TT.sum(1)
                lpi = np.divide(S.sum(1), denominator, out=np.zeros(len(T)), where=denominator > 0)
                for j in range(len(test)):
                    records.append(dict(dataset=dataset, method=method, reference=reference_name,
                        test_row=j, absolute_error=error[j], nn_similarity=nearest[j, -1],
                        nn_ramp=float(np.clip((nearest[j, -1] - .4) / .4, 0, 1)),
                        top5_similarity=float(nearest[j].mean()), per_point_global=lpi[j]))
    observations = pd.DataFrame(records)
    observations.to_csv(DEST / "per_molecule_observations.csv", index=False)
    out = []
    for (dataset, reference), frame in observations.groupby(["dataset", "reference"]):
        for subset, d in [("pooled", frame)] + list(frame.groupby("method")):
            for proxy in ("nn_similarity", "nn_ramp", "top5_similarity", "per_point_global"):
                out.append(dict(dataset=dataset, reference=reference, split=subset, proxy=proxy,
                    n=len(d), rho=float(spearmanr(d[proxy], d.absolute_error).statistic)))
    pd.DataFrame(out).to_csv(DEST / "per_molecule_correlations.csv", index=False)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    synthetic()
    molecule_errors()
    study.atomic_json(DEST / "provenance.json", dict(input_sha256=INPUTS,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="metric validation of existing predictions, not release-code splitter comparison"))
