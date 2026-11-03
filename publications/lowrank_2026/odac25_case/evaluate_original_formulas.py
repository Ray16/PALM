"""Frame-weighted evaluation on actual recorded formula vectors, offline.

Global overlap uses Monte Carlo pairs; nearest neighbours search every distinct
training formula for sampled validation configurations. Uncertainty describes
sampling error only, not split or predictive-model uncertainty.
"""
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from build_split import ROOT, fraction_matrix
from collect import atomic_json

DEST = ROOT / 'comparison_original'


def pair_mean(ref, weights_a, weights_b, seed, n=200000):
    rng = np.random.default_rng(seed)
    a = rng.choice(len(ref), n, p=weights_a / weights_a.sum())
    b = rng.choice(len(ref), n, p=weights_b / weights_b.sum())
    scores = []
    for start in range(0, n, 4096):
        dist = torch.linalg.vector_norm(ref[a[start:start + 4096]] - ref[b[start:start + 4096]], dim=1)
        scores.extend((1 / (1 + dist)).tolist())
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std(ddof=1) / np.sqrt(n))


def nearest_mean(ref, train, val, seed, queries=512):
    rng = np.random.default_rng(seed)
    q = rng.choice(len(ref), queries, p=val / val.sum())
    values = np.zeros(queries)
    overlapping = train[q] > 0
    values[overlapping] = 1.
    search = np.flatnonzero(~overlapping)
    references = np.flatnonzero(train > 0)
    for start in range(0, len(search), 32):
        positions = search[start:start + 32]
        best = torch.full((len(positions),), float('inf'))
        for offset in range(0, len(references), 8192):
            dist = torch.cdist(ref[q[positions]], ref[references[offset:offset + 8192]])
            best = torch.minimum(best, dist.min(1).values)
        values[positions] = (1 / (1 + best)).numpy()
    return dict(val_nearest_train_similarity=float(values.mean()),
        nearest_similarity_mc_stderr=float(values.std(ddof=1) / np.sqrt(queries)),
        val_queries=queries, distinct_training_formulas=len(references),
        val_fraction_with_exact_training_formula=float(val[train > 0].sum() / val.sum()))


def main():
    table = pd.read_parquet(DEST / 'original_formula_counts.parquet')
    mapping = pd.read_csv(ROOT / 'splits/Filtered/seed0/formula_assignments.csv')
    table = table.merge(mapping, left_on='formula', right_on='reduced_observed_formula', validate='one_to_one')
    X, _ = fraction_matrix(table['formula'].tolist())
    w = table['Train'].to_numpy(np.int64) + table['Val'].to_numpy(np.int64)
    # Reference scaling is configuration-weighted over the same complete pool.
    mean = np.average(X.astype(float), axis=0, weights=w)
    var = np.average((X.astype(float) - mean) ** 2, axis=0, weights=w) * w.sum() / (w.sum() - 1)
    active = var > 1e-16
    ref = torch.as_tensor(((X[:, active] - mean[active]) / np.sqrt(var[active])).astype(np.float32))
    total, total_se = pair_mean(ref, w, w, 100)
    total -= 1 / w.sum()  # remove actual configuration self-pair expectation
    arms = [('original', None, table['Train'].to_numpy(np.int64), table['Val'].to_numpy(np.int64))]
    for seed in (0, 1, 2):
        a = pd.read_csv(DEST / f'matched_lowrank_seed{seed}_groups.csv').set_index('group_id')['split']
        val = w * (table['group_id'].map(a).to_numpy() == 'val')
        arms.append(('weighted_lowrank_exact_original_counts', seed, w - val, val))
    rows = []
    for name, seed, train, val in arms:
        start = time.monotonic()
        assert int(train.sum()) == 36920036 and int(val.sum()) == 791638
        cross, cross_se = pair_mean(ref, train, val, 200)
        raw_ratio = cross / total
        ratio = raw_ratio * (1 - 1 / w.sum())
        p_cross = float(2 * train.sum() * val.sum() / w.sum() ** 2)
        ratio_se = ratio * np.sqrt((cross_se / cross) ** 2 + (total_se / total) ** 2)
        row = dict(method=name, seed=seed, train_configurations=int(train.sum()), val_configurations=int(val.sum()),
            global_share_mc=p_cross * raw_ratio, global_random_relative_mc=ratio,
            global_random_relative_mc_stderr=float(ratio_se), pair_samples=200000,
            **nearest_mean(ref, train, val, 300), evaluation_seconds=time.monotonic() - start)
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(DEST / 'comparison_recorded_formulas.csv', index=False)
    atomic_json(DEST / 'recorded_formula_evaluation.json', dict(rows=rows,
        distinct_formulas=len(table), active_element_fraction_dimensions=int(active.sum()),
        standardization='Configuration-weighted mean and sample variance over full pool; remove zero-variance dimensions',
        global_metric='Monte Carlo train-val pair mean / total pair mean; exact original fold counts',
        nearest_metric='512 configuration-weighted validation queries against every distinct training formula',
        uncertainty='Monte Carlo standard errors only; split/model uncertainty is separate',
        no_additional_downloads=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [DEST / 'original_formula_counts.parquet', ROOT / 'splits/Filtered/seed0/formula_assignments.csv'] +
            [DEST / f'matched_lowrank_seed{s}_groups.csv' for s in (0, 1, 2)]}))


if __name__ == '__main__':
    main()
