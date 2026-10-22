"""Random-relative test difficulty, with independent knot and inversion tests.

RF calibration: split seeds 0--2. Validation: split seeds 10--14, RF and XGB.
Classification difficulty is AUC(random)-AUC(alpha); regression difficulty is
RMSE(alpha)/RMSE(random)-1. Each random reference shares its split seed, model
and exact counts. Models have seed zero. All assignments and model fits are
saved; no earlier train-minus-test gap is relabelled as this new quantity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from lowrank.optimize import interpolate_to_random
from lowrank.target_gap import GapCalibrator
from splitters import SplitSpec, split

DEST = study.OUT / 'revision_20261021/calibration_v2'
ALPHAS = (0., .25, .5, .75, 1.)
CAL_SEEDS = (0, 1, 2)
VAL_SEEDS = (10, 11, 12, 13, 14)


def difficulty(task, metrics, random_metrics):
    key = 'test_auc' if task == 'classification' else 'test_rmse'
    value, reference = metrics.get(key), random_metrics.get(key)
    if value is None or reference is None:
        return None
    if task == 'classification':
        return float(reference - value)
    return float(value / reference - 1) if reference > 0 else None


def run(dataset):
    directory = DEST / dataset
    directory.mkdir(parents=True, exist_ok=True)
    sources = source_files()
    sources[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    final = directory / 'result.json'
    if final.exists():
        if json.loads(final.read_text())['source_files'] != sources:
            raise RuntimeError('completed calibration uses other sources')
        return
    b, ids, X, y, metric = study.prepare(dataset)
    study.atomic_json(directory / 'ids.json', ids)
    progress = directory / 'progress.json'
    cache = json.loads(progress.read_text()) if progress.exists() else dict(source_files=sources, rows=[])
    if cache['source_files'] != sources:
        raise RuntimeError('partial calibration uses other sources')
    rows = cache['rows']
    assignments = {}

    def save(path, labels):
        labels = np.asarray(labels)
        counts = [int(np.sum(labels == name)) for name in study.NAMES]
        if counts != targets or not set(labels) <= set(study.NAMES):
            raise AssertionError('count-preserving interpolation changed its contract')
        if path.exists() and not np.array_equal(np.load(path), labels):
            raise RuntimeError('stored assignment differs from regenerated assignment')
        np.save(path, labels)
        assignments[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return labels

    def fit(stage, seed, model, index, labels, **extra):
        matches = [r for r in rows if (r['stage'], r['seed'], r['model'], r['index']) == (stage, seed, model, index)]
        if len(matches) > 1:
            raise RuntimeError('duplicate model-fit cache key')
        if matches:
            return matches[0]
        metrics = study.model_scores(X, y, labels, b.task_type, 0, model)
        row = dict(stage=stage, seed=seed, model=model, model_seed=0, index=index, **extra, **metrics)
        rows.append(row)
        study.atomic_json(progress, study.clean(dict(source_files=sources, rows=rows)))
        print(dataset, stage, seed, model, index, metrics.get('status'), flush=True)
        return row

    from splitters.common.balanced_assignment import target_sizes
    targets = target_sizes(len(ids), [8, 1, 1]).tolist()
    optimized = {}
    for seed in (*CAL_SEEDS, *VAL_SEEDS):
        result = split('lowrank', b.feature_data,
                       SplitSpec([8, 1, 1], study.NAMES.tolist(), seed=seed, epsilon=0.),
                       allowed_size_imbalance=0., score_exact_leakage=False)
        hard = np.array([result.assignment[i] for i in ids])
        optimized[seed] = save(directory / f's{seed}__optimized.npy', hard)
        stage = 'calibration_knot' if seed in CAL_SEEDS else 'validation_knot'
        for index, alpha in enumerate(ALPHAS):
            labels = save(directory / f's{seed}__alpha{alpha:.2f}.npy',
                          interpolate_to_random(hard, alpha, seed=seed))
            for model in (('rf',) if seed in CAL_SEEDS else ('rf', 'xgb')):
                fit(stage, seed, model, index, labels, alpha=alpha)

    references = {(r['seed'], r['model']): r for r in rows
                  if r['stage'].endswith('_knot') and r['alpha'] == 0}
    for row in rows:
        row['difficulty'] = difficulty(b.task_type, row, references[row['seed'], row['model']])
    means = []
    for alpha in ALPHAS:
        values = [r['difficulty'] for r in rows if r['stage'] == 'calibration_knot' and r['alpha'] == alpha]
        if len(values) != 3 or any(v is None for v in values):
            raise RuntimeError('RF calibration lacks complete finite probe scores')
        means.append(float(np.mean(values)))
    slope, intercept = np.polyfit(ALPHAS, means, 1)
    calibrator = GapCalibrator(a=float(slope), b=float(intercept),
                              probe_alphas=list(ALPHAS), probe_gaps=means)
    lo, hi = calibrator.achievable_range
    targets_difficulty = [lo + fraction * (hi - lo) for fraction in (.25, .5, .75)]
    for seed in VAL_SEEDS:
        for index, target in enumerate(targets_difficulty):
            inv = calibrator.invert(target)
            labels = save(directory / f's{seed}__target{index}.npy',
                          interpolate_to_random(optimized[seed], inv.alpha, seed=seed))
            for model in ('rf', 'xgb'):
                row = fit('target_inversion', seed, model, index, labels,
                          alpha=inv.alpha, target_difficulty=target,
                          predicted_difficulty=inv.predicted_gap)
                row['difficulty'] = difficulty(b.task_type, row, references[seed, model])
    if len(rows) != 95 or source_files() != {k: v for k, v in sources.items() if k != str(Path(__file__).relative_to(study.ROOT))}:
        raise RuntimeError('model coverage or source identity changed')
    knots = calibrator._knots()
    record = dict(dataset=dataset, status='ok', n=len(ids), ids_sha256=hashlib.sha256((directory/'ids.json').read_bytes()).hexdigest(),
                  feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
                  target_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                  task=b.task_type, host=socket.gethostname(), threads=2, source_files=sources,
                  definition='AUC(random)-AUC(alpha)' if b.task_type == 'classification' else 'RMSE(alpha)/RMSE(random)-1',
                  calibration_seeds=list(CAL_SEEDS), validation_seeds=list(VAL_SEEDS),
                  alphas=list(ALPHAS), raw_mean_difficulties=means,
                  isotonic_mean_difficulties=knots[1].tolist(),
                  direction='increasing' if calibrator.increasing else 'decreasing',
                  target_difficulties=targets_difficulty, assignment_sha256=assignments,
                  rows=rows, old_gap_artifacts='separate legacy study; no values copied into this result')
    study.atomic_json(final, study.clean(record))
    print('COMPLETE', dataset, len(rows), 'model fits', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=study.CAL_DATASETS)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError('CPU-only calibration requires CUDA_VISIBLE_DEVICES empty')
    for dataset in args.datasets:
        run(dataset)
