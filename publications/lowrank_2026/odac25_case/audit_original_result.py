"""Offline falsification checks for the ODAC25 comparison; retains prior results."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from build_split import ROOT, fraction_matrix
from collect import atomic_json
from compare_original import assign_exact, metrics, weighted_cut
from evaluate_original_formulas import pair_mean, nearest_mean
from lowrank.nystrom import nystrom_features
from lowrank.splitters.common.pairwise_similarity import standardize_for_metric, pairwise_similarity

BASE = ROOT / 'splits/Filtered/seed0'
PREV = ROOT / 'comparison_original'
DEST = ROOT / 'audit_original_result'


def dense_optimize(K, w, initial, target):
    lab = initial.copy()
    def objective(z):
        return float((w * (z == 0)) @ K @ (w * (z == 1)))
    for iteration in range(15):
        masses = np.column_stack([w * (lab == 0), w * (lab == 1)])
        scores = K @ masses
        scores[np.arange(len(w)), lab] -= w * np.diag(K)
        candidate, status = assign_exact(w * (scores[:, 0] - scores[:, 1]), w, target)
        if objective(candidate) >= objective(lab) - 1e-9 * max(abs(objective(lab)), 1):
            break
        lab = candidate
    return lab


def main():
    DEST.mkdir(exist_ok=True)
    before = json.loads((PREV / 'provenance.json').read_text())
    for name, expected in before['output_sha256'].items():
        assert hashlib.sha256((PREV / name).read_bytes()).hexdigest() == expected, name
    data = np.load(BASE / 'group_features.npz')
    ids, X = data['ids'], data['X']
    counts = pd.read_csv(PREV / 'original_group_counts.csv').set_index('group_id').loc[ids]
    original = counts[['Train', 'Val']].to_numpy(np.int64)
    w = original.sum(1)
    target = int(original[:, 1].sum())
    B, _ = nystrom_features(X, rank=256, metric='euclidean', seed=0)
    B = B.astype(float)
    ref_group = standardize_for_metric(torch.as_tensor(X), 'euclidean')
    K = pairwise_similarity(ref_group, ref_group, 'euclidean').numpy().astype(float)
    np.fill_diagonal(K, 1.)
    table = pd.read_parquet(PREV / 'original_formula_counts.parquet')
    mapping = pd.read_csv(BASE / 'formula_assignments.csv')
    table = table.merge(mapping, left_on='formula', right_on='reduced_observed_formula', validate='one_to_one')
    F, elements = fraction_matrix(table['formula'].tolist())
    formula_weights = table['Train'].to_numpy(np.int64) + table['Val'].to_numpy(np.int64)
    mean = np.average(F.astype(float), axis=0, weights=formula_weights)
    var = np.average((F.astype(float) - mean) ** 2, axis=0, weights=formula_weights) * formula_weights.sum() / (formula_weights.sum() - 1)
    active = var > 1e-16
    ref = torch.as_tensor(((F[:, active] - mean[active]) / np.sqrt(var[active])).astype(np.float32))
    total, _ = pair_mean(ref, formula_weights, formula_weights, 100)
    total -= 1 / formula_weights.sum()
    arms = [('original', None, None)]
    diagnostics = []
    for seed in (0, 1, 2):
        initial, status = assign_exact(np.random.default_rng(seed).normal(size=len(w)), w, target)
        old = json.loads((PREV / f'matched_lowrank_seed{seed}.json').read_text())
        initial_cut = weighted_cut(B, w, initial)
        assert np.isclose(initial_cut, old['proposal_history'][0]['cut'], rtol=1e-10)
        saved = pd.read_csv(PREV / f'matched_lowrank_seed{seed}_groups.csv').set_index('group_id').loc[ids]
        optimized = (saved['split'].to_numpy() == 'val').astype(int)
        start = time.monotonic()
        dense = dense_optimize(K, w, initial, target)
        dense_s = time.monotonic() - start
        for name, lab in [('group_only_initialization', initial), ('lowrank', optimized), ('dense_kernel', dense)]:
            assert w @ lab == target
            arms.append((name, seed, lab))
            pd.DataFrame(dict(group_id=ids, split=np.where(lab, 'val', 'train'), configurations=w)).to_csv(
                DEST / f'{name}__seed{seed}_groups.csv', index=False)
            true = float((w * (lab == 0)) @ K @ (w * (lab == 1)))
            proxy = weighted_cut(B, w, lab)
            diagnostics.append(dict(method=name, seed=seed, val_groups=int(lab.sum()),
                largest_val_group_share=float(w[lab == 1].max() / target),
                top2_val_group_share=float(np.sort(w[lab == 1])[-2:].sum() / target),
                exact_kernel_cut=true, factor_cut=proxy, relative_cut_error=abs(true - proxy) / true,
                dense_optimization_seconds=dense_s if name == 'dense_kernel' else None))
    rows, independence = [], []
    for name, seed, lab in arms:
        if lab is None:
            train, val = table['Train'].to_numpy(np.int64), table['Val'].to_numpy(np.int64)
        else:
            mask = table['group_id'].map(dict(zip(ids, lab))).to_numpy(bool)
            val = formula_weights * mask
            train = formula_weights - val
        cross, _ = pair_mean(ref, train, val, 200)
        nn = nearest_mean(ref, train, val, 300)
        p_train = np.average(F, axis=0, weights=train)
        p_val = np.average(F, axis=0, weights=val)
        presence_train = np.average(F > 0, axis=0, weights=train)
        presence_val = np.average(F > 0, axis=0, weights=val)
        diff = presence_val - presence_train
        top = np.argsort(-abs(diff))[:8]
        row = dict(method=name, seed=seed,
            global_random_relative=cross / total * (1 - 1 / formula_weights.sum()), **nn,
            atomic_fraction_total_variation=float(abs(p_train - p_val).sum() / 2),
            max_element_presence_shift=float(abs(diff).max()),
            largest_presence_changes=[dict(element=elements[i], train=float(presence_train[i]), val=float(presence_val[i])) for i in top])
        rows.append(row)
        # Independent float64 SciPy reference for 32 sampled validation queries.
        rng = np.random.default_rng(405)
        q = rng.choice(len(ref), 32, p=val / val.sum())
        references = np.flatnonzero(train > 0)
        exact = 1 / (1 + cdist(ref[q].numpy().astype(float), ref[references].numpy().astype(float)).min(1))
        approx = []
        for idx in q:
            if train[idx] > 0:
                approx.append(1.)
            else:
                best = min(float(torch.cdist(ref[idx:idx + 1], ref[references[j:j + 8192]]).min())
                           for j in range(0, len(references), 8192))
                approx.append(1 / (1 + best))
        err = float(np.max(abs(np.array(approx) - exact)))
        independence.append(dict(method=name, seed=seed, float64_scipy_max_nn_difference=err))
        print(name, seed, 'global', round(row['global_random_relative'], 4),
              'NN', round(nn['val_nearest_train_similarity'], 4), 'reference error', err, flush=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != 'largest_presence_changes'} for r in rows]).to_csv(DEST / 'ablation.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(DEST / 'fidelity_and_concentration.csv', index=False)
    atomic_json(DEST / 'audit.json', dict(input_provenance_hashes_passed=True, additional_download_bytes=0,
        controls=rows, fidelity=diagnostics, independent_nn_checks=independence,
        max_independent_nn_error=max(r['float64_scipy_max_nn_difference'] for r in independence),
        note='Group-only initialization is a random-linear-cost feasible assignment, not a uniform sampler over feasible partitions. Dense comparator uses the same guarded heuristic.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
