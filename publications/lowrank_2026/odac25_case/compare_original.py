"""Offline original-vs-low-rank comparison at matched configuration counts.

The matched arm is a weighted, indivisible-group adaptation of the factor-cut
heuristic. A binary linear assignment (knapsack) enforces exact frame counts;
this is not the unmodified equal-group-capacity public splitter.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
import torch

from build_split import ROOT, family_name, reduced_formula, fraction_matrix, records
from collect import atomic_json
from lowrank.nystrom import nystrom_features
from lowrank.splitters.common.pairwise_similarity import standardize_for_metric, pairwise_similarity

DEST = ROOT / 'comparison_original'
BASE = ROOT / 'splits/Filtered/seed0'


def collect_membership():
    path = DEST / 'original_group_counts.csv'
    if all(p.exists() for p in [path, DEST / 'original_family_membership.csv', DEST / 'original_formula_counts.parquet']):
        return
    family_table = pd.read_csv(BASE / 'family_assignments.csv').set_index('inferred_family')['group_id'].to_dict()
    group_counts, family_counts, formula_counts = Counter(), Counter(), Counter()
    for original, row in records(sorted((ROOT / 'catalogues/Filtered').glob('*/co_*.json'))):
        families = sorted({family_name(n)[1] for n in row['name'].split('\x1f')})
        groups = {family_table[f] for f in families}
        assert len(groups) == 1
        n = int(row['multiplicity_sum'])
        group_counts[(groups.pop(), original)] += n
        formula_counts[(reduced_formula(row['formula']), original)] += n
        # Family counts are membership counts; aliases can refer to multiple
        # families, so these must not be summed as total configuration counts.
        for f in families:
            family_counts[(f, original)] += n
    def wide(counts, name):
        data = pd.DataFrame([(k, s, n) for (k, s), n in counts.items()], columns=[name, 'original_split', 'count'])
        data = data.pivot(index=name, columns='original_split', values='count').fillna(0).astype(np.int64)
        data['overlaps_original_folds'] = (data['Train'] > 0) & (data['Val'] > 0)
        return data
    wide(group_counts, 'group_id').to_csv(path)
    wide(family_counts, 'inferred_family').to_csv(DEST / 'original_family_membership.csv')
    wide(formula_counts, 'formula').reset_index().to_parquet(DEST / 'original_formula_counts.parquet', index=False)


def weighted_cut(B, w, labels):
    P = np.stack([(B[labels == c] * w[labels == c, None]).sum(0) for c in (0, 1)])
    return float(P[0] @ P[1])


def assign_exact(cost, w, target, seconds=10):
    """Return a verified exact-size incumbent, retaining solver termination."""
    cost = np.asarray(cost, dtype=float)
    scale = max(float(np.max(np.abs(cost))), 1e-12)
    r = milp(cost / scale, integrality=np.ones(len(w)),
        bounds=Bounds(np.zeros(len(w)), (w <= target).astype(float)),
        constraints=LinearConstraint(w.astype(float)[None, :], target, target),
        options={'time_limit': seconds, 'mip_rel_gap': .001, 'threads': 2})
    if r.x is None:
        raise RuntimeError(f'No exact-count group assignment found: {r.message}')
    labels = (r.x > .5).astype(int)
    if int(w @ labels) != target:
        raise RuntimeError('Returned incumbent violates exact integer frame counts')
    return labels, dict(status=int(r.status), message=r.message,
                        mip_gap=float(r.mip_gap), optimal=bool(r.status == 0))


def optimize(B, w, target, seed):
    rng = np.random.default_rng(seed)
    labels, status = assign_exact(rng.normal(size=len(w)), w, target)
    history = [dict(stage='initial_feasible', cut=weighted_cut(B, w, labels), solver=status)]
    for iteration in range(15):
        P = np.stack([(B[labels == c] * w[labels == c, None]).sum(0) for c in (0, 1)])
        # Self-excluded within-block scores give the standard factor-cut proposal.
        scores = B @ P.T
        scores[np.arange(len(w)), labels] -= w * (B * B).sum(1)
        candidate, status = assign_exact(w * (scores[:, 0] - scores[:, 1]), w, target)
        old, new = weighted_cut(B, w, labels), weighted_cut(B, w, candidate)
        accepted = new < old - 1e-9 * max(abs(old), 1)
        history.append(dict(stage='proposal', iteration=iteration, cut=new, accepted=accepted, solver=status))
        if not accepted:
            break
        labels = candidate
    return labels, history


def metrics(K, masses):
    train, val = masses.T
    w = train + val
    n = float(w.sum())
    # K is the exact kernel of the retained group representatives. Within-group
    # distinct frames have similarity one, and only actual self pairs are removed.
    total = float(w @ K @ w - n)
    cut = float(2 * train @ K @ val)
    expected = float(2 * train.sum() * val.sum() / (n * (n - 1)))
    nn = K[:, train > 0].max(1)
    return dict(train_configurations=int(train.sum()), val_configurations=int(val.sum()),
        global_share=cut / total, global_random_relative=(cut / total) / expected,
        val_nearest_train_similarity_frame_weighted=float(nn @ val / val.sum()),
        overlapping_groups=int(np.count_nonzero((train > 0) & (val > 0))),
        val_frames_in_overlapping_groups=int(val[train > 0].sum()))


def main():
    DEST.mkdir(exist_ok=True)
    collect_membership()
    data = np.load(BASE / 'group_features.npz')
    ids, X = data['ids'], data['X']
    counts = pd.read_csv(DEST / 'original_group_counts.csv').set_index('group_id').loc[ids]
    original = counts[['Train', 'Val']].to_numpy(np.int64)
    w = original.sum(1)
    target = int(original[:, 1].sum())
    assert int(original[:, 0].sum()) == 36920036 and target == 791638
    ref = standardize_for_metric(torch.as_tensor(X), 'euclidean')
    K = pairwise_similarity(ref, ref, 'euclidean').numpy().astype(np.float64)
    np.fill_diagonal(K, 1.)
    B, _ = nystrom_features(X, rank=256, metric='euclidean', seed=0)
    B = B.astype(float)
    input_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), BASE / 'group_features.npz', BASE / 'family_assignments.csv', DEST / 'original_group_counts.csv']}
    results = [dict(method='original', seed=None, **metrics(K, original))]
    for seed in (0, 1, 2):
        path = DEST / f'matched_lowrank_seed{seed}.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result.get('input_hashes') == input_hashes:
                results.append(result['metrics'])
                continue
        start = time.monotonic()
        lab, history = optimize(B, w, target, seed)
        runtime = time.monotonic() - start
        masses = np.column_stack([w * (lab == 0), w * (lab == 1)])
        result = dict(method='weighted_lowrank_exact_original_counts', seed=seed,
                      split_seconds=runtime, **metrics(K, masses))
        assignment = pd.DataFrame(dict(group_id=ids, split=np.where(lab, 'val', 'train'), configurations=w))
        assignment.to_csv(DEST / f'matched_lowrank_seed{seed}_groups.csv', index=False)
        atomic_json(path, dict(metrics=result, proposal_history=history, input_hashes=input_hashes,
            assignment_sha256=hashlib.sha256((DEST / f'matched_lowrank_seed{seed}_groups.csv').read_bytes()).hexdigest()))
        results.append(result)
        print(json.dumps(result), flush=True)
    pd.DataFrame(results).to_csv(DEST / 'comparison_group_proxy.csv', index=False)
    family_counts = pd.read_csv(DEST / 'original_family_membership.csv')
    formulas = pd.read_parquet(DEST / 'original_formula_counts.parquet')
    summary = dict(source='Offline pinned ODAC25 Filtered metadata; no additional downloads',
        original_split=dict(train=36920036, val=791638, published_test_download=False),
        original_overlap=dict(inferred_families=int(family_counts['overlaps_original_folds'].sum()),
            reduced_observed_formulas=int(formulas['overlaps_original_folds'].sum()),
            immutable_components=int(counts['overlaps_original_folds'].sum())),
        comparison=results,
        method='Binary weighted factor-cut heuristic with exact configuration-count MILP assignment and guarded proposals; immutable groups never broken',
        limits=['Metrics are exact for group-average composition representatives, not atomic geometry.',
                'Count matching does not control chemical or target composition.',
                'MILP optimality applies only to linear proposals, not the quadratic partition objective.',
                'Inferred CSD parents/composition groups can combine distinct structures.',
                'Timings exclude catalogue processing and factorization; this adapter differs from the plain public splitter.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(DEST / 'summary.json', summary)
    print(json.dumps(summary['original_overlap']), flush=True)


if __name__ == '__main__':
    main()
