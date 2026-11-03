"""Composition-only, immutable-family split of a pinned ODAC25 mirror.

Input: compact range-read metadata caches, never coordinate/force data.
Balance unit: independent family/composition connected components, NOT frames.
Family identifiers are explicitly inferred from trajectory names, not claimed to
be the official atoms.info.mof_name field (not retained by this mirror).
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / 'low_rank/paper/mlst_2026/release_candidate_namespaced'
sys.path.insert(0, str(PACKAGE))
from lowrank import split, SplitSpec
from lowrank.splitters.common.pairwise_similarity import standardize_for_metric, pairwise_similarity
from collect import atomic_json


@lru_cache(maxsize=None)
def reduced_formula(formula):
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    if ''.join(e + n for e, n in tokens) != formula:
        raise ValueError(f'Unsupported formula: {formula}')
    counts = Counter()
    for e, n in tokens:
        counts[e] += int(n or 1)
    gcd = math.gcd(*counts.values())
    return ''.join(e + (str(n // gcd) if n // gcd != 1 else '')
                   for e, n in sorted(counts.items()))


@lru_cache(maxsize=None)
def family_name(full_name):
    pieces = full_name.split('__', 2)
    if len(pieces) != 3 or pieces[1] not in ('mof', 'mof_plus_adsorbate', 'gcmc'):
        raise ValueError(f'Unknown ODAC25 name convention: {full_name}')
    partition, key = pieces[1:]
    key = key.removesuffix('.traj')
    host = re.split(r'_w_|_cut\d', key)[0]
    if re.match(r'^Gen\d+_', host):
        host = re.match(r'^(Gen\d+)_', host)[1]
    elif re.match(r'^[A-Z]{6}(?:\d{2})?(?:_|$)', host):
        # Conservative CSD family rule: keep polymorph suffixes, defect and
        # functionalization variants together at the six-letter parent code.
        host = host[:6]
    else:
        host = re.split(r'_(?:0\.\d+|1(?:\.0)?)_\d+(?:_|$)', host)[0]
    if not host:
        raise ValueError(f'Empty inferred family: {full_name}')
    return partition, host


@lru_cache(maxsize=None)
def infer_host_formula(full_name, formula):
    """Remove only explicit adsorbate stoichiometry in a known name convention.

    A trailing integer is a placement index, not an adsorbate count. GCMC
    occupancy is not encoded unambiguously and is never guessed.
    """
    partition, key = full_name.split('__', 2)[1:]
    if partition != 'mof_plus_adsorbate':
        return None
    if '_w_' in key:
        suffix = key.split('_w_', 1)[1]
    else:
        match = re.fullmatch(r'Gen\d+_(.+)', key)
        if not match:
            return None
        suffix = match[1]
    suffix = re.sub(r'_(?:random_)?(?:\d+|single)$', '', suffix)
    species = {'CO2': {'C': 1, 'O': 2}, 'H2O': {'H': 2, 'O': 1},
               'N2': {'N': 2}, 'O2': {'O': 2}}
    tokens = suffix.split('_')
    subtract = Counter()
    for token in tokens:
        m = re.fullmatch(r'(\d*)(CO2|H2O|N2|O2)', token)
        if not m:
            return None
        for element, n in species[m[2]].items():
            subtract[element] += int(m[1] or 1) * n
    counts = Counter({e: int(n or 1) for e, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula)})
    for element, n in subtract.items():
        counts[element] -= n
        if counts[element] < 0:
            raise ValueError(f'Impossible adsorbate subtraction: {full_name} / {formula}')
    result = ''.join(e + (str(n) if n != 1 else '') for e, n in sorted(counts.items()) if n)
    return reduced_formula(result)


def fraction_matrix(formulas):
    """118 elemental atom fractions; no incomplete property-table imputation."""
    from ase.data import atomic_numbers, chemical_symbols
    X = np.zeros((len(formulas), 118), dtype=np.float32)
    for i, formula in enumerate(formulas):
        counts = {e: int(n or 1) for e, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula)}
        total = sum(counts.values())
        for e, count in counts.items():
            X[i, atomic_numbers[e] - 1] = count / total
    return X, chemical_symbols[1:119]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def records(paths):
    for path in paths:
        data = json.loads(path.read_text())
        for row in data['records']:
            yield path.parent.name, row


def score(X, labels, random_labels, seed):
    import torch
    ref = standardize_for_metric(torch.as_tensor(X), 'euclidean')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), min(2000, len(X)), replace=False))
    S = pairwise_similarity(ref[idx], ref[idx], 'euclidean').numpy().astype(float)
    np.fill_diagonal(S, 0)
    out = {}
    for name, lab in [('lowrank', labels), ('random', random_labels)]:
        global_score = float(S[lab[idx, None] != lab[None, idx]].sum() / S.sum())
        test = np.flatnonzero(lab == 'test')
        if len(test) > 1024:
            test = np.sort(rng.choice(test, 1024, replace=False))
        train = np.flatnonzero(lab == 'train')
        nearest = []
        for block in np.array_split(test, max(1, math.ceil(len(test) / 128))):
            sim = pairwise_similarity(ref[block], ref[train], 'euclidean')
            nearest.extend(sim.max(1).values.tolist())
        out[name] = dict(global_similarity_share_sampled=global_score,
            global_sample_n=len(idx), test_query_n=len(test), training_reference_n=len(train),
            nearest_train_similarity_mean=float(np.mean(nearest)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', choices=['Filtered', 'Full'], default='Filtered')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank', type=int, default=256)
    args = parser.parse_args()
    source = ROOT / 'catalogues' / args.release
    collection = json.loads((source / 'collection_summary.json').read_text())
    if collection['partial']:
        raise RuntimeError('Cannot publish a split from an incomplete catalogue')
    paths = sorted(source.glob('*/co_*.json'))
    if len(paths) != collection['shards']:
        raise RuntimeError('Shard count mismatch')
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != collection['cache_sha256'][str(path.relative_to(ROOT))]:
            raise RuntimeError(f'Changed compact input: {path}')
    out = ROOT / 'splits' / args.release / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    uf = UnionFind()
    bare = defaultdict(set)
    observed = defaultdict(set)
    inferred = defaultdict(set)
    bare_trajectory = defaultdict(set)
    inferred_trajectory = defaultdict(set)
    alias_records = 0
    for _, row in records(paths):
        aliases = row['name'].split('\x1f')
        alias_records += len(aliases) > 1
        keys = [family_name(s) for s in aliases]
        for partition, family in keys:
            uf.find(family)
            uf.union(keys[0][1], family)
            observed[family].add(reduced_formula(row['formula']))
            if partition == 'mof':
                bare[family].add(reduced_formula(row['formula']))
        for name, (partition, family) in zip(aliases, keys):
            trajectory = name.split('__', 2)[2].removesuffix('.traj')
            if partition == 'mof':
                bare_trajectory[trajectory].add(reduced_formula(row['formula']))
            elif partition == 'mof_plus_adsorbate':
                formula = infer_host_formula(name, row['formula'])
                if formula:
                    inferred[family].add(formula)
                    inferred_trajectory[trajectory].add(formula)
    all_families = sorted(uf.parent)
    recovered = sorted(set(all_families) - set(bare))
    overlap = sorted(set(bare_trajectory) & set(inferred_trajectory))
    mismatches = [t for t in overlap if bare_trajectory[t] != inferred_trajectory[t]]
    atomic_json(out / 'host_formula_validation.json', dict(
        matched_trajectories=len(overlap), mismatch_count=len(mismatches),
        mismatch_examples=mismatches[:20], families_without_bare_partition=recovered))
    if mismatches:
        raise RuntimeError('Adsorbate subtraction disagrees with matched bare trajectories')
    for family, formulas in inferred.items():
        bare[family].update(formulas)
    no_bare = sorted(set(all_families) - set(bare))
    print(f'{len(all_families):,} inferred families; {len(bare):,} with bare formulas; '
          f'{len(no_bare):,} without bare metadata', flush=True)
    # Unknown bare compositions are reported. All feature vectors consistently
    # use observed full-system compositions; no missing bare formula is guessed.
    if no_bare:
        atomic_json(out / 'missing_bare_families.json', no_bare)
    formula_owner = {}
    for family in all_families:
        # Both repeated observed composition and known host composition create
        # immutable links. Formula prefixes keep the two meanings distinct.
        for formula in sorted(observed[family]):
            owner = formula_owner.setdefault(formula, family)
            uf.union(family, owner)
    bare_owner = {}
    for family in sorted(bare):
        for formula in sorted(bare[family]):
            uf.union(family, bare_owner.setdefault(formula, family))
    components = defaultdict(set)
    for family in all_families:
        components[uf.find(family)].add(family)
    formulas = sorted(formula_owner)
    print(f'{len(formulas):,} reduced observed formulas; {len(components):,} immutable components', flush=True)
    f_ids = formulas
    f_X, feature_names = fraction_matrix(formulas)
    f_index = {s: i for i, s in enumerate(f_ids)}
    group_ids = sorted(components)
    group_formulas = {g: sorted(set().union(*(observed[f] for f in components[g]))) for g in group_ids}
    # Equal weight per distinct observed composition within each independent group.
    X = np.stack([f_X[[f_index[f] for f in group_formulas[g]]].mean(0) for g in group_ids])
    features = {g: X[i] for i, g in enumerate(group_ids)}
    t = time.monotonic()
    result = split('lowrank', features, SplitSpec([8, 1, 1], ['train', 'val', 'test'], seed=args.seed),
        metric='euclidean', rank=args.rank, n_restarts=4, n_iter=25,
        allowed_size_imbalance=0., score_exact_leakage=False)
    elapsed = time.monotonic() - t
    assignments = result.assignment
    lab = np.array([assignments[g] for g in group_ids])
    rand = np.random.default_rng(args.seed).permutation(lab)
    random_assignment = dict(zip(group_ids, rand))
    group_rows, family_rows, formula_rows = [], [], []
    for g in group_ids:
        group_rows.append(dict(group_id=g, split=assignments[g], random_split=random_assignment[g],
            family_count=len(components[g]), formula_count=len(group_formulas[g]),
            families=';'.join(sorted(components[g])), formulas=';'.join(group_formulas[g])))
        for f in sorted(components[g]):
            family_rows.append(dict(inferred_family=f, group_id=g, split=assignments[g],
                random_split=random_assignment[g], bare_formulas=';'.join(sorted(bare[f]))))
        for formula in group_formulas[g]:
            formula_rows.append(dict(reduced_observed_formula=formula, group_id=g, split=assignments[g]))
    pd.DataFrame(group_rows).to_csv(out / 'group_assignments.csv', index=False)
    pd.DataFrame(family_rows).to_csv(out / 'family_assignments.csv', index=False)
    pd.DataFrame(formula_rows).to_csv(out / 'formula_assignments.csv', index=False)
    np.savez_compressed(out / 'group_features.npz', ids=np.array(group_ids), X=X, labels=lab)
    counts = Counter()
    for original, row in records(paths):
        families = [family_name(s)[1] for s in row['name'].split('\x1f')]
        groups = {uf.find(f) for f in families}
        if len(groups) != 1:
            raise RuntimeError('Configuration aliases cross immutable groups')
        g = groups.pop()
        counts[(original, assignments[g], 'unique_configuration_rows')] += row['multiplicity_count']
        counts[(original, assignments[g], 'multiplicity_sum')] += row['multiplicity_sum']
    count_rows = [dict(original_split=o, split=s, kind=k, count=c) for (o, s, k), c in sorted(counts.items())]
    pd.DataFrame(count_rows).to_csv(out / 'configuration_counts.csv', index=False)
    assert sum(c for (o, s, k), c in counts.items() if k == 'unique_configuration_rows') == collection['rows']
    evaluation = score(X, lab, rand, args.seed)
    package_files = sorted((PACKAGE / 'lowrank').rglob('*.py'))
    summary = dict(release=args.release, seed=args.seed, rank=args.rank, split_seconds=elapsed,
        scope='Pinned ColabFit ODAC25 mirror; not certified identical to current official archives',
        balance_unit='immutable family/composition connected components, each weighted equally',
        family_id_source='Conservative name inference; official mof_name unavailable in mirror metadata',
        representation='118 elemental atom fractions, averaged over distinct observed full-system compositions within a group; standardized Euclidean kernel',
        host_formula_reconstruction=dict(families_recovered=len(recovered) - len(no_bare),
            matched_bare_trajectory_checks=len(overlap), mismatches=len(mismatches)),
        families=len(all_families), reduced_observed_formulas=len(formulas),
        known_reduced_bare_formulas=len(bare_owner), groups=len(group_ids),
        group_counts=dict(Counter(lab)), configuration_counts=count_rows,
        alias_records=alias_records, missing_bare_families=no_bare,
        downloaded_bytes=collection['downloaded_bytes'], represented_rows=collection['rows'],
        validation=dict(all_source_shards_covered=True, all_configuration_rows_counted=True,
            configuration_aliases_disjoint=True, reduced_observed_composition_overlap=0,
            known_reduced_bare_composition_overlap=0,
            inferred_family_overlap=0, immutable_groups_broken=0),
        diagnostics=result.diagnostics, evaluation=evaluation,
        package_source_sha256={str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in package_files},
        output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*')) if p.is_file()})
    atomic_json(out / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('package_source_sha256', 'output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
