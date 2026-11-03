"""Validate frozen metadata-only ODAC25 evidence and generate manuscript values."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
CASE = PAPER.parent / 'odac25_case'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((CASE / 'evidence_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(CASE / name) != expected:
            raise RuntimeError('ODAC25 frozen evidence changed: ' + name)
    original = pd.read_csv(CASE / 'comparison_original/original_group_counts.csv').set_index('group_id')
    family = pd.read_csv(CASE / 'splits/Filtered/seed0/family_assignments.csv')
    formulas = pd.read_parquet(CASE / 'comparison_original/original_formula_counts.parquet')
    w = original[['Train', 'Val']].sum(axis=1)
    assert int(original.Train.sum()) == 36920036
    assert int(original.Val.sum()) == 791638
    assert int(formulas.Train.sum()) == 36920036 and int(formulas.Val.sum()) == 791638
    signatures = []
    for seed in (0, 1, 2):
        assignment_path = CASE / f'comparison_original/matched_lowrank_seed{seed}_groups.csv'
        a = pd.read_csv(assignment_path).set_index('group_id').loc[original.index]
        assert len(a) == 2050 and set(a.split) == {'train', 'val'}
        assert np.array_equal(a.configurations, w)
        assert int(w[a.split == 'val'].sum()) == 791638
        assert int(w[a.split == 'train'].sum()) == 36920036
        assert set(family.group_id) == set(a.index)
        signatures.append(sha(assignment_path))
    audit = json.loads((CASE / 'audit_original_result/audit.json').read_text())
    rows = pd.DataFrame(audit['controls'])
    assert len(rows) == 10 and audit['max_independent_nn_error'] < .0001
    direct = pd.read_csv(CASE / 'comparison_original/comparison_recorded_formulas.csv')
    for method, original_method in [('original', 'original'), ('lowrank', 'weighted_lowrank_exact_original_counts')]:
        left = rows[rows.method == method]
        right = direct[direct.method == original_method]
        assert np.allclose(left.global_random_relative, right.global_random_relative_mc)
        assert np.allclose(left.val_nearest_train_similarity, right.val_nearest_train_similarity)
    provenance = json.loads((CASE / 'PROVENANCE.json').read_text())
    fidelity = pd.DataFrame(audit['fidelity'])
    lr = rows[rows.method == 'lowrank']
    lr0 = lr[lr.seed == 0].iloc[0]
    controls = rows[rows.method == 'group_only_initialization']
    original_row = rows[rows.method == 'original'].iloc[0]
    concentration = fidelity[fidelity.method == 'lowrank']
    def interval(values):
        a, b = float(values.min()), float(values.max())
        return f'{a:.3f}' if f'{a:.3f}' == f'{b:.3f}' else f'{a:.3f}--{b:.3f}'
    values = dict(ODACFrames='37,711,674', ODACTrain='36,920,036', ODACVal='791,638',
        ODACUniqueRows='37,456,852', ODACGroups='2,050', ODACFamilies='2,913', ODACFormulas='205,269',
        ODACTransferMB=f"{provenance['transfer']['production_bytes'] / 1e6:.1f}",
        ODACOriginalGlobal=f'{original_row.global_random_relative:.3f}',
        ODACOriginalNN=f'{original_row.val_nearest_train_similarity:.3f}',
        ODACLowGlobal=interval(lr.global_random_relative), ODACLowNN=interval(lr.val_nearest_train_similarity),
        ODACGroupsGlobal=interval(controls.global_random_relative), ODACGroupsNN=interval(controls.val_nearest_train_similarity),
        ODACValidationGroups=f'{int(concentration.val_groups.min())}--{int(concentration.val_groups.max())}',
        ODACTopTwo=f'{100 * concentration.top2_val_group_share.iloc[0]:.1f}',
        ODACOriginalRepeated=f'{100 * original_row.val_fraction_with_exact_training_formula:.1f}',
        ODACDistinctAssignments=str(len(set(signatures))),
        ODACShiftOriginal=f'{original_row.atomic_fraction_total_variation:.3f}',
        ODACShiftLow=f'{lr0.atomic_fraction_total_variation:.3f}',
        ODACNNCheck=f"{audit['max_independent_nn_error']:.5f}",
        ODACCutError=f'{100 * fidelity[fidelity.method == "lowrank"].relative_cut_error.max():.2f}')
    pchange = next(r for r in lr0.largest_presence_changes if r['element'] == 'P')
    values.update(ODACPhosphorusVal=f"{100 * pchange['val']:.1f}", ODACPhosphorusTrain=f"{100 * pchange['train']:.1f}")
    (PAPER / 'odac25_numbers.tex').write_text('% Generated from frozen ODAC25 records.\n' + '\n'.join(
        rf'\newcommand{{\{k}}}{{{v}}}' for k, v in values.items()) + '\n')
    methods = [('original', 'Original'), ('group_only_initialization', 'Groups before optimization'),
               ('lowrank', 'Low-rank optimization'), ('dense_kernel', 'Dense-kernel control')]
    table = [r'\begin{table}[htbp]\centering', r'\begin{tabular}{lrr}', r'\toprule',
        r'Assignment & Global ratio $\downarrow$ & Mean nearest $\downarrow$\\', r'\midrule']
    for method, label in methods:
        r = rows[rows.method == method]
        table.append(label + ' & ' + interval(r.global_random_relative) + ' & ' + interval(r.val_nearest_train_similarity) + r'\\')
    table += [r'\bottomrule\end{tabular}',
        r'\caption{ODAC25 composition separation at identical configuration counts. The global ratio is normalized by the random expectation at those counts. Scores use recorded elemental fractions, configuration weights, 200,000 pair draws and 512 validation queries against every distinct training formula. Ranges span three optimizer seeds, which yield only two distinct low-rank assignments. The group control uses the same exact-count feasible initialization; it is not a uniform sampler of feasible partitions.}',
        r'\label{tab:odac25}\end{table}']
    (PAPER / 'odac25_table.tex').write_text('\n'.join(table) + '\n')
    seed_table = [r'\begin{table}[htbp]\centering\begin{tabular}{llrrr}', r'\toprule',
        r'Method & Seed & Global ratio & Mean nearest & NN MC SE\\\midrule']
    for method, label in methods:
        for _, r in rows[rows.method == method].iterrows():
            seed = '--' if pd.isna(r.seed) else str(int(r.seed))
            seed_table.append(f'{label} & {seed} & {r.global_random_relative:.4f} & {r.val_nearest_train_similarity:.4f} & {r.nearest_similarity_mc_stderr:.4f}' + r'\\')
    seed_table += [r'\bottomrule\end{tabular}', r'\caption{ODAC25 individual observations. MC SE describes configuration-query sampling error only. It does not describe uncertainty over independent MOFs, projects, or prediction models.}\end{table}']
    (PAPER / 'odac25_seed_table.tex').write_text('\n'.join(seed_table) + '\n')
    result = dict(passed=True, frozen_files=len(manifest['files']), original_counts_matched=True,
        immutable_groups_broken=0, observed_formula_overlap=0, distinct_lowrank_assignments=len(set(signatures)),
        evidence_manifest_sha256=sha(CASE / 'evidence_manifest.json'),
        generator_sha256=sha(Path(__file__)), generated_files={name: sha(PAPER / name) for name in
            ['odac25_numbers.tex', 'odac25_table.tex', 'odac25_seed_table.tex']},
        scope='Frozen weighted v0.1-factor case study; no new downloads, experiments, or predictive claims')
    (CASE / 'manuscript_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Validated ODAC25 case and generated manuscript values/tables.')


if __name__ == '__main__':
    main()
