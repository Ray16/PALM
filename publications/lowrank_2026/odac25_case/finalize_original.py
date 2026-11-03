"""Validate the matched comparison and save family maps and provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
from build_split import ROOT
from collect import atomic_json

DEST = ROOT / 'comparison_original'
BASE = ROOT / 'splits/Filtered/seed0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    family = pd.read_csv(BASE / 'family_assignments.csv')[['inferred_family', 'group_id']]
    original = pd.read_csv(DEST / 'original_group_counts.csv').set_index('group_id')
    evaluation = json.loads((DEST / 'recorded_formula_evaluation.json').read_text())
    for name, checksum in evaluation['input_hashes'].items():
        assert digest(ROOT / name) == checksum, f'Changed evaluation input: {name}'
    for seed in (0, 1, 2):
        data = pd.read_csv(DEST / f'matched_lowrank_seed{seed}_groups.csv')
        assert len(data) == len(original) and data['group_id'].is_unique
        assert set(data['group_id']) == set(original.index)
        assert (data['configurations'].to_numpy() == original.loc[data['group_id'], ['Train', 'Val']].sum(1).to_numpy()).all()
        assert data.loc[data['split'] == 'val', 'configurations'].sum() == 791638
        assert data.loc[data['split'] == 'train', 'configurations'].sum() == 36920036
        record = json.loads((DEST / f'matched_lowrank_seed{seed}.json').read_text())
        for name, checksum in record['input_hashes'].items():
            assert digest(ROOT / name) == checksum, f'Changed optimizer input: {name}'
        propagated = family.merge(data[['group_id', 'split']], on='group_id', validate='many_to_one')
        assert len(propagated) == len(family)
        propagated.to_csv(DEST / f'matched_lowrank_seed{seed}_families.csv', index=False)
    files = [p for p in DEST.iterdir() if p.is_file() and p.name != 'provenance.json']
    scripts = [ROOT / name for name in ['compare_original.py', 'evaluate_original_formulas.py',
                                        'finalize_original.py', 'test_original_comparison.py']]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        original_data_provenance='../PROVENANCE.json',
        original_data_provenance_sha256=digest(ROOT / 'PROVENANCE.json'),
        additional_download_bytes=0, device='CPU', native_thread_limit=2,
        assignments_verified_seeds=[0, 1, 2], exact_configuration_counts=True,
        independent_groups_broken=0, family_mapping_complete=True,
        script_sha256={str(p.relative_to(ROOT)): digest(p) for p in scripts},
        output_sha256={p.name: digest(p) for p in files},
        tests='Three comparison-specific tests passed; see ../comparison_tests.log',
        timing='Per-seed optimizer timings exclude metadata aggregation, kernel factorization and evaluation.')
    atomic_json(DEST / 'provenance.json', result)
    print('Verified three exact-count assignments, propagated all family maps, and saved provenance.')


if __name__ == '__main__':
    main()
