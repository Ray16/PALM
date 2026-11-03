"""Check saved split tables against every compact source row and source count."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from build_split import family_name, reduced_formula
from collect import atomic_json

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'splits/Filtered/seed0'


def main():
    with open(DEST / 'family_assignments.csv', newline='') as f:
        families = {r['inferred_family']: r for r in csv.DictReader(f)}
    with open(DEST / 'formula_assignments.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    formulas = {r['reduced_observed_formula']: r for r in rows}
    assert len(formulas) == len(rows), 'A reduced formula appears in multiple groups'
    source_counts, split_counts = Counter(), Counter()
    checked = 0
    for path in sorted((ROOT / 'catalogues/Filtered').glob('*/co_*.json')):
        d = json.loads(path.read_text())
        counted = 0
        for r in d['records']:
            mapped = [families[family_name(n)[1]] for n in r['name'].split('\x1f')]
            assert len({v['group_id'] for v in mapped}) == 1, 'Aliases cross groups'
            formula_row = formulas[reduced_formula(r['formula'])]
            assert all(v['group_id'] == formula_row['group_id'] and v['split'] == formula_row['split'] for v in mapped)
            counted += r['multiplicity_count']
            source_counts[path.parent.name] += r['multiplicity_sum']
            split_counts[mapped[0]['split']] += r['multiplicity_sum']
            checked += 1
        assert counted == d['rows'], 'Shard row count mismatch'
    official = json.loads((ROOT / 'official_release_reference.json').read_text())
    assert source_counts['Train'] == official['filtered_train_configurations']
    assert source_counts['Val'] == official['filtered_val_configurations']
    summary = json.loads((DEST / 'summary.json').read_text())
    for name, digest in summary['output_sha256'].items():
        assert hashlib.sha256((DEST / name).read_bytes()).hexdigest() == digest, name
    atomic_json(DEST / 'independent_audit.json', dict(passed=True,
        compact_source_records_checked=checked, original_configuration_counts=dict(source_counts),
        assigned_configuration_counts=dict(split_counts),
        alias_group_conflicts=0, observed_formula_group_conflicts=0,
        table_checksums_match=True,
        scope='All compact source name/formula/count records; no original coordinates/forces inspected'))
    print('All compact source records and saved assignment tables reconcile.')


if __name__ == '__main__':
    main()
