"""Apply saved assignments by original ODAC25 trajectory name, without atoms."""
import argparse
import csv
from pathlib import Path
from build_split import family_name

DEFAULT = Path(__file__).resolve().parent / 'splits/Filtered/seed0/family_assignments.csv'


class SplitLookup:
    def __init__(self, assignments=DEFAULT):
        with open(assignments, newline='') as stream:
            self.assignments = {r['inferred_family']: r['split'] for r in csv.DictReader(stream)}

    def __call__(self, name, partition=None):
        if '__' not in name:
            if partition not in ('mof', 'mof_plus_adsorbate', 'gcmc'):
                raise ValueError('Supply the original partition for an unprefixed atoms.info.name')
            name = f'ODAC25__{partition}__{name}'
        _, family = family_name(name)
        if family not in self.assignments:
            raise KeyError(f'Family {family!r} is absent from the pinned metadata catalogue; do not assign it silently')
        return self.assignments[family]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--partition', choices=['mof', 'mof_plus_adsorbate', 'gcmc'])
    parser.add_argument('--assignments', type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(SplitLookup(args.assignments)(args.name, args.partition))
