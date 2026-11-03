"""Write a reproducible source and transfer ledger from retained per-shard logs."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from collect import atomic_json

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    release = 'Filtered'
    catalogue = ROOT / 'catalogues' / release
    source_manifests = [json.loads(p.read_text()) for p in sorted(catalogue.glob('*_source.json'))]
    sources = {s['repo']: s for s in source_manifests}
    counts = Counter()
    shards = []
    for path in sorted(catalogue.glob('*/co_*.json')):
        d = json.loads(path.read_text())
        original_split = path.parent.name
        counts[(original_split, 'unique_rows')] += d['rows']
        counts[(original_split, 'multiplicity_sum')] += sum(r['multiplicity_sum'] for r in d['records'])
        shards.append(dict(repo=d['repo'], revision=d['revision'], remote_path=d['shard'],
            url=f'https://huggingface.co/datasets/{d["repo"]}/resolve/{d["revision"]}/{d["shard"]}',
            source_file_bytes=d['remote_bytes'], bytes_downloaded=d['downloaded_bytes'],
            range_requests=d['http_range_requests'], columns=d['columns'], rows=d['rows'],
            cached_file=str(path.relative_to(ROOT)), cache_sha256=sha(path),
            cache_written_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
    official = json.loads((ROOT / 'official_release_reference.json').read_text())
    for split in ['Train', 'Val']:
        assert counts[(split, 'multiplicity_sum')] == official[f'filtered_{split.lower()}_configurations']
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), release=release,
        original_dataset=official,
        original_documentation=dict(path='sources/ODAC25_DATASET.md', sha256=sha(ROOT / 'sources/ODAC25_DATASET.md')),
        mirror_repositories=[dict(repo=s['repo'], revision=s['revision'],
            url=f'https://huggingface.co/datasets/{s["repo"]}/tree/{s["revision"]}') for s in source_manifests],
        method=dict(protocol='HTTPS byte-range GET requests with Range header',
            implementation='range_io.py + pyarrow.parquet.ParquetFile.read_row_group',
            selected_columns=['names', 'chemical_formula_hill', 'multiplicity'],
            refused_responses='Any status other than 206 or mismatched Content-Range; body is closed before a full-file read',
            per_shard_byte_cap=8 * 1024**2, workers=2,
            coordinates_read=False, forces_read=False, energies_read=False,
            no_complete_parquet_or_original_archive_downloaded=True),
        transfer=dict(production_bytes=sum(s['bytes_downloaded'] for s in shards),
            represented_remote_parquet_bytes=sum(s['source_file_bytes'] for s in shards),
            known_separate_probe_bytes=json.loads((ROOT / 'probe.json').read_text())['bytes_read'],
            preliminary_failed_train_probe='Small projected-column read; exact transfer not retained. Excluded from production total.',
            api_and_documentation_bytes='Small API listings/documentation requests; not metered',
            note='Production total is exact for successful cached extraction reads; it is not an exact whole-session network counter.'),
        coverage=[dict(original_split=s, unique_configuration_rows=counts[(s, 'unique_rows')],
            multiplicity_sum=counts[(s, 'multiplicity_sum')],
            official_count=official[f'filtered_{s.lower()}_configurations'], count_matches=True)
            for s in ['Train', 'Val']],
        coverage_limit='Mirror source names/formulas and multiplicities match official aggregate counts; no row-by-row comparison to original archives or energy-label equivalence is claimed.',
        local_cache='JSON expansion is larger than network transfer; file mtimes record cache writes, not exact network-request timestamps.',
        scripts_sha256={p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}, shards=shards)
    atomic_json(ROOT / 'PROVENANCE.json', result)
    print(json.dumps({k: result[k] for k in ['transfer', 'coverage']}, indent=2))


if __name__ == '__main__':
    main()
