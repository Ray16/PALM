"""Extract only names, formulas and multiplicities from pinned ODAC25 Parquet.

The HTTP reader rejects non-range responses. No coordinates, forces, energies,
or complete Parquet shards are downloaded. Per-shard compact caches are resumable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from range_io import RangeReader

ROOT = Path(__file__).resolve().parent
COLS = ['names', 'chemical_formula_hill', 'multiplicity']


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def manifest(release, split):
    path = ROOT / 'catalogues' / release / f'{split}_source.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    repo = f'colabfit/Open_Direct_Air_Capture_ODAC2025_{split}_{release}'
    r = requests.get(f'https://huggingface.co/api/datasets/{repo}', timeout=45)
    r.raise_for_status()
    sha = r.json()['sha']
    r = requests.get(f'https://huggingface.co/api/datasets/{repo}/tree/{sha}',
                     params={'recursive': 'true', 'limit': 1000}, timeout=45)
    r.raise_for_status()
    if 'next' in r.links:
        raise RuntimeError('unexpected paginated listing')
    shards = [x for x in r.json() if x['type'] == 'file' and x['path'].startswith('co/')]
    value = dict(repo=repo, revision=sha, columns=COLS, shards=shards)
    atomic_json(path, value)
    return value


def extract(job):
    release, split, source, shard = job
    dest = ROOT / 'catalogues' / release / split / (Path(shard['path']).stem + '.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return json.loads(dest.read_text())
    url = f'https://huggingface.co/datasets/{source["repo"]}/resolve/{source["revision"]}/{shard["path"]}'
    r = RangeReader(url, shard['size'], max_bytes=8 * 1024**2)
    p = pq.ParquetFile(r)
    records = []
    for group in range(p.num_row_groups):
        t = p.read_row_group(group, columns=COLS, use_threads=False)
        if t['names'].null_count or not pc.all(pc.greater(pc.list_value_length(t['names']), 0)).as_py():
            raise RuntimeError('every row must retain at least one trajectory identity')
        # Deduplicated configurations can carry multiple original names. Preserve
        # every alias so downstream grouping can join them into one immutable unit.
        compact = pa.table({'name': pc.binary_join(t['names'], '\x1f'),
                            'formula': t['chemical_formula_hill'],
                            'multiplicity': t['multiplicity']})
        rows = compact.group_by(['name', 'formula']).aggregate([
            ('multiplicity', 'sum'), ('multiplicity', 'count')]).to_pylist()
        records.extend(rows)
    result = dict(repo=source['repo'], revision=source['revision'], shard=shard['path'],
        remote_bytes=shard['size'], rows=p.metadata.num_rows, columns=COLS,
        downloaded_bytes=r.bytes_read, http_range_requests=r.requests_made, records=records)
    atomic_json(dest, result)
    print(f'{split}/{shard["path"]}: {result["rows"]:,} rows, '
          f'{r.bytes_read:,} bytes transferred, {len(records):,} name/formula groups', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', choices=['Filtered', 'Full'], default='Filtered')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    jobs = []
    for split in ['Train', 'Val']:
        source = manifest(args.release, split)
        shards = source['shards'][:args.limit] if args.limit else source['shards']
        jobs.extend((args.release, split, source, s) for s in shards)
    # Worst-case reader cap protects against accidentally projecting large columns.
    if len(jobs) * 8 * 1024**2 > 1024**3:
        raise RuntimeError('worst-case transfer exceeds 1 GiB budget')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(extract, jobs))
    summary = dict(release=args.release, shards=len(results),
        rows=sum(r['rows'] for r in results),
        downloaded_bytes=sum(r['downloaded_bytes'] for r in results),
        remote_shard_bytes=sum(r['remote_bytes'] for r in results),
        partial=bool(args.limit), columns=COLS,
        cache_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'catalogues' / args.release).glob('*/*.json'))})
    atomic_json(ROOT / 'catalogues' / args.release / 'collection_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cache_sha256'}, indent=2))


if __name__ == '__main__':
    main()
