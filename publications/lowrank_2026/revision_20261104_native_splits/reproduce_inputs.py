"""Reproduce the pinned metadata contents without resolving a new upstream HEAD."""
from pathlib import Path
import json
import requests
import zipfile

from fetch_metadata import Budget,RemoteZip,LIMIT,HERE,sha,atomic


def main():
    budget=Budget.__new__(Budget)
    budget.path=HERE/'cache/reproduction_transfer_ledger.json'
    budget.data=json.loads(budget.path.read_text()) if budget.path.exists() else dict(limit_bytes=LIMIT,requests=[],received_bytes=0)
    budget.session=requests.Session()
    original=json.loads((HERE/'downloaded_archive_metadata.json').read_text())
    catalog=json.loads((HERE/'zip_metadata_catalog.json').read_text())
    checked=[]
    for name,files in original.items():
        missing=[entry for entry in files if not (HERE/entry['path']).exists()]
        archive=None
        if missing:archive=zipfile.ZipFile(RemoteZip(budget,catalog[name]['url']))
        try:
            for entry in files:
                path=HERE/entry['path']
                if path.exists():data=path.read_bytes()
                else:
                    info=archive.getinfo(entry['member'])
                    if info.file_size!=entry['bytes'] or info.CRC!=entry['zip_crc32']:
                        raise RuntimeError('pinned archive metadata changed')
                    data=archive.read(entry['member'])
                    if sha(data)!=entry['sha256']:raise RuntimeError('pinned archive checksum changed')
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                if sha(data)!=entry['sha256']:raise RuntimeError('existing metadata checksum mismatch')
                checked.append(entry['path'])
        finally:
            if archive is not None:archive.close()
    moses=json.loads((HERE/'moses_metadata.json').read_text())
    for entry in moses['files']:
        path=HERE/entry['path'];data=path.read_bytes() if path.exists() else budget.get(entry['url'],expected=entry['bytes'])
        if sha(data)!=entry['sha256']:raise RuntimeError('pinned MOSES checksum mismatch')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);checked.append(entry['path'])
    result=dict(passed=True,checked_metadata_objects=checked,new_transfer_bytes=budget.data['received_bytes'],
                moses_commit=moses['upstream_commit'],source_sha256=sha(Path(__file__).read_bytes()))
    atomic(HERE/'verification/reproduced_input_identity.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
