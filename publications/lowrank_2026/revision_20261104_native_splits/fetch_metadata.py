"""Size-capped metadata retrieval, including selective HTTP-range ZIP members."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import requests

HERE=Path(__file__).resolve().parent
LIMIT=200*1024**2
OBJECT_LIMIT=128*1024**2
URLS={
    'pcqm4mv2':'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip',
    'molpcba':'https://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip',
}


def sha(data):return hashlib.sha256(data).hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class Budget:
    def __init__(self):
        self.path=HERE/'transfer_ledger.json'
        self.data=json.loads(self.path.read_text()) if self.path.exists() else dict(limit_bytes=LIMIT,requests=[],received_bytes=0)
        self.session=requests.Session()

    def get(self,url,start=None,end=None,expected=None):
        requested=(end-start+1) if start is not None else expected
        if requested is None or requested>OBJECT_LIMIT or requested+self.data['received_bytes']>LIMIT:
            raise RuntimeError('metadata transfer budget exceeded before request')
        headers={'Accept-Encoding':'identity'}
        if start is not None:headers['Range']=f'bytes={start}-{end}'
        r=self.session.get(url,headers=headers,stream=True,timeout=60)
        expected_status=206 if start is not None else 200
        if r.status_code!=expected_status:
            r.close();raise RuntimeError(f'HTTP status {r.status_code}: {url}; no response payload consumed')
        if start is not None and not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
            r.close();raise RuntimeError('server ignored requested byte interval')
        chunks=[];received=0
        for chunk in r.iter_content(1024*64):
            received+=len(chunk)
            if received>requested or self.data['received_bytes']+received>LIMIT:
                r.close();raise RuntimeError('oversized metadata response')
            chunks.append(chunk)
        r.close()
        self.data['received_bytes']+=received
        self.data['requests'].append(dict(url=url,start=start,end=end,bytes=received,
                                         etag=r.headers.get('ETag'),time=time.time()))
        atomic(self.path,self.data)
        if received!=requested:raise RuntimeError(f'incomplete metadata: expected {requested}, got {received}')
        return b''.join(chunks)

    def head(self,url):
        r=self.session.head(url,allow_redirects=True,headers={'Accept-Encoding':'identity'},timeout=30)
        r.raise_for_status()
        size=int(r.headers['Content-Length'])
        return size,dict(r.headers)


class RemoteZip(io.RawIOBase):
    def __init__(self,budget,url):
        self.budget,self.url=budget,url;self.size,self.headers=budget.head(url);self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if not 0<=self.pos<=self.size:raise ValueError('invalid archive seek')
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n==0:return b''
        value=self.budget.get(self.url,self.pos,self.pos+n-1);self.pos+=n;return value


def archives(budget,fetch=False):
    inventory={};saved={}
    for name,url in URLS.items():
        remote=RemoteZip(budget,url)
        with zipfile.ZipFile(remote) as archive:
            inventory[name]=dict(url=url,archive_bytes=remote.size,headers=remote.headers,
                                  members=[dict(name=i.filename,compressed_bytes=i.compress_size,
                                                bytes=i.file_size,crc32=i.CRC) for i in archive.infolist()])
            if not fetch:continue
            if name=='pcqm4mv2':
                required=['pcqm4m-v2/raw/data.csv.gz','pcqm4m-v2/split_dict.pt']
            else:
                required=['pcba/mapping/mol.csv.gz',*[f'pcba/split/scaffold/{s}.csv.gz' for s in ['train','valid','test']]]
            saved[name]=[]
            for member in required:
                info=archive.getinfo(member)
                if info.file_size>OBJECT_LIMIT:raise RuntimeError('metadata object exceeds per-object cap')
                out=HERE/'raw'/member
                if out.exists():
                    data=out.read_bytes()
                    if len(data)!=info.file_size:raise RuntimeError('existing metadata size mismatch')
                else:
                    data=archive.read(member) # ZipFile verifies the member CRC.
                    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
                saved[name].append(dict(member=member,path=str(out.relative_to(HERE)),bytes=len(data),
                                        sha256=sha(data),zip_crc32=info.CRC))
    atomic(HERE/'zip_metadata_catalog.json',inventory)
    if saved:atomic(HERE/'downloaded_archive_metadata.json',saved)
    return inventory


def moses(budget):
    # Resolve immutable upstream source before retrieving Git-LFS metadata files.
    r=budget.session.get('https://api.github.com/repos/molecularsets/moses/commits/master',timeout=30)
    r.raise_for_status();commit=r.json()['sha'];records=[]
    for split in ['train','test','test_scaffolds']:
        relative=f'moses/dataset/data/{split}.csv.gz'
        pointer_url=f'https://raw.githubusercontent.com/molecularsets/moses/{commit}/{relative}'
        response=budget.session.get(pointer_url,timeout=30);response.raise_for_status()
        fields={line.split(' ',1)[0]:line.split(' ',1)[1] for line in response.text.splitlines() if ' ' in line}
        expected=int(fields['size']);digest=fields['oid'].split(':',1)[1]
        url=f'https://media.githubusercontent.com/media/molecularsets/moses/{commit}/{relative}'
        out=HERE/'raw/moses'/f'{split}.csv.gz'
        data=out.read_bytes() if out.exists() else budget.get(url,expected=expected)
        if len(data)!=expected or sha(data)!=digest:raise RuntimeError('MOSES LFS checksum mismatch')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        records.append(dict(split=split,url=url,pointer_url=pointer_url,path=str(out.relative_to(HERE)),
                            bytes=len(data),sha256=digest))
    atomic(HERE/'moses_metadata.json',dict(upstream_commit=commit,files=records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true');args=p.parse_args()
    budget=Budget();catalog=archives(budget,args.fetch)
    for name,v in catalog.items():
        print(name,'archive',v['archive_bytes'],'bytes')
        for i in v['members']:
            if 'split' in i['name'] or 'mapping' in i['name'] or i['name'].endswith('data.csv.gz'):
                print(i['name'],i['bytes'],'bytes')
    if args.fetch:moses(budget)
    print('Metadata bytes received:',budget.data['received_bytes'])
