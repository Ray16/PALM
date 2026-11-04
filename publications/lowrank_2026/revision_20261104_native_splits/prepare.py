"""Native partition audit and packed ECFP preparation; no coordinates or targets."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATASETS=['pcqm4mv2','moses','molpcba']


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read_native(dataset):
    start=time.perf_counter();files=[]
    if dataset=='pcqm4mv2':
        import torch
        raw=HERE/'raw/pcqm4m-v2/raw/data.csv.gz';split=HERE/'raw/pcqm4m-v2/split_dict.pt'
        # Explicit NumPy array constructors only; never unrestricted pickle load.
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('int64'))]):
            native=torch.load(split,map_location='cpu',weights_only=True)
        names=['train','valid','test-dev','test-challenge']
        frame=pd.read_csv(raw,usecols=['idx','smiles'],dtype={'smiles':str})
        if not np.array_equal(frame['idx'].values,np.arange(len(frame))):raise RuntimeError('PCQM row IDs changed')
        smiles=frame['smiles'].values;files=[raw,split]
    elif dataset=='molpcba':
        raw=HERE/'raw/pcba/mapping/mol.csv.gz'
        frame=pd.read_csv(raw,usecols=['smiles','mol_id'],dtype=str)
        if frame['mol_id'].duplicated().any():raise RuntimeError('duplicate source compound IDs')
        smiles=frame['smiles'].values;names=['train','valid','test'];native={};files=[raw]
        for name in names:
            path=HERE/'raw/pcba/split/scaffold'/(name+'.csv.gz')
            native[name]=pd.read_csv(path,header=None).iloc[:,0].to_numpy(dtype=np.int64);files.append(path)
    else:
        names=['train','test','test_scaffolds'];values=[];native={};offset=0
        for name in names:
            path=HERE/'raw/moses'/(name+'.csv.gz');files.append(path)
            v=pd.read_csv(path,usecols=['SMILES'],dtype=str)['SMILES'].values
            native[name]=np.arange(offset,offset+len(v));offset+=len(v);values.extend(v)
        smiles=np.array(values,dtype=object)
    labels=np.full(len(smiles),255,dtype=np.uint8)
    for c,name in enumerate(names):
        ids=np.asarray(native[name],dtype=np.int64)
        if len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=len(smiles)).any():raise RuntimeError('invalid native IDs')
        if (labels[ids]!=255).any():raise RuntimeError('overlapping native partitions')
        labels[ids]=c
    if (labels==255).any():raise RuntimeError('native partition does not cover metadata')
    return smiles,labels,names,dict(metadata_loading_seconds=time.perf_counter()-start,
            source_files={str(p.relative_to(HERE)):sha(p) for p in files},
            original_n=len(smiles),original_counts=np.bincount(labels,minlength=len(names)).tolist(),
            native_disjoint=True,native_complete=True,response_labels_loaded=False,
            native_construction_seconds=None,
            native_timing_note='Published fixed assignments; construction time is not supplied. Loading is not split generation.')


def init_worker(cpus,owner_dir):
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    identity=mp.current_process()._identity[0]-1;cpu=cpus[identity%len(cpus)]
    os.sched_setaffinity(0,{cpu})
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    global GENERATOR
    GENERATOR=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=False)
    atomic(Path(owner_dir)/f'worker_{os.getpid()}.json',dict(pid=os.getpid(),cpu=cpu,
           host=socket.gethostname(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))


def chunk_features(task):
    from rdkit import Chem,DataStructs
    start,smiles=task;packed=np.zeros((len(smiles),128),dtype=np.uint8)
    canonical=np.zeros(len(smiles),dtype='V16');valid=np.zeros(len(smiles),dtype=bool);failures=[]
    for i,smile in enumerate(smiles):
        try:
            mol=Chem.MolFromSmiles(str(smile))
            if mol is None or mol.GetNumAtoms()==0:raise ValueError('unparseable or empty SMILES')
            fp=GENERATOR.GetFingerprint(mol)
            packed[i]=np.frombuffer(DataStructs.BitVectToBinaryText(fp),dtype=np.uint8)
            if not packed[i].any():raise ValueError('zero fingerprint')
            key=Chem.MolToSmiles(mol,isomericSmiles=True)
            canonical[i]=np.void(hashlib.blake2b(key.encode(),digest_size=16).digest());valid[i]=True
        except Exception as e:failures.append(dict(original_row=start+i,reason=str(e)[:200]))
    return start,packed,canonical,valid,failures


def duplicate_audit(keys,labels):
    order=np.argsort(keys);ordered=keys[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    memberships=np.bitwise_or.reduceat((1<<labels[order]).astype(np.uint8),starts)
    crossing=(memberships&(memberships-1))!=0
    return dict(canonical_identity_key='BLAKE2b-128 of RDKit isomeric canonical SMILES; no chemistry deduplication applied',
                unique_canonical_keys=len(starts),repeated_canonical_records=len(keys)-len(starts),
                canonical_groups_crossing_native_folds=int(crossing.sum()))


def prepare(dataset,args):
    from rdkit import rdBase
    cache=HERE/'cache'/dataset;cache.mkdir(parents=True,exist_ok=True)
    recipe=dict(radius=2,bits=1024,include_chirality=False,packed_bit_order='little',
                rdkit_version=rdBase.rdkitVersion,producer_sha256=sha(__file__))
    completed=cache/'manifest.json'
    if completed.exists():
        old=json.loads(completed.read_text())
        if old['recipe']!=recipe or any(sha(cache/name)!=digest for name,digest in old['artifacts'].items()):
            raise RuntimeError('preparation drift; use a separately versioned cache')
        print('verified existing',dataset,old['retained_n'],flush=True);return
    owner=cache/'claim'
    try:owner.mkdir()
    except FileExistsError:raise RuntimeError('preparation already claimed; verify owner before recovery')
    atomic(owner/'parent.json',dict(pid=os.getpid(),host=socket.gethostname(),
           start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],cpus=args.cpus,parent_cpu=args.parent_cpu))
    smiles,native,names,metadata=read_native(dataset);n=len(smiles)
    print('loaded',dataset,n,metadata['original_counts'],flush=True)
    packed=np.lib.format.open_memmap(cache/'packed_all.npy',mode='w+',dtype=np.uint8,shape=(n,128))
    keys=np.lib.format.open_memmap(cache/'canonical_all.npy',mode='w+',dtype='V16',shape=(n,))
    valid=np.zeros(n,dtype=bool);failures=[];started=time.perf_counter();processed=0
    tasks=((s,smiles[s:s+2000].tolist()) for s in range(0,n,2000))
    context=mp.get_context('spawn')
    with context.Pool(len(args.cpus),initializer=init_worker,initargs=(args.cpus,str(owner))) as pool:
        for s,fp,canonical,mask,errors in pool.imap_unordered(chunk_features,tasks,chunksize=1):
            packed[s:s+len(fp)]=fp;keys[s:s+len(fp)]=canonical;valid[s:s+len(fp)]=mask
            failures.extend(errors);processed+=len(fp)
            if processed%100000<2000:print('fingerprinted',dataset,processed,'/',n,flush=True)
    seconds=time.perf_counter()-started;packed.flush();keys.flush()
    rows=np.flatnonzero(valid);labels=native[rows]
    np.save(cache/'retained_original_rows.npy',rows);np.save(cache/'native.npy',labels)
    if valid.all():
        del packed,keys
        (cache/'packed_all.npy').replace(cache/'packed.npy');(cache/'canonical_all.npy').replace(cache/'canonical.npy')
    else:
        out=np.lib.format.open_memmap(cache/'packed.npy',mode='w+',dtype=np.uint8,shape=(len(rows),128))
        outkeys=np.lib.format.open_memmap(cache/'canonical.npy',mode='w+',dtype='V16',shape=(len(rows),))
        for s in range(0,len(rows),10000):out[s:s+10000]=packed[rows[s:s+10000]];outkeys[s:s+10000]=keys[rows[s:s+10000]]
        out.flush();outkeys.flush();del out,outkeys,packed,keys
        (cache/'packed_all.npy').unlink();(cache/'canonical_all.npy').unlink()
    audit=duplicate_audit(np.load(cache/'canonical.npy',mmap_mode='r'),labels)
    atomic(cache/'parse_failures.json',failures)
    artifacts={name:sha(cache/name) for name in ['packed.npy','canonical.npy','native.npy','retained_original_rows.npy','parse_failures.json']}
    record=dict(dataset=dataset,recipe=recipe,metadata=metadata,fold_names=names,retained_n=len(rows),
                retained_counts=np.bincount(labels,minlength=len(names)).tolist(),parse_failures=len(failures),
                fingerprint_seconds=seconds,artifacts=artifacts,identity_audit=audit,
                cpu_worker_slots=args.cpus,parent_cpu=args.parent_cpu,host=socket.gethostname())
    atomic(completed,record);atomic(HERE/'preparation'/f'{dataset}.json',record)
    owner.rename(cache/'retired_claim')
    print('completed preparation',dataset,len(rows),'retained;',len(failures),'parse failures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=DATASETS)
    p.add_argument('--cpus',nargs='+',type=int,default=list(range(8,16)));p.add_argument('--parent-cpu',type=int,default=16)
    args=p.parse_args()
    import psutil
    slots=set(args.cpus+[args.parent_cpu])
    if len(slots)!=len(args.cpus)+1 or not slots<=os.sched_getaffinity(0):raise RuntimeError('CPU allocation')
    if os.getloadavg()[0]+len(slots)>os.cpu_count() or psutil.virtual_memory().available<32*2**30:raise RuntimeError('resource capacity')
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            c=process.info['cmdline'] or [];a=process.cpu_affinity()
            if c and process.pid!=os.getpid() and 'python' in Path(c[0]).name and len(a)<20 and slots&set(a):raise RuntimeError(f'pinned CPU overlap {process.pid}')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    os.sched_setaffinity(0,{args.parent_cpu})
    for dataset in args.datasets:prepare(dataset,args)
