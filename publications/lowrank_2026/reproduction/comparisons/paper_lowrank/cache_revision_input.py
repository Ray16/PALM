"""Stage identical features once for row-parallel jobs; never change identities."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import mlst_study as study
import numpy as np

ROOT=study.OUT/'revision_20261021/input_cache'


def build(dataset):
    root=ROOT/dataset;root.mkdir(parents=True,exist_ok=True)
    final=root/'metadata.json'
    if final.exists():return
    b,ids,X,y,metric=study.prepare(dataset)
    expected=json.loads((study.OUT/'revision_20261021/benchmark'/dataset/'inputs.json').read_text())
    digest=hashlib.sha256(X.tobytes()).hexdigest()
    if ids!=expected['ids'] or digest!=expected['feature_sha256']:
        raise RuntimeError('cache differs from the established benchmark inputs')
    np.save(root/'features.npy',X);np.save(root/'targets.npy',y)
    record=dict(dataset=dataset,ids=ids,metric=metric,feature_sha256=digest,
                targets_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                smiles=[b.smiles.get(i) for i in ids] if b.smiles else None,
                bundle={k:getattr(b,k) for k in ('name','category','kind','available','reason','task_type','target_name','identifier_kind','entity_type','meta')},
                source_benchmark_inputs_sha256=hashlib.sha256((study.OUT/'revision_20261021/benchmark'/dataset/'inputs.json').read_bytes()).hexdigest(),
                producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    study.atomic_json(final,study.clean(record));print(dataset,'cache verified',X.shape,digest,flush=True)


def prepare(dataset,limit=None):
    if limit is not None:raise ValueError('row-parallel cache is full-dataset only')
    from datasets.sources import DatasetBundle
    root=ROOT/dataset;r=json.loads((root/'metadata.json').read_text())
    X=np.load(root/'features.npy',mmap_mode='c');y=np.load(root/'targets.npy')
    if hashlib.sha256(X.tobytes()).hexdigest()!=r['feature_sha256'] or hashlib.sha256(y.tobytes()).hexdigest()!=r['targets_sha256']:
        raise RuntimeError('cached numerical inputs changed')
    ids=r['ids'];b=DatasetBundle(**r['bundle'])
    b.feature_data={i:X[j] for j,i in enumerate(ids)}
    b.targets={i:float(y[j]) for j,i in enumerate(ids)}
    b.smiles=dict(zip(ids,r['smiles'])) if r['smiles'] is not None else None
    b.identifiers=b.smiles
    return b,ids,X,y,r['metric']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');a=p.parse_args();build(a.dataset)
