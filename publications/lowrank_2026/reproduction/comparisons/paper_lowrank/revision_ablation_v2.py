"""Full release-code component ablation using the primary benchmark scorer.

All 18 settings, six full datasets and three split seeds. Two CPU threads,
CUDA hidden, no model fits. Timings exclude loading and scoring. Parameter
choices are fixed by the pre-existing development grid, not selected by outcome.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_benchmark import score,capacity_corridor
from revision_external_control import source_files
from splitters import SplitSpec,split

DEST=study.OUT/'revision_20261021/ablation_v2'

def sources():
    r=source_files()
    for p in [Path(__file__),Path(__file__).with_name('revision_benchmark.py')]:
        r[str(p.relative_to(study.ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return r

def run(dataset):
    if torch.cuda.is_available():raise RuntimeError('CPU ablation requires hidden CUDA')
    if torch.get_num_threads()!=2:raise RuntimeError('Ablation protocol uses two CPU threads')
    dest=DEST/dataset;dest.mkdir(parents=True,exist_ok=True)
    b,ids,X,y,metric=study.prepare(dataset)
    input_sha=hashlib.sha256(X.tobytes()).hexdigest();identity=sources()
    study.atomic_json(dest/'inputs.json',dict(dataset=dataset,n=len(ids),ids=ids,feature_sha256=input_sha,source_files=identity))
    rows=[]
    for setting,params in study.ABLATIONS.items():
        for seed in study.SEEDS:
            final=dest/f'{setting}__s{seed}.json';array=final.with_suffix('.npy')
            if final.exists():
                row=json.loads(final.read_text())
                if row['source_files']!=identity or row['feature_sha256']!=input_sha:raise RuntimeError('Existing ablation source/input drift')
                if hashlib.sha256(array.read_bytes()).hexdigest()!=row['assignment_sha256']:raise RuntimeError('Existing assignment drift')
                rows.append(row);continue
            kwargs=dict(params);kwargs.setdefault('allowed_size_imbalance',0.)
            kwargs['score_exact_leakage']=False
            start=time.perf_counter()
            res=split('lowrank',b.feature_data,SplitSpec([8,1,1],study.NAMES.tolist(),seed=seed,epsilon=0.),**kwargs)
            seconds=time.perf_counter()-start
            labels=np.asarray([res.assignment[i] for i in ids]);np.save(array,labels)
            counts=np.asarray([np.sum(labels==s) for s in study.NAMES])
            caps,floors=capacity_corridor(len(ids),[8,1,1],kwargs['allowed_size_imbalance'])
            if len(labels)!=len(ids) or set(labels)!=set(study.NAMES) or not np.all((counts>=floors)&(counts<=caps)):raise RuntimeError('Ablation retention/capacity postcondition failed')
            row=dict(dataset=dataset,setting=setting,seed=seed,status='ok',params=kwargs,
                feature_sha256=input_sha,source_files=identity,split_s=seconds,diagnostics=res.diagnostics,
                assignment_sha256=hashlib.sha256(array.read_bytes()).hexdigest(),**score(X,labels,metric,seed))
            study.atomic_json(final,study.clean(row));rows.append(study.clean(row))
            print(dataset,setting,seed,'ok',round(seconds,3),flush=True)
    if sources()!=identity:raise RuntimeError('Source changed during ablation')
    study.atomic_json(dest/'result.json',study.clean(dict(dataset=dataset,n=len(ids),ids=ids,feature_sha256=input_sha,source_files=identity,rows=rows,
        protocol='18 prespecified settings, three seeds, all retained rows; primary benchmark scorer; two CPU threads; no prediction fitting')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=study.ABLATION_DATASETS);a=p.parse_args()
    for dataset in a.datasets:run(dataset)
