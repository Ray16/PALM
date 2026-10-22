"""Release-code controls at the exact counts of each saved DataSAIL split.

The common tolerance grid retains each method's actual counts. This separate
comparison removes the remaining count difference, using the same scoring
protocol and entity identities. No timing is attributed to reused DataSAIL.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_benchmark import assign,score
from revision_external_control import source_files
from splitters import split,SplitSpec

DEST=study.OUT/'revision_20261021/datasail_matched_v2'
DATASETS=['moleculenet_'+x for x in ('freesolv','esol','bace','bbbp','clintox','sider','lipophilicity')]


def run(dataset):
    root=DEST/dataset;root.mkdir(parents=True,exist_ok=True)
    sources=source_files()
    for name in ('revision_datasail_matched_v2.py','revision_benchmark.py','revision_baselines.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(study.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    final=root/'result.json'
    if final.exists():
        if json.loads(final.read_text())['source_files']!=sources:raise RuntimeError('source drift')
        return
    b,ids,X,y,metric=study.prepare(dataset)
    study.atomic_json(root/'ids.json',ids)
    rows=[]
    for eps in (.01,.05):
        native,diag,_=assign('datasail',b,ids,X,metric,eps,0)
        counts=[int(np.sum(native==name)) for name in study.NAMES]
        for method in ('datasail','plain','combined'):
            for seed in ([0] if method=='datasail' else (0,1,2)):
                stem=f'{method}__eps{eps}__s{seed}';path=root/(stem+'.json')
                if path.exists():
                    row=json.loads(path.read_text())
                    if row['source_files']!=sources:raise RuntimeError('partial source drift')
                else:
                    if method=='datasail':labels,diagnostics=native,diag
                    else:
                        kwargs=dict(allowed_size_imbalance=0.,score_exact_leakage=False)
                        if method=='combined':kwargs.update(kernel_power=3,must_link=.4)
                        result=split('lowrank',b.feature_data,SplitSpec(counts,study.NAMES.tolist(),seed=seed,epsilon=0.),**kwargs)
                        labels=np.array([result.assignment[i] for i in ids]);diagnostics=result.diagnostics
                    if [int(np.sum(labels==name)) for name in study.NAMES]!=counts:raise AssertionError('count mismatch')
                    np.save(root/(stem+'.npy'),labels)
                    row=dict(method=method,epsilon=eps,seed=seed,diagnostics=diagnostics,source_files=sources,**score(X,labels,metric,seed))
                    study.atomic_json(path,study.clean(row))
                rows.append(row)
                print(dataset,method,eps,seed,flush=True)
    study.atomic_json(final,study.clean(dict(dataset=dataset,status='ok',n=len(ids),rows=rows,source_files=sources,
        feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),scope='exact DataSAIL fold counts, all original entities, common scoring; no reused time')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=DATASETS);a=p.parse_args()
    if torch.cuda.is_available():raise RuntimeError('CPU-only matched-count protocol')
    for dataset in a.datasets:run(dataset)
