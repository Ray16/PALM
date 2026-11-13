"""Exploratory powered-cut fidelity study. Dense K is an oracle, not a scalable backend.

An aggregate RKHS landmark adds a residual direction h=Kq-B(B' q),
normalized by sqrt(q' K q-||B' q||^2). For an exact Nystrom projection,
this is projection onto the residual of the mean feature of a sampled cohort.
No labels or evaluation outcomes enter q. Evaluate identical partitions before
and after correction, including partitions optimized under either factor.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]  # the low_rank/ package root
sys.path.insert(0, str(ROOT))
import torch
from lowrank import nystrom_features
from lowrank.optimize import optimize_restarts
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

torch.set_num_threads(2)
CACHE=Path('/nfs/lambda_stor_01/homes/rzhu/data_splitting_lowrank_reviewed/publications/lowrank_2026/correctness_20261025/verified_run/cache')
rows=[]
for dataset in ['freesolv','esol','bace']:
    source=CACHE/f'moleculenet_{dataset}.npz'
    X=np.load(source)['X']
    K=pairwise_similarity(torch.tensor(X,dtype=torch.float64),torch.tensor(X,dtype=torch.float64),'tanimoto').numpy()**3
    mass=(K.sum()-np.trace(K))/2
    for seed in range(3):
        B=nystrom_features(X,rank=256,metric='tanimoto',kernel_power=3,seed=seed)[0].astype(float)
        anchor=np.random.default_rng(seed+7949).choice(len(X),min(512,len(X)),replace=False)
        mean=B[anchor].mean(0)
        h=K[:,anchor].mean(1)-B@mean
        variance=float(K[np.ix_(anchor,anchor)].mean()-mean@mean)
        assert variance>1e-12
        augmented=np.column_stack([B,h/np.sqrt(variance)])
        labels=[]
        for F in [B,augmented]:
            lab,_,_=optimize_restarts(F,[.8,.1,.1],seed=seed,n_restarts=4,refine_restarts=2)
            labels.append(lab)
        for j in range(3):
            labels.append(np.random.default_rng(seed*3+j).permutation(labels[0]))
        errors=[]
        for lab in labels:
            cross=lab[:,None]!=lab
            exact=float(K[cross].sum()/2)
            errors.append([abs(float((F@F.T)[cross].sum()/2)-exact)/mass for F in [B,augmented]])
        row=dict(dataset=dataset,seed=seed,n=len(X),rank=B.shape[1],added_rank=1,anchor_count=len(anchor),
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),residual_variance=variance,
            errors=errors,max_error_before=max(e[0] for e in errors),max_error_after=max(e[1] for e in errors),
            exact_cut_before=float(K[labels[0][:,None]!=labels[0]].sum()/2/mass),
            exact_cut_after=float(K[labels[1][:,None]!=labels[1]].sum()/2/mass))
        rows.append(row)
        print(json.dumps(row),flush=True)
        (Path(__file__).parent/'powered_mean_pilot.json').write_text(json.dumps(dict(scope='Exploratory development datasets; dense oracle prototype; no scalable timing claim',rows=rows),indent=2)+'\n')
