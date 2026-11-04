"""Exploratory aggregate landmarks targeting incumbent fold means.
Dense oracle implementation: not a scalable timing result or production feature.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/lowrank'))
import torch
from lowrank import nystrom_features
from lowrank.optimize import optimize_restarts
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

torch.set_num_threads(2)
CACHE=Path('/nfs/lambda_stor_01/homes/rzhu/data_splitting_lowrank_reviewed/publications/lowrank_2026/correctness_20261025/verified_run/cache')
rows=[]
for dataset in ['freesolv','esol','bace']:
 X=np.load(CACHE/f'moleculenet_{dataset}.npz')['X']
 K=pairwise_similarity(torch.tensor(X,dtype=torch.float64),torch.tensor(X,dtype=torch.float64),'tanimoto').numpy()**3
 mass=(K.sum()-np.trace(K))/2
 for seed in range(3):
  B=nystrom_features(X,rank=256,metric='tanimoto',kernel_power=3,seed=seed)[0].astype(float)
  initial,_,_=optimize_restarts(B,[.8,.1,.1],seed=seed,n_restarts=4,refine_restarts=2)
  Q=np.zeros((len(X),3));rng=np.random.default_rng(seed+7949)
  for fold in range(3):
   candidates=np.flatnonzero(initial==fold)
   anchors=rng.choice(candidates,min(256,len(candidates)),replace=False)
   Q[anchors,fold]=1/len(anchors)
  H=K@Q-B@(B.T@Q)
  G=Q.T@H
  values,vectors=np.linalg.eigh((G+G.T)/2)
  keep=values>max(1e-12,values[-1]*1e-8)
  augmented=np.column_stack([B,H@(vectors[:,keep]/np.sqrt(values[keep]))])
  final,_,_=optimize_restarts(augmented,[.8,.1,.1],seed=seed,n_restarts=4,refine_restarts=2)
  errors=[]
  for lab in [initial,final]+[rng.permutation(initial) for _ in range(3)]:
   cross=lab[:,None]!=lab;exact=float(K[cross].sum()/2)
   errors.append([abs(float((F@F.T)[cross].sum()/2)-exact)/mass for F in [B,augmented]])
  row=dict(dataset=dataset,seed=seed,added_rank=int(sum(keep)),anchor_count=int(sum(np.any(Q!=0,axis=1))),errors=errors,
   exact_cut_before=float(K[initial[:,None]!=initial].sum()/2/mass),exact_cut_after=float(K[final[:,None]!=final].sum()/2/mass),
   moved_fraction=float(np.mean(initial!=final)))
  rows.append(row);print(json.dumps(row),flush=True)
  (Path(__file__).parent/'powered_fold_mean_pilot.json').write_text(json.dumps(dict(scope='Dense oracle development pilot; each fold mean samples at most 256 members; original/final plus random partitions',rows=rows),indent=2)+'\n')
