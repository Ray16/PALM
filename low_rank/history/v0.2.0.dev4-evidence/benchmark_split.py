"""Paired full public split calls; only swap batching differs. Loading/scoring excluded."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]  # the low_rank/ package root
sys.path.insert(0, str(ROOT))
import torch
from lowrank import split, SplitSpec
import lowrank.optimize as optimizer
CACHE=Path('/nfs/lambda_stor_01/homes/rzhu/data_splitting_lowrank_reviewed/publications/lowrank_2026/correctness_20261025/verified_run/cache')
torch.set_num_threads(2)
original=optimizer.swap_polish

def scalar(*args, **kwargs):
    kwargs['sim_block']=None
    return original(*args, **kwargs)

rows=[]
for dataset in ['freesolv','esol','bace']:
    path=CACHE/f'moleculenet_{dataset}.npz'
    X=np.load(path)['X']
    features={str(i):x for i,x in enumerate(X)}
    for seed in range(3):
        for batched in ([False,True] if seed%2==0 else [True,False]):
            optimizer.swap_polish=original if batched else scalar
            spec=SplitSpec([8,1,1],['train','val','test'],seed=seed)
            start=time.perf_counter()
            result=split('lowrank',features,spec,rank=256,leakage_diagnostic='none')
            seconds=time.perf_counter()-start
            labels=np.array([result.assignment[str(i)] for i in range(len(X))])
            row=dict(dataset=dataset,seed=seed,batched=batched,seconds=seconds,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                counts={str(x):int(sum(labels==x)) for x in np.unique(labels)},
                assignments=labels.tolist())
            rows.append(row)
            print(dataset,seed,batched,seconds,flush=True)
            (Path(__file__).parent/'split_results.json').write_text(json.dumps(dict(scope='Public split call; exact counts; rank256; four restarts; two refined; no scoring; CPU two threads',rows=rows),indent=2)+'\n')
