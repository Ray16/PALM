"""Returned-cut fidelity from saved landmarks; no full factor/kernel is rebuilt."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'package'))
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
from prepare import atomic,sha
from metrics import PackedMetrics,pair_sample


def transform(packed,info):
    indices=np.array(info['landmark_indices'],dtype=np.int64)
    L=torch.from_numpy(np.unpackbits(np.asarray(packed[indices]),axis=1,bitorder='little')).double()
    W=pairwise_similarity(L,L,'tanimoto')
    values,vectors=torch.linalg.eigh((W+W.T)*.5)
    keep=values>info['eigenvalue_threshold']
    if int(keep.sum())!=info['effective_rank']:raise RuntimeError('factor reconstruction rank changed')
    return L,vectors[:,keep]/torch.sqrt(values[keep])[None,:]


def pair_approximation(packed,info,a,b,block=4096):
    L,R=transform(packed,info);result=np.empty(len(a),dtype=np.float64)
    for s in range(0,len(a),block):
        factors=[]
        for ids in [a[s:s+block],b[s:s+block]]:
            X=torch.from_numpy(np.unpackbits(np.asarray(packed[ids]),axis=1,bitorder='little')).double()
            # Match the pinned producer's double construction and float32 storage.
            factors.append((pairwise_similarity(X,L,'tanimoto')@R).float().double())
        result[s:s+len(factors[0])]=(factors[0]*factors[1]).sum(1).numpy()
    return result


def absolute_interval(values):
    lo,hi=np.quantile(values,[.025,.975])
    return [0. if lo<=0<=hi else float(min(abs(lo),abs(hi))),float(max(abs(lo),abs(hi)))]


def main(args):
    torch.cuda.is_available=lambda:False;torch.set_num_threads(2)
    root=HERE/'run';rows=[]
    for method in ['exact_assignment','corridor_plain']:
        p=root/'rows'/f'{args.dataset}__{method}__s{args.seed}.json'
        if not p.exists():raise RuntimeError('wait for both returned native-cut assignments')
        rows.append((p,json.loads(p.read_text())))
    info=rows[0][1]['diagnostics']['factorization']
    if rows[1][1]['diagnostics']['factorization']['landmark_indices']!=info['landmark_indices']:
        raise RuntimeError('same-seed factor sharing assumption failed')
    packed=np.load(HERE/'cache'/args.dataset/'packed.npy',mmap_mode='r')
    a,b=pair_sample(len(packed));exact=PackedMetrics(packed,threads=2).pairs(a,b)
    approximate=pair_approximation(packed,info,a,b)
    choices=np.random.default_rng(991).integers(0,64,(1000,64));results=[]
    for path,row in rows:
        labels=np.load(root/'assignments'/row['job']['name']/'labels.npz')['labels']
        crossing=labels[a]!=labels[b];residual=(approximate-exact)*crossing
        mass=exact.reshape(64,-1).mean(1);cut=(exact*crossing).reshape(64,-1).mean(1)
        error=residual.reshape(64,-1).mean(1)
        signed_mass=error[choices].mean(1)/mass[choices].mean(1)
        signed_cut=error[choices].mean(1)/cut[choices].mean(1)
        results.append(dict(case=row['job']['name'],mass_relative_error=float(abs(error.mean()/mass.mean())),
                            cut_relative_error=float(abs(error.mean()/cut.mean())),
                            mass_error_ci95=absolute_interval(signed_mass),cut_error_ci95=absolute_interval(signed_cut),
                            row_sha256=sha(path)))
    result=dict(dataset=args.dataset,seed=args.seed,rank=info['effective_rank'],pair_count=len(a),
                results=results,source_sha256=sha(__file__),manifest_sha256=sha(root/'manifest.json'),
                scope='Post-hoc returned-cut diagnostic at fixed rank; saved-landmark factor reconstructed with the pinned double-construction/float32-storage formula; approximate Monte Carlo block-bootstrap intervals, not optimizer optimality or uniform cut guarantees')
    atomic(HERE/'verification'/f'factor_fidelity_{args.dataset}__s{args.seed}.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,default=0)
    main(p.parse_args())
