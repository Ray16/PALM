"""Independent RDKit replay of every saved original-kernel result and contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import DataStructs

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import suite


def kernel(X):
    fps=[]
    for row in X:
        fp=DataStructs.ExplicitBitVect(len(row))
        fp.SetBitsFromList(np.flatnonzero(row).tolist());fps.append(fp)
    S=np.empty((len(X),len(X)),dtype=np.float64)
    for i,fp in enumerate(fps):S[i]=DataStructs.BulkTanimotoSimilarity(fp,fps)
    np.fill_diagonal(S,0)
    return S


def reference(S,z,power=1):
    counts=np.bincount(z,minlength=3)
    cut=sum(float(S[z==a][:,z==b].sum()) for a in range(3) for b in range(a+1,3))
    P=S if power==1 else S**power
    pcut=sum(float(P[z==a][:,z==b].sum()) for a in range(3) for b in range(a+1,3))
    mass=float(S.sum())/2
    fraction=(len(z)**2-float(counts@counts))/(len(z)*(len(z)-1))
    nearest=S[z==2][:,z==0].max(1) if counts[0] and counts[2] else None
    return dict(counts=counts.tolist(),true_cut=cut,true_powered_cut=pcut,
                global_relative=cut/mass/fraction if mass and fraction else None,
                mean_nearest=float(nearest.mean()) if nearest is not None else None,
                nearest_gt06=float(np.mean(nearest>.6)) if nearest is not None else None)


def main(args):
    root=HERE/'run';manifest=json.loads((root/'manifest.json').read_text())
    if suite.sources()!=manifest['sources']:raise RuntimeError('producer source drift')
    selected=[j for j in manifest['jobs'] if j['dataset']==args.dataset]
    if len(selected)!=72:raise RuntimeError('dataset grid mismatch')
    X=np.load(suite.CACHE/(args.dataset+'.npz'))['X'];S=kernel(X)
    maxima=dict(true_cut=0.,true_powered_cut=0.,global_relative=0.,mean_nearest=0.,nearest_gt06=0.)
    checks=0;provenance={}
    for j in selected:
        p=root/'rows'/(j['name']+'.json')
        r=json.loads(p.read_text());provenance[p.name]=suite.sha(p)
        res=r['result']
        variants=[dict(variant='labels',metrics=res['metrics'])] if j['study'] in ('iterations','clock') else res['variants']
        gid=None
        if j['study'] in ('ablation','ccpart'):
            gid=np.load(root/'assignments'/j['name']/'groups.npy')
        for v in variants:
            z=np.load(root/'assignments'/j['name']/(v['variant']+'.npy'))
            pwr=j.get('power',3 if v['variant'] in ('power_only','groups_power') else 1)
            ref=reference(S,z,pwr);saved=v['metrics']
            if ref['counts']!=saved['counts']:raise AssertionError('count replay')
            for key in maxima:
                # CCPart has no powered objective; its recorded power is one.
                a,b=ref[key],saved[key]
                if a is None or b is None:
                    if a is not None or b is not None:raise AssertionError('undefined-score mismatch')
                else:
                    error=abs(a-b);maxima[key]=max(maxima[key],error)
                    if error>1e-7*max(1.,abs(a)):raise AssertionError((j['name'],v['variant'],key,a,b))
            if gid is not None and (j['study']=='ccpart' and v['variant'] in ('ccpart','matched_groups')
                                    or j['study']=='ablation' and v['variant'] in ('group_initial','groups_plain','groups_power')):
                for group in np.unique(gid):
                    if len(np.unique(z[gid==group]))!=1:raise AssertionError('group replay')
            if j['study']=='ccpart' and v['variant']=='ccpart':
                # All threshold edges crossing a split must be absent, not just
                # those retained in a truncated neighbour graph.
                for a in range(3):
                    for b in range(a+1,3):
                        block=S[z==a][:,z==b]
                        if block.size and (block>=j['threshold']).any():raise AssertionError('CCPart threshold contract')
            checks+=1
        print(j['name'],flush=True)
    result=dict(dataset=args.dataset,passed=True,cases=len(selected),assignments=checks,
                max_absolute_discrepancy=maxima,observations=provenance,
                reference='Independent RDKit ExplicitBitVect BulkTanimotoSimilarity, all pairs, float64 sums; all test queries against all training entities',
                validation_source_sha256=suite.sha(__file__),manifest_sha256=suite.sha(root/'manifest.json'))
    suite.atomic(HERE/'verification'/('replay_'+args.dataset+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args())
