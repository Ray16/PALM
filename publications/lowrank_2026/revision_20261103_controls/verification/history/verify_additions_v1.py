"""Independent replay of the nested scaling and preempted-clock additions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

import suite
import validate
HERE=Path(__file__).resolve().parent


def compare(saved,reference):
    worst=0.
    if saved['counts']!=reference['counts']:raise AssertionError('counts')
    for key in ['true_cut','true_powered_cut','global_relative','mean_nearest','nearest_gt06']:
        a,b=saved[key],reference[key]
        if a is None or b is None:
            if a is not None or b is not None:raise AssertionError('undefined scores')
        else:
            worst=max(worst,abs(a-b))
            if abs(a-b)>1e-7*max(1,abs(a)):raise AssertionError((key,a,b))
    return worst


def main(args):
    checks=[];worst=0.;identities={}
    if args.study=='scaling':
        import dense_scaling
        root=HERE/'scaling_run';m=json.loads((root/'manifest.json').read_text())
        if m['sources']!=dense_scaling.identities():raise RuntimeError('source drift')
        packed=np.load(dense_scaling.INPUT,mmap_mode='r')
        for n in [4000,16000,32000]:
            selected=[j for j in m['jobs'] if j['n']==n]
            first=selected[0]
            indices=np.load(root/'assignments'/first['name']/'indices.npy')
            X=np.unpackbits(np.asarray(packed[indices]),axis=1).astype(np.float32)
            S=validate.kernel(X)
            for j in selected:
                p=root/'rows'/(j['name']+'.json');r=json.loads(p.read_text());identities[p.name]=suite.sha(p)
                other=np.load(root/'assignments'/j['name']/'indices.npy')
                if not np.array_equal(other,indices):raise AssertionError('nested sample mismatch')
                labels=np.load(root/'assignments'/j['name']/'labels.npy')
                worst=max(worst,compare(r['result']['metrics'],validate.reference(S,labels)))
                checks.append(j['name']);print(j['name'],flush=True)
            del S
    else:
        import clock_budget
        root=HERE/'strict_clock_run';m=json.loads((root/'manifest.json').read_text())
        if m['sources']!=clock_budget.sources():raise RuntimeError('source drift')
        for ds in suite.DATASETS:
            X=np.load(suite.CACHE/(ds+'.npz'))['X'];S=validate.kernel(X)
            for j in [j for j in m['jobs'] if j['dataset']==ds]:
                p=root/'rows'/(j['name']+'.json');r=json.loads(p.read_text());identities[p.name]=suite.sha(p)
                if r['checkpoint']['elapsed']>j['budget_seconds']:raise AssertionError('late checkpoint')
                labels=np.load(root/'assignments'/j['name']/'labels.npy')
                worst=max(worst,compare(r['metrics'],validate.reference(S,labels)))
                checks.append(j['name']);print(j['name'],flush=True)
            del S
    expected=36 if args.study=='scaling' else 72
    if len(checks)!=expected:raise RuntimeError('incomplete additions')
    suite.atomic(HERE/'verification'/('replay_'+args.study+'_addition.json'),dict(
        passed=True,assignments=len(checks),max_absolute_discrepancy=worst,
        observations=identities,manifest_sha256=suite.sha(root/'manifest.json'),
        validator_sha256=suite.sha(__file__),reference='Independent RDKit all-pairs Tanimoto and original-kernel cut/nearest evaluation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['scaling','clock'],required=True);main(p.parse_args())
