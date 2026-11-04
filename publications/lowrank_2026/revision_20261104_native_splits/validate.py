"""Independent RDKit pair checks and NumPy full-training nearest-query replay."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket

import numpy as np
from rdkit import DataStructs

from prepare import atomic,sha
from metrics import PackedMetrics,pair_sample
import report

HERE=Path(__file__).resolve().parent
POPCOUNT=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)


def independent_counts(packed):
    result=np.empty(len(packed),dtype=np.int32)
    for s in range(0,len(packed),65536):result[s:s+65536]=POPCOUNT[packed[s:s+65536]].sum(1,dtype=np.int32)
    return result


def numpy_nearest(packed,counts,labels,query):
    best=0.;q=packed[query];qcount=counts[query]
    for s in range(0,len(packed),32768):
        selected=labels[s:s+32768]==0
        if not selected.any():continue
        block=packed[s:s+32768][selected]
        intersection=POPCOUNT[np.bitwise_and(block,q)].sum(1,dtype=np.int32)
        union=qcount+counts[s:s+32768][selected]-intersection
        value=np.divide(intersection,union,out=np.zeros(len(union),dtype=np.float64),where=union>0)
        best=max(best,float(value.max()))
    return best


def main(args):
    import psutil
    slots=os.sched_getaffinity(0)
    if len(slots)>2:raise RuntimeError('bind this independent replay to two CPU slots')
    if os.getloadavg()[0]+2>os.cpu_count() or psutil.virtual_memory().available<8*2**30:
        raise RuntimeError('independent replay resource capacity')
    m,rows,failures,missing=report.verified(True)
    selected=[r for r in rows if r['job']['dataset']==args.dataset and r['job']['seed']==0
              and r['job']['method'] in ['native','exact_assignment','corridor_plain']]
    if len(selected)!=3:raise RuntimeError('wait for native/exact/corridor seed-zero cases before replay')
    cache=HERE/'cache'/args.dataset;prepared=json.loads((cache/'manifest.json').read_text())
    if sha(cache/'packed.npy')!=prepared['artifacts']['packed.npy']:raise RuntimeError('fingerprint drift')
    packed=np.load(cache/'packed.npy',mmap_mode='r');counts=independent_counts(packed)
    engine=PackedMetrics(packed,threads=2);a,b=pair_sample(len(packed));sim=engine.pairs(a,b)
    chosen=np.random.default_rng(7001).choice(len(a),1024,replace=False);pair_error=0.
    for index in chosen:
        fa=DataStructs.CreateFromBinaryText(packed[a[index]].tobytes())
        fb=DataStructs.CreateFromBinaryText(packed[b[index]].tobytes())
        reference=DataStructs.TanimotoSimilarity(fa,fb)
        pair_error=max(pair_error,abs(reference-sim[index]))
    if pair_error>1e-12:raise AssertionError('RDKit pair discrepancy')
    comparisons=[];query_error=0.
    for row in selected:
        job=row['job'];base=HERE/'run/assignments'/job['name']
        labels=np.load(base/'labels.npz')['labels'];saved=np.load(base/'nearest_audit.npz')
        tasks=[];observed=[]
        for c,name in enumerate(row['fold_names'][1:],1):
            queries=saved[f'fold{c}_queries'][:8];nearest=saved[f'fold{c}_nearest'][:8]
            tasks.extend(queries.tolist());observed.extend(nearest.tolist())
        def check_query(query):return numpy_nearest(packed,counts,labels,query)
        with ThreadPoolExecutor(max_workers=2) as pool:reference=np.array(list(pool.map(check_query,tasks)))
        error=float(np.max(np.abs(reference-np.array(observed))));query_error=max(query_error,error)
        if error>1e-12:raise AssertionError('independent nearest-training discrepancy')
        # Reproduce all pair-block sufficient statistics for these assignments.
        blocks=np.load(base/'pair_blocks.npz')['means']
        calculated=np.stack([sim.reshape(64,-1).mean(1),(sim*(labels[a]!=labels[b])).reshape(64,-1).mean(1)])
        if not np.allclose(calculated,blocks,rtol=0,atol=1e-14):raise AssertionError('pair-block reconstruction')
        comparisons.append(dict(case=job['name'],query_count=len(tasks),queries=tasks,
                                max_absolute_error=error,reference_values=reference.tolist(),
                                row_sha256=sha(HERE/'run/rows'/(job['name']+'.json'))))
        print('independent full-training replay',job['name'],len(tasks),'queries',flush=True)
    result=dict(passed=True,dataset=args.dataset,independent_pair_checks=1024,
                max_pair_discrepancy=pair_error,max_query_discrepancy=query_error,
                full_training_query_checks=sum(v['query_count'] for v in comparisons),
                comparisons=comparisons,validator_sha256=sha(__file__),
                manifest_sha256=sha(HERE/'run/manifest.json'),host=socket.gethostname(),
                cpu_affinity=sorted(slots),scope='RDKit pairs and independent bounded-memory NumPy maxima against every training entity for eight saved queries per held-out role in native/exact/corridor seed-zero cases; all saved-case aggregations checked separately')
    atomic(HERE/'verification'/('independent_'+args.dataset+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);main(p.parse_args())
