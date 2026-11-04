"""Frozen native-split comparison, fresh-process CPU profiles and exact audits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PUB=HERE.parent
sys.path.insert(0,str(PUB/'package'))
from prepare import atomic,sha,DATASETS


def sources():
    paths=[HERE/n for n in ['run.py','metrics.py','packed_metrics.cpp','PROTOCOL.md','prepare.py','fetch_metadata.py']]
    paths+=sorted((PUB/'package/lowrank').rglob('*.py'))
    return {str(p.relative_to(PUB)):sha(p) for p in paths}


def jobs():
    result=[]
    for dataset in DATASETS:
        result.append(dict(dataset=dataset,method='native',seed=0))
        for seed in [0,1,2]:
            for method in ['random','exact_assignment','corridor_plain']:
                result.append(dict(dataset=dataset,method=method,seed=seed))
    for j in result:j['name']=f"{j['dataset']}__{j['method']}__s{j['seed']}"
    return result


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def artifact(root,job,name,**arrays):
    path=root/'assignments'/job['name']/(name+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    return str(path.relative_to(root)),sha(path)


def case(args):
    import torch
    from lowrank import split,SplitSpec
    from lowrank.splitters.common.balanced_assignment import capacity_corridor
    from metrics import PackedMetrics,pair_sample,score
    torch.cuda.is_available=lambda:False;torch.set_num_threads(16);torch.set_num_interop_threads(1)
    root=args.output;m=json.loads((root/'manifest.json').read_text());j=m['jobs'][args.index]
    if m['sources']!=sources():raise RuntimeError('producer drift')
    if sorted(os.sched_getaffinity(0))!=m['cpus']:raise RuntimeError('CPU affinity mismatch')
    cache=HERE/'cache'/j['dataset'];manifest=json.loads((cache/'manifest.json').read_text())
    if sha(cache/'manifest.json')!=m['inputs'][j['dataset']]['manifest_sha256']:raise RuntimeError('input manifest drift')
    for name,digest in manifest['artifacts'].items():
        if sha(cache/name)!=digest:raise RuntimeError('prepared input changed: '+name)
    started=time.perf_counter();packed=np.load(cache/'packed.npy',mmap_mode='r')
    native=np.load(cache/'native.npy');names=manifest['fold_names'];k=len(names);n=len(native)
    input_load=time.perf_counter()-started;counts=np.bincount(native,minlength=k)
    warmX=np.unpackbits(np.asarray(packed[:100]),axis=1,bitorder='little').astype(np.float32)
    split('lowrank',dict(enumerate(warmX)),SplitSpec(counts.tolist(),names,seed=0),
          rank=32,metric='tanimoto',score_exact_leakage=False,fm=False)
    seconds=None;rss=None;params=None;diagnostics={};adapter_seconds=0.
    if j['method']=='native':labels=native.copy()
    elif j['method']=='random':
        t=time.perf_counter();labels=native[np.random.default_rng(j['seed']).permutation(n)]
        seconds=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    else:
        t=time.perf_counter()
        X=np.unpackbits(np.asarray(packed),axis=1,bitorder='little').astype(np.float32)
        data=dict(enumerate(X));adapter_seconds=time.perf_counter()-t
        epsilon=0. if j['method']=='exact_assignment' else .01
        params=dict(rank=256,metric='tanimoto',allowed_size_imbalance=epsilon,
                    score_exact_leakage=False,leakage_diagnostic='none',n_restarts=4,
                    n_iter=25,refine_restarts=2,fm=j['method']=='corridor_plain')
        print('starting public split',j['name'],n,flush=True)
        t=time.perf_counter()
        result=split('lowrank',data,SplitSpec(counts.tolist(),names,seed=j['seed']),**params)
        seconds=time.perf_counter()-t
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        labels=np.fromiter((names.index(result.assignment[i]) for i in range(n)),dtype=np.uint8,count=n)
        diagnostics=clean(result.diagnostics)
        del result,data,X
        print('split complete',j['name'],seconds,'seconds; independent audit begins',flush=True)
    actual=np.bincount(labels,minlength=k)
    epsilon=.01 if j['method']=='corridor_plain' else 0.
    caps,floors=capacity_corridor(n,counts.tolist(),epsilon)
    if not ((actual>=floors)&(actual<=caps)).all():raise RuntimeError('capacity violation')
    artifacts=dict([artifact(root,j,'labels',labels=labels)])
    # Native construction cost is unknown. Its small label-loading allocation
    # is not reported as a competing generation runtime or memory measurement.
    audit_start=time.perf_counter();engine=PackedMetrics(packed,threads=16)
    a,b=pair_sample(n);sim=engine.pairs(a,b)
    metrics,queries=score(engine,labels,names,a,b,sim,seed=j['seed'],query_count=128)
    arrays={}
    for c,name in enumerate(names[1:],1):
        arrays[f'fold{c}_queries']=queries[name]['queries'];arrays[f'fold{c}_nearest']=queries[name]['nearest']
    artifacts.update([artifact(root,j,'nearest_audit',**arrays)])
    # Each case stores pair statistics sufficient to replay the paired contrast.
    pair_blocks=np.stack([sim.reshape(64,-1).mean(1),
                          (sim*(labels[a]!=labels[b])).reshape(64,-1).mean(1)])
    artifacts.update([artifact(root,j,'pair_blocks',means=pair_blocks)])
    atomic(root/'rows'/(j['name']+'.json'),dict(job=j,status='ok',n=n,fold_names=names,
           native_counts=counts.tolist(),realized_counts=actual.tolist(),size_tolerance=epsilon,
           split_seconds=seconds,peak_process_rss_gib=rss,params=params,diagnostics=diagnostics,
           input_loading_seconds=input_load,input_adapter_seconds=adapter_seconds,
           audit_seconds=time.perf_counter()-audit_start,metrics=metrics,artifacts=artifacts,
           source_input_manifest_sha256=sha(cache/'manifest.json'),manifest_sha256=sha(root/'manifest.json'),
           host=socket.gethostname(),cpu_affinity=sorted(os.sched_getaffinity(0)),
           native_construction_time_status='not supplied; never equated with metadata loading',
           original_n=manifest['metadata']['original_n'],parse_failures=manifest['parse_failures']))
    print('completed',j['name'],metrics['global_relative'],metrics['mean_heldout_nearest'],flush=True)


def resource_check(cpus):
    import psutil
    if not set(cpus)<=os.sched_getaffinity(0) or len(cpus)!=16:raise RuntimeError('CPU allocation')
    if os.getloadavg()[0]+len(cpus)>os.cpu_count():raise RuntimeError('CPU load capacity')
    if psutil.virtual_memory().available<120*2**30:raise RuntimeError('insufficient available memory')
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            c=process.info['cmdline'] or [];a=process.cpu_affinity()
            if c and process.pid!=os.getpid() and 'python' in Path(c[0]).name and len(a)<32 and set(cpus)&set(a):
                raise RuntimeError(f'CPU overlap with {process.pid}')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return dict(online_cpus=os.cpu_count(),load=os.getloadavg(),available_gib=psutil.virtual_memory().available/2**30)


def plan(args):
    from metrics import load
    cpu=list(range(24,40));resources=resource_check(cpu);load()
    inputs={}
    for d in DATASETS:
        path=HERE/'cache'/d/'manifest.json';record=json.loads(path.read_text())
        if record['recipe']['producer_sha256']!=sha(HERE/'prepare.py'):raise RuntimeError('fingerprint producer drift')
        inputs[d]=dict(manifest_sha256=sha(path),n=record['retained_n'],counts=record['retained_counts'])
    path=args.output/'manifest.json'
    if path.exists():raise RuntimeError('fresh production directory required')
    atomic(path,dict(jobs=jobs(),sources=sources(),inputs=inputs,cpus=cpu,host=socket.gethostname(),
                    python=sys.executable,threads=16,max_attempts=1,timeout_seconds=1200,
                    packed_metric_binary_sha256=sha(HERE/'cache/packed_metrics.so'),
                    compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
                    preflight=resources,policy='Immutable sources and inputs; one 16-core fresh CPU process at a time; native construction times unknown; independent sampled queries search complete training folds'))
    print('planned',len(jobs()),flush=True)


def launch(args):
    root=args.output;m=json.loads((root/'manifest.json').read_text())
    if m['sources']!=sources():raise RuntimeError('source drift')
    if sha(HERE/'cache/packed_metrics.so')!=m['packed_metric_binary_sha256']:raise RuntimeError('metric binary drift')
    atomic(root/'launcher.json',dict(pid=os.getpid(),host=socket.gethostname(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    consecutive=0
    for i,j in enumerate(m['jobs']):
        if (root/'rows'/(j['name']+'.json')).exists() or (root/'failures'/(j['name']+'.json')).exists():continue
        resource_check(m['cpus'])
        claim=root/'claims'/j['name'];claim.parent.mkdir(parents=True,exist_ok=True)
        try:claim.mkdir()
        except FileExistsError:raise RuntimeError('held claim; verify owner rather than reclaim by age')
        atomic(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),
               start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',NUMPY_MADVISE_HUGEPAGE='0',
                                        OMP_MAX_ACTIVE_LEVELS='1',OMP_PROC_BIND='false')
        for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='16'
        cpus=','.join(map(str,m['cpus']))
        command=['taskset','-c',cpus,m['python'],str(Path(__file__).resolve()),'case','--output',str(root),'--index',str(i)]
        log=root/'logs'/(j['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:
            p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            atomic(claim/'child.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],command=command))
            try:code=p.wait(timeout=m['timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                code=-999
        complete=(root/'rows'/(j['name']+'.json')).exists()
        if code or not complete:
            atomic(root/'failures'/(j['name']+'.json'),dict(job=j,exit_code=code,log=str(log),
                   status='end_to_end_budget_exhausted' if code==-999 else 'failed',manifest_sha256=sha(root/'manifest.json')))
            consecutive+=1
        else:consecutive=0
        retired=root/'retired_claims'/j['name'];retired.parent.mkdir(parents=True,exist_ok=True);claim.rename(retired)
        print(j['name'],code,flush=True)
        atomic(root/'progress.json',dict(expected=len(m['jobs']),complete=len(list((root/'rows').glob('*.json'))),failed=len(list((root/'failures').glob('*.json')))))
        if consecutive>=3:raise RuntimeError('three consecutive failures; circuit breaker')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','launch','case'])
    p.add_argument('--output',type=Path,default=HERE/'run');p.add_argument('--index',type=int)
    args=p.parse_args();args.output=args.output.resolve();globals()[args.action](args)
