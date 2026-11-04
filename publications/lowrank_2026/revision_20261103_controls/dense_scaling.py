"""Nested ChEMBL dense controls, with independent immutable run manifest."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import time

import suite

HERE=Path(__file__).resolve().parent
INPUT=HERE.parents[4]/'low_rank/data/chembl/ecfp2048_packed.npy'


def identities():
    return dict(**suite.sources(),**{str(p.relative_to(HERE.parent)):suite.sha(p)
                for p in [Path(__file__),HERE/'SCALING_PLAN.md']})


def main(args):
    import numpy as np
    import torch
    import psutil
    root=args.output
    if args.action=='plan':
        packed_sha=suite.sha(INPUT)
        old=json.loads((HERE.parent/'revision_20261026/scaling/cpu_run/manifest.json').read_text())
        if packed_sha!=old['packed_sha256']:raise RuntimeError('ChEMBL input changed')
        items=[]
        for n in [4000,16000,32000]:
            for seed in range(3):
                for arm,rank in [('dense',0),('factor',64),('factor',256),('factor',512)]:
                    items.append(dict(name=f'chembl__n{n}__s{seed}__{arm}__r{rank}',
                                      study='iterations',dataset='chembl37',n=n,seed=seed,
                                      arm=arm,rank=rank,power=1,epsilon=.01))
        slots={16,17,20,21}
        if os.getloadavg()[0]+len(slots)>os.cpu_count():raise RuntimeError('CPU capacity')
        if psutil.virtual_memory().available<64*2**30:raise RuntimeError('memory capacity')
        for process in psutil.process_iter(['pid','cmdline']):
            try:
                c=process.info['cmdline'] or [];aff=process.cpu_affinity()
                if c and process.pid!=os.getpid() and 'python' in Path(c[0]).name and len(aff)<20 and slots&set(aff):
                    raise RuntimeError(f'CPU overlap {process.pid} {aff}')
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if (root/'manifest.json').exists():raise RuntimeError('fresh output required')
        suite.atomic(root/'manifest.json',dict(jobs=items,sources=identities(),input_sha256=packed_sha,
                     cpus=['16,17','20,21'],host=socket.gethostname(),python=sys.executable,
                     timeout_seconds=1200,threads=2,max_attempts=1,
                     preflight=dict(load=os.getloadavg(),available_memory_gib=psutil.virtual_memory().available/2**30)))
        print('planned',len(items));return
    m=json.loads((root/'manifest.json').read_text())
    if m['sources']!=identities():raise RuntimeError('producer drift')
    if args.action=='case':
        torch.cuda.is_available=lambda:False;torch.set_num_threads(2)
        j=m['jobs'][args.index];packed=np.load(INPUT,mmap_mode='r')
        indices=np.sort(np.random.default_rng(0).permutation(len(packed))[:j['n']])
        X=np.unpackbits(np.asarray(packed[indices]),axis=1).astype(np.float32)
        from lowrank.optimize import balanced_lloyd
        balanced_lloyd(np.ones((20,4),np.float32),[8,1,1],n_iter=1)
        artifacts={};suite.record(root,j,'indices',indices,artifacts)
        print('starting',j['name'],flush=True)
        result=suite.optimization_case(X,j,root,artifacts)
        suite.atomic(root/'rows'/(j['name']+'.json'),dict(job=j,n=len(X),status='ok',result=result,
                     artifacts=artifacts,feature_sha256=__import__('hashlib').sha256(X.tobytes()).hexdigest(),
                     manifest_sha256=suite.sha(root/'manifest.json'),host=socket.gethostname(),
                     cpu_affinity=sorted(os.sched_getaffinity(0))))
        print('completed',j['name'],flush=True);return
    q=queue.Queue()
    for i,j in enumerate(m['jobs']):
        if not (root/'rows'/(j['name']+'.json')).exists() and not (root/'failures'/(j['name']+'.json')).exists():q.put((i,j))
    def worker(cpus):
        consecutive=0
        while True:
            try:i,j=q.get_nowait()
            except queue.Empty:return
            claim=root/'claims'/j['name'];claim.parent.mkdir(parents=True,exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            suite.atomic(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),
                         start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],cpus=cpus))
            env=os.environ.copy()
            for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
            env['CUDA_VISIBLE_DEVICES']=''
            log=root/'logs'/(j['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            command=['taskset','-c',cpus,m['python'],str(Path(__file__).resolve()),'case','--output',str(root),'--index',str(i)]
            with log.open('w') as f:
                p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                suite.atomic(claim/'child.json',dict(pid=p.pid,command=command,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]))
                try:code=p.wait(timeout=m['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(p.pid,signal.SIGTERM);p.wait();code=-999
            if code or not (root/'rows'/(j['name']+'.json')).exists():
                suite.atomic(root/'failures'/(j['name']+'.json'),dict(job=j,exit_code=code,log=str(log)));consecutive+=1
            else:consecutive=0
            retired=root/'retired_claims'/j['name'];retired.parent.mkdir(parents=True,exist_ok=True)
            os.replace(claim,retired);print(j['name'],code,flush=True)
            if consecutive>=3:return
    suite.atomic(root/'launcher.json',dict(pid=os.getpid(),host=socket.gethostname(),
                 start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,m['cpus']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','launch','case'])
    p.add_argument('--output',type=Path,default=HERE/'scaling_run');p.add_argument('--index',type=int)
    args=p.parse_args();args.output=args.output.resolve();main(args)
