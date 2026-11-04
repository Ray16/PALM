"""Preempted equal-wall-clock controls with before-deadline checkpoints only.

The timed optimizer runs in a disposable subprocess. The supervising case
terminates it at the deadline and evaluates the last completed checkpoint.
Imports, input loading, warm-up and subsequent evaluation are outside the clock.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import time

import suite
HERE=Path(__file__).resolve().parent


def sources():
    return dict(**suite.sources(),**{str(Path(__file__).resolve().relative_to(HERE.parent)):suite.sha(__file__)})


def optimize(root,index):
    import numpy as np
    import torch
    torch.cuda.is_available=lambda:False;torch.set_num_threads(2)
    from lowrank.optimize import balanced_lloyd,fm_polish
    m=json.loads((root/'manifest.json').read_text());j=m['jobs'][index]
    X=np.load(suite.CACHE/(j['dataset']+'.npz'))['X']
    state=root/'state'/j['name'];state.mkdir(parents=True,exist_ok=True)
    balanced_lloyd(np.ones((20,4),np.float32),[8,1,1],n_iter=1)
    initial=suite.init_labels(len(X),j['seed'])
    np.save(state/'initial.npy',initial)
    # Ready before the start signal: a feasible output exists even if kernel
    # construction is interrupted. No partial numerical operation is accepted.
    suite.atomic(state/'checkpoint.json',dict(path='initial.npy',elapsed=0.,initial=True))
    started=time.monotonic();deadline=started+j['budget_seconds']
    suite.atomic(state/'start.json',dict(monotonic=started,deadline=deadline,pid=os.getpid()))
    ops,_,_=suite.representation(X,j)
    if time.monotonic()>=deadline:return
    best_cost=ops.cost(initial,3);counter=0;restart=0
    def retain(labels,value):
        nonlocal best_cost,counter
        if value>=best_cost or time.monotonic()>=deadline:return
        path=state/f'candidate_{counter}.npy';np.save(path,labels)
        elapsed=time.monotonic()-started
        if elapsed>=j['budget_seconds']:return
        # Atomic pointer publication may itself cross the deadline; the final
        # read checks the declared completion time, which includes serialization.
        suite.atomic(state/'checkpoint.json',dict(path=path.name,elapsed=elapsed,initial=False))
        best_cost=value;counter+=1
    while time.monotonic()<deadline:
        current=suite.init_labels(len(X),j['seed']+restart)
        for _ in range(25):
            if time.monotonic()>=deadline:return
            proposal=balanced_lloyd(ops,[8,1,1],init_labels=current,n_iter=1,balance_slack=.01)
            value=ops.cost(proposal,3);retain(proposal,value)
            same=np.array_equal(current,proposal);current=proposal
            if same:break
        for _ in range(len(X)):
            if time.monotonic()>=deadline:return
            current,moves=fm_polish(ops,current,[8,1,1],epsilon=.01,max_moves=1)
            retain(current,ops.cost(current,3))
            if not moves:break
        restart+=1


def case(root,index):
    import numpy as np
    from oracles import independent_metrics
    m=json.loads((root/'manifest.json').read_text());j=m['jobs'][index]
    if m['sources']!=sources():raise RuntimeError('strict-clock source drift')
    X=np.load(suite.CACHE/(j['dataset']+'.npz'))['X']
    if suite.sha(suite.CACHE/(j['dataset']+'.npz'))!=m['inputs'][j['dataset']]:raise RuntimeError('input drift')
    command=[sys.executable,str(Path(__file__).resolve()),'optimize','--output',str(root),'--index',str(index)]
    process=subprocess.Popen(command,start_new_session=True)
    state=root/'state'/j['name'];startup=time.monotonic()
    while not (state/'start.json').exists():
        if process.poll() is not None:raise RuntimeError('optimizer failed before clock start')
        if time.monotonic()-startup>60:
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError('optimizer startup timeout')
        time.sleep(.005)
    start=json.loads((state/'start.json').read_text())
    while process.poll() is None and time.monotonic()<start['deadline']:
        time.sleep(min(.002,max(0.,start['deadline']-time.monotonic())))
    if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    elapsed=time.monotonic()-start['monotonic']
    checkpoint=json.loads((state/'checkpoint.json').read_text())
    if checkpoint['elapsed']>j['budget_seconds']:raise RuntimeError('late checkpoint')
    labels=np.load(state/checkpoint['path']);suite.check(labels,len(X),.01)
    artifacts={};suite.record(root,j,'labels',labels,artifacts)
    metrics=independent_metrics(X,labels)
    suite.atomic(root/'rows'/(j['name']+'.json'),dict(job=j,status='ok',n=len(X),metrics=metrics,
                 timed_process_seconds=elapsed,checkpoint=checkpoint,
                 termination_overrun_seconds=max(0.,elapsed-j['budget_seconds']),
                 manifest_sha256=suite.sha(root/'manifest.json'),artifacts=artifacts,
                 cpu_affinity=sorted(os.sched_getaffinity(0)),host=socket.gethostname()))
    print('completed',j['name'],flush=True)


def main(args):
    root=args.output
    if args.action=='plan':
        old=json.loads((HERE/'run/manifest.json').read_text())
        jobs=[dict(j,name='preempted__'+j['name']) for j in old['jobs'] if j['study']=='clock']
        if (root/'manifest.json').exists():raise RuntimeError('fresh output required')
        if os.getloadavg()[0]+4>os.cpu_count():raise RuntimeError('CPU capacity')
        suite.atomic(root/'manifest.json',dict(jobs=jobs,sources=sources(),inputs=old['inputs'],
                     cpus=['10,11','12,13'],host=socket.gethostname(),python=sys.executable,
                     threads=2,max_attempts=1,policy='Terminate timed subprocess at deadline; score last serialized before-deadline feasible checkpoint; no late candidate accepted'))
        return
    if args.action=='case':case(root,args.index);return
    if args.action=='optimize':optimize(root,args.index);return
    m=json.loads((root/'manifest.json').read_text());q=queue.Queue()
    if m['sources']!=sources():raise RuntimeError('source drift')
    for i,j in enumerate(m['jobs']):
        if not (root/'rows'/(j['name']+'.json')).exists() and not (root/'failures'/(j['name']+'.json')).exists():q.put((i,j))
    def worker(cpus):
        failures=0
        while True:
            try:i,j=q.get_nowait()
            except queue.Empty:return
            # One launcher owns this immutable queue, with atomic per-case claims.
            claim=root/'claims'/j['name'];claim.parent.mkdir(parents=True,exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            suite.atomic(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),cpus=cpus,
                         start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
            log=root/'logs'/(j['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('w') as f:
                p=subprocess.Popen(['taskset','-c',cpus,m['python'],str(Path(__file__).resolve()),'case',
                                    '--output',str(root),'--index',str(i)],stdout=f,stderr=subprocess.STDOUT,env=env)
                suite.atomic(claim/'child.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]))
                code=p.wait()
            if code or not (root/'rows'/(j['name']+'.json')).exists():
                suite.atomic(root/'failures'/(j['name']+'.json'),dict(job=j,exit_code=code,log=str(log)));failures+=1
            else:failures=0
            retired=root/'retired_claims'/j['name'];retired.parent.mkdir(parents=True,exist_ok=True)
            os.replace(claim,retired);print(j['name'],code,flush=True)
            if failures>=3:return
    suite.atomic(root/'launcher.json',dict(pid=os.getpid(),host=socket.gethostname(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,m['cpus']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','launch','case','optimize'])
    p.add_argument('--output',type=Path,default=HERE/'strict_clock_run');p.add_argument('--index',type=int)
    args=p.parse_args();args.output=args.output.resolve();main(args)
