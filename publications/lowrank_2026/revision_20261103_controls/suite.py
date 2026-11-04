"""Frozen, resumable CPU controls for the constrained low-rank methods paper."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
REPO = HERE.parents[2]
CACHE = REPO / "data/lowrank_cache"
DEPENDENCIES = HERE.parents[4] / "runtime/controls_dependencies"
sys.path[:0] = [str(PUB / "package"), str(HERE / "vendor"), str(DEPENDENCIES)]
DATASETS = ["moleculenet_" + s for s in
            ("freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sources():
    paths = [HERE / "suite.py", HERE / "oracles.py", HERE / "PLAN.md",
             *sorted((PUB / "package/lowrank").rglob("*.py")),
             *sorted((HERE / "vendor").rglob("*.py"))]
    return {str(p.relative_to(PUB)): sha(p) for p in paths}


def jobs():
    result = []
    for dataset in DATASETS:
        for seed in range(3):
            for power in [1, 3]:
                for epsilon in [0., .01]:
                    for arm, rank in [("dense",0),("factor",64),("factor",256),("factor",512)]:
                        result.append(dict(study="iterations", dataset=dataset,
                                           seed=seed, power=power, epsilon=epsilon,
                                           arm=arm, rank=rank))
            for seconds in [1., 5.]:
                for arm in ["dense", "factor"]:
                    result.append(dict(study="clock", dataset=dataset, seed=seed,
                                       power=1, epsilon=.01, arm=arm, rank=256,
                                       budget_seconds=seconds))
            result.append(dict(study="ablation",dataset=dataset,seed=seed))
            for threshold in [.4,.6,.8]:
                result.append(dict(study="ccpart",dataset=dataset,seed=seed,threshold=threshold))
    for j in result:
        fields = [j["study"],j["dataset"],f"s{j['seed']}"]
        fields.extend(f"{k}{v}" for k,v in j.items() if k not in ("study","dataset","seed"))
        j["name"] = "__".join(fields)
    return result


def init_labels(n, seed):
    import numpy as np
    from lowrank.splitters.common.balanced_assignment import target_sizes
    labels = np.repeat(np.arange(3),target_sizes(n,[8,1,1]))
    np.random.default_rng(seed).shuffle(labels)
    return labels


def record(output, job, name, value, artifacts):
    import numpy as np
    path = output / "assignments" / job["name"] / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path,value)
    artifacts[str(path.relative_to(output))] = sha(path)


def check(labels, n, epsilon, gid=None, ratios=None):
    import numpy as np
    from lowrank.splitters.common.balanced_assignment import capacity_corridor
    if labels.shape != (n,) or np.any(labels < 0) or np.any(labels > 2):
        raise AssertionError("missing/invalid assignment")
    cap,floor=capacity_corridor(n,[8,1,1] if ratios is None else ratios,epsilon)
    count=np.bincount(labels,minlength=3)
    if not ((count>=floor)&(count<=cap)).all():
        raise AssertionError(f"capacity: {count}, {floor}, {cap}")
    if gid is not None:
        lo,hi=np.full(int(gid.max())+1,3),np.full(int(gid.max())+1,-1)
        np.minimum.at(lo,gid,labels);np.maximum.at(hi,gid,labels)
        if not np.array_equal(lo,hi):raise AssertionError("broken group")


def representation(X, job):
    from lowrank.nystrom import nystrom_features
    from lowrank.factors import factor_ops
    from oracles import DenseOps,build_dense
    if job["arm"]=="dense":
        K=build_dense(X,job["power"])
        return DenseOps(K),None,dict(storage_bytes=K.numel()*K.element_size())
    B,_,info=nystrom_features(X,rank=job["rank"],metric="tanimoto",
                            kernel_power=job["power"],seed=job["seed"],return_info=True)
    return factor_ops(B),B,dict(storage_bytes=B.nbytes,factor_info=info)


def optimization_case(X,job,output,artifacts):
    import numpy as np
    from lowrank.optimize import balanced_lloyd,fm_polish
    from oracles import independent_metrics
    labels=initial=init_labels(len(X),job["seed"])
    record(output,job,"initial",initial,artifacts)
    started=time.perf_counter()
    ops,B,info=representation(X,job)
    built=time.perf_counter()-started
    moves,steps=0,0
    if job["study"]=="iterations":
        labels=balanced_lloyd(ops,[8,1,1],init_labels=initial,n_iter=25,
                              balance_slack=job["epsilon"])
        assigned=time.perf_counter()-started
        labels,moves=fm_polish(ops,labels,[8,1,1],epsilon=job["epsilon"],max_moves=len(X))
        # The routine may stop early; 25 is the maximum, not an observed count.
        info.update(assignment_iteration_limit=25,refinement_limit=len(X))
    else:
        deadline=started+job["budget_seconds"]
        # Same one-step current routines in both arms; inspect time between
        # operations. An operation already in progress is allowed to finish.
        best=initial.copy()
        best_cost=ops.cost(best,3) if time.perf_counter()<deadline else None
        restart=0
        while time.perf_counter()<deadline:
            current=init_labels(len(X),job["seed"]+restart)
            for _ in range(25):
                if time.perf_counter()>=deadline:break
                proposal=balanced_lloyd(ops,[8,1,1],init_labels=current,n_iter=1,
                                       balance_slack=job["epsilon"])
                steps+=1
                same=np.array_equal(proposal,current)
                current=proposal
                val=ops.cost(current,3)
                if best_cost is None or val<best_cost:best,best_cost=current.copy(),val
                if same:break
            for _ in range(len(X)):
                if time.perf_counter()>=deadline:break
                current,nmove=fm_polish(ops,current,[8,1,1],epsilon=job["epsilon"],max_moves=1)
                moves+=nmove
                val=ops.cost(current,3)
                if best_cost is None or val<best_cost:best,best_cost=current.copy(),val
                if not nmove:break
            restart+=1
        labels=best
        assigned=None
        info.update(completed_restarts=restart,one_step_assignment_calls=steps,
                    construction_exhausted_budget=built>=job["budget_seconds"])
    elapsed=time.perf_counter()-started
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    # Profiling stops before all independent scoring and artifact writes.
    check(labels,len(X),job["epsilon"])
    record(output,job,"labels",labels,artifacts)
    value=independent_metrics(X,labels,job["power"],B=B)
    return dict(seconds=elapsed,construction_seconds=built,assignment_endpoint_seconds=assigned,
                native_peak_rss_gib=peak,refinement_operations=int(moves),
                overrun_seconds=max(0.,elapsed-job.get("budget_seconds",elapsed)),
                metrics=value,diagnostics=info)


def ablation_case(X,job,output,artifacts):
    import numpy as np
    from lowrank.nystrom import nystrom_features
    from lowrank.optimize import balanced_lloyd,fm_polish,optimize_restarts
    from lowrank.grouped import must_link_groups,grouped_split
    from oracles import independent_metrics
    from lowrank.splitters.common.balanced_assignment import target_sizes
    seed=job['seed']; initial=init_labels(len(X),seed)
    record(output,job,'initial',initial,artifacts)
    t=time.perf_counter();B,_=nystrom_features(X,256,'tanimoto',seed=seed)
    factor_seconds=time.perf_counter()-t
    variants=[]
    def add(name,fn,factor=B,groups=None,construction=factor_seconds,power=1):
        started=time.perf_counter(); lab,diag=fn(); elapsed=time.perf_counter()-started
        check(lab,len(X),.01,groups)
        record(output,job,name,lab,artifacts)
        variants.append(dict(variant=name,metrics=independent_metrics(X,lab,power,B=factor),
                             optimization_seconds=elapsed,construction_seconds=construction,
                             diagnostics=diag))
        return lab
    add('initial',lambda:(initial.copy(),{}))
    add('assignment_only',lambda:(balanced_lloyd(B,[8,1,1],init_labels=initial,balance_slack=.01),{}))
    add('refinement_only',lambda:fm_polish(B,initial,[8,1,1],epsilon=.01,max_moves=len(X)))
    def both():
        z=balanced_lloyd(B,[8,1,1],init_labels=initial,balance_slack=.01)
        return fm_polish(B,z,[8,1,1],epsilon=.01,max_moves=len(X))
    add('one_restart',both)
    def multi(factor):
        z,_,diag=optimize_restarts(factor,[8,1,1],seed=seed,n_restarts=4,
                                 refine_restarts=2,balance_slack=.01)
        return z,diag
    add('four_restarts',lambda:multi(B))
    t=time.perf_counter();U,_=nystrom_features(X,256,'tanimoto',seed=seed,landmark='uniform')
    uniform_time=time.perf_counter()-t
    add('uniform_landmarks',lambda:multi(U),factor=U,construction=uniform_time)
    t=time.perf_counter();P,_=nystrom_features(X,256,'tanimoto',seed=seed,kernel_power=3)
    power_time=time.perf_counter()-t
    add('power_only',lambda:multi(P),factor=P,construction=power_time,power=3)
    # Fix capped groups once and preserve them strictly in both factor settings.
    t=time.perf_counter()
    gid=must_link_groups(X,'tanimoto',.4,max(1,int(.05*len(X))),k=32)
    group_time=time.perf_counter()-t
    record(output,job,'groups',gid,artifacts)
    from lowrank.grouped import _weighted_peel,_whole_group_assignment
    from lowrank.splitters.common.balanced_assignment import capacity_corridor
    w=np.bincount(gid);caps,floors=capacity_corridor(len(X),[8,1,1],.01)
    gl=_weighted_peel(np.random.default_rng(seed).random((len(w),3)),w,caps,floors)
    counts=np.bincount(gl,weights=w,minlength=3)
    if not ((counts>=floors)&(counts<=caps)).all():
        gl=_whole_group_assignment(w,caps,floors,gl,time_limit=30)
    feasible=gl[gid]
    add('group_initial',lambda:(feasible,{}),groups=gid,construction=group_time+factor_seconds)
    def grouped(factor):
        return grouped_split(factor,gid,[8,1,1],.01,n_restarts=1,n_iter=25,
                             init_labels=feasible,seed=seed,group_policy='strict',
                             feasibility_seconds=30,refine_restarts=1)
    add('groups_plain',lambda:grouped(B),groups=gid,construction=group_time+factor_seconds)
    add('groups_power',lambda:grouped(P),factor=P,groups=gid,power=3,
        construction=group_time+power_time)
    return dict(variants=variants,group_count=len(w),largest_group=int(w.max()),
                group_construction_seconds=group_time,
                timing_scope='component calls; shared representations; no per-variant RSS comparison')


def ccpart_case(X,job,output,artifacts):
    import numpy as np
    import pandas as pd
    import polars as pl
    from hestia.partition import ccpart
    from lowrank.nystrom import nystrom_features
    from lowrank.optimize import balanced_lloyd,fm_polish
    from lowrank.grouped import grouped_split,GroupInfeasibleError,GroupFeasibilityUnresolved
    from oracles import independent_metrics
    n=len(X);perm=np.random.default_rng(job['seed']).permutation(n)
    record(output,job,'permutation',perm,artifacts)
    # Full float64 Tanimoto pairs, streamed; include self edges for all isolated
    # vertices, and every unordered threshold edge. No neighbour truncation.
    start=time.perf_counter(); A=np.asarray(X[perm],dtype=np.float64)
    norm=np.einsum('ij,ij->i',A,A);rows=[];cols=[];values=[]
    for s in range(0,n,256):
        dot=A[s:s+256]@A.T;den=norm[s:s+len(dot),None]+norm[None,:]-dot
        S=np.divide(dot,den,out=np.zeros_like(dot),where=den>0)
        r,c=np.nonzero((S>=job['threshold']) & (np.arange(s,s+len(dot))[:,None]<=np.arange(n)[None,:]))
        rows.append(r+s);cols.append(c);values.append(S[r,c])
    sim=pl.DataFrame({'query':np.concatenate(rows),'target':np.concatenate(cols),
                      'metric':np.concatenate(values)})
    similarity_seconds=time.perf_counter()-start
    train,test,valid,gid=ccpart(pd.DataFrame(index=np.arange(n)),sim,
                              test_size=.1,valid_size=.1,threshold=job['threshold'],verbose=0)
    seconds=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    zp=np.full(n,-1,dtype=np.int64)
    for c,ids in enumerate((train,valid,test)):
        if (zp[ids]>=0).any():raise AssertionError('duplicate native IDs')
        zp[ids]=c
    labels=np.empty(n,dtype=np.int64);labels[perm]=zp
    groups=np.empty(n,dtype=np.int64);groups[perm]=gid
    if (labels<0).any():raise AssertionError('CCPart omitted entities')
    # Native sizes are part of the result, not repaired or treated as matched.
    counts=np.bincount(labels,minlength=3)
    check(labels,n,0,groups,ratios=counts)
    record(output,job,'ccpart',labels,artifacts);record(output,job,'groups',groups,artifacts)
    variants=[dict(variant='ccpart',metrics=independent_metrics(X,labels),seconds=seconds,
                   similarity_seconds=similarity_seconds,native_peak_rss_gib=peak)]
    if min(counts)==0:
        return dict(variants=variants,matched_status='empty_native_fold',
                    group_count=len(np.unique(groups)),largest_group=int(np.bincount(groups).max()))
    t=time.perf_counter();B,_=nystrom_features(X,256,'tanimoto',seed=job['seed'])
    factor_seconds=time.perf_counter()-t
    for variant in ['matched_counts','matched_groups']:
        t=time.perf_counter()
        if variant=='matched_counts':
            z=balanced_lloyd(B,counts,init_labels=labels,n_iter=25)
            z,moves=fm_polish(B,z,counts,epsilon=0,max_moves=n)
            diag=dict(refinement_operations=moves)
        else:
            z,diag=grouped_split(B,groups,counts,0,n_restarts=1,n_iter=25,
                                init_labels=labels,seed=job['seed'],group_policy='strict',
                                feasibility_seconds=30,refine_restarts=1)
        elapsed=time.perf_counter()-t
        check(z,n,0,groups if variant=='matched_groups' else None,ratios=counts)
        record(output,job,variant,z,artifacts)
        variants.append(dict(variant=variant,metrics=independent_metrics(X,z,B=B),
                             seconds=elapsed+factor_seconds,diagnostics=diag))
    return dict(variants=variants,matched_status='ok',group_count=len(np.unique(groups)),
                largest_group=int(np.bincount(groups).max()),
                timing_scope='native includes complete threshold similarities and components; matched controls reuse native groups')


def case(args):
    import numpy as np
    import torch
    # Explicit CPU-only protocol; no GPU reservation or GPU computation occurs.
    torch.cuda.is_available=lambda:False
    torch.set_num_threads(2)
    manifest=json.loads((args.output/'manifest.json').read_text())
    if manifest['sources']!=sources():raise RuntimeError('producer source drift')
    job=manifest['jobs'][args.index]
    p=CACHE/(job['dataset']+'.npz')
    if sha(p)!=manifest['inputs'][job['dataset']]:raise RuntimeError('input drift')
    X=np.load(p)['X']
    # Identical small warm-up, excluded from split clocks; included in process RSS.
    from lowrank.optimize import balanced_lloyd
    balanced_lloyd(np.ones((20,4),np.float32),[8,1,1],n_iter=1)
    artifacts={};started=time.time()
    if job['study'] in ('iterations','clock'):result=optimization_case(X,job,args.output,artifacts)
    elif job['study']=='ablation':result=ablation_case(X,job,args.output,artifacts)
    else:result=ccpart_case(X,job,args.output,artifacts)
    atomic(args.output/'rows'/(job['name']+'.json'),dict(
        job=job,status='ok',n=len(X),result=result,artifacts=artifacts,
        manifest_sha256=sha(args.output/'manifest.json'),
        host=socket.gethostname(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        started=started,finished=time.time()))
    print('completed',job['name'],flush=True)


def plan(args):
    import psutil
    selected=[j for j in jobs() if not args.studies or j['study'] in args.studies]
    if args.datasets:selected=[j for j in selected if j['dataset'] in args.datasets]
    if args.seeds is not None:selected=[j for j in selected if j['seed'] in args.seeds]
    prior=json.loads((PUB/'revision_20261026/run/manifest.json').read_text())
    inputs={d:sha(CACHE/(d+'.npz')) for d in sorted({j['dataset'] for j in selected})}
    if any(v!=prior['inputs'][d]['cache_sha256'] for d,v in inputs.items()):
        raise RuntimeError('caches differ from original study')
    flat=[int(c) for cp in args.cpus for c in cp.split(',')]
    if len(flat)!=len(set(flat)) or not set(flat)<=os.sched_getaffinity(0):
        raise RuntimeError('invalid/disjoint CPU allocation')
    if os.getloadavg()[0]+len(flat)>os.cpu_count():raise RuntimeError('CPU capacity')
    if psutil.virtual_memory().available<32*2**30:raise RuntimeError('memory capacity')
    conflicts=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            command=process.info['cmdline'] or []
            affinity=process.cpu_affinity()
            if (process.pid!=os.getpid() and command and len(affinity)<20
                and 'python' in Path(command[0]).name and set(flat)&set(affinity)):
                conflicts.append(dict(pid=process.pid,affinity=affinity,command=command))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if conflicts:raise RuntimeError('CPU allocations overlap pinned Python processes: '+str(conflicts))
    m=dict(jobs=selected,sources=sources(),inputs=inputs,cpus=args.cpus,
           python=sys.executable,host=socket.gethostname(),threads=2,
           timeout_seconds=600,max_attempts=1,cache_root=str(CACHE),
           dependencies=str(DEPENDENCIES),hestia_commit=(HERE/'vendor/HESTIA_COMMIT').read_text().strip(),
           preflight=dict(load=os.getloadavg(),available_memory_gib=psutil.virtual_memory().available/2**30,
                          online_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0))),
           allocation_conflicts=conflicts,
           versions={x:importlib.metadata.version(x) for x in ['numpy','torch','scipy','pandas','scikit-learn']})
    path=args.output/'manifest.json'
    if path.exists():raise RuntimeError('choose a fresh run directory')
    atomic(path,m);print('planned',len(selected))


def launch(args):
    import queue
    m=json.loads((args.output/'manifest.json').read_text())
    if sources()!=m['sources']:raise RuntimeError('source drift')
    if m['host']!=socket.gethostname():raise RuntimeError('host mismatch')
    if os.getloadavg()[0]+2*len(m['cpus'])>os.cpu_count():raise RuntimeError('CPU capacity')
    q=queue.Queue()
    for i,j in enumerate(m['jobs']):
        if not (args.output/'rows'/(j['name']+'.json')).exists() and not (args.output/'failures'/(j['name']+'.json')).exists():q.put((i,j))
    def worker(cpus):
        failures=0
        while not q.empty():
            try:i,j=q.get_nowait()
            except queue.Empty:return
            claim=args.output/'claims'/j['name'];claim.parent.mkdir(parents=True,exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            atomic(claim/'owner.json',dict(host=socket.gethostname(),pid=os.getpid(),
                   start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],cpus=cpus))
            env=os.environ.copy()
            for k in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS']:env[k]='2'
            env['CUDA_VISIBLE_DEVICES']=''
            log=args.output/'logs'/(j['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            command=['taskset','-c',cpus,m['python'],str(HERE/'suite.py'),'case','--output',str(args.output),'--index',str(i)]
            try:
                with log.open('w') as f:
                    p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    atomic(claim/'child.json',dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],command=command))
                    try:code=p.wait(timeout=m['timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                        code=-999
                complete=(args.output/'rows'/(j['name']+'.json')).exists()
                if code!=0 or not complete:
                    atomic(args.output/'failures'/(j['name']+'.json'),dict(job=j,exit_code=code,log=str(log),manifest_sha256=sha(args.output/'manifest.json')))
                    failures+=1
                else:failures=0
                print(j['name'],code,flush=True)
            finally:
                # Retire claim metadata atomically on NFS; never infer stale
                # ownership from age or delete another producer's claim.
                retired=args.output/'retired_claims'/j['name'];retired.parent.mkdir(parents=True,exist_ok=True)
                os.replace(claim,retired)
                q.task_done()
            if failures>=3:return
    atomic(args.output/'launcher.json',dict(pid=os.getpid(),host=socket.gethostname(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    with ThreadPoolExecutor(max_workers=len(m['cpus'])) as pool:
        list(pool.map(worker,m['cpus']))
    atomic(args.output/'progress.json',dict(expected=len(m['jobs']),completed=len(list((args.output/'rows').glob('*.json'))),failures=len(list((args.output/'failures').glob('*.json')))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','case','launch'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int)
    p.add_argument('--cpus',nargs='+',default=['8,9','10,11','12,13','14,15'])
    p.add_argument('--studies',nargs='+');p.add_argument('--datasets',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    args=p.parse_args();args.output=args.output.resolve()
    globals()[args.action](args)
