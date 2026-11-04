"""Plan relocated reruns and execute one frozen numerical case on assigned CPUs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import sys

import suite


def main(args):
    import psutil
    root=args.output.resolve()
    module=suite
    if args.protocol=='scaling':
        import dense_scaling as module
    elif args.protocol=='clock':
        import clock_budget as module
    if args.action=='plan':
        flat=[int(x) for cp in args.cpus for x in cp.split(',')]
        if len(flat)!=len(set(flat)) or not set(flat)<=os.sched_getaffinity(0):raise RuntimeError('invalid CPU allocation')
        if os.getloadavg()[0]+len(flat)>os.cpu_count():raise RuntimeError('CPU load capacity')
        for process in psutil.process_iter(['pid','cmdline']):
            try:
                cmd=process.info['cmdline'] or [];affinity=process.cpu_affinity()
                if cmd and process.pid!=os.getpid() and 'python' in Path(cmd[0]).name and len(affinity)<20 and set(flat)&set(affinity):
                    raise RuntimeError(f'overlapping Python allocation {process.pid}')
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if args.protocol=='clock' and args.cache_root and args.cache_root.resolve()!=suite.CACHE:
            raise ValueError('clock subprocess requires the repository-relative prepared cache location')
        if args.cache_root:suite.CACHE=args.cache_root.resolve()
        if args.protocol=='scaling' and args.packed_input:module.INPUT=args.packed_input.resolve()
        if args.protocol=='primary':
            suite.plan(argparse.Namespace(output=root,cpus=args.cpus,studies=args.studies,datasets=args.datasets,seeds=args.seeds))
        else:
            module.main(argparse.Namespace(action='plan',output=root,index=None))
        manifest=json.loads((root/'manifest.json').read_text())
        if args.datasets:manifest['jobs']=[j for j in manifest['jobs'] if j['dataset'] in args.datasets]
        if args.seeds is not None:manifest['jobs']=[j for j in manifest['jobs'] if j['seed'] in args.seeds]
        if not manifest['jobs']:raise RuntimeError('no selected cases')
        manifest.update(cpus=args.cpus,portable_adapter_sha256=suite.sha(__file__),
                        protocol=args.protocol,actual_cache_root=str(suite.CACHE),
                        actual_packed_input=str(module.INPUT) if args.protocol=='scaling' else None,
                        portable_host=socket.gethostname())
        suite.atomic(root/'manifest.json',manifest)
        print('planned',len(manifest['jobs']),'cases; use taskset and portable.py case --index INDEX')
        return
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['portable_adapter_sha256']!=suite.sha(__file__):raise RuntimeError('portable adapter drift')
    affinity=','.join(map(str,sorted(os.sched_getaffinity(0))))
    if affinity not in manifest['cpus']:raise RuntimeError('run this case under an assigned two-core taskset')
    for key in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','POLARS_MAX_THREADS']:os.environ[key]='2'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    suite.CACHE=Path(manifest['actual_cache_root'])
    if args.protocol=='scaling':module.INPUT=Path(manifest['actual_packed_input'])
    if manifest['protocol']!=args.protocol:raise RuntimeError('protocol mismatch')
    if args.protocol=='primary':suite.case(argparse.Namespace(output=root,index=args.index))
    elif args.protocol=='scaling':module.main(argparse.Namespace(action='case',output=root,index=args.index))
    else:module.case(root,args.index)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','case'])
    p.add_argument('--protocol',choices=['primary','scaling','clock'],default='primary')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int)
    p.add_argument('--cpus',nargs='+',default=['10,11'])
    p.add_argument('--datasets',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--studies',nargs='+');p.add_argument('--cache-root',type=Path);p.add_argument('--packed-input',type=Path)
    main(p.parse_args())
