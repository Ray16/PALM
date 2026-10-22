"""CPU runner avoiding the shared node's saturated NUMA memory domain."""
import ctypes
import faulthandler
import json
import os
import platform
import runpy
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(60,repeat=True)
libc=ctypes.CDLL(None,use_errno=True)
if platform.system()!='Linux' or platform.machine()!='x86_64':
    raise RuntimeError('This cluster memory-placement helper requires Linux x86-64')
if libc.prctl(41,1,0,0,0)!=0:
    raise OSError(ctypes.get_errno(),'cannot disable THP for this worker')
node=int(os.environ.get('LOWRANK_MEMORY_NODE','0'))
mask=ctypes.c_ulong(1<<node)
if libc.syscall(238,2,ctypes.byref(mask),64)!=0:
    raise OSError(ctypes.get_errno(),'cannot bind worker memory to the preflighted NUMA node')
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
if torch.cuda.is_available():raise RuntimeError('CPU protocol exposes a GPU')
print(json.dumps(dict(cpu_runner=True,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                      affinity=sorted(os.sched_getaffinity(0)),memory_node=node,transparent_hugepages=False)),flush=True)
script=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,os.path.dirname(script))
try:runpy.run_path(script,run_name='__main__')
finally:faulthandler.cancel_dump_traceback_later()
