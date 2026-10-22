"""CPU-only entry point with bounded pools and diagnostic stack snapshots.

Set wait-policy environment variables before this process starts. This wrapper
changes worker initialization, not split/model settings or saved-output logic.
"""
import faulthandler
import json
import os
import runpy
import sys

faulthandler.enable()
faulthandler.dump_traceback_later(60,repeat=True)
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
if torch.cuda.is_available():
    raise RuntimeError('CPU protocol unexpectedly exposes a GPU')
print(json.dumps(dict(cpu_runner=True,threads=torch.get_num_threads(),
                      interop_threads=torch.get_num_interop_threads(),
                      affinity=sorted(os.sched_getaffinity(0)),
                      wait_policy=os.environ.get('OMP_WAIT_POLICY'),
                      nvml_based_check=os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK'))),flush=True)
script=sys.argv[1]
sys.argv=sys.argv[1:]
sys.path.insert(0,os.path.dirname(script))
try:
    runpy.run_path(script,run_name='__main__')
finally:
    faulthandler.cancel_dump_traceback_later()
