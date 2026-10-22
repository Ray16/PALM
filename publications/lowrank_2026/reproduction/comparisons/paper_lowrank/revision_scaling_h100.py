"""One fresh-process scaling case for a reservation-owned GPU worker.

Actual public split call, rank256, four restarts, 25 iterations, epsilon .01.
Plain and exact-tiled combined settings are measured separately; no inference
about unmeasured grouped sizes is made. Feature loading and scoring are outside
the split clock; cold-process RSS and peak GPU allocation are recorded.
"""
from __future__ import annotations
import argparse
import faulthandler
import hashlib
import json
import resource
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_external_control import source_files
from splitters import SplitSpec, split
from splitters.common.balanced_assignment import capacity_corridor

DEST = study.OUT / "revision_20261021/scaling_h100"


def run(method, n):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU scaling requires a live reservation-owned CUDA device")
    directory = DEST / f"{method}__n{n}"
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "result.json"
    sources = source_files()
    sources[str(Path(__file__).relative_to(study.ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if final.exists():
        if json.loads(final.read_text())["source_files"] != sources:
            raise RuntimeError("existing scaling case uses another source")
        return
    print("loading packed fingerprints", method, n, flush=True)
    packed = np.load(study.ROOT / "low_rank/data/chembl/ecfp2048_packed.npy", mmap_mode="r")
    indices = np.sort(np.random.default_rng(0).permutation(len(packed))[:n])
    if len(indices) != n:
        raise ValueError("requested scaling sample exceeds cached ChEMBL corpus")
    X = np.unpackbits(np.asarray(packed[indices]), axis=1).astype(np.float32)
    print("unpacked features", X.shape, flush=True)
    fd = dict(enumerate(X))
    kwargs = dict(rank=256, allowed_size_imbalance=.01, score_exact_leakage=False)
    if method == "combined":
        kwargs.update(kernel_power=3, must_link=.4, must_link_backend="tiled",
                      must_link_row_block=256, must_link_ref_block=8192)
    warm_kwargs = dict(kwargs, rank=32)
    split("lowrank", {i: X[i] for i in range(100)},
          SplitSpec([8, 1, 1], study.NAMES.tolist(), epsilon=.01), **warm_kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print("starting timed split", method, n, flush=True)
    start = time.perf_counter()
    result = split("lowrank", fd, SplitSpec([8, 1, 1], study.NAMES.tolist(), seed=0, epsilon=.01), **kwargs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    labels = np.array([result.assignment[i] for i in range(n)])
    codes = np.array([list(study.NAMES).index(s) for s in labels], dtype=np.int8)
    counts = np.bincount(codes, minlength=3)
    caps, floors = capacity_corridor(n, [8, 1, 1], .01)
    if not np.all((counts >= floors) & (counts <= caps)):
        raise AssertionError("scaling output violates fold-size corridor")
    np.save(directory / "assignment_codes.npy", codes)
    np.save(directory / "sample_indices.npy", indices)
    record = dict(method=method, n=n, status="ok", split_s=seconds, source_files=sources,
        host=socket.gethostname(), device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, counts=counts.tolist(),
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        sample_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        packed_input_shape=list(packed.shape), params=kwargs, diagnostics=result.diagnostics,
        timing_scope="warm public split call; feature loading and evaluation excluded",
        memory_scope="fresh process RSS includes loaded features and warm-up; GPU peak reset after warm-up",
        label_order=study.NAMES.tolist())
    study.atomic_json(final, study.clean(record))
    print(method, n, seconds, record["peak_gpu_gib"], flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(120, repeat=True)
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["plain", "combined"])
    p.add_argument("--n", required=True, type=int)
    a = p.parse_args()
    run(a.method, a.n)
    faulthandler.cancel_dump_traceback_later()
