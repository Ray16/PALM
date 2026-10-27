"""Native CPU backend timings and sampled recall on prepared public fingerprints."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--package", type=Path, required=True)
p.add_argument("--run", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.package))
from lowrank.neighbours import audited_neighbours
import torch
if torch.cuda.is_available() or len(os.sched_getaffinity(0)) != 2 or torch.get_num_threads() != 2:
    raise RuntimeError("native check requires two CPU slots, two threads and hidden CUDA")
records = []
for dataset in ("moleculenet_bbbp", "moleculenet_tox21"):
    X = np.load(a.run / "cache" / f"{dataset}.npz")["X"]
    for backend in ("tiled", "nndescent", "auto"):
        start = time.perf_counter()
        (idx, sim), info = audited_neighbours(X, "tanimoto", 32, backend=backend,
            seed=0, candidates=64, n_jobs=1, audit_queries=64, minimum_recall=.98)
        if idx.shape != (len(X), 32) or not np.isfinite(sim).all() or np.any(idx == np.arange(len(X))[:, None]):
            raise RuntimeError("invalid native neighbour output")
        records.append(dict(dataset=dataset, n=len(X), backend=backend,
                            wall_seconds=time.perf_counter() - start, diagnostics=info))
        print(dataset, backend, round(records[-1]["wall_seconds"], 3),
              info["selected_backend"], info["index_recall"], flush=True)
out = a.run / "neighbour_native.json"
out.write_text(json.dumps(dict(cpu_affinity=sorted(os.sched_getaffinity(0)), threads=2,
    protocol="same CPU interpreter; exact, ANN, auto in fixed order; first ANN includes compilation; sampled recall only",
    records=records), indent=2, allow_nan=False) + "\n")
