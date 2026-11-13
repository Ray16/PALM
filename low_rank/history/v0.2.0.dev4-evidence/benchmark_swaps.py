"""CPU paired refinement benchmark; same input, shortlist, counts and move budget."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]  # the low_rank/ package root
sys.path.insert(0, str(ROOT))
import torch
from lowrank.factors import FactorOps
from lowrank.optimize import swap_polish

torch.set_num_threads(2)
rows = []
for n, rank in [(2000, 64), (10000, 256)]:
    for seed in range(3):
        rng = np.random.default_rng(seed)
        B = rng.normal(size=(n, rank)).astype(np.float32)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        labels = rng.permutation(np.repeat(np.arange(3), [n*8//10, n//10, n//10]))
        ops = FactorOps(B)
        # Alternate order, and warm up both operation shapes outside the clock.
        ops.column(0)
        ops.block(torch.arange(min(n, 4096)), torch.arange(8))
        for batched in ([False, True] if seed % 2 == 0 else [True, False]):
            lab = ops.labels(labels, 3)
            mass, diagonal = ops.mass(lab, 3), ops.diagonal()
            start = time.perf_counter()
            final, moves = swap_polish(ops.column, lab, mass, diagonal, 3,
                max_swaps=20, sim_block=ops.block if batched else None)
            seconds = time.perf_counter() - start
            rows.append(dict(n=n, rank=rank, seed=seed, batched=batched,
                seconds=seconds, moves=moves, cost=ops.cost(final, 3),
                counts=np.bincount(final.numpy()).tolist(),
                assignment_sha256=hashlib.sha256(final.numpy().tobytes()).hexdigest()))
            print(json.dumps(rows[-1]), flush=True)
output = dict(scope='20-swap refinement only; not full split-call timing',
              threads=2, cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
              torch=torch.__version__, numpy=np.__version__, rows=rows)
(Path(__file__).parent/'swap_results.json').write_text(json.dumps(output, indent=2)+'\n')
