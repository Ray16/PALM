"""GPU round-robin process pool — the one copy.

The benchmark sweeps run one worker per dataset, each pinned to its own GPU so
parallel workers do not contend for a single device's memory. Extracted from
``benchmark_lowrank`` / ``benchmark_baselines`` / ``benchmark_astartes``.

A worker is a top-level (picklable) function ``worker(item, gpu_id)`` that calls
:func:`pin_gpu` **before importing torch**, then does its work.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, List


def pin_gpu(gpu_id: int) -> None:
    """Pin the current process to ``gpu_id``. Must run before any torch import."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def run_round_robin(worker: Callable, items: Iterable, workers: int = 8,
                    gpus: int = 8, flatten: bool = False) -> List:
    """Run ``worker(item, gpu_id)`` over ``items`` on a spawn pool.

    Each item ``i`` is pinned to GPU ``i % gpus``. ``spawn`` is required for CUDA
    in child processes. If ``flatten`` and each worker returns a list, the
    per-item lists are concatenated.
    """
    import multiprocessing as mp

    items = list(items)
    ctx = mp.get_context("spawn")
    jobs = [(it, i % gpus) for i, it in enumerate(items)]
    with ctx.Pool(workers) as pool:
        out = pool.starmap(worker, jobs)
    if flatten:
        out = [r for sub in out for r in sub]
    return out
