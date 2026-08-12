"""CUDA-synced wall-clock timing — the one copy.

Extracted from ``omol25_scaling`` / ``uma_scaling``. ``_time`` reports the median
over CUDA-synced repeats so launch overhead and run-to-run variance don't inflate
the first timed point; ``warmup`` runs a throwaway call so CUDA/torch and
Mt-KaHyPar are initialized before the timed region.
"""

from __future__ import annotations

import time

import numpy as np


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, reps):
    """Median wall time of ``fn()`` over ``reps`` CUDA-synced runs.

    Returns ``(t_median, last_result)``.
    """
    ts, out = [], None
    for _ in range(reps):
        _sync(); t0 = time.time(); out = fn(); _sync()
        ts.append(time.time() - t0)
    return float(np.median(ts)), out


def warmup(fn):
    """Run ``fn()`` once (discarding the result) to warm CUDA / Mt-KaHyPar."""
    _sync()
    fn()
    _sync()
