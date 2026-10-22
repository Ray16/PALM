"""Split-ratio bookkeeping: exact target sizes, the capacity corridor, and a
balanced greedy assignment.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def target_sizes(n: int, splits: Sequence[float]) -> np.ndarray:
    """Exact per-block sizes at the requested ratio, summing to ``n``.

    Uses largest-remainder rounding so the sizes sum to exactly ``n``.
    """
    total = float(sum(splits))
    raw = np.array([n * s / total for s in splits])
    sizes = np.floor(raw).astype(int)
    remainder = n - int(sizes.sum())
    for c in np.argsort(-(raw - sizes))[:remainder]:
        sizes[c] += 1
    return sizes


def capacity_corridor(n: int, splits: Sequence[float], epsilon: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-block ``(caps, floors)`` for the ``(1 +/- epsilon)`` balance corridor.

    With real-valued targets ``t_c = n * s_c/total``: ``caps = floor(t_c(1+epsilon))``
    and ``floors = ceil(t_c(1-epsilon))``, so every size inside the corridor is
    within ``epsilon`` of its target. Feasibility is resolved globally by anchoring
    on the integer :func:`target_sizes` (which sum to ``n``): each corridor is
    widened only as far as needed to contain that block's integer target, so
    ``epsilon=0`` gives ``caps == floors == target_sizes`` (exact sizes).
    """
    total = float(sum(splits))
    t = np.array([n * s / total for s in splits])
    sizes = target_sizes(n, splits)
    caps = np.maximum(np.floor(t * (1 + epsilon) + 1e-9).astype(int), sizes)
    floors = np.minimum(np.ceil(t * (1 - epsilon) - 1e-9).astype(int), sizes)
    return caps, floors


def balanced_assign(scores, sizes: np.ndarray):
    """Assign n points to k blocks of *exactly* ``sizes`` maximizing chosen scores.

    ``scores``: (n, k) torch tensor, higher = better fit of point i to block c.
    Exact/optimal for k == 2 (sort by block-preference difference); greedy
    regret-ordered "peel" for k > 2 (fully vectorized, k-1 sorts). Returns a
    (n,) long tensor of labels.
    """
    import torch

    n, k = scores.shape
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]
        order = torch.argsort(diff, descending=True)
        lab = torch.zeros(n, dtype=torch.long, device=scores.device)
        lab[order[:int(sizes[1])]] = 1
        return lab
    dev = scores.device
    remaining = torch.arange(n, device=dev)
    lab = torch.full((n,), k - 1, dtype=torch.long, device=dev)
    for c in range(k - 1):
        sub = scores[remaining]
        other = sub[:, c + 1:].max(dim=1).values
        pref = sub[:, c] - other
        order = torch.argsort(pref, descending=True)
        take = order[:int(sizes[c])]
        lab[remaining[take]] = c
        keep = torch.ones(remaining.numel(), dtype=torch.bool, device=dev)
        keep[take] = False
        remaining = remaining[keep]
    return lab
