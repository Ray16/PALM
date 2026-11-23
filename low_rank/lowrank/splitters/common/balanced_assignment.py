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


def _select_top_mask(pref, t: int):
    """Return a boolean mask of length ``len(pref)`` with ``True`` at the top ``t`` positions.

    Uses ``torch.topk`` on the smaller of ``t`` or ``n - t`` when the cutoff
    boundary is strictly separated (no tie at rank ``t``), and falls back to
    ``torch.argsort(pref, descending=True)`` when a tie straddles the boundary so
    tie-breaking remains bit-identical to full sorting.
    """
    import torch

    n = pref.numel()
    mask = torch.zeros(n, dtype=torch.bool, device=pref.device)
    if t <= 0:
        return mask
    if t >= n:
        mask.fill_(True)
        return mask
    if n > 128:
        if t <= n // 2:
            vals, idx = torch.topk(pref, t + 1, largest=True, sorted=True)
            if float(vals[t - 1]) > float(vals[t]):
                mask[idx[:t]] = True
                return mask
        else:
            rem = n - t
            vals, idx = torch.topk(pref, rem + 1, largest=False, sorted=True)
            if float(vals[rem - 1]) < float(vals[rem]):
                mask.fill_(True)
                mask[idx[:rem]] = False
                return mask
    order = torch.argsort(pref, descending=True)
    mask[order[:t]] = True
    return mask


def _peel_exact_order(scores, sizes: np.ndarray, block_order: Sequence[int]):
    import torch

    n, k = scores.shape
    dev = scores.device
    remaining = torch.arange(n, device=dev)
    lab = torch.full((n,), int(block_order[-1]), dtype=torch.long, device=dev)
    for pos in range(k - 1):
        c = int(block_order[pos])
        future_blocks = [int(b) for b in block_order[pos + 1 :]]
        sub = scores[remaining]
        other = (
            sub[:, future_blocks[0]]
            if len(future_blocks) == 1
            else sub[:, future_blocks].max(dim=1).values
        )
        pref = sub[:, c] - other
        take_mask = _select_top_mask(pref, int(sizes[c]))
        lab[remaining[take_mask]] = c
        remaining = remaining[~take_mask]
    return lab


def balanced_assign(scores, sizes: np.ndarray):
    """Assign n points to k blocks of *exactly* ``sizes`` maximizing chosen scores.

    ``scores``: (n, k) torch tensor, higher = better fit of point i to block c.
    Exact/optimal for k == 2 (sort by block-preference difference); for k > 2,
    evaluates both the forward regret-ordered peel and the minority-first
    (ascending block size) peel and returns the higher-scoring assignment
    (fully vectorized). Returns a (n,) long tensor of labels.
    """
    import torch

    n, k = scores.shape
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]
        take_mask = _select_top_mask(diff, int(sizes[1]))
        lab = torch.zeros(n, dtype=torch.long, device=scores.device)
        lab[take_mask] = 1
        return lab
    forward = list(range(k))
    lab = _peel_exact_order(scores, sizes, forward)
    minority_first = [int(c) for c in np.argsort(sizes, kind="stable")]
    if minority_first != forward:
        alt = _peel_exact_order(scores, sizes, minority_first)
        if float(scores.gather(1, alt[:, None]).sum()) > float(
            scores.gather(1, lab[:, None]).sum()
        ):
            lab = alt
    return lab
