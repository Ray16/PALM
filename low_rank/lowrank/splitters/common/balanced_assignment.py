"""Split-ratio bookkeeping: exact target sizes, capacity corridors, and balanced peel assignments."""

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


def capacity_corridor(
    n: int, splits: Sequence[float], epsilon: float
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


def _peel_corridor_order(
    scores, caps: Sequence[int], floors: Sequence[int], block_order: Sequence[int]
):
    """Greedy regret-ordered peel in the specified ``block_order`` within ``[floors, caps]``."""
    import torch

    n, _ = scores.shape
    dev = scores.device
    ord_caps = [int(caps[c]) for c in block_order]
    ord_flrs = [int(floors[c]) for c in block_order]
    m_blocks = len(block_order)
    suf_cap = [0] * (m_blocks + 1)
    suf_flr = [0] * (m_blocks + 1)
    for j in range(m_blocks - 1, -1, -1):
        suf_cap[j] = suf_cap[j + 1] + ord_caps[j]
        suf_flr[j] = suf_flr[j + 1] + ord_flrs[j]

    remaining = torch.arange(n, device=dev)
    lab = torch.full((n,), int(block_order[-1]), dtype=torch.long, device=dev)
    for pos in range(m_blocks - 1):
        c = int(block_order[pos])
        future_blocks = [int(b) for b in block_order[pos + 1 :]]
        R = remaining.numel()
        sub = scores[remaining]
        other = (
            sub[:, future_blocks[0]]
            if len(future_blocks) == 1
            else sub[:, future_blocks].max(dim=1).values
        )
        pref = sub[:, c] - other
        t_min = max(0, min(max(ord_flrs[pos], R - suf_cap[pos + 1]), R))
        t_max = max(t_min, min(min(ord_caps[pos], R - suf_flr[pos + 1]), R))
        if t_min == t_max:
            t = t_min
        else:
            natural = int((pref > 0).sum().item())
            t = min(max(natural, t_min), t_max)
        take_mask = _select_top_mask(pref, t)
        lab[remaining[take_mask]] = c
        remaining = remaining[~take_mask]
    return lab


def _peel_exact_order(scores, sizes: np.ndarray, block_order: Sequence[int]):
    """Greedy regret-ordered peel with exact block sizes in ``block_order``."""
    return _peel_corridor_order(scores, sizes, sizes, block_order)


def corridor_assign(scores, sizes: Sequence[int], caps: Sequence[int], floors: Sequence[int]):
    """Assign n points to k blocks maximizing chosen scores, sizes within ``[floors, caps]``.

    - **k == 2**: exact/optimal (concave in the single split point).
    - **k > 2**: evaluates both the forward regret-based greedy peel and the
      minority-first (ascending target size) peel, returning the higher-scoring
      corridor-valid assignment. When ``floors == caps == sizes``, reproduces
      :func:`balanced_assign`.
    """
    import torch

    n, k = scores.shape
    dev = scores.device
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]
        lo = max(int(floors[1]), n - int(caps[0]))
        hi = min(int(caps[1]), n - int(floors[0]))
        if lo == hi:
            m = lo
        else:
            natural = int((diff > 0).sum().item())
            m = min(max(natural, lo), hi)
        take_mask = _select_top_mask(diff, m)
        lab = torch.zeros(n, dtype=torch.long, device=dev)
        lab[take_mask] = 1
        return lab

    forward = list(range(k))
    lab = _peel_corridor_order(scores, caps, floors, forward)
    minority_first = [int(c) for c in np.argsort(sizes, kind="stable")]
    if minority_first != forward:
        alt = _peel_corridor_order(scores, caps, floors, minority_first)
        if float(scores.gather(1, alt[:, None]).sum()) > float(
            scores.gather(1, lab[:, None]).sum()
        ):
            lab = alt
    return lab


def balanced_assign(scores, sizes: np.ndarray):
    """Assign n points to k blocks of *exactly* ``sizes`` maximizing chosen scores."""
    return corridor_assign(scores, sizes, sizes, sizes)
