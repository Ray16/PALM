"""Balanced optimization of the low-rank leakage in factor space.

``balanced_lloyd`` minimizes cross-block similarity by alternating block sums and
balanced reassignment (exact target sizes); ``fm_polish`` is a monotone
Fiduccia–Mattheyses single-move refinement on the same factor-space objective.
This module is where the *optimizer* extensions land (tunable balance slack,
multi-objective weights, spectral init).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from PALM.splitters.common.balanced_assignment import (balanced_assign,
                                                       capacity_corridor, target_sizes)
from PALM.splitters.common.fiduccia_mattheyses import fiduccia_mattheyses_loop


def corridor_assign(scores, sizes, caps, floors):
    """Assign n points to k blocks maximizing chosen scores, sizes within a corridor.

    Unlike :func:`balanced_assign` (which pins *exact* sizes), this only requires
    each block's size to lie in ``[floors[c], caps[c]]`` — giving the optimizer the
    slack to cut more cleanly.

    - **k == 2**: exact/optimal (concave in the single split point).
    - **k > 2**: a regret-based greedy *peel* (the corridor generalization of
      :func:`balanced_assign`'s k>2 branch). Blocks are peeled in order; each block
      ``c`` takes its top-preference remaining points — where preference is
      ``score[:,c] − max score over the still-unassigned blocks`` — and the count is
      the *natural* number that prefer ``c`` (``pref > 0``), clamped into
      ``[floors[c], caps[c]]`` and further clamped to keep the later blocks feasible
      (leave ≥ Σ future floors, ≤ Σ future caps). This is a heuristic (exact k>2 is a
      transportation problem, and the FM polish refines it afterwards), but it has two
      guarantees: it always yields corridor-valid sizes, and when the corridor is
      degenerate (``floors == caps == sizes``) it reproduces the exact assignment. It
      never scores below the exact peel — the corridor only frees points toward the
      block they prefer. O(n·k) with k−1 sorts, fully vectorized.
    """
    import torch

    n, k = scores.shape
    dev = scores.device
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]                # >0 => prefers block 1
        order = torch.argsort(diff, descending=True)
        m = int((diff > 0).sum().item())                  # natural block-1 count
        m = min(max(m, int(floors[1])), int(caps[1]))     # clamp into the corridor
        lab = torch.zeros(n, dtype=torch.long, device=dev)
        lab[order[:m]] = 1
        return lab

    caps = [int(x) for x in caps]
    floors = [int(x) for x in floors]
    suf_cap = [0] * (k + 1)                                # Σ caps[j], j >= idx
    suf_flr = [0] * (k + 1)                                # Σ floors[j], j >= idx
    for j in range(k - 1, -1, -1):
        suf_cap[j] = suf_cap[j + 1] + caps[j]
        suf_flr[j] = suf_flr[j + 1] + floors[j]

    remaining = torch.arange(n, device=dev)
    lab = torch.full((n,), k - 1, dtype=torch.long, device=dev)   # last block = leftovers
    for c in range(k - 1):
        R = remaining.numel()
        sub = scores[remaining]
        other = sub[:, c + 1:].max(dim=1).values          # best still-unassigned alternative
        pref = sub[:, c] - other
        order = torch.argsort(pref, descending=True)
        natural = int((pref > 0).sum().item())            # #points that prefer block c
        # keep later blocks feasible: leave >= future floors, <= future caps
        t_min = max(floors[c], R - suf_cap[c + 1])
        t_max = min(caps[c], R - suf_flr[c + 1])
        t_min = max(0, min(t_min, R))
        t_max = max(t_min, min(t_max, R))                  # guard degenerate corridors
        t = min(max(natural, t_min), t_max)
        take = order[:t]
        lab[remaining[take]] = c
        keep = torch.ones(R, dtype=torch.bool, device=dev)
        keep[take] = False
        remaining = remaining[keep]
    return lab


def interpolate_to_random(labels, alpha, seed=0):
    """Interpolate a split toward a random one, preserving exact block sizes.

    The controllable-hardness dial (Step 3). ``alpha=1`` leaves the split unchanged
    (hardest / lowest leakage); ``alpha=0`` fully randomizes it (easiest / highest
    leakage, ~in-distribution). Permutes the labels of a ``(1-alpha)`` fraction of
    points — a permutation keeps every block's size exact — so leakage rises
    monotonically toward the random baseline as ``alpha`` falls.
    """
    labels = np.asarray(labels).copy()
    n = len(labels)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    n_shuffle = int(round((1.0 - alpha) * n))
    if n_shuffle >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, n_shuffle, replace=False)
        labels[idx] = rng.permutation(labels[idx])
    return labels


def balanced_lloyd(B, splits, epsilon=0.05, n_iter=25, init_labels=None, seed=0,
                   balance_slack=0.0):
    """Balanced-Lloyd minimization of the low-rank leakage in B-space.

    Alternates: block sums from current labels, then reassign every point to the
    most-similar block subject to a size constraint. ``balance_slack`` sets how far
    block sizes may deviate from the target ratio: ``0.0`` pins exact sizes (the
    default, back-compatible); ``>0`` opens a ``(1 ± balance_slack)`` corridor the
    optimizer can exploit to lower leakage (the leakage↔balance tradeoff knob).
    O(n·r·k) per iteration. Returns the final labels (numpy int array).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    n, r = Bt.shape
    k = len(splits)
    sizes = target_sizes(n, splits)
    use_corridor = balance_slack > 0.0
    if use_corridor:
        caps, floors = capacity_corridor(n, splits, balance_slack)
    self_sim = (Bt * Bt).sum(1)

    if init_labels is None:
        gen = np.random.default_rng(seed)
        order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sizes)])
        gen.shuffle(order)
        lab = torch.as_tensor(order, dtype=torch.long, device=device)
    else:
        lab = torch.as_tensor(np.asarray(init_labels), dtype=torch.long, device=device)

    for _ in range(n_iter):
        P = torch.zeros(k, r, device=device, dtype=Bt.dtype)
        P.index_add_(0, lab, Bt)
        scores = Bt @ P.T
        scores[torch.arange(n, device=device), lab] -= self_sim   # exclude self
        new_lab = (corridor_assign(scores, sizes, caps, floors) if use_corridor
                   else balanced_assign(scores, sizes))
        if torch.equal(new_lab, lab):
            break
        lab = new_lab
    return lab.cpu().numpy()


def fm_polish(B, labels, splits: Sequence[float], epsilon: float = 0.05,
              max_moves: Optional[int] = None, tol: float = 1e-6) -> Tuple[np.ndarray, int]:
    """Monotone FM polish on the low-rank factor-space leakage. Returns ``(labels, n_moves)``."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    n, k = Bt.shape[0], len(splits)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    self_sim = (Bt * Bt).sum(1)

    P = torch.zeros(k, Bt.shape[1], device=device, dtype=Bt.dtype)
    P.index_add_(0, lab, Bt)
    S = Bt @ P.T

    caps, floors = capacity_corridor(n, splits, epsilon)
    caps_t = torch.as_tensor(caps, device=device)
    floors_t = torch.as_tensor(floors, device=device)

    def sim_col(v):
        return Bt @ Bt[v]

    lab, moves = fiduccia_mattheyses_loop(lab, S, sim_col, caps_t, floors_t,
                                          self_term=self_sim, tol=tol, max_moves=max_moves)
    return lab.cpu().numpy(), moves
