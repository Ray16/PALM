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

from lowrank.splitters.common.balanced_assignment import (balanced_assign,
                                                       capacity_corridor, target_sizes)
from lowrank.splitters.common.fiduccia_mattheyses import fiduccia_mattheyses_loop


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
      can score below the exact-size greedy peel: enlarging a feasible set does
      not guarantee that a greedy heuristic finds a better answer. The outer
      guarded iteration checks the actual quadratic objective. O(n·k + k·n·log n)
      with k−1 sorts, fully vectorized.
    """
    import torch

    n, k = scores.shape
    dev = scores.device
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]                # >0 => prefers block 1
        order = torch.argsort(diff, descending=True)
        m = int((diff > 0).sum().item())                  # natural block-1 count
        # Both blocks constrain the count. With unequal targets the smaller
        # block's relative corridor can be tighter than the larger one's.
        lo = max(int(floors[1]), n - int(caps[0]))
        hi = min(int(caps[1]), n - int(floors[0]))
        m = min(max(m, lo), hi)
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

    The controllable-hardness dial. ``alpha=1`` leaves the split unchanged
    (hardest / lowest leakage); ``alpha=0`` fully randomizes it (easiest,
    ~in-distribution). A seeded forward Fisher–Yates pass over a fixed random order
    of the points is run for its first ``m = round((1-alpha)·n)`` steps; each step is
    one label transposition, so block sizes stay exact and the perturbations are
    **nested** — for a fixed ``seed``, the split at a lower ``alpha`` is the split at
    a higher ``alpha`` plus further swaps, and ``alpha=0`` is a uniform random
    permutation of the labels.

    Nesting makes the dial coherent but does **not** guarantee that leakage rises
    monotonically as ``alpha`` falls: an individual swap can lower the objective.
    The leakage/gap trend is an empirical, in-expectation relationship (see
    ``docs/FINDINGS.md`` Step 3), not a pointwise guarantee.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"hardness/alpha must be a finite value in [0, 1]; got {alpha!r}")
    labels = np.asarray(labels).copy()
    n = len(labels)
    m = int(round((1.0 - alpha) * n))
    if m >= 2:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        # j_t ~ U{0..t}, drawn for all n steps so the stream is independent of alpha
        js = (rng.random(n) * np.arange(1, n + 1)).astype(np.int64)
        for t in range(1, m):
            a, b = order[t], order[js[t]]
            labels[a], labels[b] = labels[b], labels[a]
    return labels


def balanced_lloyd(B, splits, epsilon=0.05, n_iter=25, init_labels=None, seed=0,
                   balance_slack=0.0):
    """Balanced-Lloyd minimization of the low-rank leakage in B-space.

    The legacy epsilon argument is retained for call compatibility and
    is unused here; balance_slack controls the actual corridor.

    Alternates: block sums from current labels, then reassign every point to the
    most-similar block subject to a size constraint. ``balance_slack`` sets how far
    block sizes may deviate from the target ratio: ``0.0`` pins exact sizes (the
    default, back-compatible); ``>0`` opens a ``(1 ± balance_slack)`` corridor the
    optimizer can exploit to lower leakage (the leakage↔balance tradeoff knob).
    Simultaneous self-excluded assignments need not improve the quadratic
    objective. Proposals are therefore accepted only when their block-sum norm
    does not decrease; a rejected proposal ends this restart. This is a local
    heuristic, not a global optimality certificate.
    Returns the final labels (numpy int array).
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
        new_P = torch.zeros_like(P)
        new_P.index_add_(0, new_lab, Bt)
        # Total factor sum is invariant, so leakage decreases exactly when
        # sum_c ||P_c||^2 increases. Accumulate these small norms in float64.
        if (new_P.double().square().sum() < P.double().square().sum()).item():
            break
        lab = new_lab
    return lab.cpu().numpy()


def fm_polish(B, labels, splits: Sequence[float], epsilon: float = 0.05,
              max_moves: Optional[int] = None, tol: float = 1e-6) -> Tuple[np.ndarray, int]:
    """Monotone FM polish on the low-rank factor-space leakage. Returns ``(labels, n_moves)``.

    Single-entity moves inside the ``epsilon`` corridor. When the corridor is
    degenerate (``epsilon=0`` -> exact sizes, where no single move is legal) a
    size-preserving pairwise **swap** polish runs instead, so exact sizes are kept.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    n, k = Bt.shape[0], len(splits)
    # as_tensor aliases NumPy memory on CPU; refinement must not mutate the
    # caller's initial partition (GPU copies previously hid this difference).
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device).clone()
    self_sim = (Bt * Bt).sum(1)

    P = torch.zeros(k, Bt.shape[1], device=device, dtype=Bt.dtype)
    P.index_add_(0, lab, Bt)
    S = Bt @ P.T

    def sim_col(v):
        return Bt @ Bt[v]

    caps, floors = capacity_corridor(n, splits, epsilon)
    if np.array_equal(caps, floors):
        lab, moves = swap_polish(sim_col, lab, S, self_sim, k, tol=tol, max_swaps=max_moves)
        return lab.cpu().numpy(), moves
    caps_t = torch.as_tensor(caps, device=device)
    floors_t = torch.as_tensor(floors, device=device)

    lab, moves = fiduccia_mattheyses_loop(lab, S, sim_col, caps_t, floors_t,
                                          self_term=self_sim, tol=tol, max_moves=max_moves)
    return lab.cpu().numpy(), moves


def swap_polish(sim_col, lab, S, self_sim, k: int, tol: float = 1e-6,
                max_swaps: Optional[int] = None, n_cand: int = 8):
    """Greedy monotone pairwise-swap polish (block sizes exactly preserved).

    Swapping ``u`` (block a) with ``v`` (block b) reduces the leakage by
    ``r_u(a→b) + r_v(b→a) − 2·sim(u, v)``, where ``r`` is the single-move reduction.
    Each round, for every ordered block pair, the ``n_cand`` best single movers ``u``
    are paired with their best partner ``v``; the best improving swap is applied.
    ``sim_col(v)`` returns the (n,) similarity column of ``v``.
    Mutates ``lab``/``S``; returns ``(lab, n_swaps)``.
    """
    import torch

    n = lab.numel()
    max_swaps = max_swaps if max_swaps is not None else n
    swaps = 0
    while swaps < max_swaps:
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_sim   # own-block mass, self excl.
        members = [torch.nonzero(lab == c).squeeze(1) for c in range(k)]
        best = (tol, None)
        for a in range(k):
            for b in range(k):
                ia, ib = members[a], members[b]
                if a == b or ia.numel() == 0 or ib.numel() == 0:
                    continue
                ru = S[ia, b] - cur[ia]
                rv = S[ib, a] - cur[ib]
                top = torch.topk(ru, min(n_cand, ia.numel())).indices
                for t in top.tolist():
                    u = int(ia[t])
                    gain = ru[t] + rv - 2.0 * sim_col(u)[ib]
                    j = int(torch.argmax(gain))
                    g = float(gain[j])
                    if g > best[0]:
                        best = (g, (u, int(ib[j]), a, b))
        if best[1] is None:
            break
        u, v, a, b = best[1]
        du, dv = sim_col(u), sim_col(v)
        S[:, a] += dv - du
        S[:, b] += du - dv
        lab[u], lab[v] = b, a
        swaps += 1
    return lab, swaps
