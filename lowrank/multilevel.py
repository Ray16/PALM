"""Multilevel FM refinement for the low-rank leakage objective (Direction #1).

The flat optimizer is best-of-restarts ``balanced_lloyd`` + a single-move
``fm_polish``. Single-move FM is a purely *local* descent: it gets stuck in local
minima it cannot escape one node at a time, and the caller caps it at n<=200k so
the largest-n regime (the low-rank method's whole moat) gets no polish at all.

This module is a **coarsen -> refine -> uncoarsen** V-cycle that fixes both. The
key algebraic fact it exploits:

    factor_leakage(B, labels) = 0.5 (||s||^2 - sum_c ||p_c||^2),   p_c = sum_{i in c} B_i

is *linear in the B rows*. So contracting a set of rows into one super-node whose
B is the **sum** of its members' B rows preserves the objective **exactly**; we
only carry a per-super-node ``weight`` = how many original rows it stands for, and
the balance constraint becomes a constraint on summed weights. Refining on the
coarse graph moves whole clusters at once (escaping local minima), then each
finer level polishes the projection.

Design guarantees (k = 2, the split arity used throughout PALM):
- **Never worse than flat.** ``multilevel_split(..., seed_labels=L)`` also runs the
  finest-level (unit-weight) FM from ``L`` and returns whichever candidate has
  lower leakage. With unit weights the weighted FM here is identical to
  ``optimize.fm_polish``, so the returned split is <= flat-FM-from-``L`` by
  construction.
- **Balance stays in the corridor.** A max-super-node-weight cap keeps every
  level's assignment inside the ``(1 +/- epsilon)`` corridor; projection preserves
  weighted block sizes exactly and the weighted FM only makes corridor-respecting
  moves, so the finest candidate is corridor-valid by construction (no
  imbalance-for-leakage cheating).
- **No size cap.** Everything is O(n.r) memory / O(n.r) per FM move, matmuls are
  blocked, so it runs at any n.

k > 2 falls back to a flat weighted FM from ``seed_labels`` (== current behaviour).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from PALM.splitters.common.balanced_assignment import (capacity_corridor,
                                                       target_sizes)


# --------------------------------------------------------------------------- #
# small torch helpers
# --------------------------------------------------------------------------- #
def _obj(Bt, lab, k):
    """factor_leakage in torch (no numpy roundtrip): 0.5(||s||^2 - sum_c||p_c||^2)."""
    import torch
    P = torch.zeros(k, Bt.shape[1], device=Bt.device, dtype=Bt.dtype)
    P.index_add_(0, lab, Bt)
    s = P.sum(0)
    return 0.5 * float((s @ s) - (P * P).sum())


def _weighted_sizes(lab, weights, k):
    import torch
    ws = torch.zeros(k, device=lab.device, dtype=weights.dtype)
    ws.index_add_(0, lab, weights)
    return ws


# --------------------------------------------------------------------------- #
# coarsening: heavy-edge matching in B-space + contraction
# --------------------------------------------------------------------------- #
def _match(Bt, weights, seed, max_node_weight, block=1024):
    """Heavy-edge matching: pair each row with a similar row, respecting a weight cap.

    Returns ``(group, n_super)`` where ``group[i]`` is the super-node id of row i.
    Similarity is ``B_i . B_j`` (the leakage kernel). Uses a blocked nearest-neighbour
    pass (never forms the full n x n matrix), then a vectorised mutual-NN match
    followed by a greedy pass over the leftovers, ordered by edge weight. Pairs are
    refused when the merged weight would exceed ``max_node_weight`` (keeps the
    balance corridor feasible at every level).
    """
    import torch

    n = Bt.shape[0]
    nn = torch.empty(n, dtype=torch.long, device=Bt.device)
    wv = torch.empty(n, dtype=Bt.dtype, device=Bt.device)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = Bt[s:e] @ Bt.T                                   # (be, n)
        rows = torch.arange(e - s, device=Bt.device)
        sim[rows, torch.arange(s, e, device=Bt.device)] = -float("inf")   # exclude self
        wv[s:e], nn[s:e] = sim.max(1)

    matched = torch.zeros(n, dtype=torch.bool, device=Bt.device)
    group = torch.full((n,), -1, dtype=torch.long, device=Bt.device)
    w = weights

    # mutual nearest neighbours, vectorised, one pair per edge (i < nn[i]), weight ok
    idx = torch.arange(n, device=Bt.device)
    mutual = (nn[nn] == idx) & (idx < nn) & ((w + w[nn]) <= max_node_weight)
    pi = idx[mutual]
    pj = nn[pi]
    g = 0
    if pi.numel():
        gids = torch.arange(pi.numel(), device=Bt.device)
        group[pi] = gids
        group[pj] = gids
        matched[pi] = True
        matched[pj] = True
        g = int(pi.numel())

    # greedy leftover pass over unmatched, strongest edges first (fixed nn)
    un = idx[~matched]
    order = un[torch.argsort(wv[un], descending=True)]
    nn_c = nn.cpu().numpy()
    w_c = w.cpu().numpy()
    matched_c = matched.cpu().numpy()
    group_c = group.cpu().numpy()
    for i in order.cpu().numpy():
        if matched_c[i]:
            continue
        j = int(nn_c[i])
        if matched_c[j] or j == i or (w_c[i] + w_c[j]) > max_node_weight:
            continue
        group_c[i] = g
        group_c[j] = g
        matched_c[i] = matched_c[j] = True
        g += 1
    # remaining rows become singleton super-nodes
    rem = np.where(~matched_c)[0]
    group_c[rem] = np.arange(g, g + len(rem))
    g += len(rem)
    return torch.as_tensor(group_c, device=Bt.device, dtype=torch.long), g


def _contract(Bt, weights, group, n_super):
    import torch
    Bc = torch.zeros(n_super, Bt.shape[1], device=Bt.device, dtype=Bt.dtype)
    Bc.index_add_(0, group, Bt)
    wc = torch.zeros(n_super, device=Bt.device, dtype=weights.dtype)
    wc.index_add_(0, group, weights)
    return Bc, wc


# --------------------------------------------------------------------------- #
# weighted assignment + weighted Lloyd (k = 2)  -- coarsest initial partition
# --------------------------------------------------------------------------- #
def _weighted_assign_k2(scores, weights, tgt1):
    """Assign to 2 blocks maximising preference, block-1 weight filled up to ``tgt1``.

    Sort by preference for block 1; take the strongest nodes into block 1 until the
    next would overshoot ``tgt1`` (so block-1 weight lands in ``(tgt1 - max_wt, tgt1]``,
    inside the corridor given the weight cap).
    """
    import torch
    diff = scores[:, 1] - scores[:, 0]
    order = torch.argsort(diff, descending=True)
    cum = torch.cumsum(weights[order], 0)
    take = cum <= tgt1
    lab = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    lab[order[take]] = 1
    return lab


def _weighted_lloyd_k2(Bt, weights, splits, n_iter, n_restarts, seed):
    import torch
    n, r = Bt.shape
    self_sim = (Bt * Bt).sum(1)
    tgt = target_sizes(int(weights.sum().item()), splits)
    tgt1 = float(tgt[1])
    best_lab, best_obj = None, float("inf")
    for rs in range(n_restarts):
        gen = torch.Generator(device=Bt.device).manual_seed(seed + rs)
        order = torch.randperm(n, generator=gen, device=Bt.device)
        cum = torch.cumsum(weights[order], 0)
        lab = torch.zeros(n, dtype=torch.long, device=Bt.device)
        lab[order[cum <= tgt1]] = 1                       # random corridor init
        for _ in range(n_iter):
            P = torch.zeros(2, r, device=Bt.device, dtype=Bt.dtype)
            P.index_add_(0, lab, Bt)
            scores = Bt @ P.T
            scores[torch.arange(n, device=Bt.device), lab] -= self_sim
            new = _weighted_assign_k2(scores, weights, tgt1)
            if torch.equal(new, lab):
                break
            lab = new
        obj = _obj(Bt, lab, 2)
        if obj < best_obj:
            best_obj, best_lab = obj, lab
    return best_lab


# --------------------------------------------------------------------------- #
# weighted single-move FM  (unit weights == optimize.fm_polish exactly)
# --------------------------------------------------------------------------- #
def _weighted_fm(Bt, lab, weights, caps, floors, self_sim, tol=1e-6, max_moves=None):
    """Monotone single-move FM on factor-space leakage with weighted capacities.

    Move v: a->b allowed iff ``wsize[b]+w[v] <= caps[b]`` and ``wsize[a]-w[v] >=
    floors[a]``; applied only when it lowers leakage. With all weights == 1 this is
    identical to :func:`PALM.lowrank.optimize.fm_polish`.
    """
    import torch
    n, r = Bt.shape
    k = caps.shape[0]
    P = torch.zeros(k, r, device=Bt.device, dtype=Bt.dtype)
    P.index_add_(0, lab, Bt)
    S = Bt @ P.T
    wsize = _weighted_sizes(lab, weights, k)
    max_moves = max_moves if max_moves is not None else n
    moves = 0
    ar = torch.arange(n, device=Bt.device)
    while moves < max_moves:
        recv = (wsize[None, :] + weights[:, None]) <= caps[None, :]    # (n,k) per-node feasibility
        S_recv = S.clone()
        S_recv[~recv] = -float("inf")
        best_val, best_b = S_recv.max(1)
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_sim
        reduction = best_val - cur
        reduction[best_b == lab] = -float("inf")
        can_leave = (wsize[lab] - weights) >= floors[lab]
        reduction[~can_leave] = -float("inf")
        v = int(torch.argmax(reduction).item())
        if not torch.isfinite(reduction[v]) or reduction[v].item() <= tol:
            break
        a, b = int(lab[v].item()), int(best_b[v].item())
        col = Bt @ Bt[v]
        S[:, a] -= col
        S[:, b] += col
        lab[v] = b
        wsize[a] -= weights[v]
        wsize[b] += weights[v]
        moves += 1
    return lab, moves


# --------------------------------------------------------------------------- #
# the V-cycle
# --------------------------------------------------------------------------- #
def multilevel_split(B, splits: Sequence[float], epsilon: float = 0.05, seed: int = 0,
                     n_iter: int = 25, n_restarts: int = 4, coarsest: int = 256,
                     max_levels: int = 40, seed_labels=None,
                     fm_max_moves: Optional[int] = None) -> np.ndarray:
    """Multilevel FM refinement of the low-rank leakage. Returns numpy int labels.

    ``coarsest``: stop coarsening at <= this many super-nodes. ``seed_labels``: an
    existing split (e.g. best-of-restarts Lloyd) to also refine and beat — pass it to
    guarantee the result is no worse than flat FM from that seed. k != 2 falls back
    to a flat weighted FM from ``seed_labels``.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(np.asarray(B), dtype=torch.float32, device=device)
    n, _ = Bt.shape
    k = len(splits)
    caps_np, floors_np = capacity_corridor(n, splits, epsilon)
    caps = torch.as_tensor(caps_np, device=device, dtype=torch.float32)
    floors = torch.as_tensor(floors_np, device=device, dtype=torch.float32)
    self_sim = (Bt * Bt).sum(1)
    ones = torch.ones(n, device=device, dtype=torch.float32)

    def _fm_from(lab_t, Bl, w, ss):
        return _weighted_fm(Bl, lab_t.clone(), w, caps, floors, ss,
                            max_moves=fm_max_moves)[0]

    # ---- k != 2: flat weighted FM from seed (== current behaviour) --------- #
    if k != 2:
        base = (np.asarray(seed_labels) if seed_labels is not None
                else _lloyd_fallback(B, splits, seed))
        lab = torch.as_tensor(base, dtype=torch.long, device=device)
        lab = _fm_from(lab, Bt, ones, self_sim)
        return lab.cpu().numpy()

    # ---- build the coarsening hierarchy ------------------------------------ #
    tgt = target_sizes(n, splits)
    max_node_weight = max(2, int(0.5 * epsilon * float(min(tgt))))
    level_B: List = [Bt]
    level_w: List = [ones]
    mappings: List = []
    curB, curw = Bt, ones
    while curB.shape[0] > coarsest and len(mappings) < max_levels:
        group, m = _match(curB, curw, seed + len(mappings), max_node_weight)
        if m >= curB.shape[0]:                     # no contraction possible -> stop
            break
        curB, curw = _contract(curB, curw, group, m)
        mappings.append(group)
        level_B.append(curB)
        level_w.append(curw)

    # ---- initial partition at the coarsest level --------------------------- #
    lab = _weighted_lloyd_k2(curB, curw, splits, n_iter, n_restarts, seed)
    lab = _fm_from(lab, curB, curw, (curB * curB).sum(1))

    # ---- uncoarsen: project down + weighted FM at every finer level -------- #
    for lvl in range(len(mappings) - 1, -1, -1):
        lab = lab[mappings[lvl]]                    # project super-node labels to children
        Bl, w = level_B[lvl], level_w[lvl]
        lab = _fm_from(lab, Bl, w, (Bl * Bl).sum(1))

    cand = lab
    best = cand
    best_obj = _obj(Bt, cand, 2)

    # ---- guarantee: also refine the seed and keep the better --------------- #
    if seed_labels is not None:
        s = torch.as_tensor(np.asarray(seed_labels), dtype=torch.long, device=device)
        s = _fm_from(s, Bt, ones, self_sim)
        s_obj = _obj(Bt, s, 2)
        if s_obj < best_obj:
            best, best_obj = s, s_obj

    return best.cpu().numpy()


def _lloyd_fallback(B, splits, seed):
    """balanced_lloyd import kept local to avoid a cycle at module import time."""
    from .optimize import balanced_lloyd
    return balanced_lloyd(B, splits, seed=seed)
