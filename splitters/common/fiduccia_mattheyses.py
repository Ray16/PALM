"""Fiduccia–Mattheyses single-move polish, shared by both splitters.

The move-selection loop is identical for the hypergraph (exact pairwise leakage)
and low-rank (factor-space leakage) backends; only how the per-block score matrix
``S`` and the moved node's similarity column are produced differs.
:func:`fiduccia_mattheyses_loop` is that common loop;
:func:`fiduccia_mattheyses_exact` and :func:`fiduccia_mattheyses_lowrank` supply
the two score/column providers.

``S[i, c]`` = sum of similarity from point i to the members of block c. A move of
node v from block a to b changes the cross-block leakage by ``S[v,b] - S[v,a]``
(with the self term handled per backend), and is applied only when it reduces
leakage and respects the capacity/floor corridor. Monotone by construction.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .balanced_assignment import capacity_corridor
from .pairwise_similarity import pairwise_similarity, standardize_for_metric


def fiduccia_mattheyses_loop(lab, S, sim_column: Callable[[int], "object"], caps_t,
                             floors_t, self_term=None, tol: float = 1e-6,
                             max_moves: Optional[int] = None):
    """Greedy monotone FM loop. Mutates ``lab``/``S`` in place; returns ``(lab, moves)``.

    Args:
        lab: (n,) torch long tensor of current block labels.
        S: (n, k) torch tensor, ``S[i,c]`` = similarity mass of i to block c.
        sim_column: ``v -> (n,)`` torch tensor of similarity of every node to v.
        caps_t, floors_t: (k,) torch tensors — the balance corridor.
        self_term: optional (n,) torch tensor subtracted from the current-block
            score (used when ``S`` includes self-similarity); ``None`` = zeros.
        tol: minimum leakage reduction (raw similarity units) to bother moving.
        max_moves: cap on applied moves (default n).
    """
    import torch

    n, k = S.shape
    sizes = torch.bincount(lab, minlength=k).to(torch.float32)
    if self_term is None:
        self_term = torch.zeros(n, device=S.device, dtype=S.dtype)

    max_moves = max_moves if max_moves is not None else n
    moves = 0
    while moves < max_moves:
        can_recv = sizes < caps_t
        can_leave = sizes > floors_t
        S_recv = S.clone()
        S_recv[:, ~can_recv] = -float("inf")
        best_val, best_b = S_recv.max(dim=1)
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_term
        reduction = best_val - cur
        reduction[best_b == lab] = -float("inf")
        reduction[~can_leave[lab]] = -float("inf")
        v = int(torch.argmax(reduction).item())
        if not torch.isfinite(reduction[v]) or reduction[v].item() <= tol:
            break
        a, b = int(lab[v].item()), int(best_b[v].item())
        col = sim_column(v)
        S[:, a] -= col
        S[:, b] += col
        lab[v] = b
        sizes[a] -= 1
        sizes[b] += 1
        moves += 1
    return lab, moves


def fiduccia_mattheyses_exact(labels, X, splits: Sequence[float], metric: str,
                              epsilon: float = 0.05, max_moves: Optional[int] = None,
                              block: int = 4096, tol_frac: float = 1e-4
                              ) -> Tuple[np.ndarray, int, dict]:
    """FM polish on the EXACT pairwise leakage ``L = sum_{i,j diff block} sim(i,j)``.

    Returns ``(labels, n_moves, info)``.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n, k = Xt.shape[0], len(splits)
    ref = standardize_for_metric(Xt, metric)

    def sim_col(v):
        s = pairwise_similarity(ref, ref[v][None, :], metric).squeeze(1)
        s[v] = 0.0
        return s

    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    onehot = torch.zeros(n, k, device=device)
    onehot[torch.arange(n, device=device), lab] = 1.0
    # S[i, c] = sum_{u in block c} sim(i, u), self excluded (diag zeroed per block)
    S = torch.zeros(n, k, device=device)
    for s0 in range(0, n, block):
        e0 = min(s0 + block, n)
        sim = pairwise_similarity(ref[s0:e0], ref, metric)
        rg = torch.arange(s0, e0, device=device)
        sim[torch.arange(e0 - s0, device=device), rg] = 0.0
        S[s0:e0] = sim @ onehot

    caps, floors = capacity_corridor(n, splits, epsilon)
    caps_t = torch.as_tensor(caps, device=device)
    floors_t = torch.as_tensor(floors, device=device)
    lab, moves = fiduccia_mattheyses_loop(lab, S, sim_col, caps_t, floors_t,
                                          self_term=None, tol=tol_frac, max_moves=max_moves)
    return lab.cpu().numpy(), moves, {}


def fiduccia_mattheyses_lowrank(B, labels, splits: Sequence[float], epsilon: float = 0.05,
                                max_moves: Optional[int] = None, tol: float = 1e-6
                                ) -> Tuple[np.ndarray, int]:
    """FM polish on the low-rank factor-space leakage ``0.5(||s||^2 - sum||p_c||^2)``.

    Returns ``(labels, n_moves)``.
    """
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
