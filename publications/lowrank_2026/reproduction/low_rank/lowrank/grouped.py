"""Must-link groups for the low-rank splitter: keep near-twins on the same side.

The sum objective (``factor_leakage``) is dominated by the O(n²) weak background pairs,
so it will happily trade a few strong (near-duplicate / analog) pairs across the split
for many weak ones — the reason low-rank wins global L(pi) but has the worst
nearest-neighbour leakage of the three splitters (comparisons/FINDINGS.md). DataSAIL avoids
this structurally by moving whole clusters; this module optimizes capped groups
in factor space, with preservation conditional on capacity repair:

  1. ``must_link_groups`` — capped single linkage over the exact top-k similarity graph:
     edges with ``sim >= tau`` are merged strongest-first, refusing any merge that would
     exceed ``max_group`` points. Every point ends in a group; most stay singletons.
  2. The optimizer then works on *groups*. With ``g`` = sum of a group's B-rows, moving a
     group from block a to b changes the leakage by exactly ``g·p_a − g·p_b − |g|²``, and
     swapping u (in a) with v (in b) by the same expression with ``d = u − v``. So
     weighted assignment and FM operate on the exact cut of the approximate kernel,
     with whole groups retained unless balance repair dissolves them.

When greedy whole-group repair cannot restore the corridor, :func:`_repair` can
break a smallest group into singletons (reported as ``broken_groups``). A final
postcondition rejects invalid capacities; no universal feasibility or
pair-threshold guarantee is claimed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from splitters.common.balanced_assignment import capacity_corridor
from splitters.common.pairwise_similarity import pairwise_similarity, standardize_for_metric


# ---------------------------------------------------------------- grouping

def topk_neighbours(X, metric: str, k: int = 32, block: int = 2048):
    """Exact top-k similarity neighbours (self excluded). Returns ``(idx, sim)`` (n, k)."""
    import torch

    Xt = standardize_for_metric(torch.as_tensor(np.asarray(X), dtype=torch.float32), metric)
    n = Xt.shape[0]
    k = min(k, n - 1)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        S = pairwise_similarity(Xt[s:s + block], Xt, metric)
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -float("inf")
        v, i = torch.topk(S, k, dim=1)
        idx[s:s + block], sim[s:s + block] = i.numpy(), v.numpy()
    return idx, sim


def must_link_groups(X, metric: str, tau: float, max_group: int, k: int = 32,
                     nbrs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     cap_rule: str = "freeze") -> np.ndarray:
    """Capped single-linkage groups of the ``sim >= tau`` top-k graph. Returns (n,) group ids.

    ``cap_rule`` decides what happens when a merge would exceed ``max_group``:

    * ``"freeze"`` (default): the larger component stops growing at that similarity. Each group
      then keeps every top-k pair stronger than its own stopping level, and dense regions are
      grouped at a higher effective threshold than sparse ones. Without it, a tau below the
      dataset's chaining point lets weaker edges fill capped groups with arbitrary chunks of one
      giant component (full BACE at tau=0.4: 1502/1513 points in 49 cap-sized groups).
    * ``"skip"``: skip only that merge and keep merging weaker edges (the original rule).
    """
    if cap_rule not in ("freeze", "skip"):
        raise ValueError(f"cap_rule must be 'freeze' or 'skip'; got {cap_rule!r}")
    n = len(X)
    idx, sim = nbrs if nbrs is not None else topk_neighbours(X, metric, k=k)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols, vals = idx.ravel(), sim.ravel()
    keep = vals >= tau
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.argsort(-vals, kind="stable")
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    frozen = np.zeros(n, dtype=bool)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in order:                                   # Kruskal, strongest edge first
        a, b = find(rows[e]), find(cols[e])
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        if size[a] + size[b] > max_group:
            if cap_rule == "freeze":
                frozen[a] = True
            continue
        if frozen[a] or frozen[b]:
            continue
        parent[b] = a
        size[a] += size[b]
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def resolve_tau(tau, metric: str, X=None, seed: int = 0) -> float:
    """``tau`` as given, or ``"auto"``: 0.6 for tanimoto (the near-analog boundary used by the
    leakage tail metrics), else the 0.995 quantile of the dataset's pair similarities."""
    if tau != "auto":
        return float(tau)
    if metric == "tanimoto":
        return 0.6
    import torch
    from splitters.common.leakage_metrics import _pair_sim_quantiles
    ref = standardize_for_metric(torch.as_tensor(np.asarray(X), dtype=torch.float32), metric)
    return _pair_sim_quantiles(ref, metric, (0.995,), seed=seed)[0]


# ---------------------------------------------------------------- weighted optimizer

def _weighted_peel(scores, w, caps, floors):
    """Weighted corridor assignment of m items to k blocks (greedy regret peel).

    Same shape as :func:`lowrank.optimize.corridor_assign`, but each item carries a
    weight (its point count). Block c takes items in preference order up to its
    natural weight clamped into the corridor; items that would overshoot the cap are
    skipped so smaller ones can fill the remainder. The last block takes the rest;
    any residual violation is fixed later by :func:`_repair`.
    """
    m, k = scores.shape
    lab = np.full(m, k - 1, dtype=np.int64)
    remaining = np.arange(m)
    suf_cap = np.concatenate([np.cumsum(caps[::-1])[::-1], [0]])
    suf_flr = np.concatenate([np.cumsum(floors[::-1])[::-1], [0]])
    for c in range(k - 1):
        sub = scores[remaining]
        pref = sub[:, c] - sub[:, c + 1:].max(axis=1)
        order = np.argsort(-pref, kind="stable")
        W = w[remaining].sum()
        natural = w[remaining][pref > 0].sum()
        t_min = max(floors[c], W - suf_cap[c + 1])
        t_max = max(t_min, min(caps[c], W - suf_flr[c + 1]))
        target = min(max(natural, t_min), t_max)
        w_ord = w[remaining[order]]
        p = int(np.searchsorted(np.cumsum(w_ord), target, side="right"))   # prefix that fits whole
        take, cum = order[:p].tolist(), int(w_ord[:p].sum())
        if cum < target:
            for j in order[p:]:                           # preference order, skip overshoot
                wj = w[remaining[j]]
                if cum + wj <= target:
                    take.append(j)
                    cum += wj
                if cum >= target:
                    break
        if cum < t_min:                                   # top up with anything that fits
            taken = set(take)
            for j in order:
                if j not in taken and cum + w[remaining[j]] <= t_max:
                    take.append(j)
                    cum += w[remaining[j]]
                    if cum >= t_min:
                        break
        take = np.asarray(take, dtype=np.int64)
        lab[remaining[take]] = c
        remaining = np.delete(remaining, take)
    return lab


class _State:
    """Group-level factor-space state: P (k,r) block sums, M (m,k) = G·P^T, sizes."""

    def __init__(self, G, w, lab, k):
        self.G, self.w, self.lab, self.k = G, w, lab, k
        self.self_sim = np.einsum("ij,ij->i", G, G)
        onehot = np.zeros((k, len(lab)))
        onehot[lab, np.arange(len(lab))] = 1.0
        self.P = onehot @ G
        self.M = G @ self.P.T
        self.sizes = np.bincount(lab, weights=w, minlength=k).astype(np.int64)

    def leakage(self):
        s = self.P.sum(0)
        return 0.5 * float(s @ s - (self.P * self.P).sum())

    def own(self):
        return self.M[np.arange(len(self.lab)), self.lab] - self.self_sim

    def move(self, u, b):
        a = self.lab[u]
        col = self.G @ self.G[u]
        self.M[:, a] -= col
        self.M[:, b] += col
        self.P[a] -= self.G[u]
        self.P[b] += self.G[u]
        self.sizes[a] -= self.w[u]
        self.sizes[b] += self.w[u]
        self.lab[u] = b


def _repair(B, G, w, lab, k, caps, floors, members):
    """Restore the corridor: move the best-fitting item that fixes a violation; if no whole
    group fits, break the smallest multi-point group of the over-full block into singletons.
    Moves update the factor-space state incrementally (exact); only a break rebuilds it.
    Returns ``(G, w, lab, members, n_broken)``."""
    broken = 0
    st = _State(G, w, lab, k)
    for _ in range(10 * len(B) + 10):
        over = np.nonzero(st.sizes > caps)[0]
        under = np.nonzero(st.sizes < floors)[0]
        if not len(over) and not len(under):
            break
        best = None
        donors = np.flatnonzero(st.sizes > floors)
        for src in donors:
            u_src = np.flatnonzero(st.lab == src)
            w_src = st.w[u_src]
            ok_u = st.sizes[src] - w_src >= floors[src]
            own = st.M[u_src, src] - st.self_sim[u_src]
            for b in np.flatnonzero(st.sizes < caps):
                # A move must reduce an existing violation without creating one.
                # Its weight may exceed the excess/deficit when both final sizes
                # fit the corridor. Check every donor before dissolving a group.
                if b == src or (st.sizes[src] <= caps[src] and st.sizes[b] >= floors[b]):
                    continue
                ok = ok_u & (st.sizes[b] + w_src <= caps[b])
                if ok.any():
                    sc = np.where(ok, st.M[u_src, b] - own, -np.inf)
                    j = int(np.argmax(sc))
                    if best is None or sc[j] > best[0]:
                        best = (sc[j], int(u_src[j]), int(b))
        if best is not None:
            st.move(best[1], best[2])
            continue
        # No directly feasible whole-group move reduces the violation. Break a
        # smallest group in an overfull block (or a donor to an underfull one).
        sources = over if len(over) else donors
        multi = np.flatnonzero(np.isin(st.lab, sources) & (st.w > 1))
        if not len(multi):
            break
        u = int(multi[np.argmin(st.w[multi])])          # first minimum, as before
        src = st.lab[u]
        pts = members[u]
        G = np.vstack([np.delete(st.G, u, 0), B[pts]])
        w = np.concatenate([np.delete(st.w, u), np.ones(len(pts), dtype=np.int64)])
        lab = np.concatenate([np.delete(st.lab, u), np.full(len(pts), src)])
        members = members[:u] + members[u + 1:] + [[p] for p in pts]
        st = _State(G, w, lab, k)
        broken += 1
    return st.G, st.w, st.lab, members, broken


def _fm_groups(st: _State, caps, floors, tol=1e-6, max_moves=None, n_cand=8):
    """Monotone FM on groups: best feasible single move, else best feasible pair swap."""
    m, k = len(st.lab), st.k
    max_moves = max_moves if max_moves is not None else 4 * m
    moves = 0
    while moves < max_moves:
        own = st.own()
        red = st.M - own[:, None]                          # reduction of moving u -> b
        red[np.arange(m), st.lab] = -np.inf
        ok_leave = st.sizes[st.lab] - st.w >= floors[st.lab]
        ok_recv = st.sizes[None, :] + st.w[:, None] <= caps[None, :]
        red[~ok_leave] = -np.inf
        red[~ok_recv] = -np.inf
        u, b = np.unravel_index(int(np.argmax(red)), red.shape)
        if red[u, b] > tol:
            st.move(u, b)
            moves += 1
            continue
        best = (tol, None)                                 # pair swaps (exact-size corridors)
        blocks = [np.nonzero(st.lab == c)[0] for c in range(k)]
        G_blk = [st.G[ix] for ix in blocks]                # one copy per block, not per candidate
        for a in range(k):
            ia = blocks[a]
            for bb in range(k):
                ib = blocks[bb]
                if a == bb or not len(ia) or not len(ib):
                    continue
                ru = st.M[ia, bb] - own[ia]
                rv = st.M[ib, a] - own[ib]
                top = np.argsort(-ru)[:n_cand]
                uu = ia[top]
                dw = st.w[uu][:, None] - st.w[ib][None, :]  # net weight a -> bb, (cand, |ib|)
                feas = ((st.sizes[a] - dw >= floors[a]) & (st.sizes[a] - dw <= caps[a])
                        & (st.sizes[bb] + dw >= floors[bb]) & (st.sizes[bb] + dw <= caps[bb]))
                if not feas.any():
                    continue
                gain = ru[top][:, None] + rv[None, :] - 2.0 * (st.G[uu] @ G_blk[bb].T)
                gain[~feas] = -np.inf
                t, j = np.unravel_index(int(np.argmax(gain)), gain.shape)
                if gain[t, j] > best[0]:
                    best = (float(gain[t, j]), (int(uu[t]), int(ib[j]), a, bb))
        if best[1] is None:
            break
        uu, vv, a, bb = best[1]
        st.move(uu, bb)
        st.move(vv, a)
        moves += 1
    return moves


def grouped_split(B, gid, splits: Sequence[float], epsilon: float, n_restarts: int = 4,
                  n_iter: int = 25, seed: int = 0, fm: bool = True, init_labels=None):
    """Minimize factor-space leakage with groups and explicit capacity repair.

    Returns ``(labels (n,), info dict)``. ``epsilon`` is the two-sided size corridor
    (``0`` -> exact sizes, enforced by group swaps). ``init_labels`` (point labels, e.g.
    the unconstrained point-level solution) adds a warm-started restart: each group
    takes its members' majority label, then repair + Lloyd + FM as usual.
    """
    B = np.asarray(B, dtype=np.float64)
    n, k = B.shape[0], len(splits)
    m = int(gid.max()) + 1
    members = [[] for _ in range(m)]
    for i, g in enumerate(gid):
        members[g].append(i)
    G0 = np.zeros((m, B.shape[1]))
    np.add.at(G0, gid, B)
    w0 = np.bincount(gid, minlength=m).astype(np.int64)
    caps, floors = capacity_corridor(n, splits, epsilon)
    caps, floors = caps.astype(np.int64), floors.astype(np.int64)

    best = None
    starts = ([None] if init_labels is not None else []) + list(range(n_restarts))
    for r in starts:
        G, w, mem = G0.copy(), w0.copy(), list(members)
        if r is None:                                     # warm start: group majority vote
            votes = np.zeros((m, k))
            np.add.at(votes, (gid, np.asarray(init_labels)), 1.0)
            lab = votes.argmax(1)
        else:
            rng = np.random.default_rng(seed + r)
            lab = _weighted_peel(rng.random((m, k)), w, caps, floors)
        G, w, lab, mem, broken = _repair(B, G, w, lab, k, caps, floors, mem)
        for _ in range(n_iter):                           # weighted balanced Lloyd
            st = _State(G, w, lab, k)
            old_G, old_w, old_mem, old_broken = G, w, mem, broken
            sc = st.M.copy()
            sc[np.arange(len(lab)), lab] -= st.self_sim
            new = _weighted_peel(sc, w, caps, floors)
            G, w, new, mem, b2 = _repair(B, G, w, new, k, caps, floors, mem)
            if _State(G, w, new, k).leakage() > st.leakage():
                G, w, mem, broken = old_G, old_w, old_mem, old_broken
                break
            broken += b2
            if len(new) == len(lab) and np.array_equal(new, lab):
                break
            lab = new
        st = _State(G, w, lab, k)
        moves = _fm_groups(st, caps, floors) if fm else 0
        obj = st.leakage()
        if best is None or obj < best[0]:
            best = (obj, st.lab.copy(), mem, broken, moves, r is None)
    obj, lab, mem, broken, moves, warm = best
    labels = np.empty(n, dtype=np.int64)
    for g, pts in enumerate(mem):
        labels[pts] = lab[g]
    sz = np.bincount(labels, minlength=k)
    corridor_ok = bool(((sz <= caps) & (sz >= floors)).all())
    if not corridor_ok:
        raise RuntimeError("group balance repair failed to meet the requested size corridor")
    return labels, dict(factor_leakage=obj, n_groups=m, largest_group=int(w0.max()),
                        grouped_points=int((w0[gid] > 1).sum()), broken_groups=int(broken),
                        fm_moves=int(moves), warm_start_won=bool(warm), corridor_ok=corridor_ok)
