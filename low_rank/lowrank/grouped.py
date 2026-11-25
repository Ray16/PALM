"""Must-link groups for the low-rank splitter: keep near-twins on the same side.

Capped groups encourage strongly related entities to remain together:

  1. ``must_link_groups`` — capped linkage over the top-k similarity graph:
     edges with ``sim >= tau`` are merged strongest-first, refusing any merge that would
     exceed ``max_group`` points. Every point ends in a group; most stay singletons.
  2. The optimizer then works on *groups*. With ``g`` = sum of a group's B-rows, moving a
     group from block a to b changes the leakage by ``g·p_a − g·p_b − |g|²``, and
     swapping u (in a) with v (in b) by the same expression with ``d = u − v``.

When greedy whole-group repair cannot restore the corridor, :func:`_repair` can
break a smallest group into singletons or two coherent subgroups (reported as
``broken_groups``), or under ``group_policy='strict'`` fall back to a bounded
single-threaded HiGHS MILP (:func:`_whole_group_assignment`).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from lowrank.splitters.common.balanced_assignment import capacity_corridor
from lowrank.splitters.common.pairwise_similarity import standardize_for_metric


# ── Top-k Neighbour Graph & Capped Linkage Grouping ──────────────────────────


def topk_neighbours(
    X,
    metric: str,
    k: int = 32,
    block: int = 2048,
    *,
    row_block: Optional[int] = None,
    reference_block: int = 8192,
    device: Optional[str] = None,
):
    """Exact top-k similarity neighbours (self excluded). Returns ``(idx, sim)`` (n, k).

    Delegates to :func:`lowrank.neighbours.tiled_neighbours` so exact search
    uses two-dimensional similarity tiles and visible GPU acceleration rather
    than materializing ``(block, n)`` slices on CPU.
    """
    from .neighbours import tiled_neighbours

    r_block = row_block if row_block is not None else min(block, 256)
    ref_block = max(r_block, reference_block)
    return tiled_neighbours(
        X,
        metric,
        k=k,
        row_block=r_block,
        reference_block=ref_block,
        device=device,
    )


def must_link_groups(
    X,
    metric: str,
    tau: float,
    max_group: int,
    k: int = 32,
    nbrs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cap_rule: str = "freeze",
    *,
    B: Optional[np.ndarray] = None,
    linkage: str = "single",
) -> np.ndarray:
    """Capped linkage groups of the ``sim >= tau`` top-k graph. Returns (n,) group ids."""
    if cap_rule not in ("freeze", "skip"):
        raise ValueError(f"cap_rule must be 'freeze' or 'skip'; got {cap_rule!r}")
    if linkage not in ("single", "centroid"):
        raise ValueError(f"linkage must be 'single' or 'centroid'; got {linkage!r}")
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
    G_sum = (
        np.asarray(B, dtype=np.float64).copy()
        if (linkage == "centroid" and B is not None and np.asarray(B).shape[1] > 0)
        else None
    )

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in order:
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
        if G_sum is not None and (size[a] > 1 or size[b] > 1):
            na = float(np.linalg.norm(G_sum[a]))
            nb = float(np.linalg.norm(G_sum[b]))
            if na > 1e-12 and nb > 1e-12:
                if float(np.dot(G_sum[a], G_sum[b]) / (na * nb)) < tau:
                    continue
        parent[b] = a
        size[a] += size[b]
        if G_sum is not None:
            G_sum[a] += G_sum[b]
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def resolve_tau(tau, metric: str, X=None, seed: int = 0) -> float:
    """``tau`` as given, or ``"auto"``: 0.6 for tanimoto, else 0.995 pair-sim quantile."""
    if tau != "auto":
        return float(tau)
    if metric == "tanimoto":
        return 0.6
    import torch
    from lowrank.splitters.common.leakage_metrics import _pair_sim_quantiles

    ref = standardize_for_metric(
        torch.as_tensor(np.asarray(X), dtype=torch.float32), metric
    )
    return _pair_sim_quantiles(ref, metric, (0.995,), seed=seed)[0]


# ── Group-Level Factor State & Weighted Corridor Peel ────────────────────────


def _weighted_peel(scores, w, caps, floors):
    """Weighted corridor assignment of m items to k blocks (greedy regret peel)."""
    m, k = scores.shape
    lab = np.full(m, k - 1, dtype=np.int64)
    remaining = np.arange(m)
    suf_cap = np.concatenate([np.cumsum(caps[::-1])[::-1], [0]])
    suf_flr = np.concatenate([np.cumsum(floors[::-1])[::-1], [0]])
    for c in range(k - 1):
        sub = scores[remaining]
        pref = sub[:, c] - sub[:, c + 1 :].max(axis=1)
        order = np.argsort(-pref, kind="stable")
        W = w[remaining].sum()
        natural = w[remaining][pref > 0].sum()
        t_min = max(floors[c], W - suf_cap[c + 1])
        t_max = max(t_min, min(caps[c], W - suf_flr[c + 1]))
        target = min(max(natural, t_min), t_max)
        w_ord = w[remaining[order]]
        p = int(np.searchsorted(np.cumsum(w_ord), target, side="right"))
        take, cum = order[:p].tolist(), int(w_ord[:p].sum())
        if cum < target:
            for j in order[p:]:
                wj = w[remaining[j]]
                if cum + wj <= target:
                    take.append(j)
                    cum += wj
                if cum >= target:
                    break
        if cum < t_min:
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


def _contract_local(local, members, n: int):
    """Contract an (n, n) symmetric sparse local penalty onto group supernodes."""
    if local is None:
        return None
    from scipy import sparse

    m = len(members)
    rows = np.concatenate([np.asarray(pts, dtype=np.int64) for pts in members])
    cols = np.concatenate([np.full(len(pts), g, dtype=np.int64) for g, pts in enumerate(members)])
    G_map = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n, m))
    Rg = (G_map.T @ sparse.csr_matrix(local, dtype=np.float64) @ G_map).tocsr()
    Rg.setdiag(0.0)
    Rg.eliminate_zeros()
    Rg.sort_indices()
    return Rg


class _State:
    """Group-level factor-space state: ``P`` (k, r) block sums, ``M = G·P^T``, ``sizes``."""

    def __init__(self, G, w, lab, k, fold_weights=None, local_g=None):
        from scipy import sparse
        from .factors import validate_fold_weights

        self.G, self.w, self.lab, self.k = G, w, lab, k
        self.W = validate_fold_weights(fold_weights, k)
        self.self_sim = np.einsum("ij,ij->i", G, G)
        self.P = np.zeros((k, G.shape[1]), dtype=np.float64)
        for c in range(k):
            mask = lab == c
            if mask.any():
                self.P[c] = G[mask].sum(0)
        self.M = G @ self.P.T
        self.sizes = np.bincount(lab, weights=w, minlength=k).astype(np.int64)
        self.R = local_g
        if local_g is not None:
            m = len(lab)
            onehot = sparse.csr_matrix(
                (np.ones(m, dtype=np.float64), (np.arange(m), lab)), shape=(m, k)
            )
            self.R_mass = (local_g @ onehot).toarray()
            self.R_deg = np.asarray(local_g.sum(axis=1)).ravel()
        else:
            self.R_mass = None
            self.R_deg = None

    def leakage(self) -> float:
        base = 0.5 * float(np.sum(self.W * (self.P @ self.P.T)))
        if self.R is not None:
            coo = self.R.tocoo()
            base += 0.5 * float(coo.data[self.lab[coo.row] != self.lab[coo.col]].sum())
        return base

    def costs(self) -> np.ndarray:
        c = self.M @ self.W.T - self.self_sim[:, None] * self.W[:, self.lab].T
        if self.R_mass is not None:
            c = c + (self.R_deg[:, None] - self.R_mass)
        return c

    def reductions(self) -> np.ndarray:
        cost = self.costs()
        return cost[np.arange(len(self.lab)), self.lab, None] - cost

    def own(self) -> np.ndarray:
        return self.M[np.arange(len(self.lab)), self.lab] - self.self_sim

    def move(self, u: int, b: int) -> None:
        a = self.lab[u]
        col = self.G @ self.G[u]
        self.M[:, a] -= col
        self.M[:, b] += col
        if self.R is not None:
            start, end = self.R.indptr[u], self.R.indptr[u + 1]
            idx = self.R.indices[start:end]
            vals = self.R.data[start:end]
            self.R_mass[idx, a] -= vals
            self.R_mass[idx, b] += vals
        self.P[a] -= self.G[u]
        self.P[b] += self.G[u]
        self.sizes[a] -= self.w[u]
        self.sizes[b] += self.w[u]
        self.lab[u] = b


# ── Capacity Repair & Strict Whole-Group MILP Feasibility ────────────────────


class GroupInfeasibleError(ValueError):
    """Whole-group retention and the specified capacities are proven incompatible."""


class GroupFeasibilityUnresolved(RuntimeError):
    """No feasible assignment found before the solver limit; not an infeasibility proof."""


def _bisect_group_points(B, pts: Sequence[int], split_weight: int):
    """Split a group of size >2 into two coherent subgroups of sizes ``(split_weight, len - split_weight)``."""
    pts = list(pts)
    m = len(pts)
    split_weight = max(1, min(int(split_weight), m - 1))
    sub = np.asarray(B[pts], dtype=np.float64)
    if sub.shape[1] == 0:
        return [pts[:split_weight], pts[split_weight:]]
    centered = sub - sub.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    _, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
    proj = centered @ evecs[:, -1]
    order = np.argsort(proj, kind="stable")
    return [
        [pts[i] for i in order[:split_weight]],
        [pts[i] for i in order[split_weight:]],
    ]


def _repair(
    B,
    G,
    w,
    lab,
    k,
    caps,
    floors,
    members,
    allow_break: bool = True,
    fold_weights=None,
    break_mode: str = "singletons",
    local=None,
):
    """Restore the capacity corridor via greedy whole-group moves or group dissolution."""
    if break_mode not in ("singletons", "bisect"):
        raise ValueError(
            f"break_mode must be 'singletons' or 'bisect'; got {break_mode!r}"
        )
    broken = 0
    st = _State(G, w, lab, k, fold_weights, _contract_local(local, members, len(B)))
    for _ in range(10 * len(B) + 10):
        over = np.nonzero(st.sizes > caps)[0]
        under = np.nonzero(st.sizes < floors)[0]
        if not len(over) and not len(under):
            break
        best = None
        donors = np.flatnonzero(st.sizes > floors)
        if len(donors):
            all_reductions = st.reductions()
        for src in donors:
            u_src = np.flatnonzero(st.lab == src)
            w_src = st.w[u_src]
            ok_u = st.sizes[src] - w_src >= floors[src]
            reductions = all_reductions[u_src]
            for b in np.flatnonzero(st.sizes < caps):
                if b == src or (
                    st.sizes[src] <= caps[src] and st.sizes[b] >= floors[b]
                ):
                    continue
                ok = ok_u & (st.sizes[b] + w_src <= caps[b])
                if ok.any():
                    sc = np.where(ok, reductions[:, b], -np.inf)
                    j = int(np.argmax(sc))
                    if best is None or sc[j] > best[0]:
                        best = (sc[j], int(u_src[j]), int(b))
        if best is not None:
            st.move(best[1], best[2])
            continue
        if not allow_break:
            break
        sources = over if len(over) else donors
        multi = np.flatnonzero(np.isin(st.lab, sources) & (st.w > 1))
        if not len(multi):
            break
        u = int(multi[np.argmin(st.w[multi])])
        src = st.lab[u]
        pts = members[u]
        if break_mode == "bisect" and len(pts) > 2:
            if st.sizes[src] > caps[src]:
                needed = int(st.sizes[src] - caps[src])
            elif len(under):
                needed = int(floors[under[0]] - st.sizes[under[0]])
            else:
                needed = len(pts) // 2
            split_w = max(1, min(needed, len(pts) - 1))
            sub_members = _bisect_group_points(B, pts, split_w)
            sub_G = np.stack([B[p].sum(0) for p in sub_members])
            sub_w = np.array([len(p) for p in sub_members], dtype=np.int64)
            G = np.vstack([np.delete(st.G, u, 0), sub_G])
            w = np.concatenate([np.delete(st.w, u), sub_w])
            lab = np.concatenate([np.delete(st.lab, u), np.full(len(sub_members), src)])
            members = members[:u] + members[u + 1 :] + sub_members
        else:
            G = np.vstack([np.delete(st.G, u, 0), B[pts]])
            w = np.concatenate([np.delete(st.w, u), np.ones(len(pts), dtype=np.int64)])
            lab = np.concatenate([np.delete(st.lab, u), np.full(len(pts), src)])
            members = members[:u] + members[u + 1 :] + [[p] for p in pts]
        st = _State(G, w, lab, k, fold_weights, _contract_local(local, members, len(B)))
        broken += 1
    return st.G, st.w, st.lab, members, broken


def _whole_group_assignment(weights, caps, floors, preferred, time_limit: float = 5.0):
    """Find a feasible whole-group packing using a single-threaded sparse binary MILP."""
    import warnings
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    w = np.asarray(weights, dtype=np.int64)
    m, k = len(w), len(caps)
    if w.max() > max(caps):
        raise GroupInfeasibleError("a supplied group exceeds every fold capacity")
    cols = np.arange(m * k)
    rows = np.concatenate([np.repeat(np.arange(m), k), m + np.tile(np.arange(k), m)])
    values = np.concatenate([np.ones(m * k), np.repeat(w, k)])
    matrix = coo_matrix(
        (values, (rows, np.tile(cols, 2))), shape=(m + k, m * k)
    ).tocsc()
    objective = np.ones((m, k))
    objective[np.arange(m), preferred] = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected.*threads.*",
            category=RuntimeWarning,
        )
        result = milp(
            objective.ravel(),
            integrality=np.ones(m * k),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(
                matrix, np.r_[np.ones(m), floors], np.r_[np.ones(m), caps]
            ),
            options={"time_limit": time_limit, "threads": 1},
        )
    if result.x is not None:
        values = result.x.reshape(m, k)
        lab = values.argmax(1)
        counts = np.bincount(lab, weights=w, minlength=k)
        if (
            np.allclose(values.sum(1), 1, atol=1e-6)
            and np.allclose(values, np.eye(k)[lab], atol=1e-6)
            and ((counts >= floors) & (counts <= caps)).all()
        ):
            return lab
    if result.status == 2:
        raise GroupInfeasibleError(
            "whole-group packing is infeasible at the requested fold capacities; "
            "allow a size corridor or choose count_first explicitly"
        )
    raise GroupFeasibilityUnresolved(
        f"whole-group feasibility unresolved (solver status {result.status}): "
        f"{result.message}"
    )


# ── Group-Level FM Polish, Driver & Hardness Interpolation ───────────────────


def _fm_groups(st: _State, caps, floors, tol: float = 1e-6, max_moves=None, n_cand: int = 8) -> int:
    """Monotone FM on groups: best feasible single move, else best feasible pair swap."""
    m, k = len(st.lab), st.k
    max_moves = max_moves if max_moves is not None else 4 * m
    moves = 0
    while moves < max_moves:
        gains = st.reductions()
        red = gains.copy()
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
        best = (tol, None)
        blocks = [np.nonzero(st.lab == c)[0] for c in range(k)]
        G_blk = [st.G[ix] for ix in blocks]
        for a in range(k):
            ia = blocks[a]
            for bb in range(k):
                ib = blocks[bb]
                if a == bb or not len(ia) or not len(ib):
                    continue
                ru = gains[ia, bb]
                rv = gains[ib, a]
                top = np.argsort(-ru)[:n_cand]
                uu = ia[top]
                dw = st.w[uu][:, None] - st.w[ib][None, :]
                feas = (
                    (st.sizes[a] - dw >= floors[a])
                    & (st.sizes[a] - dw <= caps[a])
                    & (st.sizes[bb] + dw >= floors[bb])
                    & (st.sizes[bb] + dw <= caps[bb])
                )
                if not feas.any():
                    continue
                gain = (
                    ru[top][:, None]
                    + rv[None, :]
                    - 2.0 * st.W[a, bb] * (st.G[uu] @ G_blk[bb].T)
                )
                if st.R is not None:
                    gain = gain - 2.0 * st.R[uu][:, ib].toarray()
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


def _expand_group_labels(members: Sequence[Sequence[int]], group_labels: np.ndarray, n: int) -> np.ndarray:
    labels = np.empty(n, dtype=np.int64)
    for g, pts in enumerate(members):
        labels[pts] = group_labels[g]
    return labels


def grouped_split(
    B,
    gid,
    splits: Sequence[float],
    epsilon: float,
    n_restarts: int = 4,
    n_iter: int = 25,
    seed: int = 0,
    fm: bool = True,
    init_labels=None,
    group_policy: str = "count_first",
    feasibility_seconds: float = 5.0,
    fold_weights=None,
    refine_restarts=None,
    break_mode: str = "singletons",
    local=None,
):
    """Minimize factor-space leakage with groups and explicit capacity repair."""
    if group_policy not in ("count_first", "strict"):
        raise ValueError("group_policy must be count_first or strict")
    if break_mode not in ("singletons", "bisect"):
        raise ValueError("break_mode must be 'singletons' or 'bisect'")
    if not np.isfinite(feasibility_seconds) or feasibility_seconds <= 0:
        raise ValueError("feasibility_seconds must be finite and positive")
    if refine_restarts is not None:
        from .diagnostics import positive_integer

        positive_integer("refine_restarts", refine_restarts)
    B = np.asarray(B, dtype=np.float64)
    gid = np.asarray(gid)
    if gid.shape != (len(B),) or not np.issubdtype(gid.dtype, np.integer):
        raise ValueError("one integer group id is required per factor row")
    _, gid = np.unique(gid, return_inverse=True)
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

    def in_corridor(lab_vec, w_vec):
        sz = np.bincount(lab_vec, weights=w_vec, minlength=k)
        return bool(((sz >= floors) & (sz <= caps)).all())

    best = None
    candidates = []
    feasibility_solves = 0
    rejected_capacity_proposals = 0
    strict = group_policy == "strict"
    starts = ([None] if init_labels is not None else []) + list(range(n_restarts))

    for r in starts:
        G, w, mem = G0.copy(), w0.copy(), list(members)
        if r is None:
            votes = np.zeros((m, k))
            np.add.at(votes, (gid, np.asarray(init_labels)), 1.0)
            lab = votes.argmax(1)
        else:
            rng = np.random.default_rng(seed + r)
            lab = _weighted_peel(rng.random((m, k)), w, caps, floors)
        G, w, lab, mem, broken = _repair(
            B,
            G,
            w,
            lab,
            k,
            caps,
            floors,
            mem,
            allow_break=not strict,
            fold_weights=fold_weights,
            break_mode=break_mode,
            local=local,
        )
        if strict and not in_corridor(lab, w):
            if best is not None:
                lab = best[1].copy()
            else:
                lab = _whole_group_assignment(
                    w, caps, floors, lab, time_limit=feasibility_seconds
                )
                feasibility_solves += 1
        for _ in range(n_iter):
            st = _State(G, w, lab, k, fold_weights, _contract_local(local, mem, n))
            old_G, old_w, old_mem, old_broken = G, w, mem, broken
            new = _weighted_peel(-st.costs(), w, caps, floors)
            G, w, new, mem, b2 = _repair(
                B,
                G,
                w,
                new,
                k,
                caps,
                floors,
                mem,
                allow_break=not strict,
                fold_weights=fold_weights,
                break_mode=break_mode,
                local=local,
            )
            if strict and not in_corridor(new, w):
                rejected_capacity_proposals += 1
                G, w, mem, broken = old_G, old_w, old_mem, old_broken
                break
            if (
                _State(G, w, new, k, fold_weights, _contract_local(local, mem, n)).leakage()
                > st.leakage()
            ):
                G, w, mem, broken = old_G, old_w, old_mem, old_broken
                break
            broken += b2
            if len(new) == len(lab) and np.array_equal(new, lab):
                break
            lab = new
        st = _State(G, w, lab, k, fold_weights, _contract_local(local, mem, n))
        obj = st.leakage()
        candidates.append((obj, st, mem, broken, r is None))
        if best is None or obj < best[0]:
            best = (obj, st.lab.copy(), mem, broken, 0, r is None)

    candidates.sort(key=lambda item: item[0])
    refined_count = (
        min(len(candidates), refine_restarts or len(candidates)) if fm else 0
    )
    for before, st, mem, broken, warm in candidates[:refined_count]:
        initial = st.lab.copy()
        moves = _fm_groups(st, caps, floors)
        obj = st.leakage()
        if obj > before:
            st = _State(st.G, st.w, initial, k, fold_weights, st.R)
            obj, moves = before, 0
        if obj < best[0]:
            best = (obj, st.lab.copy(), mem, broken, moves, warm)

    obj, lab, mem, broken, moves, warm = best
    labels = _expand_group_labels(mem, lab, n)
    sz = np.bincount(labels, minlength=k)
    corridor_ok = bool(((sz <= caps) & (sz >= floors)).all())
    if not corridor_ok:
        raise RuntimeError(
            "group balance repair failed to meet the requested size corridor"
        )
    minimum, maximum = np.full(m, k), np.full(m, -1)
    np.minimum.at(minimum, gid, labels)
    np.maximum.at(maximum, gid, labels)
    intact = bool(np.array_equal(minimum, maximum))
    if strict and not intact:
        raise RuntimeError("strict group-preservation postcondition failed")
    return labels, dict(
        factor_leakage=obj,
        n_groups=m,
        largest_group=int(w0.max()),
        grouped_points=int((w0[gid] > 1).sum()),
        broken_groups=int(broken),
        fm_moves=int(moves),
        warm_start_won=bool(warm),
        corridor_ok=corridor_ok,
        group_policy=group_policy,
        all_groups_preserved=bool(intact),
        refined_group_restarts=refined_count,
        feasibility_solves=feasibility_solves,
        rejected_capacity_proposals=rejected_capacity_proposals,
    )


def interpolate_groups_to_random(
    labels,
    gid,
    alpha: float,
    splits: Sequence[float],
    epsilon: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Interpolate a grouped split toward random while keeping every intact group whole."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"hardness/alpha must be a finite value in [0, 1]; got {alpha!r}"
        )
    labels = np.asarray(labels, dtype=np.int64).copy()
    gid = np.asarray(gid)
    _, inv = np.unique(gid, return_inverse=True)
    n, k = len(labels), len(splits)
    m = int(inv.max()) + 1 if n else 0
    if m < 2 or alpha == 1.0:
        return labels
    members = [[] for _ in range(m)]
    for i, g in enumerate(inv):
        members[g].append(i)
    w = np.array([len(pts) for pts in members], dtype=np.int64)
    g_lab = np.array([labels[pts[0]] for pts in members], dtype=np.int64)
    for g, pts in enumerate(members):
        if not np.all(labels[pts] == g_lab[g]):
            raise ValueError(
                "interpolate_groups_to_random requires intact initial groups"
            )
    caps, floors = capacity_corridor(n, splits, epsilon)
    caps, floors = caps.astype(np.int64), floors.astype(np.int64)
    sizes = np.bincount(g_lab, weights=w, minlength=k).astype(np.int64)
    steps = int(round((1.0 - alpha) * m))
    rng = np.random.default_rng(seed)
    order = rng.permutation(m)
    partners = (rng.random(m) * np.arange(1, m + 1)).astype(np.int64)
    for t in range(1, steps):
        u, v = int(order[t]), int(order[partners[t]])
        a, b = int(g_lab[u]), int(g_lab[v])
        if a == b:
            continue
        dw = w[u] - w[v]
        if (
            floors[a] <= sizes[a] - dw <= caps[a]
            and floors[b] <= sizes[b] + dw <= caps[b]
        ):
            sizes[a] -= dw
            sizes[b] += dw
            g_lab[u], g_lab[v] = b, a
    return _expand_group_labels(members, g_lab, n)
