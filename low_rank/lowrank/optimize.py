"""Balanced optimization of the low-rank leakage in factor space.

``balanced_lloyd`` minimizes cross-block similarity by alternating block sums and
balanced reassignment (exact target sizes or a ``(1 ± balance_slack)`` corridor);
``fm_polish`` is a monotone Fiduccia–Mattheyses single-move / pairwise-swap
refinement on the same factor-space objective.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np

from lowrank.splitters.common.balanced_assignment import (
    _peel_corridor_order,
    _select_top_mask,
    balanced_assign,
    capacity_corridor,
    corridor_assign,
    target_sizes,
)
from lowrank.splitters.common.fiduccia_mattheyses import fiduccia_mattheyses_loop


# ── Hardness Interpolation ───────────────────────────────────────────────────


def interpolate_to_random(labels, alpha: float, seed: int = 0) -> np.ndarray:
    """Interpolate a split toward a random one, preserving exact block sizes.

    ``alpha=1`` leaves the split unchanged (hardest / lowest leakage); ``alpha=0``
    fully randomizes it. A seeded forward Fisher–Yates pass over a fixed random
    order of the points is run for its first ``m = round((1-alpha)·n)`` steps so
    perturbations are nested across ``alpha`` for a fixed ``seed``.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            f"hardness/alpha must be a finite value in [0, 1]; got {alpha!r}"
        )
    labels = np.asarray(labels).copy()
    n = len(labels)
    m = int(round((1.0 - alpha) * n))
    if m >= 2:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        js = (rng.random(n) * np.arange(1, n + 1)).astype(np.int64)
        for t in range(1, m):
            a, b = order[t], order[js[t]]
            labels[a], labels[b] = labels[b], labels[a]
    return labels


# ── Stratified Plan & Initialization Helpers ─────────────────────────────────


def _validate_optimizer_modes(init: str, linearization: str) -> None:
    if init not in ("random", "spectral", "hybrid"):
        raise ValueError(f"init must be 'random', 'spectral' or 'hybrid'; got {init!r}")
    if linearization not in ("self_excluded", "full"):
        raise ValueError(
            f"linearization must be 'self_excluded' or 'full'; got {linearization!r}"
        )


def _build_strata_plan(
    strata, n: int, splits: Sequence[float], balance_slack: float, device
):
    """Precompute per-stratum indices, exact target sizes and corridors summing to global targets."""
    import torch

    if strata is None:
        return None
    st = np.asarray(strata)
    if st.shape != (n,):
        raise ValueError(
            f"strata must have one label per row ({n}); got shape {st.shape}"
        )
    _, inv = np.unique(st, return_inverse=True)
    m_strata = int(inv.max()) + 1 if n else 0
    k = len(splits)
    global_sizes = target_sizes(n, splits)
    total = float(sum(splits))
    ratios = np.array([s / total for s in splits], dtype=np.float64)

    s_counts = np.bincount(inv, minlength=m_strata)
    raw = s_counts[:, None] * ratios[None, :]
    s_sizes = np.floor(raw).astype(int)
    frac = raw - s_sizes
    col_deficits = global_sizes - s_sizes.sum(axis=0)
    row_deficits = s_counts - s_sizes.sum(axis=1)

    pairs = [
        (-frac[s, c], s, c)
        for s in range(m_strata)
        for c in range(k)
        if row_deficits[s] > 0 and col_deficits[c] > 0
    ]
    pairs.sort()
    for _, s, c in pairs:
        if row_deficits[s] > 0 and col_deficits[c] > 0:
            s_sizes[s, c] += 1
            row_deficits[s] -= 1
            col_deficits[c] -= 1
    for s in range(m_strata):
        while row_deficits[s] > 0:
            c = int(np.argmax(col_deficits))
            s_sizes[s, c] += 1
            row_deficits[s] -= 1
            col_deficits[c] -= 1

    plan = []
    for s in range(m_strata):
        idx_np = np.flatnonzero(inv == s)
        idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        sz = s_sizes[s]
        if balance_slack > 0.0:
            t = len(idx_np) * ratios
            caps = np.maximum(np.floor(t * (1 + balance_slack) + 1e-9).astype(int), sz)
            flrs = np.minimum(np.ceil(t * (1 - balance_slack) - 1e-9).astype(int), sz)
        else:
            caps, flrs = sz.copy(), sz.copy()
        plan.append((idx_np, idx_t, sz, caps, flrs))
    return plan


def _assign_with_plan(scores, sizes, caps, floors, use_corridor: bool, strata_plan):
    import torch

    if strata_plan is None:
        return (
            corridor_assign(scores, sizes, caps, floors)
            if use_corridor
            else balanced_assign(scores, sizes)
        )
    lab = torch.empty(scores.shape[0], dtype=torch.long, device=scores.device)
    for _, idx_t, sz_s, caps_s, flrs_s in strata_plan:
        if len(idx_t) == 0:
            continue
        sub_scores = scores[idx_t]
        lab[idx_t] = (
            corridor_assign(sub_scores, sz_s, caps_s, flrs_s)
            if use_corridor
            else balanced_assign(sub_scores, sz_s)
        )
    return lab


def _random_init_labels(
    sizes, seed: int, device, strata_plan=None, n: Optional[int] = None
):
    import torch

    gen = np.random.default_rng(seed)
    if strata_plan is None:
        order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sizes)])
        gen.shuffle(order)
        return torch.as_tensor(order, dtype=torch.long, device=device)
    out = np.empty(int(n), dtype=np.int64)
    for idx_np, _, sz_s, _, _ in strata_plan:
        if len(idx_np) == 0:
            continue
        order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sz_s)])
        gen.shuffle(order)
        out[idx_np] = order
    return torch.as_tensor(out, dtype=torch.long, device=device)


def _centered_spectral_basis(ops, k: int):
    """Compute standardized top centered principal components of factor ``B`` in O(n·r²)."""
    import torch

    n, r = ops.shape
    top_m = min(r, max(k * 2, 6))
    if ops.sparse:
        mean = np.asarray(ops.B.mean(axis=0)).ravel().astype(np.float64)
        gram = (ops.B.T @ ops.B).toarray().astype(np.float64) - n * np.outer(mean, mean)
        _, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
        V = evecs[:, -top_m:][:, ::-1].copy()
        Z_base = np.asarray(ops.B @ V, dtype=np.float64) - (mean @ V)[None, :]
        basis = torch.as_tensor(Z_base, dtype=torch.float32, device=ops.device)
    else:
        mean = ops.B.mean(dim=0, dtype=torch.float64)
        gram = torch.zeros((r, r), dtype=torch.float64, device=ops.device)
        for s in range(0, n, 4096):
            blk = ops.B[s : s + 4096].double()
            gram += blk.T @ blk
        gram -= n * torch.outer(mean, mean)
        _, evecs = torch.linalg.eigh(0.5 * (gram + gram.T))
        V = evecs[:, -top_m:].flip(1)
        basis = (ops.B @ V.to(ops.B.dtype)) - (mean @ V).to(ops.B.dtype)[None, :]
    std = basis.std(dim=0, keepdim=True).clamp_min(1e-6)
    return basis / std


def _spectral_init_labels(
    ops,
    splits,
    sizes,
    caps,
    floors,
    use_corridor: bool,
    seed: int,
    strata_plan=None,
    _cached_basis=None,
):
    """Initialize fold labels along centered principal directions of factor ``B``."""
    import torch

    n, r = ops.shape
    k = len(splits)
    if r == 0 or n <= k:
        return _random_init_labels(sizes, seed, ops.device, strata_plan, n)
    Z_base_t = (
        _centered_spectral_basis(ops, k) if _cached_basis is None else _cached_basis
    )
    top_m = Z_base_t.shape[1]
    rng = np.random.default_rng(seed + 7919)
    raw_proj = rng.normal(size=(top_m, k))
    proj = np.linalg.qr(raw_proj)[0][:, :k] if top_m >= k else raw_proj
    proj_t = torch.as_tensor(proj, dtype=torch.float32, device=ops.device)
    scores = Z_base_t @ proj_t
    return _assign_with_plan(scores, sizes, caps, floors, use_corridor, strata_plan)


# ── Balanced-Lloyd Optimization ──────────────────────────────────────────────


def _batched_balanced_lloyd(
    ops,
    splits,
    *,
    n_iter: int = 25,
    seed: int = 0,
    n_restarts: int = 4,
    balance_slack: float = 0.0,
    local=None,
    fold_weights=None,
    init: str = "random",
    strata=None,
    linearization: str = "self_excluded",
    init_labels=None,
    return_diagnostics: bool = False,
):
    """Run ``n_restarts`` of balanced-Lloyd simultaneously with fused GEMMs."""
    import torch
    from .factors import validate_fold_weights

    _validate_optimizer_modes(init, linearization)
    device = ops.device
    local_term = None
    if local is not None:
        from .local import SparseTerm

        local_term = SparseTerm(local, device)
    n, r = ops.shape
    k = len(splits)
    W_np = validate_fold_weights(fold_weights, k)
    W = torch.as_tensor(W_np, device=device)
    sizes = target_sizes(n, splits)
    use_corridor = balance_slack > 0.0
    caps, floors = (
        capacity_corridor(n, splits, balance_slack)
        if use_corridor
        else (sizes, sizes)
    )
    strata_plan = _build_strata_plan(strata, n, splits, balance_slack, device)
    self_sim = ops.diagonal()
    rows = torch.arange(n, device=device)

    def cost_from_P(P_f64, lab):
        gram = (P_f64 @ P_f64.T).cpu().numpy()
        base = 0.5 * float(np.sum(W_np * gram))
        return base + (local_term.cross(lab) if local_term is not None else 0)

    if init_labels is not None:
        labs = [ops.labels(init_labels, k)]
        n_restarts = 1
    else:
        use_spectral = [
            (init == "spectral") or (init == "hybrid" and idx >= (n_restarts + 1) // 2)
            for idx in range(n_restarts)
        ]
        cached_basis = (
            _centered_spectral_basis(ops, k)
            if (any(use_spectral) and r > 0 and n > k)
            else None
        )
        labs = [
            _spectral_init_labels(
                ops,
                splits,
                sizes,
                caps,
                floors,
                use_corridor,
                seed + idx,
                strata_plan,
                _cached_basis=cached_basis,
            )
            if use_spectral[idx]
            else _random_init_labels(sizes, seed + idx, device, strata_plan, n)
            for idx in range(n_restarts)
        ]

    P_f64_list = ops.batched_sums(labs, k, precise=True)
    costs = [cost_from_P(P_f64_list[idx], labs[idx]) for idx in range(n_restarts)]
    use_f32_sums = not ops.sparse and n <= 2048
    P_mass_list = (
        ops.batched_sums(labs, k, precise=False) if use_f32_sums else P_f64_list
    )

    active = list(range(n_restarts))
    stop_reasons = ["max_iter"] * n_restarts
    iter_counts = [0] * n_restarts
    rejected_proposals = 0

    for _ in range(n_iter):
        if not active:
            break
        masses = ops.mass_from_sums([P_mass_list[i] for i in active], k)
        changed_indices, changed_labs = [], []
        for idx, M in zip(active, masses):
            iter_counts[idx] += 1
            lab = labs[idx]
            if fold_weights is None:
                scores = M
                if linearization == "self_excluded":
                    scores[rows, lab] -= self_sim
            else:
                sub = (
                    self_sim[:, None] * W[:, lab].T
                    if linearization == "self_excluded"
                    else 0.0
                )
                scores = -(M.double() @ W.T - sub)
            if local_term is not None:
                scores += local_term.block_mass(lab, k)
            new_lab = _assign_with_plan(
                scores, sizes, caps, floors, use_corridor, strata_plan
            )
            if not torch.equal(new_lab, lab):
                changed_indices.append(idx)
                changed_labs.append(new_lab)
            else:
                stop_reasons[idx] = "converged"
        if not changed_indices:
            break
        new_P_f64 = ops.batched_sums(changed_labs, k, precise=True)
        new_P_mass = (
            ops.batched_sums(changed_labs, k, precise=False)
            if use_f32_sums
            else new_P_f64
        )
        next_active = []
        for idx, new_lab, P64, Pm in zip(
            changed_indices, changed_labs, new_P_f64, new_P_mass
        ):
            new_cost = cost_from_P(P64, new_lab)
            if new_cost > costs[idx]:
                rejected_proposals += 1
                stop_reasons[idx] = "non_improving_proposal"
                continue
            labs[idx] = new_lab
            costs[idx] = new_cost
            P_f64_list[idx] = P64
            P_mass_list[idx] = Pm
            next_active.append(idx)
        active = next_active

    candidates = [
        (costs[idx], seed + idx, labs[idx].cpu().numpy()) for idx in range(n_restarts)
    ]
    if not return_diagnostics:
        return candidates
    return candidates, dict(
        rejected_lloyd_proposals=int(rejected_proposals),
        lloyd_stop_reasons=stop_reasons,
        lloyd_iterations=iter_counts,
        restart_dispersion=float(np.std(costs)) if len(costs) > 1 else 0.0,
        linearization=linearization,
    )


def balanced_lloyd(
    B,
    splits,
    epsilon=0.05,
    n_iter=25,
    init_labels=None,
    seed=0,
    balance_slack=0.0,
    local=None,
    fold_weights=None,
    init: str = "random",
    strata=None,
    linearization: str = "self_excluded",
):
    """Balanced-Lloyd minimization of the low-rank leakage in B-space.

    The legacy ``epsilon`` argument is retained for call compatibility and is
    unused here; ``balance_slack`` controls the actual corridor.
    """
    from .factors import factor_ops

    ops = factor_ops(B)
    res = _batched_balanced_lloyd(
        ops,
        splits,
        n_iter=n_iter,
        seed=seed,
        n_restarts=1,
        balance_slack=balance_slack,
        local=local,
        fold_weights=fold_weights,
        init=init,
        strata=strata,
        linearization=linearization,
        init_labels=init_labels,
    )
    return res[0][2]


# ── Monotone FM & Pairwise-Swap Refinement ───────────────────────────────────


def fm_polish(
    B,
    labels,
    splits: Sequence[float],
    epsilon: float = 0.05,
    max_moves: Optional[int] = None,
    tol: float = 1e-6,
    local=None,
    fold_weights=None,
    strata=None,
) -> Tuple[np.ndarray, int]:
    """Monotone FM polish on the low-rank factor-space leakage. Returns ``(labels, n_moves)``."""
    import torch
    from .factors import factor_ops

    ops = factor_ops(B)
    device = ops.device
    n, k = ops.shape[0], len(splits)
    lab = ops.labels(labels, k)
    initial = lab.clone()
    self_sim = ops.diagonal()

    S = ops.mass(lab, k)
    if local is not None:
        from .local import SparseTerm

        local = SparseTerm(local, device)
        S += local.block_mass(lab, k)

    def sim_col(v):
        out = ops.column(v)
        return out + local.column(v) if local is not None else out

    def checked(final_labels, moves):
        before = ops.cost(initial, k, fold_weights) + (
            local.cross(initial) if local is not None else 0
        )
        after = ops.cost(final_labels, k, fold_weights) + (
            local.cross(final_labels) if local is not None else 0
        )
        return (
            (initial.cpu().numpy(), 0)
            if after > before
            else (final_labels.cpu().numpy(), moves)
        )

    caps, floors = capacity_corridor(n, splits, epsilon)
    if fold_weights is not None:
        global_mass = ops.mass(lab, k)
        local_mass = (
            local.block_mass(lab, k)
            if local is not None
            else torch.zeros_like(global_mass)
        )
        lab, moves = _weighted_polish(
            ops,
            lab,
            global_mass,
            local_mass,
            self_sim,
            fold_weights,
            caps,
            floors,
            local,
            tol,
            max_moves,
            strata=strata,
        )
        return checked(lab, moves)
    if np.array_equal(caps, floors) or strata is not None:
        if not ops.sparse and local is None and strata is None and n > 256:
            lab, moves = _batched_swap_polish(
                ops, lab, self_sim, k, tol=tol, max_swaps=max_moves
            )
        else:
            lab, moves = swap_polish(
                sim_col,
                lab,
                S,
                self_sim,
                k,
                tol=tol,
                max_swaps=max_moves,
                sim_block=ops.block if local is None else None,
                strata=strata,
            )
        return checked(lab, moves)

    caps_t = torch.as_tensor(caps, device=device)
    floors_t = torch.as_tensor(floors, device=device)
    if not ops.sparse and local is None and n > 256:
        lab, moves = _batched_fm_polish(
            ops,
            lab,
            self_sim,
            caps_t,
            floors_t,
            tol=tol,
            max_moves=max_moves,
        )
    else:
        lab, moves = fiduccia_mattheyses_loop(
            lab,
            S,
            sim_col,
            caps_t,
            floors_t,
            self_term=self_sim,
            tol=tol,
            max_moves=max_moves,
        )
    return checked(lab, moves)


def _batched_fm_polish(
    ops,
    lab,
    self_sim,
    caps_t,
    floors_t,
    *,
    tol: float = 1e-6,
    max_moves: Optional[int] = None,
    pool_size: Optional[int] = None,
):
    """Boundary-pool FM polish for dense factors: O(C·r) per move, O(n·k·r) per pool refresh."""
    import torch

    n = lab.numel()
    k = int(caps_t.numel())
    max_moves = max_moves if max_moves is not None else n
    pool_cap = min(n, pool_size if pool_size is not None else max(256, 64 * k))
    sizes = torch.bincount(lab, minlength=k).to(torch.float32)
    moves = 0

    while moves < max_moves:
        P = ops.sums(lab, k, precise=True)
        S = ops.mass_from_sums([P], k)[0]
        can_recv = sizes < caps_t
        can_leave = sizes > floors_t
        S_recv = S.clone()
        S_recv[:, ~can_recv] = -float("inf")
        S_recv.scatter_(1, lab[:, None], -float("inf"))
        best_val, _ = S_recv.max(dim=1)
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_sim
        reduction = best_val - cur
        reduction[~can_leave[lab]] = -float("inf")
        if float(reduction.max().item()) <= tol:
            break

        c_size = min(pool_cap, n)
        pool_idx = torch.topk(reduction, c_size).indices
        B_pool = ops.B[pool_idx]
        S_pool = S[pool_idx].clone()
        lab_pool = lab[pool_idx].clone()
        self_pool = self_sim[pool_idx]
        S_pool_recv = torch.empty_like(S_pool)
        pool_moves = 0
        budget = min(max_moves - moves, c_size)

        while pool_moves < budget:
            can_recv = sizes < caps_t
            can_leave = sizes > floors_t
            S_pool_recv.copy_(S_pool)
            S_pool_recv[:, ~can_recv] = -float("inf")
            S_pool_recv.scatter_(1, lab_pool[:, None], -float("inf"))
            pv, pb = S_pool_recv.max(dim=1)
            pcur = S_pool.gather(1, lab_pool[:, None]).squeeze(1) - self_pool
            pred = pv - pcur
            pred[pb == lab_pool] = -float("inf")
            pred[~can_leave[lab_pool]] = -float("inf")
            pi = int(torch.argmax(pred).item())
            if not torch.isfinite(pred[pi]) or float(pred[pi].item()) <= tol:
                break
            a, b = int(lab_pool[pi].item()), int(pb[pi].item())
            bv = B_pool[pi]
            col_pool = B_pool @ bv
            S_pool[:, a] -= col_pool
            S_pool[:, b] += col_pool
            lab_pool[pi] = b
            lab[pool_idx[pi]] = b
            sizes[a] -= 1
            sizes[b] += 1
            moves += 1
            pool_moves += 1

        if pool_moves == 0:
            break
    return lab, moves


def _batched_swap_polish(
    ops,
    lab,
    self_sim,
    k: int,
    *,
    tol: float = 1e-6,
    max_swaps: Optional[int] = None,
    n_cand: int = 8,
    per_fold_pool: Optional[int] = None,
):
    """Boundary-pool pairwise-swap polish for dense factors at exact fold counts."""
    import torch

    n = lab.numel()
    max_swaps = max_swaps if max_swaps is not None else n
    pf_cap = per_fold_pool if per_fold_pool is not None else max(64, 32 * k)
    swaps = 0

    while swaps < max_swaps:
        P = ops.sums(lab, k, precise=True)
        S = ops.mass_from_sums([P], k)[0]
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_sim
        S_other = S.clone()
        S_other.scatter_(1, lab[:, None], -float("inf"))
        margin = S_other.max(dim=1).values - cur

        pool_parts = []
        for c in range(k):
            idx_c = torch.nonzero(lab == c).squeeze(1)
            if idx_c.numel() == 0:
                continue
            take = min(pf_cap, int(idx_c.numel()))
            top_c = torch.topk(margin[idx_c], take).indices
            pool_parts.append(idx_c[top_c])
        if len(pool_parts) < 2:
            break

        pool_idx = torch.cat(pool_parts)
        B_pool = ops.B[pool_idx]
        S_pool = S[pool_idx].clone()
        lab_pool = lab[pool_idx].clone()
        self_pool = self_sim[pool_idx]
        members = [torch.nonzero(lab_pool == c).squeeze(1) for c in range(k)]
        pool_swaps = 0
        budget = min(max_swaps - swaps, int(pool_idx.numel()))

        while pool_swaps < budget:
            pcur = S_pool.gather(1, lab_pool[:, None]).squeeze(1) - self_pool
            best = (tol, None)
            for a in range(k):
                ia = members[a]
                if ia.numel() == 0:
                    continue
                for b in range(k):
                    if a == b:
                        continue
                    ib = members[b]
                    if ib.numel() == 0:
                        continue
                    ru = S_pool[ia, b] - pcur[ia]
                    rv = S_pool[ib, a] - pcur[ib]
                    top = torch.topk(ru, min(n_cand, int(ia.numel()))).indices
                    sim_blk = B_pool[ia[top]] @ B_pool[ib].T
                    gains = ru[top, None] + rv[None, :] - 2.0 * sim_blk
                    flat = int(torch.argmax(gains.reshape(-1)).item())
                    ci, j = divmod(flat, int(ib.numel()))
                    g = float(gains[ci, j].item())
                    if g > best[0]:
                        best = (g, (int(ia[top[ci]].item()), int(ib[j].item()), a, b))
            if best[1] is None:
                break
            pu, pv, a, b = best[1]
            diff = B_pool @ (B_pool[pv] - B_pool[pu])
            S_pool[:, a] += diff
            S_pool[:, b] -= diff
            lab_pool[pu], lab_pool[pv] = b, a
            lab[pool_idx[pu]], lab[pool_idx[pv]] = b, a
            members[a] = torch.nonzero(lab_pool == a).squeeze(1)
            members[b] = torch.nonzero(lab_pool == b).squeeze(1)
            swaps += 1
            pool_swaps += 1

        if pool_swaps == 0:
            break
    return lab, swaps


def swap_polish(
    sim_col,
    lab,
    S,
    self_sim,
    k: int,
    tol: float = 1e-6,
    max_swaps: Optional[int] = None,
    n_cand: int = 8,
    sim_block=None,
    strata=None,
):
    """Greedy monotone pairwise-swap polish (block and optional stratum sizes preserved)."""
    import torch

    n = lab.numel()
    max_swaps = max_swaps if max_swaps is not None else n
    swaps = 0
    if strata is not None:
        _, st_inv = np.unique(np.asarray(strata), return_inverse=True)
        n_strata = int(st_inv.max()) + 1 if n else 0
        st_t = torch.as_tensor(st_inv, dtype=torch.long, device=lab.device)
        strata_groups = [
            [torch.nonzero((lab == c) & (st_t == s)).squeeze(1) for c in range(k)]
            for s in range(n_strata)
        ]
    else:
        strata_groups = [[torch.nonzero(lab == c).squeeze(1) for c in range(k)]]
        st_t = None

    while swaps < max_swaps:
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_sim
        best = (tol, None)
        for s_idx, members in enumerate(strata_groups):
            for a in range(k):
                for b in range(k):
                    ia, ib = members[a], members[b]
                    if a == b or ia.numel() == 0 or ib.numel() == 0:
                        continue
                    ru = S[ia, b] - cur[ia]
                    rv = S[ib, a] - cur[ib]
                    top = torch.topk(ru, min(n_cand, ia.numel())).indices
                    if sim_block is not None:
                        similarity = sim_block(ib, ia[top]).T
                        gains = ru[top, None] + rv[None, :] - 2.0 * similarity
                        flat = int(torch.argmax(gains.reshape(-1)))
                        candidate, j = divmod(flat, len(ib))
                        g = float(gains[candidate, j])
                        if g > best[0]:
                            best = (
                                g,
                                (int(ia[top[candidate]]), int(ib[j]), a, b, s_idx),
                            )
                    else:
                        for t in top.tolist():
                            u = int(ia[t])
                            gain = ru[t] + rv - 2.0 * sim_col(u)[ib]
                            j = int(torch.argmax(gain))
                            g = float(gain[j])
                            if g > best[0]:
                                best = (g, (u, int(ib[j]), a, b, s_idx))
        if best[1] is None:
            break
        u, v, a, b, s_idx = best[1]
        du, dv = sim_col(u), sim_col(v)
        S[:, a] += dv - du
        S[:, b] += du - dv
        lab[u], lab[v] = b, a
        if st_t is None:
            strata_groups[0][a] = torch.nonzero(lab == a).squeeze(1)
            strata_groups[0][b] = torch.nonzero(lab == b).squeeze(1)
        else:
            strata_groups[s_idx][a] = torch.nonzero(
                (lab == a) & (st_t == s_idx)
            ).squeeze(1)
            strata_groups[s_idx][b] = torch.nonzero(
                (lab == b) & (st_t == s_idx)
            ).squeeze(1)
        swaps += 1
    return lab, swaps


def _weighted_polish(
    ops,
    lab,
    M,
    R,
    diagonal,
    weights,
    caps,
    floors,
    local,
    tol,
    max_moves,
    strata=None,
):
    """Monotone feasible moves/swaps for an explicit symmetric fold-pair cost."""
    import torch
    from .factors import validate_fold_weights

    n, k = len(lab), len(caps)
    W = torch.as_tensor(validate_fold_weights(weights, k), device=ops.device)
    caps = torch.as_tensor(caps, device=ops.device)
    floors = torch.as_tensor(floors, device=ops.device)
    sizes = torch.bincount(lab, minlength=k)
    rows = torch.arange(n, device=ops.device)
    st_t = None
    if strata is not None:
        _, st_inv = np.unique(np.asarray(strata), return_inverse=True)
        st_t = torch.as_tensor(st_inv, dtype=torch.long, device=ops.device)
    moves = 0

    def apply(u, destination):
        source = int(lab[u])
        column = ops.column(u)
        M[:, source] -= column
        M[:, destination] += column
        if local is not None:
            column = local.column(u)
            R[:, source] -= column
            R[:, destination] += column
        sizes[source] -= 1
        sizes[destination] += 1
        lab[u] = destination

    for _ in range(n if max_moves is None else max_moves):
        cross = M.double() @ W.T - diagonal[:, None] * W[:, lab].T
        gains = cross[rows, lab, None] - cross + R - R[rows, lab, None]
        gains[rows, lab] = -torch.inf
        if st_t is None:
            feasible = (sizes[lab] - 1 >= floors[lab])[:, None] & (sizes + 1 <= caps)[
                None, :
            ]
            legal = gains.masked_fill(~feasible, -torch.inf)
            best_value, index = legal.reshape(-1).max(0)
            if float(best_value) > tol:
                apply(int(index) // k, int(index) % k)
                moves += 1
                continue
        best = (tol, None)
        for a in range(k):
            ia_all = torch.nonzero(lab == a).ravel()
            for b in range(a + 1, k):
                ib_all = torch.nonzero(lab == b).ravel()
                if not len(ia_all) or not len(ib_all):
                    continue
                if st_t is None:
                    strata_pairs = [(ia_all, ib_all)]
                else:
                    n_st = int(st_t.max()) + 1
                    strata_pairs = [
                        (ia_all[st_t[ia_all] == s], ib_all[st_t[ib_all] == s])
                        for s in range(n_st)
                    ]
                for ia, ib in strata_pairs:
                    if not len(ia) or not len(ib):
                        continue
                    candidates = ia[torch.topk(gains[ia, b], min(8, len(ia))).indices]
                    if local is None:
                        similarity = ops.block(ib, candidates).T
                        values = (
                            gains[candidates, b][:, None]
                            + gains[ib, a][None, :]
                            - 2 * W[a, b] * similarity
                        )
                        flat = int(torch.argmax(values.reshape(-1)))
                        cand_idx, j = divmod(flat, len(ib))
                        val = float(values[cand_idx, j])
                        if val > best[0]:
                            best = (
                                val,
                                (int(candidates[cand_idx]), int(ib[j]), a, b),
                            )
                    else:
                        for u in candidates.tolist():
                            edge = W[a, b] * ops.column(u)[ib] + local.column(u)[ib]
                            values = gains[u, b] + gains[ib, a] - 2 * edge
                            j = int(torch.argmax(values))
                            if float(values[j]) > best[0]:
                                best = (float(values[j]), (u, int(ib[j]), a, b))
        if best[1] is None:
            break
        u, v, a, b = best[1]
        apply(u, b)
        apply(v, a)
        moves += 1
    return lab, moves


# ── Multi-Restart Shortlist Driver ───────────────────────────────────────────


def optimize_restarts(
    B,
    splits,
    *,
    seed=0,
    n_restarts=4,
    n_iter=25,
    balance_slack=0.0,
    fm=True,
    refine_restarts=None,
    local=None,
    fold_weights=None,
    tol=1e-6,
    init: str = "random",
    strata=None,
    linearization: str = "self_excluded",
):
    """Refine a bounded shortlist before selecting the winning restart."""
    from .diagnostics import positive_integer
    from .factors import factor_ops

    positive_integer("n_restarts", n_restarts)
    if refine_restarts is not None:
        positive_integer("refine_restarts", refine_restarts)
    _validate_optimizer_modes(init, linearization)
    ops = factor_ops(B)
    local_cost = None
    if local is not None:
        from .local import SparseTerm

        local_cost = SparseTerm(local)

    def objective(labels):
        return ops.cost(labels, len(splits), fold_weights) + (
            local_cost.cross(labels) if local_cost is not None else 0
        )

    t_lloyd0 = time.perf_counter()
    candidates, lloyd_diag = _batched_balanced_lloyd(
        ops,
        splits,
        n_iter=n_iter,
        seed=seed,
        n_restarts=n_restarts,
        balance_slack=balance_slack,
        local=local,
        fold_weights=fold_weights,
        init=init,
        strata=strata,
        linearization=linearization,
        return_diagnostics=True,
    )
    lloyd_seconds = time.perf_counter() - t_lloyd0
    candidates.sort(key=lambda item: item[0])
    best_cost, winner, best_labels = candidates[0]
    before = best_cost
    records = []
    chosen_moves = 0
    count = min(n_restarts, refine_restarts or n_restarts) if fm else 0
    t_ref0 = time.perf_counter()
    for cost, candidate_seed, labels in candidates[:count]:
        refined, moves = fm_polish(
            ops,
            labels,
            splits,
            epsilon=balance_slack,
            tol=tol,
            local=local,
            fold_weights=fold_weights,
            strata=strata,
        )
        value = objective(refined)
        records.append(
            dict(seed=candidate_seed, before=cost, after=value, moves=int(moves))
        )
        if value < best_cost:
            best_cost, winner, best_labels, chosen_moves = (
                value,
                candidate_seed,
                refined,
                moves,
            )
    refine_seconds = time.perf_counter() - t_ref0
    return (
        best_labels,
        best_cost,
        dict(
            factor_storage="csr" if ops.sparse else "dense",
            optimizer_device=str(ops.device),
            winning_restart_seed=int(winner),
            best_unrefined_objective=float(before),
            refined_restarts=count,
            fm_moves=int(chosen_moves),
            refinement_candidates=records,
            init=init,
            lloyd_seconds=round(lloyd_seconds, 6),
            refine_seconds=round(refine_seconds, 6),
            **lloyd_diag,
        ),
    )
