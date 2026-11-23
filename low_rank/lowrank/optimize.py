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

from lowrank.splitters.common.balanced_assignment import (
    _select_top_mask,
    balanced_assign,
    capacity_corridor,
    target_sizes,
)
from lowrank.splitters.common.fiduccia_mattheyses import fiduccia_mattheyses_loop


def _peel_corridor_order(scores, caps, floors, block_order):
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
        natural = int((pref > 0).sum().item())
        t_min = max(ord_flrs[pos], R - suf_cap[pos + 1])
        t_max = min(ord_caps[pos], R - suf_flr[pos + 1])
        t_min = max(0, min(t_min, R))
        t_max = max(t_min, min(t_max, R))
        t = min(max(natural, t_min), t_max)
        take_mask = _select_top_mask(pref, t)
        lab[remaining[take_mask]] = c
        remaining = remaining[~take_mask]
    return lab


def corridor_assign(scores, sizes, caps, floors):
    """Assign n points to k blocks maximizing chosen scores, sizes within a corridor.

    Unlike :func:`balanced_assign` (which pins *exact* sizes), this only requires
    each block's size to lie in ``[floors[c], caps[c]]`` — giving the optimizer the
    slack to cut more cleanly.

    - **k == 2**: exact/optimal (concave in the single split point).
    - **k > 2**: evaluates both the forward regret-based greedy peel and the
      minority-first (ascending target size) peel, returning the higher-scoring
      corridor-valid assignment. When the corridor is degenerate
      (``floors == caps == sizes``) it reproduces :func:`balanced_assign`.
      O(n·k + k·n·log n), fully vectorized.
    """
    import torch

    n, k = scores.shape
    dev = scores.device
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]  # >0 => prefers block 1
        m = int((diff > 0).sum().item())  # natural block-1 count
        # Both blocks constrain the count. With unequal targets the smaller
        # block's relative corridor can be tighter than the larger one's.
        lo = max(int(floors[1]), n - int(caps[0]))
        hi = min(int(caps[1]), n - int(floors[0]))
        m = min(max(m, lo), hi)
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
        raise ValueError(
            f"hardness/alpha must be a finite value in [0, 1]; got {alpha!r}"
        )
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


def _build_strata_plan(strata, n: int, splits: Sequence[float], balance_slack: float, device):
    """Precompute per-stratum indices, exact target sizes and corridors summing to global targets."""
    import torch

    if strata is None:
        return None
    st = np.asarray(strata)
    if st.shape != (n,):
        raise ValueError(f"strata must have one label per row ({n}); got shape {st.shape}")
    _, inv = np.unique(st, return_inverse=True)
    m_strata = int(inv.max()) + 1 if n else 0
    k = len(splits)
    global_sizes = target_sizes(n, splits)
    total = float(sum(splits))
    ratios = np.array([s / total for s in splits], dtype=np.float64)

    # First assign floor(n_s * ratio_c) to each stratum, then distribute remainders
    # so per-stratum sizes sum to n_s AND column sums equal global_sizes exactly.
    s_counts = np.bincount(inv, minlength=m_strata)
    raw = s_counts[:, None] * ratios[None, :]
    s_sizes = np.floor(raw).astype(int)
    frac = raw - s_sizes
    col_deficits = global_sizes - s_sizes.sum(axis=0)
    row_deficits = s_counts - s_sizes.sum(axis=1)
    # Greedy largest-fraction matching of row deficits to column deficits.
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


def _random_init_labels(sizes, seed: int, device, strata_plan=None, n: Optional[int] = None):
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
    """Initialize fold labels along centered principal directions of factor ``B``.

    Unlike a uniform random partition (whose fold-sum difference ``P_train - P_test``
    collapses onto the global degree vector ``0.6 * sum(B)`` on iteration 1 for an
    80/20 split), projecting onto the top eigenvectors of the centered r-by-r
    covariance ``(B - mu)^T (B - mu)`` cuts across the principal geometric axes of
    the factor manifold in O(n*r) time.
    """
    import torch

    n, r = ops.shape
    k = len(splits)
    if r == 0 or n <= k:
        return _random_init_labels(sizes, seed, ops.device, strata_plan, n)

    if _cached_basis is None:
        if ops.sparse:
            mean = np.asarray(ops.B.mean(axis=0)).ravel().astype(np.float64)
            gram = (ops.B.T @ ops.B).toarray().astype(np.float64) - n * np.outer(mean, mean)
            evals, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
            top_m = min(r, max(k * 2, 6))
            V = evecs[:, -top_m:][:, ::-1].copy()
            Z_base = np.asarray(ops.B @ V, dtype=np.float64) - (mean @ V)[None, :]
            Z_base_t = torch.as_tensor(Z_base, dtype=torch.float32, device=ops.device)
        else:
            mean = ops.B.mean(dim=0, dtype=torch.float64)
            gram = torch.zeros((r, r), dtype=torch.float64, device=ops.device)
            for s in range(0, n, 4096):
                blk = ops.B[s : s + 4096].double()
                gram += blk.T @ blk
            gram -= n * torch.outer(mean, mean)
            evals, evecs = torch.linalg.eigh(0.5 * (gram + gram.T))
            top_m = min(r, max(k * 2, 6))
            V = evecs[:, -top_m:].flip(1)
            Z_base_t = (ops.B @ V.to(ops.B.dtype)) - (mean @ V).to(ops.B.dtype)[None, :]
        std = Z_base_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        Z_base_t = Z_base_t / std
    else:
        Z_base_t = _cached_basis

    top_m = Z_base_t.shape[1]
    rng = np.random.default_rng(seed + 7919)
    # Orthogonal / diverse projection from the top principal components into k fold scores
    raw_proj = rng.normal(size=(top_m, k))
    if top_m >= k:
        q, _ = np.linalg.qr(raw_proj)
        proj = q[:, :k]
    else:
        proj = raw_proj
    proj_t = torch.as_tensor(proj, dtype=torch.float32, device=ops.device)
    scores = Z_base_t @ proj_t
    return _assign_with_plan(scores, sizes, caps, floors, use_corridor, strata_plan)


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

    from .factors import factor_ops, validate_fold_weights

    if init not in ("random", "spectral", "hybrid"):
        raise ValueError(f"init must be 'random', 'spectral' or 'hybrid'; got {init!r}")
    if linearization not in ("self_excluded", "full"):
        raise ValueError(
            f"linearization must be 'self_excluded' or 'full'; got {linearization!r}"
        )
    ops = factor_ops(B)
    device = ops.device
    if local is not None:
        from .local import SparseTerm

        local = SparseTerm(local, device)
    n, r = ops.shape
    k = len(splits)
    W = torch.as_tensor(validate_fold_weights(fold_weights, k), device=device)
    sizes = target_sizes(n, splits)
    use_corridor = balance_slack > 0.0
    caps, floors = (
        capacity_corridor(n, splits, balance_slack)
        if use_corridor
        else (sizes, sizes)
    )
    strata_plan = _build_strata_plan(strata, n, splits, balance_slack, device)
    self_sim = ops.diagonal()

    if init_labels is None:
        if init == "spectral":
            lab = _spectral_init_labels(
                ops, splits, sizes, caps, floors, use_corridor, seed, strata_plan
            )
        else:
            lab = _random_init_labels(sizes, seed, device, strata_plan, n)
    else:
        lab = ops.labels(init_labels, k)

    old_cost = ops.cost(lab, k, fold_weights) + (
        local.cross(lab) if local is not None else 0
    )

    for _ in range(n_iter):
        M = ops.mass(lab, k)
        if fold_weights is None:
            scores = M
            if linearization == "self_excluded":
                scores[torch.arange(n, device=device), lab] -= self_sim
        else:
            sub = (
                self_sim[:, None] * W[:, lab].T
                if linearization == "self_excluded"
                else 0.0
            )
            scores = -(M.double() @ W.T - sub)
        if local is not None:
            scores += local.block_mass(lab, k)
        new_lab = _assign_with_plan(
            scores, sizes, caps, floors, use_corridor, strata_plan
        )
        if torch.equal(new_lab, lab):
            break
        new_cost = ops.cost(new_lab, k, fold_weights) + (
            local.cross(new_lab) if local is not None else 0
        )
        if new_cost > old_cost:
            break
        lab = new_lab
        old_cost = new_cost
    return lab.cpu().numpy()


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
    """Monotone FM polish on the low-rank factor-space leakage. Returns ``(labels, n_moves)``.

    Single-entity moves inside the ``epsilon`` corridor. When the corridor is
    degenerate (``epsilon=0`` -> exact sizes, where no single move is legal) a
    size-preserving pairwise **swap** polish runs instead, so exact sizes are kept.
    """
    import torch

    from .factors import factor_ops

    ops = factor_ops(B)
    device = ops.device
    n, k = ops.shape[0], len(splits)
    # as_tensor aliases NumPy memory on CPU; refinement must not mutate the
    # caller's initial partition (GPU copies previously hid this difference).
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

    def checked(labels, moves):
        before = ops.cost(initial, k, fold_weights) + (
            local.cross(initial) if local is not None else 0
        )
        after = ops.cost(labels, k, fold_weights) + (
            local.cross(labels) if local is not None else 0
        )
        return (
            (initial.cpu().numpy(), 0)
            if after > before
            else (labels.cpu().numpy(), moves)
        )

    caps, floors = capacity_corridor(n, splits, epsilon)
    if fold_weights is not None:
        # Keep the factor and local row/fold masses separate: fold-pair weights
        # apply to the global kernel; the local penalty keeps its own definition.
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
    """Greedy monotone pairwise-swap polish (block sizes exactly preserved).

    Swapping ``u`` (block a) with ``v`` (block b) reduces the leakage by
    ``r_u(a→b) + r_v(b→a) − 2·sim(u, v)``, where ``r`` is the single-move reduction.
    Each round, for every ordered block pair, the ``n_cand`` best single movers ``u``
    are paired with their best partner ``v``; the best improving swap is applied.
    ``sim_col(v)`` returns the (n,) similarity column of ``v``.
    ``sim_block(rows, columns)`` optionally batches the same candidate search.
    When ``strata`` is given, swaps only pair entities within the same stratum so
    per-stratum fold counts remain invariant.
    Mutates ``lab``/``S``; returns ``(lab, n_swaps)``.
    """
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
        cur = (
            S.gather(1, lab[:, None]).squeeze(1) - self_sim
        )  # own-block mass, self excl.
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
                        # Candidate-major ordering preserves the scalar search's
                        # first-maximum tie rule. Only destination rows are needed.
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
        # Update only the two affected fold member lists in the affected stratum.
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
                        (
                            ia_all[st_t[ia_all] == s],
                            ib_all[st_t[ib_all] == s],
                        )
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
                            edge = W[a, b] * ops.column(u)[ib]
                            edge += local.column(u)[ib]
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


def _batched_balanced_lloyd(
    ops,
    splits,
    *,
    n_iter=25,
    seed=0,
    n_restarts=4,
    balance_slack=0.0,
    local=None,
    fold_weights=None,
    init="random",
    strata=None,
    linearization="self_excluded",
    return_diagnostics=False,
):
    """Run all ``n_restarts`` of balanced-Lloyd simultaneously with fused GEMMs."""
    import torch
    from .factors import validate_fold_weights

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

    # Precompute centered basis once if any restart uses spectral initialization.
    use_spectral = [
        (init == "spectral") or (init == "hybrid" and idx >= (n_restarts + 1) // 2)
        for idx in range(n_restarts)
    ]
    cached_basis = None
    if any(use_spectral) and r > 0 and n > k:
        if ops.sparse:
            mean = np.asarray(ops.B.mean(axis=0)).ravel().astype(np.float64)
            gram = (ops.B.T @ ops.B).toarray().astype(np.float64) - n * np.outer(
                mean, mean
            )
            _, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
            top_m = min(r, max(k * 2, 6))
            V = evecs[:, -top_m:][:, ::-1].copy()
            Z_base = np.asarray(ops.B @ V, dtype=np.float64) - (mean @ V)[None, :]
            cached_basis = torch.as_tensor(
                Z_base, dtype=torch.float32, device=ops.device
            )
        else:
            mean = ops.B.mean(dim=0, dtype=torch.float64)
            gram = torch.zeros((r, r), dtype=torch.float64, device=ops.device)
            for s in range(0, n, 4096):
                blk = ops.B[s : s + 4096].double()
                gram += blk.T @ blk
            gram -= n * torch.outer(mean, mean)
            _, evecs = torch.linalg.eigh(0.5 * (gram + gram.T))
            top_m = min(r, max(k * 2, 6))
            V = evecs[:, -top_m:].flip(1)
            cached_basis = (ops.B @ V.to(ops.B.dtype)) - (mean @ V).to(ops.B.dtype)[
                None, :
            ]
        std = cached_basis.std(dim=0, keepdim=True).clamp_min(1e-6)
        cached_basis = cached_basis / std

    labs = []
    for idx in range(n_restarts):
        if use_spectral[idx]:
            lab = _spectral_init_labels(
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
        else:
            lab = _random_init_labels(sizes, seed + idx, device, strata_plan, n)
        labs.append(lab)

    # Compute initial fold sums once and reuse them for both cost and iteration-0 GEMM.
    P_f64_list = ops.batched_sums(labs, k, precise=True)
    costs = [cost_from_P(P_f64_list[idx], labs[idx]) for idx in range(n_restarts)]
    use_f32_sums = not ops.sparse and n <= 2048
    P_mass_list = (
        ops.batched_sums(labs, k, precise=False)
        if use_f32_sums
        else P_f64_list
    )

    active = list(range(n_restarts))
    stop_reasons = ["max_iter"] * n_restarts
    iter_counts = [0] * n_restarts
    rejected_proposals = 0
    for _ in range(n_iter):
        if not active:
            break
        masses = ops.mass_from_sums([P_mass_list[i] for i in active], k)
        changed_indices = []
        changed_labs = []
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
    """Refine a bounded shortlist before selecting the winning restart.

    The unrefined best is retained as an incumbent. Default refinement covers all
    starts; refine_restarts=1 is the old compute budget. Candidate counts and the
    selected seed are reported, so extra work is visible rather than hidden.
    """
    import time
    from .factors import factor_ops
    from .diagnostics import positive_integer

    positive_integer("n_restarts", n_restarts)
    if refine_restarts is not None:
        positive_integer("refine_restarts", refine_restarts)
    if init not in ("random", "spectral", "hybrid"):
        raise ValueError(f"init must be 'random', 'spectral' or 'hybrid'; got {init!r}")
    if linearization not in ("self_excluded", "full"):
        raise ValueError(
            f"linearization must be 'self_excluded' or 'full'; got {linearization!r}"
        )
    ops = factor_ops(B)
    if local is not None:
        from .local import SparseTerm

        local_cost = SparseTerm(local)
    else:
        local_cost = None

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
