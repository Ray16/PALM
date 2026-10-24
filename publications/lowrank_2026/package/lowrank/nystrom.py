"""Rank-revealing, tiled Nyström factorization with empirical rank selection.

``rank='auto'`` increases landmarks until sampled entrywise residuals meet a
tolerance or a rank/array-budget limit is reached. An independent audit follows
selection. Neither check is a uniform cut-error or optimization certificate.
"""
from __future__ import annotations
import numpy as np
from .diagnostics import factor_pair_error, positive_integer, prepare_reference
from .splitters.common.feature_preparation import choose_metric
from .splitters.common.pairwise_similarity import pairwise_similarity

LANDMARKS = ("kmeans++", "uniform")


def _kmeanspp_landmarks(Xt, n_landmarks, metric, seed):
    """Distinct rows with (1-max_similarity)^2 weights, not Hilbert k-means++.

    Zero residual weights fall back to uniform *unselected* rows.
    """
    import torch
    n = len(Xt)
    n_landmarks = min(n_landmarks, n)
    gen = torch.Generator(device=Xt.device).manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=gen, device=Xt.device).item())
    chosen = [first]
    selected = torch.zeros(n, dtype=torch.bool, device=Xt.device)
    selected[first] = True
    min_dist = 1 - pairwise_similarity(Xt, Xt[first:first + 1], metric).squeeze(1)
    for _ in range(n_landmarks - 1):
        weights = min_dist.clamp(min=0).square()
        weights[selected] = 0
        if float(weights.sum()) <= 0:
            weights = (~selected).to(Xt.dtype)
        nxt = int(torch.multinomial(weights, 1, generator=gen).item())
        chosen.append(nxt)
        selected[nxt] = True
        sim = pairwise_similarity(Xt, Xt[nxt:nxt + 1], metric).squeeze(1)
        min_dist = torch.minimum(min_dist, 1 - sim)
    return np.asarray(chosen, dtype=np.int64)


def _array_plan(n, d, r, block_size, memory_mb):
    # Conservative controlled-array model, NOT a hard process-RSS limit.
    fixed = 8 * n * d + 4 * n * r + 64 * r * r + 16 * r * d + 64 * n
    per_row = 32 * r + 48 * d
    block = min(n, block_size)
    if memory_mb is not None:
        budget = int(memory_mb * 1024**2)
        block = min(block, (budget - fixed) // max(1, per_row))
        if block < 1:
            raise MemoryError(f"rank {r} exceeds factor_memory_mb={memory_mb}; "
                              "reduce rank/input size or increase the factor-array budget")
    return block, fixed + block * per_row


def nystrom_features(X, rank=256, metric=None, landmark="kmeans++", seed=0, kernel_power=1,
                     *, return_info=False, rank_min=32, rank_max=512, rank_tolerance=.05,
                     rank_probe_pairs=2048, block_size=2048, factor_memory_mb=None,
                     eigen_rtol=1e-8):
    """Return ``(B, metric)`` or, with ``return_info``, ``(B, metric, info)``.

    The landmark kernel is diagonalized in float64. Small eigenvalues are
    discarded rather than floored/amplified. Output may have fewer columns than
    requested, including zero for a zero kernel. Point/landmark products are tiled.

    ``factor_memory_mb`` bounds a conservative estimate of controlled arrays,
    excluding caller-owned inputs, framework overhead and downstream optimization.
    Fixed ranks fail when they cannot fit; auto stops at its affordable limit.
    """
    import torch
    auto = isinstance(rank, str) and rank == "auto"
    if not auto:
        positive_integer("rank", rank)
    for name, value in (("rank_min", rank_min), ("rank_max", rank_max),
                        ("rank_probe_pairs", rank_probe_pairs), ("block_size", block_size),
                        ("kernel_power", kernel_power)):
        positive_integer(name, value)
    if rank_min > rank_max:
        raise ValueError("rank_min must not exceed rank_max")
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be finite and positive")
    if not np.isfinite(eigen_rtol) or not 0 <= eigen_rtol < 1:
        raise ValueError("eigen_rtol must be finite and in [0, 1)")
    if factor_memory_mb is not None and (not np.isfinite(factor_memory_mb) or factor_memory_mb <= 0):
        raise ValueError("factor_memory_mb must be finite and positive")
    if landmark not in LANDMARKS:
        raise ValueError(f"unknown landmark method {landmark!r}; expected one of {LANDMARKS}")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    metric = choose_metric(X) if metric is None else metric
    n, d = X.shape
    maximum = min(n, rank_max if auto else rank)
    if n < 1:
        raise ValueError("features must have at least one row")
    if auto and factor_memory_mb is not None:
        lo, hi = 0, maximum
        while lo < hi:
            mid = (lo + hi + 1) // 2
            try:
                _array_plan(n, d, mid, block_size, factor_memory_mb)
                lo = mid
            except MemoryError:
                hi = mid - 1
        maximum = lo
    if maximum < min(n, rank_min) and auto:
        raise MemoryError("factor-array budget cannot fit rank_min; lower rank_min explicitly")
    count = min(maximum, rank_min) if auto else maximum
    _array_plan(n, d, count, block_size, factor_memory_mb)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = prepare_reference(X, metric, device)
    history = []
    while True:
        rows, predicted_bytes = _array_plan(n, d, count, block_size, factor_memory_mb)
        if landmark == "uniform":
            gen = torch.Generator(device=device).manual_seed(seed)
            idx = torch.randperm(n, generator=gen, device=device)[:count].cpu().numpy()
        else:
            idx = _kmeanspp_landmarks(Xt, count, metric, seed)
        L = Xt[torch.as_tensor(idx, device=device)].double()
        W = pairwise_similarity(L, L, metric).pow(kernel_power)
        W = (W + W.T) * .5
        evals, evecs = torch.linalg.eigh(W)
        largest = max(float(evals[-1]), 0.)
        threshold = largest * max(eigen_rtol, count * np.finfo(np.float64).eps)
        keep = evals > threshold
        transform = evecs[:, keep] / torch.sqrt(evals[keep])[None, :]
        del W, evecs
        B = np.empty((n, int(keep.sum())), dtype=np.float32)
        for start in range(0, n, rows):
            C = pairwise_similarity(Xt[start:start + rows].double(), L, metric).pow(kernel_power)
            B[start:start + rows] = (C @ transform).float().cpu().numpy()
        del C, transform, L
        info = dict(requested_rank="auto" if auto else int(rank), landmark_count=int(count), effective_rank=B.shape[1],
                    landmark_indices=idx.tolist(), eigenvalue_threshold=threshold,
                    discarded_eigenvalues=count - B.shape[1], minimum_eigenvalue=float(evals[0]),
                    row_block_size=rows, estimated_array_bytes=predicted_bytes,
                    memory_scope="factor arrays only; excludes framework and optimizer",
                    rank_limit=maximum, kernel_power=int(kernel_power), adaptive=auto)
        if not auto:
            break
        probe = factor_pair_error(B, Xt, metric, pairs=rank_probe_pairs,
                                  seed=seed + 104729, kernel_power=kernel_power, block_size=rows)
        history.append(dict(landmark_count=count, effective_rank=B.shape[1], **probe))
        summary = probe["relative_pair_mae"]
        passed = summary["ci95"] is not None and summary["ci95"][1] <= rank_tolerance
        if passed or count == maximum:
            info.update(selection_history=history, selection_tolerance_met=bool(passed),
                        rank_tolerance=float(rank_tolerance),
                        stop_reason="sample_tolerance" if passed else "rank_or_memory_limit")
            break
        del B
        count = min(2 * count, maximum)
    if auto:
        audit = factor_pair_error(B, Xt, metric, pairs=rank_probe_pairs,
                                  seed=seed + 130363, kernel_power=kernel_power,
                                  block_size=info["row_block_size"])
        interval = audit["relative_pair_mae"]["ci95"]
        info.update(audit=audit, audit_tolerance_met=bool(interval is not None and interval[1] <= rank_tolerance))
    return (B, metric, info) if return_info else (B, metric)
