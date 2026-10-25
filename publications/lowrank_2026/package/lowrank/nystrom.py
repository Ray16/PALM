"""Rank-revealing, tiled Nyström factorization with empirical rank selection.

``rank='auto'`` increases landmarks until the selected kernel/cut check meets a
tolerance or a rank/array-budget limit is reached. An independent audit follows
selection. Neither check is a uniform cut-error or optimization certificate.
"""

from __future__ import annotations
import numpy as np
import warnings
from .diagnostics import factor_pair_error, positive_integer, prepare_reference
from .splitters.common.feature_preparation import choose_metric
from .splitters.common.pairwise_similarity import (
    pairwise_similarity,
    normalize_cosine,
    validate_features,
)

LANDMARKS = ("kmeans++", "uniform")


class AccuracyWarning(UserWarning):
    """The requested empirical fidelity check did not establish its tolerance."""


class AccuracyError(RuntimeError):
    """The caller required a fidelity check that did not pass."""


class _LandmarkSampler:
    """Extend one seeded landmark sequence; do not restart selection at each rank."""

    def __init__(self, Xt, metric, seed, method):
        import torch

        self.X, self.metric, self.method = Xt, metric, method
        eligible = (
            Xt.ne(0).any(1)
            if metric != "euclidean"
            else torch.ones(len(Xt), dtype=torch.bool, device=Xt.device)
        )
        self.pool = torch.nonzero(eligible).ravel()
        self.generator = torch.Generator(device=Xt.device).manual_seed(seed)
        self.chosen = []
        self.selected = torch.zeros(len(Xt), dtype=torch.bool, device=Xt.device)
        self.min_dist = torch.ones(len(Xt), dtype=Xt.dtype, device=Xt.device)
        self.eligible = eligible
        self.uniform = (
            self.pool[
                torch.randperm(
                    len(self.pool), generator=self.generator, device=Xt.device
                )
            ]
            if method == "uniform"
            else None
        )

    def extend(self, count):
        import torch

        count = min(count, len(self.pool))
        while len(self.chosen) < count:
            if self.uniform is not None:
                nxt = int(self.uniform[len(self.chosen)])
            else:
                weights = self.min_dist.clamp(min=0).square()
                weights[~self.eligible | self.selected] = 0
                if float(weights.sum()) <= 0:
                    weights = (self.eligible & ~self.selected).to(self.X.dtype)
                nxt = int(torch.multinomial(weights, 1, generator=self.generator))
            self.chosen.append(nxt)
            self.selected[nxt] = True
            if self.uniform is None:
                sim = pairwise_similarity(
                    self.X, self.X[nxt : nxt + 1], self.metric
                ).squeeze(1)
                self.min_dist = torch.minimum(self.min_dist, 1 - sim)
        return np.asarray(self.chosen, dtype=np.int64)


def _kmeanspp_landmarks(Xt, n_landmarks, metric, seed):
    """Distinct rows with (1-max_similarity)^2 weights, not Hilbert k-means++.

    Zero residual weights fall back to uniform *unselected* rows.
    """
    return _LandmarkSampler(Xt, metric, seed, "kmeans++").extend(n_landmarks)


def _array_plan(n, d, r, block_size, memory_mb):
    # Conservative controlled-array model, NOT a hard process-RSS limit.
    fixed = 16 * n * d + 4 * n * r + 64 * r * r + 16 * r * d + 64 * n
    per_row = 32 * r + 48 * d
    block = min(n, block_size)
    if memory_mb is not None:
        budget = int(memory_mb * 1024**2)
        block = min(block, (budget - fixed) // max(1, per_row))
        if block < 1:
            raise MemoryError(
                f"rank {r} exceeds factor_memory_mb={memory_mb}; "
                "reduce rank/input size or increase the factor-array budget"
            )
    return block, fixed + block * per_row


def _direct_array_plan(n, d, block_size, memory_mb):
    fixed = 12 * n * d + 64 * n
    per_row = 16 * d
    rows = min(n, block_size)
    if memory_mb is not None:
        rows = min(rows, (int(memory_mb * 1024**2) - fixed) // max(1, per_row))
        if rows < 1:
            raise MemoryError("exact cosine factor exceeds factor_memory_mb")
    return rows, fixed + rows * per_row


def nystrom_features(
    X,
    rank=256,
    metric=None,
    landmark="kmeans++",
    seed=0,
    kernel_power=1,
    *,
    return_info=False,
    rank_min=32,
    rank_max=512,
    rank_tolerance=0.05,
    rank_probe_pairs=None,
    block_size=2048,
    factor_memory_mb=None,
    eigen_rtol=1e-8,
    on_failure="warn",
    factor_backend="nystrom",
    rank_check="pair_mae",
    cut_labels=None,
    fold_weights=None,
    _partitioner=None,
    _prepared=False,
):
    """Return ``(B, metric)`` or, with ``return_info``, ``(B, metric, info)``.

    The landmark kernel is diagonalized in float64. Small eigenvalues are
    discarded rather than floored/amplified. Output may have fewer columns than
    requested, including zero for a zero kernel. Point/landmark products are tiled.

    ``factor_memory_mb`` bounds a conservative estimate of controlled arrays,
    excluding caller-owned inputs, framework overhead and downstream optimization.
    Fixed ranks fail when they cannot fit; auto stops at its affordable limit.
    """
    import torch

    if on_failure not in ("warn", "raise", "return"):
        raise ValueError("on_failure must be warn, raise or return")
    if factor_backend not in ("nystrom", "auto"):
        raise ValueError("factor_backend must be nystrom or auto")
    if rank_check not in ("pair_mae", "cut_error"):
        raise ValueError("rank_check must be pair_mae or cut_error")
    if rank_probe_pairs is None:
        rank_probe_pairs = 32768 if rank_check == "cut_error" else 2048
    if rank_check == "cut_error" and cut_labels is None and _partitioner is None:
        raise ValueError(
            "cut_error requires a supplied partition or the splitter's partition optimizer"
        )
    auto = isinstance(rank, str) and rank == "auto"
    if not auto:
        positive_integer("rank", rank)
    for name, value in (
        ("rank_min", rank_min),
        ("rank_max", rank_max),
        ("rank_probe_pairs", rank_probe_pairs),
        ("block_size", block_size),
        ("kernel_power", kernel_power),
    ):
        positive_integer(name, value)
    if rank_min > rank_max:
        raise ValueError("rank_min must not exceed rank_max")
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be finite and positive")
    if not np.isfinite(eigen_rtol) or not 0 <= eigen_rtol < 1:
        raise ValueError("eigen_rtol must be finite and in [0, 1)")
    if factor_memory_mb is not None and (
        not np.isfinite(factor_memory_mb) or factor_memory_mb <= 0
    ):
        raise ValueError("factor_memory_mb must be finite and positive")
    if landmark not in LANDMARKS:
        raise ValueError(
            f"unknown landmark method {landmark!r}; expected one of {LANDMARKS}"
        )
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    metric = choose_metric(X) if metric is None else metric
    validate_features(X, metric)
    n, d = X.shape
    eligible = n if metric == "euclidean" else int(np.any(X != 0, axis=1).sum())
    maximum = min(eligible, rank_max if auto else rank)
    direct = (
        factor_backend == "auto"
        and metric == "cosine"
        and kernel_power == 1
        and d <= maximum
    )
    if n < 1:
        raise ValueError("features must have at least one row")
    if auto and factor_memory_mb is not None and not direct:
        lo, hi = 0, maximum
        while lo < hi:
            mid = (lo + hi + 1) // 2
            try:
                _array_plan(n, d, mid, block_size, factor_memory_mb)
                lo = mid
            except MemoryError:
                hi = mid - 1
        maximum = lo
    if maximum < min(eligible, rank_min) and auto:
        raise MemoryError(
            "factor-array budget cannot fit rank_min; lower rank_min explicitly"
        )
    count = min(maximum, rank_min) if auto else maximum
    if direct:
        _direct_array_plan(n, d, block_size, factor_memory_mb)
    else:
        _array_plan(n, d, count, block_size, factor_memory_mb)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = (
        torch.as_tensor(X, dtype=torch.float32, device=device)
        if _prepared
        else prepare_reference(X, metric, device)
    )
    sampler = None if direct else _LandmarkSampler(Xt, metric, seed, landmark)
    if direct:
        count = d
    stages = 1 + max(0, int(np.ceil(np.log2(max(1, maximum) / max(1, count)))))
    history = []
    cached_W = None
    while True:
        rows, predicted_bytes = (
            _direct_array_plan(n, d, block_size, factor_memory_mb)
            if direct
            else _array_plan(n, d, count, block_size, factor_memory_mb)
        )
        if direct:
            B = np.empty((n, d), dtype=np.float32)
            for start in range(0, n, rows):
                B[start : start + rows] = (
                    normalize_cosine(Xt[start : start + rows]).cpu().numpy()
                )
            idx = np.empty(0, dtype=np.int64)
            threshold = 0.0
            minimum_eigenvalue = None
            discarded = 0
        elif count == 0:
            B = np.empty((n, 0), dtype=np.float32)
            idx = np.empty(0, dtype=np.int64)
            threshold = 0.0
            minimum_eigenvalue = 0.0
            discarded = 0
        else:
            idx = sampler.extend(count)
            L = Xt[torch.as_tensor(idx, device=device)].double()
            previous = 0 if cached_W is None else len(cached_W)
            W = torch.empty(count, count, dtype=torch.float64, device=device)
            if previous:
                W[:previous, :previous] = cached_W
                extension = pairwise_similarity(L, L[previous:], metric).pow(
                    kernel_power
                )
                W[:, previous:] = extension
                W[previous:, :] = extension.T
            else:
                W[:] = pairwise_similarity(L, L, metric).pow(kernel_power)
            cached_W = W
            evals, evecs = torch.linalg.eigh((W + W.T) * 0.5)
            threshold = max(float(evals[-1]), 0.0) * max(
                eigen_rtol, count * np.finfo(np.float64).eps
            )
            keep = evals > threshold
            transform = evecs[:, keep] / torch.sqrt(evals[keep])[None, :]
            B = np.empty((n, int(keep.sum())), dtype=np.float32)
            for start in range(0, n, rows):
                C = pairwise_similarity(
                    Xt[start : start + rows].double(), L, metric
                ).pow(kernel_power)
                B[start : start + rows] = (C @ transform).float().cpu().numpy()
                del C
            minimum_eigenvalue = float(evals[0])
            discarded = count - B.shape[1]
            del transform, L, evecs, evals
        info = dict(
            requested_rank="auto" if auto else int(rank),
            landmark_count=0 if direct else int(count),
            effective_rank=B.shape[1],
            landmark_indices=idx.tolist(),
            eigenvalue_threshold=threshold,
            discarded_eigenvalues=discarded,
            minimum_eigenvalue=minimum_eigenvalue,
            row_block_size=rows,
            estimated_array_bytes=predicted_bytes,
            memory_scope="factor arrays only; excludes framework and optimizer",
            rank_limit=maximum,
            kernel_power=int(kernel_power),
            adaptive=auto,
            factor_backend="exact_cosine" if direct else "nystrom",
            rank_check=rank_check,
            probe_pair_budget=int(rank_probe_pairs),
            selection_reuses_landmarks=not direct,
        )
        if not auto:
            break
        labels = _partitioner(B, metric) if _partitioner is not None else cut_labels
        probe = factor_pair_error(
            B,
            Xt,
            metric,
            pairs=rank_probe_pairs,
            seed=seed + 104729,
            kernel_power=kernel_power,
            block_size=rows,
            confidence=1 - 0.05 / stages,
            labels=labels if rank_check == "cut_error" else None,
            fold_weights=fold_weights,
        )
        history.append(
            dict(
                landmark_count=info["landmark_count"],
                effective_rank=B.shape[1],
                **probe,
            )
        )
        summary = probe["relative_pair_mae"]
        passed = summary["ci95"] is not None and summary["ci95"][1] <= rank_tolerance
        if passed or count == maximum or direct:
            info.update(
                selection_history=history,
                selection_tolerance_met=bool(passed),
                rank_tolerance=float(rank_tolerance),
                stop_reason="sample_tolerance"
                if passed
                else "exact_feature_map"
                if direct
                else "rank_or_memory_limit",
            )
            break
        del B
        count = min(2 * count, maximum)
    if auto:
        audit = factor_pair_error(
            B,
            Xt,
            metric,
            pairs=rank_probe_pairs,
            seed=seed + 130363,
            kernel_power=kernel_power,
            block_size=info["row_block_size"],
            labels=labels if rank_check == "cut_error" else None,
            fold_weights=fold_weights,
        )
        interval = audit["relative_pair_mae"]["ci95"]
        info.update(
            audit=audit,
            audit_tolerance_met=bool(
                interval is not None and interval[1] <= rank_tolerance
            ),
        )
        if not (info["selection_tolerance_met"] and info["audit_tolerance_met"]):
            message = (
                f"rank fidelity tolerance {rank_tolerance:g} was not established "
                f"at factor dimension {B.shape[1]} ({rank_check}); inspect factorization diagnostics"
            )
            if on_failure == "raise":
                raise AccuracyError(message)
            if on_failure == "warn":
                warnings.warn(message, AccuracyWarning, stacklevel=2)
    return (B, metric, info) if return_info else (B, metric)
