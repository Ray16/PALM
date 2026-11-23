"""Rank-revealing, tiled Nyström factorization with empirical rank selection.

``rank='auto'`` increases landmarks until the selected kernel/cut check meets a
tolerance or a rank/array-budget limit is reached. An independent audit follows
selection. Neither check is a uniform cut-error or optimization certificate.
"""

from __future__ import annotations

import warnings

import numpy as np

from .diagnostics import factor_pair_error, positive_integer, prepare_reference
from .splitters.common.feature_preparation import choose_metric
from .splitters.common.pairwise_similarity import (
    normalize_cosine,
    pairwise_similarity,
    validate_features,
)

LANDMARKS = ("kmeans++", "uniform", "kmeans||")


class AccuracyWarning(UserWarning):
    """The requested empirical fidelity check did not establish its tolerance."""


class AccuracyError(RuntimeError):
    """The caller required a fidelity check that did not pass."""


# ── Seeded Incremental Landmark Sampling ─────────────────────────────────────


class _LandmarkSampler:
    """Extend one seeded landmark sequence; do not restart selection at each rank."""

    def __init__(
        self, Xt, metric, seed, method, *, kernel_power=1, power_landmarks=False
    ):
        import torch

        self.X, self.metric, self.method = Xt, metric, method
        self.kernel_power = int(kernel_power)
        self.power_landmarks = bool(power_landmarks or method == "kmeans||")
        self._X_norm = normalize_cosine(Xt) if metric == "cosine" else None
        self._card = Xt.sum(1) if metric == "tanimoto" else None
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

    def _sim_to_indices(self, idx_tensor):
        import torch

        if self.metric == "cosine":
            return self._X_norm @ self._X_norm[idx_tensor].T
        if self.metric == "tanimoto":
            inter = self.X @ self.X[idx_tensor].T
            union = self._card[:, None] + self._card[idx_tensor][None, :] - inter
            return torch.where(union > 0, inter / union, torch.zeros_like(inter))
        return pairwise_similarity(self.X, self.X[idx_tensor], self.metric)

    def _update_min_dist(self, idx_tensor):
        import torch

        sim = self._sim_to_indices(idx_tensor)
        if self.power_landmarks and self.kernel_power > 1:
            sim = sim.pow(self.kernel_power)
        max_sim = sim.squeeze(1) if sim.shape[1] == 1 else sim.max(dim=1).values
        self.min_dist = torch.minimum(self.min_dist, 1 - max_sim)

    def _active_weights(self):
        weights = self.min_dist.clamp(min=0).square()
        weights[~self.eligible | self.selected] = 0
        if float(weights.sum()) <= 0:
            weights = (self.eligible & ~self.selected).to(self.X.dtype)
        return weights

    def extend(self, count: int) -> np.ndarray:
        import torch

        count = min(count, len(self.pool))
        if self.method == "kmeans||" and len(self.chosen) < count:
            if not self.chosen:
                weights = (self.eligible & ~self.selected).to(self.X.dtype)
                first = int(torch.multinomial(weights, 1, generator=self.generator))
                self.chosen.append(first)
                self.selected[first] = True
                self._update_min_dist(torch.tensor([first], device=self.X.device))
            remaining = count - len(self.chosen)
            n_rounds = min(5, remaining)
            batch_size = max(1, (remaining + n_rounds - 1) // max(1, n_rounds))
            while len(self.chosen) < count:
                take = min(batch_size, count - len(self.chosen))
                weights = self._active_weights()
                avail = int((weights > 0).sum().item())
                if avail < take:
                    weights = (self.eligible & ~self.selected).to(self.X.dtype)
                    avail = int((weights > 0).sum().item())
                take = min(take, avail)
                if take < 1:
                    break
                picks = torch.multinomial(
                    weights, take, replacement=False, generator=self.generator
                )
                for nxt in picks.tolist():
                    self.chosen.append(int(nxt))
                    self.selected[int(nxt)] = True
                self._update_min_dist(picks)
            return np.asarray(self.chosen, dtype=np.int64)

        while len(self.chosen) < count:
            if self.uniform is not None:
                nxt = int(self.uniform[len(self.chosen)])
            else:
                nxt = int(
                    torch.multinomial(
                        self._active_weights(), 1, generator=self.generator
                    )
                )
            self.chosen.append(nxt)
            self.selected[nxt] = True
            if self.uniform is None:
                self._update_min_dist(torch.tensor([nxt], device=self.X.device))
        return np.asarray(self.chosen, dtype=np.int64)


def _kmeanspp_landmarks(
    Xt, n_landmarks, metric, seed, *, kernel_power=1, power_landmarks=False
):
    """Distinct rows with ``(1 - max_similarity)^2`` weights, not Hilbert k-means++."""
    return _LandmarkSampler(
        Xt,
        metric,
        seed,
        "kmeans++",
        kernel_power=kernel_power,
        power_landmarks=power_landmarks,
    ).extend(n_landmarks)


# ── Array Budget & Probe Evaluation Helpers ──────────────────────────────────


def _array_plan(n, d, r, block_size, memory_mb):
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


def _evaluate_rank_probe(
    B,
    Xt,
    metric,
    *,
    pairs,
    seed,
    kernel_power,
    block_size,
    confidence,
    rank_check,
    rank_tolerance,
    cut_rel_tol,
    cut_abs_floor,
    labels,
    fold_weights,
):
    """Run one pair/cut probe and return ``(probe_dict, passed, error_established)``."""
    is_cut = rank_check in ("cut_error", "cut_relative_error")
    eff_rel_tol = (
        cut_rel_tol
        if rank_check == "cut_relative_error"
        else rank_tolerance
        if rank_check == "cut_error"
        else cut_rel_tol
    )
    probe = factor_pair_error(
        B,
        Xt,
        metric,
        pairs=pairs,
        seed=seed,
        kernel_power=kernel_power,
        block_size=block_size,
        confidence=confidence,
        labels=labels if is_cut else None,
        fold_weights=fold_weights,
        cut_rel_tol=eff_rel_tol,
        cut_abs_floor=cut_abs_floor,
    )
    if rank_check == "cut_relative_error":
        acc = probe["acceptance"]
        passed = bool(acc["accepted"])
        rel_ci = probe["cut_relative_error"].get("ci95")
        error_established = bool(
            acc["status"] == "rejected"
            and rel_ci is not None
            and rel_ci[0] > acc["rel_tol"]
        )
    else:
        interval = probe["relative_pair_mae"]["ci95"]
        passed = bool(interval is not None and interval[1] <= rank_tolerance)
        error_established = bool(interval is not None and interval[0] > rank_tolerance)
    return probe, passed, error_established


def _build_nystrom_block(
    Xt,
    sampler: _LandmarkSampler,
    count: int,
    rows: int,
    metric: str,
    kernel_power: int,
    eigen_rtol: float,
    cached_W,
):
    """Extend landmarks to ``count``, update landmark kernel ``W``, and project ``B``."""
    import torch

    n = len(Xt)
    device = Xt.device
    idx = sampler.extend(count)
    L = Xt[torch.as_tensor(idx, device=device)].double()
    previous = 0 if cached_W is None else len(cached_W)
    W = torch.empty(count, count, dtype=torch.float64, device=device)
    if previous:
        W[:previous, :previous] = cached_W
        extension = pairwise_similarity(L, L[previous:], metric).pow(kernel_power)
        W[:, previous:] = extension
        W[previous:, :] = extension.T
    else:
        W[:] = pairwise_similarity(L, L, metric).pow(kernel_power)
    evals, evecs = torch.linalg.eigh((W + W.T) * 0.5)
    threshold = max(float(evals[-1]), 0.0) * max(
        eigen_rtol, count * np.finfo(np.float64).eps
    )
    keep = evals > threshold
    transform = evecs[:, keep] / torch.sqrt(evals[keep])[None, :]
    B = np.empty((n, int(keep.sum())), dtype=np.float32)
    for start in range(0, n, rows):
        C = pairwise_similarity(Xt[start : start + rows].double(), L, metric).pow(
            kernel_power
        )
        B[start : start + rows] = (C @ transform).float().cpu().numpy()
        del C
    min_eval = float(evals[0])
    discarded = count - B.shape[1]
    return B, idx, W, threshold, min_eval, discarded


# ── Public Nyström Entry Point ───────────────────────────────────────────────


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
    rank_probe_policy="fixed",
    rank_probe_max_pairs=None,
    block_size=2048,
    factor_memory_mb=None,
    eigen_rtol=1e-8,
    on_failure="warn",
    factor_backend="nystrom",
    rank_check="pair_mae",
    cut_labels=None,
    fold_weights=None,
    cut_rel_tol=0.1,
    cut_abs_floor=1e-3,
    power_landmarks=False,
    _partitioner=None,
    _prepared=False,
):
    """Return ``(B, metric)`` or, with ``return_info``, ``(B, metric, info)``."""
    import torch

    if on_failure not in ("warn", "raise", "return"):
        raise ValueError("on_failure must be warn, raise or return")
    if factor_backend not in ("nystrom", "auto"):
        raise ValueError("factor_backend must be nystrom or auto")
    if rank_check not in ("pair_mae", "cut_error", "cut_relative_error"):
        raise ValueError("rank_check must be pair_mae, cut_error or cut_relative_error")
    if rank_probe_policy not in ("fixed", "adaptive"):
        raise ValueError("rank_probe_policy must be fixed or adaptive")
    is_cut_check = rank_check in ("cut_error", "cut_relative_error")
    if rank_probe_pairs is None:
        rank_probe_pairs = (
            32768 if is_cut_check and rank_probe_policy == "fixed" else 2048
        )
    positive_integer("rank_probe_pairs", rank_probe_pairs)
    if rank_probe_max_pairs is None:
        rank_probe_max_pairs = (
            max(32768, rank_probe_pairs)
            if rank_probe_policy == "adaptive"
            else rank_probe_pairs
        )
    positive_integer("rank_probe_max_pairs", rank_probe_max_pairs)
    if rank_probe_max_pairs < rank_probe_pairs:
        raise ValueError("rank_probe_max_pairs must not be below rank_probe_pairs")
    if rank_probe_policy == "fixed" and rank_probe_max_pairs != rank_probe_pairs:
        raise ValueError("a different maximum probe count requires adaptive policy")
    if is_cut_check and cut_labels is None and _partitioner is None:
        raise ValueError(
            f"{rank_check} requires a supplied partition or the splitter's partition optimizer"
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
    if n < 1:
        raise ValueError("features must have at least one row")
    eligible = n if metric == "euclidean" else int(np.any(X != 0, axis=1).sum())
    maximum = min(eligible, rank_max if auto else rank)
    direct = (
        factor_backend == "auto"
        and metric == "cosine"
        and kernel_power == 1
        and d <= maximum
    )
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
    sampler = (
        None
        if direct
        else _LandmarkSampler(
            Xt,
            metric,
            seed,
            landmark,
            kernel_power=kernel_power,
            power_landmarks=power_landmarks,
        )
    )
    if direct:
        count = d
    stages = 1 + max(0, int(np.ceil(np.log2(max(1, maximum) / max(1, count)))))
    probe_stages = 1 + int(np.ceil(np.log2(rank_probe_max_pairs / rank_probe_pairs)))
    joint = rank_probe_policy == "adaptive"
    selection_confidence = 1 - (0.025 if joint else 0.05) / (stages * probe_stages)
    probe_pairs = rank_probe_pairs
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
            threshold, minimum_eigenvalue, discarded = 0.0, None, 0
        elif count == 0:
            B = np.empty((n, 0), dtype=np.float32)
            idx = np.empty(0, dtype=np.int64)
            threshold, minimum_eigenvalue, discarded = 0.0, 0.0, 0
        else:
            B, idx, cached_W, threshold, minimum_eigenvalue, discarded = (
                _build_nystrom_block(
                    Xt,
                    sampler,
                    count,
                    rows,
                    metric,
                    kernel_power,
                    eigen_rtol,
                    cached_W,
                )
            )
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
            probe_policy=rank_probe_policy,
            maximum_probe_pairs=int(rank_probe_max_pairs),
            selection_grid_checks=int(stages * probe_stages),
            selection_confidence=float(selection_confidence),
            joint_selection_audit_confidence=0.95 if joint else None,
            selection_reuses_landmarks=not direct,
        )
        if not auto:
            break
        labels = _partitioner(B, metric) if _partitioner is not None else cut_labels
        while True:
            probe, passed, error_established = _evaluate_rank_probe(
                B,
                Xt,
                metric,
                pairs=probe_pairs,
                seed=seed + 104729,
                kernel_power=kernel_power,
                block_size=rows,
                confidence=selection_confidence,
                rank_check=rank_check,
                rank_tolerance=rank_tolerance,
                cut_rel_tol=cut_rel_tol,
                cut_abs_floor=cut_abs_floor,
                labels=labels,
                fold_weights=fold_weights,
            )
            more_probes = (
                joint
                and not passed
                and not error_established
                and not probe["exact"]
                and probe_pairs < rank_probe_max_pairs
            )
            history.append(
                dict(
                    landmark_count=info["landmark_count"],
                    effective_rank=B.shape[1],
                    requested_pairs=int(probe_pairs),
                    decision="accept"
                    if passed
                    else "increase_probes"
                    if more_probes
                    else "increase_rank"
                    if count < maximum and not direct
                    else "budget_exhausted",
                    **probe,
                )
            )
            if not more_probes:
                break
            probe_pairs = min(2 * probe_pairs, rank_probe_max_pairs)
        if passed or count == maximum or direct:
            info.update(
                selection_history=history,
                selection_tolerance_met=bool(passed),
                rank_tolerance=float(
                    cut_rel_tol
                    if rank_check == "cut_relative_error"
                    else rank_tolerance
                ),
                selected_probe_pairs=int(probe_pairs),
                selection_pair_evaluations=sum(h["sample_pairs"] for h in history),
                stop_reason="sample_tolerance"
                if passed
                else "exact_feature_map"
                if direct
                else "rank_probe_or_memory_limit"
                if joint
                else "rank_or_memory_limit",
            )
            break
        del B
        count = min(2 * count, maximum)

    if auto:
        audit, audit_passed, _ = _evaluate_rank_probe(
            B,
            Xt,
            metric,
            pairs=probe_pairs,
            seed=seed + 130363,
            kernel_power=kernel_power,
            block_size=info["row_block_size"],
            confidence=0.975 if joint else 0.95,
            rank_check=rank_check,
            rank_tolerance=rank_tolerance,
            cut_rel_tol=cut_rel_tol,
            cut_abs_floor=cut_abs_floor,
            labels=labels,
            fold_weights=fold_weights,
        )
        info.update(audit=audit, audit_tolerance_met=audit_passed)
        if not (info["selection_tolerance_met"] and info["audit_tolerance_met"]):
            message = (
                f"rank fidelity tolerance {info['rank_tolerance']:g} was not established "
                f"at factor dimension {B.shape[1]} ({rank_check}); inspect factorization diagnostics"
            )
            if on_failure == "raise":
                raise AccuracyError(message)
            if on_failure == "warn":
                warnings.warn(message, AccuracyWarning, stacklevel=2)
    return (B, metric, info) if return_info else (B, metric)
