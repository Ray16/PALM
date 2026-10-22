"""Nyström low-rank factorization  S ~= B B^T  for the low-rank splitter.

Picks ``rank`` landmark rows, forms the point-landmark (C) and landmark-landmark
(W) similarity blocks, and returns ``B = C · W^{-1/2}`` so that ``B @ B.T ~= S``.
Tanimoto/cosine are positive-semidefinite kernels. The cut decomposes exactly for
the approximating matrix in factor space, with residual error for the true kernel.

``landmark`` selects the landmark rows: ``kmeans++`` (default) or ``uniform``.

**Fidelity bound (why this matters).** The factor-space objective equals the true
cross-split similarity up to the Nyström reconstruction error: for any split,
``|factor_leakage − true_leakage| ≤ n·‖S − BBᵀ‖₂``.

Tighter-sketch variants (ridge-leverage-score landmarks, ``ridge`` regularization of
``W^{-1/2}``, and adaptive energy-rank) were evaluated and gave negligible leakage
benefit over plain k-means++ — the split objective is robust to approximation error,
so the reconstruction bound is loose in practice. They were removed as a documented
negative result; the study and pre-strip code live under
``archive/nystrom_variants/`` (see ``docs/FINDINGS.md`` Step 2).
"""

from __future__ import annotations

import numpy as np

from splitters.common.feature_preparation import choose_metric
from splitters.common.pairwise_similarity import (check_metric, pairwise_similarity,
                                                  standardize_for_metric)

LANDMARKS = ("kmeans++", "uniform")


def _kmeanspp_landmarks(Xt, n_landmarks, metric, seed):
    """Pick landmark rows by k-means++ (D^2-weighted) sampling on the GPU."""
    import torch

    n = Xt.shape[0]
    n_landmarks = min(n_landmarks, n)
    gen = torch.Generator(device=Xt.device).manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=gen, device=Xt.device).item())
    chosen = [first]
    sim_to_set = pairwise_similarity(Xt, Xt[first][None, :], metric).squeeze(1)
    min_dist = 1.0 - sim_to_set
    for _ in range(n_landmarks - 1):
        weights = torch.clamp(min_dist, min=0.0) ** 2
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        nxt = int(torch.multinomial(weights, 1, generator=gen).item())
        chosen.append(nxt)
        sim_new = pairwise_similarity(Xt, Xt[nxt][None, :], metric).squeeze(1)
        min_dist = torch.minimum(min_dist, 1.0 - sim_new)
    return np.asarray(chosen, dtype=np.int64)


def nystrom_features(X, rank=256, metric=None, landmark="kmeans++", seed=0, kernel_power=1):
    """Nyström low-rank factor ``B`` with ``B @ B.T ~= S``. Returns ``(B, metric)``.

    ``landmark`` ∈ {``kmeans++`` (default), ``uniform``}. The factor is
    ``B = C · W^{-1/2}`` with tiny eigenvalues clamped for numerical stability.
    For ``euclidean`` the features are z-scored first (:func:`standardize_for_metric`)
    so the factor approximates the same kernel the leakage scorers use.

    ``kernel_power`` p factorizes the elementwise power ``S**p`` instead of ``S``
    (still PSD for integer p, by the Schur product theorem). p > 1 sharpens the kernel
    toward its tail: strong (analog) pairs keep most of their weight while the weak
    background pairs that dominate the plain sum objective are suppressed.
    """
    import torch

    if landmark not in LANDMARKS:
        raise ValueError(f"unknown landmark method {landmark!r}; expected one of {LANDMARKS}")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError(f"rank must be an integer >= 1; got {rank!r}")
    if (isinstance(kernel_power, bool)
            or not isinstance(kernel_power, (int, np.integer)) or kernel_power < 1):
        raise ValueError("kernel_power must be an integer >= 1 (PSD Schur power)")
    if metric is None:
        metric = choose_metric(X)
    check_metric(metric)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = standardize_for_metric(torch.as_tensor(X, dtype=torch.float32, device=device), metric)
    n = Xt.shape[0]
    rank = min(rank, n)

    if landmark == "uniform":
        gen = torch.Generator(device=device).manual_seed(seed)
        idx = torch.randperm(n, generator=gen, device=device)[:rank].cpu().numpy()
    else:
        idx = _kmeanspp_landmarks(Xt, rank, metric, seed)
    L = Xt[torch.as_tensor(idx, device=device)]

    C = pairwise_similarity(Xt, L, metric)                  # (n, r) point-landmark
    W = pairwise_similarity(L, L, metric)                   # (r, r) landmark-landmark
    if kernel_power != 1:
        C, W = C ** kernel_power, W ** kernel_power
    evals, evecs = torch.linalg.eigh(W)
    evals = torch.clamp(evals, min=1e-6)
    W_inv_sqrt = (evecs * (1.0 / torch.sqrt(evals))[None, :]) @ evecs.T
    B = C @ W_inv_sqrt                                                     # (n, r)
    return B.cpu().numpy().astype(np.float32), metric
