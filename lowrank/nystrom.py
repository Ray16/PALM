"""Nyström low-rank factorization  S ~= B B^T  for the low-rank splitter.

Picks ``rank`` landmark rows, forms the point-landmark (C) and landmark-landmark
(W) similarity blocks, and returns ``B = C · W^{-1/2}`` so that ``B @ B.T ~= S``.
Tanimoto/cosine are valid PD kernels, so the leakage decomposes exactly in the
r-dim factor space. This module is where the *approximation* extensions land
(leverage-score landmarks, adaptive rank, ridge-regularized W).
"""

from __future__ import annotations

import numpy as np

from PALM.splitters.common.feature_preparation import choose_metric
from PALM.splitters.common.pairwise_similarity import pairwise_similarity


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


def nystrom_features(X, rank=256, metric=None, landmark="kmeans++", seed=0):
    """Nyström low-rank factor ``B`` with ``B @ B.T ~= S``. Returns ``(B, metric)``."""
    import torch

    if metric is None:
        metric = choose_metric(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
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
    evals, evecs = torch.linalg.eigh(W)
    evals = torch.clamp(evals, min=1e-6)
    W_inv_sqrt = (evecs * (1.0 / torch.sqrt(evals))[None, :]) @ evecs.T
    B = C @ W_inv_sqrt
    return B.cpu().numpy().astype(np.float32), metric
