"""Nyström low-rank factorization  S ~= B B^T  for the low-rank splitter.

Picks ``rank`` landmark rows, forms the point-landmark (C) and landmark-landmark
(W) similarity blocks, and returns ``B = C · W^{-1/2}`` so that ``B @ B.T ~= S``.
Tanimoto/cosine are valid PD kernels, so the leakage decomposes exactly in the
r-dim factor space.

Approximation quality controls (Step 2):
- ``landmark``: ``kmeans++`` (default), ``uniform``, or ``leverage`` (approximate
  ridge-leverage-score sampling — the theoretically-preferred Nyström sketch).
- ``ridge``: regularize ``W^{-1/2}`` by adding ``ridge·λ_max`` to the eigenvalues,
  instead of only clamping tiny ones (more stable factor at small eigenvalues).
- ``energy``: adaptive rank — keep only the top eigen-directions capturing this
  fraction of W's spectral energy, so the factor width matches the kernel's
  effective rank.

**Fidelity bound (why this matters).** The factor-space objective equals the true
cross-split similarity up to the Nyström reconstruction error: for any split,
``|factor_leakage − true_leakage| ≤ n·‖S − BBᵀ‖₂``. Tightening ``S ≈ BBᵀ`` (better
landmarks / rank / ridge) therefore tightens how faithfully the optimizer's
objective tracks the leakage we actually report — a prerequisite for *targeting* a
leakage value (Step 3). ``experiments/nystrom_fidelity.py`` measures it.
"""

from __future__ import annotations

from typing import Optional

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


def _leverage_landmarks(Xt, n_landmarks, metric, seed, oversample=4):
    """Approximate ridge-leverage-score landmark sampling (Musco-Musco style).

    Builds a preliminary uniform sketch of ``oversample·rank`` columns, estimates
    each row's ridge leverage from it, then samples ``rank`` landmarks ∝ leverage.
    O(n·s) memory/time, no full kernel.
    """
    import torch

    n = Xt.shape[0]
    n_landmarks = min(n_landmarks, n)
    s = min(oversample * n_landmarks, n)
    gen = torch.Generator(device=Xt.device).manual_seed(seed)
    idx0 = torch.randperm(n, generator=gen, device=Xt.device)[:s]
    L0 = Xt[idx0]
    C0 = pairwise_similarity(Xt, L0, metric)              # (n, s)
    W0 = pairwise_similarity(L0, L0, metric)              # (s, s)
    lam = float(torch.diagonal(W0).mean()) * 0.1 + 1e-6   # ridge
    W0_reg_inv = torch.linalg.pinv(W0 + lam * torch.eye(s, device=Xt.device, dtype=W0.dtype))
    tau = ((C0 @ W0_reg_inv) * C0).sum(1)                 # (n,) approx ridge leverage
    tau = torch.clamp(tau, min=0.0)
    if float(tau.sum()) <= 0:
        tau = torch.ones_like(tau)
    idx = torch.multinomial(tau, n_landmarks, replacement=False, generator=gen)
    return idx.cpu().numpy().astype(np.int64)


def nystrom_features(X, rank=256, metric=None, landmark="kmeans++", seed=0,
                     ridge: float = 0.0, energy: Optional[float] = None):
    """Nyström low-rank factor ``B`` with ``B @ B.T ~= S``. Returns ``(B, metric)``.

    ``landmark`` ∈ {kmeans++, uniform, leverage}; ``ridge`` ≥ 0 regularizes
    ``W^{-1/2}`` (fraction of λ_max added to eigenvalues); ``energy`` ∈ (0,1) selects
    an adaptive rank capturing that fraction of W's spectrum. Defaults reproduce the
    original factorization exactly.
    """
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
    elif landmark == "leverage":
        idx = _leverage_landmarks(Xt, rank, metric, seed)
    else:
        idx = _kmeanspp_landmarks(Xt, rank, metric, seed)
    L = Xt[torch.as_tensor(idx, device=device)]

    C = pairwise_similarity(Xt, L, metric)                  # (n, r) point-landmark
    W = pairwise_similarity(L, L, metric)                   # (r, r) landmark-landmark
    evals, evecs = torch.linalg.eigh(W)
    evals = torch.clamp(evals, min=1e-6)
    if ridge > 0:
        evals = evals + ridge * float(evals.max())

    if energy is not None and 0.0 < energy < 1.0:
        # adaptive rank: keep the top eigen-directions capturing `energy` of the spectrum
        order = torch.argsort(evals, descending=True)
        cum = torch.cumsum(evals[order], 0) / evals.sum()
        k = int((cum < energy).sum().item()) + 1
        keep = order[:k]
        B = C @ (evecs[:, keep] * (1.0 / torch.sqrt(evals[keep]))[None, :])   # (n, k)
    else:
        W_inv_sqrt = (evecs * (1.0 / torch.sqrt(evals))[None, :]) @ evecs.T
        B = C @ W_inv_sqrt                                                     # (n, r)
    return B.cpu().numpy().astype(np.float32), metric
