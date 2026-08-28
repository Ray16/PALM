"""Shared helpers for the OMol25 leakage study: feature loading, non-negative
scaling, and DataSAIL-style L(pi) via the low-rank factorization (scalable) and
exact cosine (for validation).

L(pi) definition (DataSAIL form): the fraction of total pairwise similarity that
crosses split boundaries,
    L = 1 - sum_c ||p_c||^2 / ||s||^2 ,   p_c = sum_{i in block c} B_i,  s = sum_c p_c,
where B B^T approximates the similarity matrix. Similarity here is COSINE over a
non-negative structural descriptor (composition | elemental | 3D-RDF | charge/spin),
so it lies in [0, 1] like DataSAIL's Tanimoto — ECFP is not used because OMol25
has no SMILES and includes metal complexes RDKit can't reliably perceive.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = "/nfs/lambda_stor_01/homes/rzhu/PALM/data/omol25/_cache"


def load_features(mmap: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return (features [N,115], meta DataFrame) from the cache."""
    X = np.load(os.path.join(CACHE_DIR, "features.npy"), mmap_mode="r" if mmap else None)
    meta = pd.read_parquet(os.path.join(CACHE_DIR, "meta.parquet"))
    return X, meta


def fit_nonneg_scale(X: np.ndarray, n_sample: int = 300_000, seed: int = 0) -> np.ndarray:
    """Per-column 1/std scale (no centering, so features stay >= 0) from a sample.

    Equalizes the descriptor blocks (composition ~[0,1] vs mass ~10-100 vs natoms
    ~hundreds) so cosine similarity is not dominated by the large-magnitude columns.
    Returns the scale vector; apply as ``X * scale``.
    """
    n = X.shape[0]
    idx = np.random.default_rng(seed).choice(n, size=min(n_sample, n), replace=False)
    sample = np.asarray(X[np.sort(idx)], dtype=np.float64)
    std = sample.std(axis=0)
    std[std == 0] = 1.0
    return (1.0 / std).astype(np.float32)


def build_factor(X_scaled: np.ndarray, rank: int = 512, seed: int = 0,
                 chunk: int = 500_000):
    """Nystrom low-rank factor B (cosine kernel) for a possibly-huge X. Returns B [N,r].

    Landmarks are chosen on a manageable random subsample (k-means++ is O(n*r) and
    would be slow at ~10^7), then C = cos(X, L) is built in row chunks on GPU.
    """
    import torch
    from PALM.splitters.common.pairwise_similarity import pairwise_similarity

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, d = X_scaled.shape
    rank = min(rank, n)
    rng = np.random.default_rng(seed)
    land_idx = np.sort(rng.choice(n, size=rank, replace=False))
    L = torch.as_tensor(np.asarray(X_scaled[land_idx]), dtype=torch.float32, device=dev)

    W = pairwise_similarity(L, L, "cosine")                       # (r, r)
    evals, evecs = torch.linalg.eigh(W)
    evals = torch.clamp(evals, min=1e-6)
    W_inv_sqrt = (evecs * (1.0 / torch.sqrt(evals))[None, :]) @ evecs.T

    B = np.empty((n, rank), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xt = torch.as_tensor(np.asarray(X_scaled[s:e]), dtype=torch.float32, device=dev)
        C = pairwise_similarity(Xt, L, "cosine")                  # (chunk, r)
        B[s:e] = (C @ W_inv_sqrt).cpu().numpy()
    return B


def lpi_from_factor(B: np.ndarray, labels: np.ndarray, n_blocks: int,
                    chunk: int = 1_000_000) -> float:
    """DataSAIL-form L(pi) from a precomputed factor B (scalable, O(n*r))."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = B.shape[1]
    P = torch.zeros(n_blocks, r, device=dev)
    lab_t = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev)
    for s in range(0, B.shape[0], chunk):
        e = min(s + chunk, B.shape[0])
        Bt = torch.as_tensor(B[s:e], dtype=torch.float32, device=dev)
        P.index_add_(0, lab_t[s:e], Bt)
    svec = P.sum(0)
    total = float(svec @ svec)
    within = float((P * P).sum())
    return max(0.0, 1.0 - within / total) if total > 0 else 0.0


def lpi_exact_cosine(X_scaled: np.ndarray, labels: np.ndarray, block: int = 4096) -> float:
    """Exact cosine L(pi) (O(n^2)) — for validating the factorized estimate on a subsample."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xn = torch.nn.functional.normalize(
        torch.as_tensor(np.asarray(X_scaled), dtype=torch.float32, device=dev), dim=1)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev)
    n = Xn.shape[0]
    total = torch.zeros((), device=dev)
    cross = torch.zeros((), device=dev)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = Xn[s:e] @ Xn.T
        total += sim.sum()
        cross += (sim * (lab[s:e][:, None] != lab[None, :]).float()).sum()
    return float((cross / total).item())
