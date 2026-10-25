"""The pairwise similarity kernel, in one place.

``pairwise_similarity(A, B, metric)`` returns the (|A| x |B|) similarity between
the rows of two torch tensors:

- ``tanimoto``  : Jaccard over binary vectors, ``inter / (|A| + |B| - inter)``
- ``cosine``    : inner product of L2-normalized rows
- ``euclidean`` : ``1 / (1 + dist)`` — a bounded similarity on the *raw* rows

This is the kernel the low-rank splitter has always used. It performs **no**
feature standardization; the euclidean branch is a raw ``1/(1+cdist)``. Callers
that want the standardized-euclidean behaviour prepare the matrix
with :func:`standardize_for_metric` first (the k-NN helpers do this).
"""

from __future__ import annotations

KNOWN_METRICS = ("tanimoto", "cosine", "euclidean")
COSINE_EPS = 1e-12


def validate_features(X, metric):
    """Validate once at an API boundary; kernels below assume validated tensors."""
    import numpy as np

    check_metric(metric)
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("features must be a nonempty finite two-dimensional matrix")
    if metric == "tanimoto" and not np.all((X == 0) | (X == 1)):
        raise ValueError(
            "tanimoto requires binary 0/1 fingerprints; negative/count-valued features are unsupported"
        )
    return X


def normalize_cosine(X):
    torch = _get_torch()
    # Accumulate the norm in float64 so large finite float32 inputs do not
    # overflow to an infinite norm and silently become zero vectors.
    work = X.double()
    norm = torch.linalg.vector_norm(work, dim=1, keepdim=True).clamp_min(COSINE_EPS)
    return (work / norm).to(X.dtype)


def aligned_similarity(A, B, metric):
    """Same kernel as pairwise_similarity, evaluated only for aligned row pairs."""
    torch = _get_torch()
    check_metric(metric)
    if metric == "tanimoto":
        dot = (A * B).sum(1)
        union = A.sum(1) + B.sum(1) - dot
        return torch.where(union > 0, dot / union, torch.zeros_like(dot))
    if metric == "cosine":
        return (normalize_cosine(A) * normalize_cosine(B)).sum(1)
    return 1 / (1 + torch.linalg.vector_norm(A - B, dim=1))


def check_metric(metric: str) -> str:
    """Return ``metric`` if it is a known kernel name, else raise ``ValueError``."""
    if metric not in KNOWN_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {KNOWN_METRICS}")
    return metric


def _get_torch():
    import torch

    return torch


def pairwise_similarity(A, B, metric: str):
    """Pairwise similarity between rows of ``A`` (a x d) and ``B`` (b x d).

    ``A`` and ``B`` are torch float tensors on the same device. Returns an
    (a x b) torch tensor.
    """
    torch = _get_torch()
    check_metric(metric)
    if metric == "tanimoto":
        inter = A @ B.T
        card_a = A.sum(1)[:, None]
        card_b = B.sum(1)[None, :]
        union = card_a + card_b - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(inter))
    if metric == "cosine":
        An = normalize_cosine(A)
        Bn = normalize_cosine(B)
        return An @ Bn.T
    # euclidean -> bounded similarity on the raw rows
    return 1.0 / (1.0 + torch.cdist(A, B))


def standardize_for_metric(Xt, metric: str):
    """Return the reference matrix used by the k-NN builders for ``metric``.

    - ``euclidean`` : z-score each column, ``(X - mean) / (std + 1e-8)`` — so a
      subsequent raw ``1/(1+cdist)`` reproduces the standardized
      distance. (cosine/tanimoto need no prep; :func:`pairwise_similarity`
      normalizes cosine internally.)
    """
    if metric == "euclidean":
        # Keep the historical sample scaling for n>1, while a single featured
        # record (possible on an n-D axis) must not produce NaN descriptors.
        correction = 1 if Xt.shape[0] > 1 else 0
        # Scale before computing statistics to keep squared deviations finite.
        # eps/scale preserves the original (std + eps) regularization in input units.
        tiny = _get_torch().finfo(Xt.dtype).tiny
        scale = Xt.abs().amax(0).clamp_min(tiny)
        work = Xt / scale
        denominator = (work.std(0, correction=correction) + 1e-8 / scale).clamp_min(
            tiny
        )
        return (work - work.mean(0)) / denominator
    return Xt
