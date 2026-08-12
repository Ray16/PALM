"""The pairwise similarity kernel, in one place.

``pairwise_similarity(A, B, metric)`` returns the (|A| x |B|) similarity between
the rows of two torch tensors:

- ``tanimoto``  : Jaccard over binary vectors, ``inter / (|A| + |B| - inter)``
- ``cosine``    : inner product of L2-normalized rows
- ``euclidean`` : ``1 / (1 + dist)`` — a bounded similarity on the *raw* rows

This is the kernel the low-rank splitter has always used. It performs **no**
feature standardization; the euclidean branch is a raw ``1/(1+cdist)``. Callers
that want the hypergraph's standardized-euclidean behaviour prepare the matrix
with :func:`standardize_for_metric` first (the k-NN helpers do this).
"""

from __future__ import annotations


def _get_torch():
    import torch
    return torch


def pairwise_similarity(A, B, metric: str):
    """Pairwise similarity between rows of ``A`` (a x d) and ``B`` (b x d).

    ``A`` and ``B`` are torch float tensors on the same device. Returns an
    (a x b) torch tensor.
    """
    torch = _get_torch()
    if metric == "tanimoto":
        inter = A @ B.T
        card_a = A.sum(1)[:, None]
        card_b = B.sum(1)[None, :]
        union = card_a + card_b - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(inter))
    if metric == "cosine":
        An = torch.nn.functional.normalize(A, dim=1)
        Bn = torch.nn.functional.normalize(B, dim=1)
        return An @ Bn.T
    # euclidean -> bounded similarity on the raw rows
    return 1.0 / (1.0 + torch.cdist(A, B))


def standardize_for_metric(Xt, metric: str):
    """Return the reference matrix used by the k-NN builders for ``metric``.

    - ``euclidean`` : z-score each column, ``(X - mean) / (std + 1e-8)`` — so a
      subsequent raw ``1/(1+cdist)`` reproduces the hypergraph's standardized
      distance. (cosine/tanimoto need no prep; :func:`pairwise_similarity`
      normalizes cosine internally.)
    """
    if metric == "euclidean":
        return (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)
    return Xt
