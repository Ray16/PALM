"""k-NN neighbour lists and weighted similarity-graph edges.

GPU (torch) construction with a CPU (sklearn) fallback, chunked over query rows
so peak memory is O(block * n) rather than O(n^2). These back the hypergraph and
graph-edge-cut splitters — the *construction* half of the method, where the
neighbourhood definition (k, metric, thresholding, degree caps) is tuned.

``metric`` is one of ``"tanimoto"`` / ``"cosine"`` / ``"euclidean"``; the
euclidean branch standardizes columns first (via
:func:`~PALM.splitters.common.pairwise_similarity.standardize_for_metric`) so a
raw ``1/(1+cdist)`` reproduces the historical hypergraph distances.
"""

from __future__ import annotations

import logging

import numpy as np

from PALM.splitters.common.pairwise_similarity import (pairwise_similarity,
                                                       standardize_for_metric)

logger = logging.getLogger(__name__)

# Similarity weights are floats in [0, 1]; Mt-KaHyPar needs positive integer
# edge weights, so we scale before rounding.
WEIGHT_SCALE = 1000


def knn_torch(X, k, metric, block=4096):
    """Top-k neighbours per row on the GPU. Returns ``(neigh[n,k], sims[n,k])``."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    k = min(k, n - 1)
    ref = standardize_for_metric(Xt, metric)

    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    all_sim = torch.empty((n, k), dtype=torch.float32, device=device)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = pairwise_similarity(ref[s:e], ref, metric)
        rows = torch.arange(s, e, device=device)
        sim[torch.arange(e - s, device=device), rows] = -1.0   # mask self
        ts, ti = torch.topk(sim, k, dim=1)
        all_sim[s:e], all_idx[s:e] = ts, ti

    return all_idx.cpu().numpy(), all_sim.clamp_min(0).cpu().numpy()


def knn_sklearn(X, k, metric):
    """CPU fallback k-NN. Returns ``(neigh[n,k], sims[n,k])``."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    k = min(k, n - 1)
    if metric == "tanimoto":
        nn = NearestNeighbors(n_neighbors=k + 1, metric="jaccard")
        Xf = X
    elif metric == "cosine":
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        Xf = X
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        Xf = StandardScaler().fit_transform(X)
    nn.fit(Xf)
    dist, idx = nn.kneighbors(Xf)
    idx, dist = idx[:, 1:], dist[:, 1:]      # drop the self column
    sim = 1.0 - dist if metric in ("tanimoto", "cosine") else 1.0 / (1.0 + dist)
    return idx, np.clip(sim, 0, None)


def k_nearest_neighbors(X, k, metric, use_gpu=True):
    """k-NN neighbours + similarities, GPU with sklearn fallback.

    Returns ``(neigh[n,k] int, sims[n,k] float)``.
    """
    if use_gpu:
        try:
            return knn_torch(X, k, metric)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("  GPU k-NN failed (%s); falling back to sklearn", exc)
    return knn_sklearn(X, k, metric)


def knn_graph_edges_torch(X, k, metric, threshold=None, max_deg=256, block=4096):
    """Directed (row -> col) similarity edges on the GPU.

    Keeps, per node, its top-k neighbours plus (if ``threshold`` is set) any
    neighbour with sim >= threshold, capped at ``max_deg`` per node. Returns
    ``(rows, cols, sims)`` numpy arrays for edges with sim > 0.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    kk = min(k, n - 1)
    K = min(max(kk, max_deg) if threshold is not None else kk, n - 1)
    ref = standardize_for_metric(Xt, metric)

    rows_all, cols_all, sims_all = [], [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = pairwise_similarity(ref[s:e], ref, metric)
        rows_g = torch.arange(s, e, device=device)
        sim[torch.arange(e - s, device=device), rows_g] = -1.0   # mask self
        vals, idx = torch.topk(sim, K, dim=1)
        ranks = torch.arange(K, device=device)[None, :]
        keep = ranks < kk
        if threshold is not None:
            keep = keep | (vals >= threshold)
        keep = keep & (vals > 0)
        bi, ki = torch.nonzero(keep, as_tuple=True)
        rows_all.append(rows_g[bi])
        cols_all.append(idx[bi, ki])
        sims_all.append(vals[bi, ki])

    rows = torch.cat(rows_all).cpu().numpy()
    cols = torch.cat(cols_all).cpu().numpy()
    sims = torch.cat(sims_all).clamp_min(0).cpu().numpy() if sims_all else np.array([])
    return rows, cols, sims


def build_knn_hyperedges(X, k=15, metric="tanimoto", use_gpu=True):
    """One weighted k-NN similarity hyperedge per node ({node} U k neighbours).

    Returns ``(hyperedges: list[list[int]], weights: list[int])``, weight =
    ``round(mean_sim * WEIGHT_SCALE) + 1`` (>= 1).
    """
    neigh, sims = k_nearest_neighbors(X, k, metric, use_gpu=use_gpu)
    hyperedges, weights = [], []
    for i in range(neigh.shape[0]):
        members = [i] + [int(j) for j in neigh[i]]
        w = int(round(float(sims[i].mean()) * WEIGHT_SCALE)) + 1
        hyperedges.append(members)
        weights.append(w)
    return hyperedges, weights


def build_knn_graph(X, k=15, metric="tanimoto", threshold=None, max_deg=256, use_gpu=True):
    """Symmetrized weighted k-NN graph. Returns ``(edges, weights)``.

    Each undirected edge weight is ``round(sim * WEIGHT_SCALE) + 1`` (keeping the
    max similarity per pair) so an integer CUT objective sums to the (scaled)
    cross-split similarity — the leakage numerator.
    """
    n = X.shape[0]
    if use_gpu:
        try:
            rows, cols, sims = knn_graph_edges_torch(X, k, metric, threshold, max_deg)
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("  GPU k-NN graph failed (%s); falling back to sklearn", exc)
            neigh, s = knn_sklearn(X, k, metric)
            rows = np.repeat(np.arange(n), neigh.shape[1])
            cols = neigh.ravel()
            sims = s.ravel()
    else:
        neigh, s = knn_sklearn(X, k, metric)
        rows = np.repeat(np.arange(n), neigh.shape[1])
        cols = neigh.ravel()
        sims = s.ravel()

    if len(rows) == 0:
        return [], []

    a = np.minimum(rows, cols).astype(np.int64)
    b = np.maximum(rows, cols).astype(np.int64)
    key = a * n + b
    uniq, inv = np.unique(key, return_inverse=True)
    w = np.zeros(len(uniq), dtype=np.float64)
    np.maximum.at(w, inv, sims)
    ua = (uniq // n).astype(int)
    ub = (uniq % n).astype(int)
    edges = list(zip(ua.tolist(), ub.tolist()))
    weights = [int(round(float(x) * WEIGHT_SCALE)) + 1 for x in w]
    return edges, weights
