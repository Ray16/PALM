"""Optional approximate candidates with exact Jaccard edge verification.

NN-Descent searches packed binary fingerprints. Only candidate discovery is
approximate: all returned edge weights are recomputed from bit intersections.
This does not certify recall or enforce constraints on unreturned neighbours.
"""
from __future__ import annotations

import numpy as np


def tiled_neighbours(X, metric, k=32, *, row_block=256, reference_block=8192,
                     device=None, queries=None):
    """Exact neighbours with bounded two-dimensional similarity tiles.

    Uses CUDA when available, or CPU otherwise. Arithmetic remains quadratic for
    all-row queries; temporary pairwise storage is O(row_block*reference_block).
    The complete feature matrix and output neighbour graph are still retained.
    Ties can choose different equal-similarity indices across devices/backends.
    """
    import torch
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity, standardize_for_metric
    for name, value in (("k", k), ("row_block", row_block), ("reference_block", reference_block)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = standardize_for_metric(torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device), metric)
    n = len(ref)
    k = min(k, max(0, n - 1))
    queries = np.arange(n) if queries is None else np.asarray(queries, dtype=np.int64)
    if queries.ndim != 1 or np.any((queries < 0) | (queries >= n)):
        raise ValueError("queries must be valid row indices")
    indices = np.empty((len(queries), k), dtype=np.int64)
    scores = np.empty((len(queries), k), dtype=np.float32)
    if k == 0:
        return indices, scores
    for start in range(0, len(queries), row_block):
        q = queries[start:start + row_block]
        best = torch.full((len(q), k), -torch.inf, device=device)
        ids = torch.zeros((len(q), k), dtype=torch.long, device=device)
        for r in range(0, n, reference_block):
            S = pairwise_similarity(ref[q], ref[r:r + reference_block], metric)
            self_rows = np.flatnonzero((q >= r) & (q < r + S.shape[1]))
            if len(self_rows):
                S[torch.as_tensor(self_rows, device=device),
                  torch.as_tensor(q[self_rows] - r, device=device)] = -torch.inf
            val, col = S.topk(min(k, S.shape[1]), dim=1)
            combined = torch.cat([best, val], dim=1)
            combined_ids = torch.cat([ids, col + r], dim=1)
            best, which = combined.topk(k, dim=1)
            ids = combined_ids.gather(1, which)
        indices[start:start + len(q)] = ids.cpu().numpy()
        scores[start:start + len(q)] = best.cpu().numpy()
    return indices, scores


def approximate_tanimoto_neighbours(X, k=32, *, candidates=64, seed=0, n_jobs=1):
    """Return (indices, similarities) for verified ANN candidates, self excluded.

    Requires the optional ``pynndescent`` dependency and binary input. Fixed
    ``n_jobs=1`` is recommended for reproducible construction. Candidate width
    controls the speed/recall tradeoff; it is not an exact search guarantee.
    """
    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError("ANN grouping requires pip install 'lowrank-split[ann]'") from exc

    X = np.asarray(X)
    if X.ndim != 2 or not np.all((X == 0) | (X == 1)):
        raise ValueError("ANN Tanimoto neighbours require a binary feature matrix")
    n = len(X)
    k = min(k, n - 1)
    if k < 1:
        return np.empty((n, 0), dtype=np.int64), np.empty((n, 0), dtype=np.float32)
    if candidates < k + 1:
        raise ValueError("must_link_candidates must be at least must_link_k + 1")
    packed = np.packbits(X.astype(np.uint8), axis=1)
    index = NNDescent(packed, metric="bit_jaccard", n_neighbors=min(n, candidates),
                      random_state=seed, n_jobs=n_jobs, low_memory=True,
                      max_candidates=min(60, candidates), verbose=False)
    idx = np.asarray(index.neighbor_graph[0], dtype=np.int64)
    popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    card = popcount[packed].sum(axis=1, dtype=np.int32)
    out_i = np.empty((n, k), dtype=np.int64)
    out_s = np.empty((n, k), dtype=np.float32)
    # Limit temporary packed candidate arrays, independent of full n.
    block = max(1, min(256, 8_000_000 // (idx.shape[1] * packed.shape[1])))
    for start in range(0, n, block):
        end = min(n, start + block)
        ids = idx[start:end]
        valid = (ids >= 0) & (ids < n) & (ids != np.arange(start, end)[:, None])
        safe = np.clip(ids, 0, n - 1)
        inter = popcount[np.bitwise_and(packed[start:end, None, :], packed[safe])].sum(
            axis=2, dtype=np.int32)
        union = card[start:end, None] + card[safe] - inter
        sim = np.divide(inter, union, out=np.zeros(inter.shape, dtype=np.float32), where=union > 0)
        sim[~valid] = -np.inf
        # Similarity then id: stable tie handling, independent of candidate order.
        order = np.lexsort((safe, -sim), axis=1)[:, :k]
        out_i[start:end] = np.take_along_axis(safe, order, axis=1)
        out_s[start:end] = np.take_along_axis(sim, order, axis=1)
    return out_i, out_s
