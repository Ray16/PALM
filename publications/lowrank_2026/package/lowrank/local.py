"""Experimental normalized global cut plus a positive local pairwise penalty.

This is deliberately distinct from the archived signed residual correction.
The union top-k graph receives a similarity ramp; each objective component is
normalized by its own off-diagonal mass before applying the explicit weight.
An absent local graph is reported, not interpreted as absence of all analogues.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .diagnostics import positive_integer


def local_graph(X, metric, *, k=32, threshold=.4, neighbours=None):
    positive_integer("local_k", k)
    if not np.isfinite(threshold) or not -1 <= threshold < 1:
        raise ValueError("local_threshold must be finite and in [-1, 1)")
    if neighbours is None:
        from .neighbours import tiled_neighbours
        neighbours = tiled_neighbours(X, metric, k)
    idx, sim = neighbours
    n = len(X)
    idx, sim = np.asarray(idx), np.asarray(sim)
    if idx.shape != sim.shape or idx.ndim != 2 or idx.shape[0] != n:
        raise ValueError("neighbour indices/scores must have shape (n, k)")
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    vals = np.maximum(0., (sim.ravel().astype(np.float64) - threshold) / (1 - threshold))
    valid = (rows != cols) & (cols >= 0) & (cols < n) & np.isfinite(vals) & (vals > 0)
    graph = sparse.coo_matrix((vals[valid], (rows[valid], cols[valid])), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0); graph.eliminate_zeros()
    mass = float(graph.sum() / 2)
    info = dict(edges=graph.nnz // 2, offdiagonal_mass=mass, k=int(k), threshold=float(threshold),
                definition="union top-k pairwise similarity ramp; no hyperedges",
                status="active" if mass > 0 else "no_positive_edges_in_graph")
    return graph, info


def normalize_components(B, graph, weight):
    """Return factor and sparse terms for L_global/mass_global + weight*L_local/mass_local."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("local_weight must be finite and non-negative")
    total = B.sum(0, dtype=np.float64)
    # Accumulate row norms in bounded blocks; avoid a full float64 factor copy.
    diagonal = sum(float(np.square(B[s:s + 2048].astype(np.float64)).sum())
                   for s in range(0, len(B), 2048))
    global_mass = .5 * (float(total @ total) - diagonal)
    if global_mass <= np.finfo(float).eps:
        raise ValueError("normalized local objective requires positive global factor mass")
    local_mass = float(graph.sum() / 2)
    return (B / np.float32(np.sqrt(global_mass)),
            graph * (weight / local_mass) if local_mass > 0 else graph.copy(),
            dict(global_factor_mass=global_mass, local_mass=local_mass, weight=float(weight)))


class SparseTerm:
    """Small adapter for exact sparse contributions inside the existing optimizer."""
    def __init__(self, matrix, device="cpu"):
        import torch
        self.R = sparse.csr_matrix(matrix, dtype=np.float64)
        self.R.sum_duplicates(); self.R.sort_indices()
        if (self.R.shape[0] != self.R.shape[1] or not np.isfinite(self.R.data).all()
                or np.any(self.R.data < 0) or np.any(self.R.diagonal() != 0)
                or (self.R - self.R.T).nnz):
            raise ValueError("local penalty must be finite, non-negative, symmetric and zero-diagonal")
        self.device, self.torch = device, torch

    def block_mass(self, labels, k):
        lab = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
        onehot = sparse.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), k))
        return self.torch.as_tensor((self.R @ onehot).toarray(), dtype=self.torch.float32, device=self.device)

    def column(self, v):
        out = self.torch.zeros(self.R.shape[0], dtype=self.torch.float32, device=self.device)
        start, end = self.R.indptr[v:v + 2]
        ix = self.torch.as_tensor(self.R.indices[start:end], device=self.device)
        out[ix] = self.torch.as_tensor(self.R.data[start:end], dtype=out.dtype, device=self.device)
        return out

    def cross(self, labels):
        lab = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
        coo = self.R.tocoo()
        return .5 * float(coo.data[lab[coo.row] != lab[coo.col]].sum())
