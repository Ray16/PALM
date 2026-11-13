"""Sparse k-NN correction for the low-rank kernel:  K ~= B Bᵀ + R.

The Nyström factor B captures the *global* (meso-scale) structure of the kernel, but a
rank-r sketch cannot resolve many tiny tight groups (near-duplicates, small analog
series): their strong pairs are smeared out, so the sum objective is blind to them —
exactly the local leakage low-rank was worst at (comparisons/FINDINGS.md). ``R`` puts that
part back exactly where it matters: on the exact top-k neighbour pairs it holds the
residual ``K_ij − b_i·b_j``, so ``BBᵀ + R`` equals the true kernel on every k-NN pair and
the Nyström approximation elsewhere. Optionally, edges with ``sim >= tau`` get an extra
``boost`` — a soft must-link that makes cutting a near-analog pair expensive.

R is symmetric with zero diagonal and O(n·k) non-zeros, so the optimizer stays O(n·(r+k)).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse as sp


def sparse_residual(X, B, metric: str, k: int = 16, kernel_power: int = 1,
                    tau: Optional[float] = None, boost: float = 0.0,
                    nbrs=None) -> sp.csr_matrix:
    """Symmetric sparse residual ``R`` on the exact top-k graph (see module docstring)."""
    from lowrank.grouped import topk_neighbours

    n = len(X)
    idx, sim = nbrs if nbrs is not None else topk_neighbours(X, metric, k=k)
    idx, sim = idx[:, :k], sim[:, :k].astype(np.float64)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    kern = np.clip(sim.ravel(), 0.0, None) ** kernel_power
    approx = np.einsum("ij,ij->i", B[rows].astype(np.float64), B[cols].astype(np.float64))
    val = kern - approx
    if tau is not None and boost > 0:
        val = val + boost * (sim.ravel() >= tau)
    # symmetrize: each unordered pair once (duplicates from i->j and j->i are averaged)
    R = sp.coo_matrix((val, (rows, cols)), shape=(n, n)).tocsr()
    cnt = sp.coo_matrix((np.ones_like(val), (rows, cols)), shape=(n, n)).tocsr()
    R, cnt = R + R.T, cnt + cnt.T
    R.data /= cnt.data
    R.setdiag(0.0)
    R.eliminate_zeros()
    return R.tocsr()


class SparseTerm:
    """Torch-side helpers for R inside the Lloyd / FM loops."""

    def __init__(self, R: sp.csr_matrix, device):
        import torch

        self.R = R.tocsr()
        self.device = device
        self.torch = torch

    def block_mass(self, lab, k):
        """(n, k): Σ_j R_ij [lab_j == c]."""
        lab_np = lab.cpu().numpy() if hasattr(lab, "cpu") else np.asarray(lab)
        onehot = sp.csr_matrix((np.ones(len(lab_np)), (np.arange(len(lab_np)), lab_np)),
                               shape=(len(lab_np), k))
        out = (self.R @ onehot).toarray()
        return self.torch.as_tensor(out, dtype=self.torch.float32, device=self.device)

    def column(self, v):
        """Dense (n,) column R[:, v] (R is symmetric, so this is row v)."""
        col = self.torch.zeros(self.R.shape[0], dtype=self.torch.float32, device=self.device)
        s, e = self.R.indptr[v], self.R.indptr[v + 1]
        if e > s:
            col[self.torch.as_tensor(self.R.indices[s:e], device=self.device)] = \
                self.torch.as_tensor(self.R.data[s:e], dtype=self.torch.float32,
                                     device=self.device)
        return col

    def cross(self, labels):
        """Σ over cross-block unordered pairs of R_ij."""
        lab = np.asarray(labels)
        coo = self.R.tocoo()
        m = lab[coo.row] != lab[coo.col]
        return 0.5 * float(coo.data[m].sum())
