"""Dense/CSR factor operations without materializing a pairwise kernel.

Sparse factors stay sparse. Only fold sums (rank by number of folds), row/fold
scores and individual similarity columns become dense. This is essential for
exact categorical identity factors with many distinct repeated values.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse


def validate_fold_weights(weights, k):
    W = (
        np.ones((k, k)) - np.eye(k)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        W.shape != (k, k)
        or not np.isfinite(W).all()
        or (W < 0).any()
        or not np.allclose(W, W.T, rtol=0, atol=1e-12)
        or np.any(np.diag(W) != 0)
        or not np.any(W > 0)
    ):
        raise ValueError(
            "fold_weights must be a finite, non-negative, symmetric k-by-k matrix with zero diagonal and a positive pair"
        )
    return 0.5 * (W + W.T)


class FactorOps:
    def __init__(self, B):
        import torch

        self.torch = torch
        self.sparse = sparse.issparse(B)
        if self.sparse:
            self.B = sparse.csr_matrix(B, dtype=np.float32)
            self.B.sum_duplicates()
            self.B.sort_indices()
            finite = np.isfinite(self.B.data).all()
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.B = torch.as_tensor(B, dtype=torch.float32, device=self.device)
            finite = bool(torch.isfinite(self.B).all())
        self.shape = self.B.shape
        if len(self.shape) != 2 or self.shape[0] < 1 or not finite:
            raise ValueError("factor must be a nonempty finite dense or CSR matrix")

    def labels(self, labels, k):
        lab = (
            labels.detach().cpu().numpy()
            if hasattr(labels, "detach")
            else np.asarray(labels)
        )
        if (
            lab.shape != (self.shape[0],)
            or not np.issubdtype(lab.dtype, np.integer)
            or np.any((lab < 0) | (lab >= k))
        ):
            raise ValueError(
                "factor labels must be one valid integer fold index per row"
            )
        return self.torch.as_tensor(
            lab, dtype=self.torch.long, device=self.device
        ).clone()

    def sums(self, labels, k, precise=False):
        torch = self.torch
        if self.sparse:
            lab = labels.cpu().numpy()
            onehot = np.zeros(
                (self.shape[0], k), dtype=np.float64 if precise else np.float32
            )
            onehot[np.arange(len(lab)), lab] = 1
            result = np.asarray(self.B.T @ onehot).T
            return torch.as_tensor(result, device=self.device)
        dtype = torch.float64 if precise else self.B.dtype
        P = torch.zeros(k, self.shape[1], dtype=dtype, device=self.device)
        if precise:
            for start in range(0, self.shape[0], 4096):
                P.index_add_(
                    0,
                    labels[start : start + 4096],
                    self.B[start : start + 4096].double(),
                )
        else:
            P.index_add_(0, labels, self.B)
        return P

    def mass(self, labels, k):
        P = self.sums(labels, k)
        if self.sparse:
            return self.torch.as_tensor(
                np.asarray(self.B @ P.numpy().T), dtype=self.torch.float32
            )
        return self.B @ P.T

    def diagonal(self):
        if self.sparse:
            return self.torch.as_tensor(
                np.asarray(self.B.multiply(self.B).sum(1)).ravel()
            )
        return self.torch.einsum("ij,ij->i", self.B, self.B)

    def column(self, v):
        if self.sparse:
            return self.torch.as_tensor((self.B @ self.B[v].T).toarray().ravel())
        return self.B @ self.B[v]

    def cost(self, labels, k, weights=None):
        W = validate_fold_weights(weights, k)
        P = self.sums(self.labels(labels, k), k, precise=True)
        gram = (P @ P.T).cpu().numpy()
        return 0.5 * float(np.sum(W * gram))


def factor_ops(B):
    return B if isinstance(B, FactorOps) else FactorOps(B)


def offdiagonal_mass(B):
    if sparse.issparse(B):
        total = np.asarray(B.sum(0, dtype=np.float64)).ravel()
        diagonal = float(B.astype(np.float64).multiply(B).sum())
    else:
        B = np.asarray(B)
        total = B.sum(0, dtype=np.float64)
        diagonal = sum(
            float(np.square(B[s : s + 4096].astype(np.float64)).sum())
            for s in range(0, len(B), 4096)
        )
    return 0.5 * (float(total @ total) - diagonal)
