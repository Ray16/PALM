"""Direct dense-kernel oracle for the unchanged current optimizer.

This adapter is an experimental control, not a replacement public splitter.
It deliberately stores S rather than factoring it or computing a full square root.
"""
from __future__ import annotations

import numpy as np
import torch

from lowrank.factors import FactorOps, validate_fold_weights


class DenseOps(FactorOps):
    def __init__(self, kernel):
        self.torch = torch
        self.K = torch.as_tensor(kernel, dtype=torch.float32, device="cpu")
        if self.K.ndim != 2 or self.K.shape[0] != self.K.shape[1]:
            raise ValueError("square kernel required")
        if not torch.isfinite(self.K).all():
            raise ValueError("finite kernel required")
        self.shape = (len(self.K), len(self.K))
        self.device, self.sparse = "cpu", False

    def mass(self, labels, k):
        onehot = torch.nn.functional.one_hot(labels, k).to(self.K.dtype)
        return self.K @ onehot

    def diagonal(self):
        return self.K.diagonal()

    def column(self, v):
        return self.K[:, v]

    def cost(self, labels, k, weights=None):
        lab = self.labels(labels, k)
        W = validate_fold_weights(weights, k)
        # Sum each unordered crossing block once in float64. No n-by-n float64
        # copy or factorization is hidden in the direct-kernel control.
        members = [torch.nonzero(lab == c).ravel() for c in range(k)]
        value = 0.0
        for a in range(k):
            for b in range(a + 1, k):
                value += W[a, b] * float(
                    self.K[members[a]][:, members[b]].sum(dtype=torch.float64)
                )
        return value


def independent_metrics(X, labels, power=1, B=None, block=256):
    """Full NumPy float64 original-kernel evaluation, independent of optimizer.

    The same returned partition is scored under S and S**power. Approximate
    cuts are calculated only from factor sums, with errors against S**power.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(X)
    if labels.shape != (n,) or np.any(labels < 0) or np.any(labels > 2):
        raise ValueError("invalid or missing labels")
    counts = np.bincount(labels, minlength=3)
    norm = np.einsum("ij,ij->i", X, X)
    raw, cross, total, powered_total, powered_cross = 0., 0., 0., 0., 0.
    nearest = []
    train = labels == 0
    for s in range(0, n, block):
        e = min(s + block, n)
        dot = X[s:e] @ X.T
        den = norm[s:e, None] + norm[None, :] - dot
        S = np.divide(dot, den, out=np.zeros_like(dot), where=den > 0)
        # All published molecular inputs have nonzero fingerprints.
        S[np.arange(e-s), np.arange(s, e)] = 0
        mask = labels[s:e, None] != labels[None, :]
        total += float(S.sum())
        cross += float(S[mask].sum())
        P = S if power == 1 else S ** power
        powered_total += float(P.sum())
        powered_cross += float(P[mask].sum())
        q = labels[s:e] == 2
        if q.any() and train.any():
            nearest.extend(S[q][:, train].max(1).tolist())
    mass, cut, pmass, pcut = total/2, cross/2, powered_total/2, powered_cross/2
    random_fraction = (n*n - float(counts @ counts)) / (n*(n-1)) if n > 1 else 0
    result = dict(
        counts=counts.tolist(), retention=1., true_cut=cut,
        true_powered_cut=pcut, kernel_mass=mass, powered_kernel_mass=pmass,
        global_relative=cut / mass / random_fraction if mass and random_fraction else None,
        mean_nearest=float(np.mean(nearest)) if nearest else None,
        nearest_gt06=float(np.mean(np.array(nearest) > .6)) if nearest else None,
    )
    if B is not None:
        sums = np.zeros((3, B.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, np.asarray(B, dtype=np.float64))
        approx = sum(float(sums[a] @ sums[b]) for a in range(3) for b in range(a+1,3))
        error = abs(pcut - approx)
        result.update(approximate_cut=approx, absolute_cut_error=error,
                      cut_relative_error=error/pcut if pcut > 0 else None,
                      cut_relative_status="defined" if pcut > 0 else "zero_true_cut",
                      mass_relative_error=error/pmass if pmass > 0 else None)
    return result


def build_dense(X, power):
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity
    Xt = torch.as_tensor(X, dtype=torch.float32)
    S = torch.empty((len(X), len(X)), dtype=torch.float32)
    for s in range(0, len(X), 256):
        tile = pairwise_similarity(Xt[s:s+256], Xt, "tanimoto")
        S[s:s+len(tile)] = tile if power == 1 else tile.pow(power)
    return S
