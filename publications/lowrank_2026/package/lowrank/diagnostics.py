"""Bounded pair sampling, never an exact exclusion certificate.

Uniform off-diagonal pairs are sampled with replacement. Normal 95% intervals
use delta-method standard errors conditional on the data and supplied split.
They describe Monte Carlo variability, not future-experiment uncertainty.
"""
from __future__ import annotations
import numpy as np
from .splitters.common.pairwise_similarity import check_metric, standardize_for_metric


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")


def prepare_reference(X, metric, device="cpu"):
    import torch
    check_metric(metric)
    X = np.asarray(X)
    if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("features must be a nonempty finite two-dimensional matrix")
    if metric == "tanimoto" and (X < 0).any():
        raise ValueError("tanimoto requires non-negative features")
    return standardize_for_metric(torch.as_tensor(X, dtype=torch.float32, device=device), metric)


def sample_pairs(n, count, seed):
    positive_integer("sample_pairs", count)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    i = rng.integers(n, size=count)
    j = rng.integers(n - 1, size=count)
    j += j >= i
    return i, j


def paired_similarity(ref, i, j, metric, *, kernel_power=1, block_size=1024):
    """Evaluate aligned pairs without constructing a sample-by-sample matrix."""
    import torch
    positive_integer("kernel_power", kernel_power)
    positive_integer("block_size", block_size)
    out = np.empty(len(i), dtype=np.float64)
    for start in range(0, len(i), block_size):
        end = start + block_size
        a = ref[torch.as_tensor(i[start:end], device=ref.device)].double()
        b = ref[torch.as_tensor(j[start:end], device=ref.device)].double()
        if metric == "tanimoto":
            dot = (a * b).sum(1)
            union = a.sum(1) + b.sum(1) - dot
            sim = torch.where(union > 0, dot / union, torch.zeros_like(dot))
        elif metric == "cosine":
            sim = torch.nn.functional.cosine_similarity(a, b, dim=1)
        else:
            sim = 1 / (1 + torch.linalg.vector_norm(a - b, dim=1))
        out[start:end] = sim.pow(kernel_power).cpu().numpy()
    return out


def ratio_summary(numerator, denominator):
    """Approximate ratio interval; undefined mass remains explicit and JSON finite."""
    x, y = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ratio samples must be matching finite vectors")
    if not len(x):
        return dict(estimate=None, standard_error=None, ci95=None, status="no_offdiagonal_pairs")
    den = float(y.mean())
    if den <= np.finfo(float).eps:
        return dict(estimate=None, standard_error=None, ci95=None, status="nonpositive_or_negligible_mass")
    value = float(x.mean() / den)
    se = float(np.std((x - value * y) / den, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    return dict(estimate=value, standard_error=se,
                ci95=[value - 1.96 * se, value + 1.96 * se] if se is not None else None,
                status="estimated")


def factor_pair_error(B, ref, metric, *, pairs=2048, seed=0, kernel_power=1, block_size=1024):
    """Sampled entry MAE / mean absolute true powered similarity, with a normal CI.

    This is entrywise error, not cut error. Independent seeds must be used for
    adaptive rank selection and held-out reporting.
    """
    i, j = sample_pairs(len(ref), pairs, seed)
    exact = paired_similarity(ref, i, j, metric, kernel_power=kernel_power, block_size=block_size)
    residual = np.empty(len(i), dtype=np.float64)
    for s in range(0, len(i), block_size):
        a = np.asarray(B[i[s:s + block_size]], dtype=np.float64)
        b = np.asarray(B[j[s:s + block_size]], dtype=np.float64)
        residual[s:s + block_size] = np.abs(exact[s:s + block_size] - (a * b).sum(1))
    if len(exact) and np.all(exact == 0) and np.all(residual == 0):
        summary = dict(estimate=0., standard_error=0., ci95=[0., 0.], status="exact_zero_on_sample")
    else:
        summary = ratio_summary(residual, np.abs(exact))
    return dict(relative_pair_mae=summary, sample_pairs=len(i), seed=int(seed),
                normalization="mean absolute off-diagonal powered kernel similarity",
                interval="normal delta-method 95%; Monte Carlo only; not a uniform bound")


def sampled_leakage(X, labels, metric="tanimoto", *, sample_pairs_count=8192,
                    seed=0, kernel_power=1, block_size=1024):
    """O(samples*d) estimate of original-kernel cross-fold/total mass ratio.

    Power defaults to one independently of the optimization kernel. Nonpositive
    sampled total mass is undefined. Exact diagnostics are a separate operation.
    """
    ref = prepare_reference(X, metric)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(ref):
        raise ValueError("one label is required per feature row")
    i, j = sample_pairs(len(ref), sample_pairs_count, seed)
    weights = paired_similarity(ref, i, j, metric, kernel_power=kernel_power, block_size=block_size)
    result = ratio_summary(weights * (labels[i] != labels[j]), weights)
    result.update(sample_pairs=len(i), seed=int(seed), metric=metric, kernel_power=int(kernel_power),
                  interval="normal delta-method 95%; Monte Carlo only", exact=False)
    return result
