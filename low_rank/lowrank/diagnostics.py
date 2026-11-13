"""Support-aware exact checks and conservative bounded-sample diagnostics.

Known zero rows can be excluded because both numerator and denominator vanish
outside their support. Small pair populations are enumerated; otherwise pairs
are sampled uniformly with replacement. Empirical Bernstein mean bounds
(Maurer & Pontil, 2009, Theorem 4), with a union bound over both tails and ratio
components, replace zero-width normal intervals. They are conditional on a fixed
factor/partition, not a guarantee for every possible split or future data.
"""

from __future__ import annotations
import numpy as np
from .splitters.common.pairwise_similarity import (
    aligned_similarity,
    standardize_for_metric,
    validate_features,
)

EXACT_PAIR_LIMIT = 65536


def positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")


def prepare_reference(X, metric, device="cpu"):
    import torch

    X = validate_features(X, metric)
    ref = torch.as_tensor(X, dtype=torch.float32, device=device)
    if not torch.isfinite(ref).all():
        raise ValueError("features exceed the supported float32 range")
    return standardize_for_metric(ref, metric)


def sample_pairs(n, count, seed):
    positive_integer("sample_pairs", count)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    i = rng.integers(n, size=count)
    j = rng.integers(n - 1, size=count)
    return i, j + (j >= i)


def _audit_pairs(ref, metric, count, seed, B=None):
    """Condition only on analytically zero contributions, never sampled zero mass."""
    positive_integer("sample_pairs", count)
    pool = np.arange(len(ref))
    if metric in ("tanimoto", "cosine"):
        active = ref.ne(0).any(1).cpu().numpy()
        outside = False
        if B is not None:
            for start in range(0, len(ref), 1024):
                if np.any(
                    np.asarray(B)[start : start + 1024][~active[start : start + 1024]]
                    != 0
                ):
                    outside = True
                    break
        if not outside:
            pool = np.flatnonzero(active)
    population = len(pool) * (len(pool) - 1)
    if population <= EXACT_PAIR_LIMIT:
        # At most 65536 aligned evaluations, independent of full dataset size.
        i = np.repeat(np.arange(len(pool)), len(pool))
        j = np.tile(np.arange(len(pool)), len(pool))
        keep = i != j
        return pool[i[keep]], pool[j[keep]], True, len(pool)
    i, j = sample_pairs(len(pool), count, seed)
    return pool[i], pool[j], False, len(pool)


def paired_similarity(ref, i, j, metric, *, kernel_power=1, block_size=1024):
    import torch

    positive_integer("kernel_power", kernel_power)
    positive_integer("block_size", block_size)
    out = np.empty(len(i), dtype=np.float64)
    for start in range(0, len(i), block_size):
        a = ref[
            torch.as_tensor(i[start : start + block_size], device=ref.device)
        ].double()
        b = ref[
            torch.as_tensor(j[start : start + block_size], device=ref.device)
        ].double()
        out[start : start + block_size] = (
            aligned_similarity(a, b, metric).pow(kernel_power).cpu().numpy()
        )
    return out


def _mean_interval(x, bounds, alpha):
    low, high = bounds
    mean = float(x.mean())
    if len(x) < 2:
        return low, high
    log = np.log(4 / alpha)  # two one-sided Theorem-4 bounds, each at alpha/2
    radius = np.sqrt(2 * float(x.var(ddof=1)) * log / len(x))
    radius += 7 * (high - low) * log / (3 * (len(x) - 1))
    return max(low, mean - radius), min(high, mean + radius)


def ratio_summary(
    numerator,
    denominator,
    *,
    numerator_bounds=(0.0, 1.0),
    denominator_bounds=(0.0, 1.0),
    confidence=0.95,
    exact=False,
):
    x, y = (
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    if (
        x.shape != y.shape
        or x.ndim != 1
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("ratio samples must be matching finite vectors")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    for values, bounds in ((x, numerator_bounds), (y, denominator_bounds)):
        if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("sample bounds must be finite and ordered")
        if np.any(values < bounds[0] - 1e-10) or np.any(values > bounds[1] + 1e-10):
            raise ValueError("sample violates its declared kernel bound")
    common = dict(
        confidence=float(confidence),
        exact=bool(exact),
        interval="exact population"
        if exact
        else "empirical Bernstein; simultaneous ratio-component bounds",
    )
    if not len(x) or float(y.mean()) <= np.finfo(float).eps:
        return dict(
            estimate=None,
            standard_error=None,
            ci95=None,
            status="no_positive_mass" if exact else "insufficient_mass",
            **common,
        )
    den = float(y.mean())
    value = float(x.mean() / den)
    se = (
        float(np.std((x - value * y) / den, ddof=1) / np.sqrt(len(x)))
        if len(x) > 1
        else None
    )
    if exact:
        return dict(
            estimate=value,
            standard_error=0.0,
            ci95=[value, value],
            status="exact",
            **common,
        )
    # Each component receives half the total failure probability; dependence
    # between numerator and denominator is allowed by the union bound.
    nx = _mean_interval(x, numerator_bounds, (1 - confidence) / 2)
    dy = _mean_interval(y, denominator_bounds, (1 - confidence) / 2)
    interval = None
    if dy[0] > 0:
        corners = [a / b for a in nx for b in dy]
        interval = [min(corners), max(corners)]
    return dict(
        estimate=value,
        standard_error=se,
        ci95=interval,
        status="estimated" if interval is not None else "insufficient_mass",
        **common,
    )


def factor_pair_error(
    B,
    ref,
    metric,
    *,
    pairs=2048,
    seed=0,
    kernel_power=1,
    block_size=1024,
    confidence=0.95,
    labels=None,
    fold_weights=None,
):
    """Entrywise MAE, or the absolute returned-cut error, relative to |kernel| mass.

    Exact checks enumerate the analytically nonzero support for small populations.
    For sampled cuts, bound the signed residual mean before taking its absolute
    value; this estimates cut error rather than the larger entrywise MAE.
    """
    B = np.asarray(B)
    if B.ndim != 2 or len(B) != len(ref) or not np.isfinite(B).all():
        raise ValueError(
            "audit factor must be a finite matrix aligned to reference rows"
        )
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(ref),):
            raise ValueError(
                "audit partition must have one fold label per reference row"
            )
        if fold_weights is not None:
            from .factors import validate_fold_weights

            fold_weights = validate_fold_weights(fold_weights, len(fold_weights))
            if not np.issubdtype(labels.dtype, np.integer) or np.any(
                (labels < 0) | (labels >= len(fold_weights))
            ):
                raise ValueError(
                    "weighted audit labels must be integer indices into fold_weights"
                )
    i, j, exact_population, active_rows = _audit_pairs(ref, metric, pairs, seed, B)
    truth = paired_similarity(
        ref, i, j, metric, kernel_power=kernel_power, block_size=block_size
    )
    residual = np.empty(len(i), dtype=np.float64)
    for s in range(0, len(i), block_size):
        a, b = (
            B[i[s : s + block_size]].astype(np.float64),
            B[j[s : s + block_size]].astype(np.float64),
        )
        residual[s : s + block_size] = truth[s : s + block_size] - (a * b).sum(1)
    maximum_norm = max(
        (
            float(np.square(B[s : s + block_size].astype(np.float64)).sum(1).max())
            for s in range(0, len(B), block_size)
        ),
        default=0.0,
    )
    bound = 1.0 + maximum_norm
    if labels is None:
        numerator, nb = np.abs(residual), (0.0, bound)
    else:
        labels = np.asarray(labels)
        weights = (
            (labels[i] != labels[j]).astype(float)
            if fold_weights is None
            else fold_weights[labels[i], labels[j]]
        )
        numerator = residual * weights
        bound *= (
            max(1.0, float(np.max(fold_weights))) if fold_weights is not None else 1.0
        )
        nb = (-bound, bound)
    if exact_population and np.all(truth == 0) and np.all(residual == 0):
        summary = dict(
            estimate=0.0,
            standard_error=0.0,
            ci95=[0.0, 0.0],
            status="proven_zero_offdiagonal",
            exact=True,
        )
    else:
        summary = ratio_summary(
            numerator,
            np.abs(truth),
            numerator_bounds=nb,
            confidence=confidence,
            exact=exact_population,
        )
    if labels is not None and summary["estimate"] is not None:
        summary["estimate"] = abs(summary["estimate"])
        if summary["ci95"] is not None:
            lo, hi = summary["ci95"]
            summary["ci95"] = [
                0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)),
                max(abs(lo), abs(hi)),
            ]
    return dict(
        relative_pair_mae=summary,
        criterion="cut_error" if labels is not None else "pair_mae",
        sample_pairs=len(i),
        eligible_rows=active_rows,
        seed=int(seed),
        normalization="mean absolute off-diagonal powered kernel similarity",
        exact=exact_population,
        confidence=float(confidence),
    )


def sampled_leakage(
    X,
    labels,
    metric="tanimoto",
    *,
    sample_pairs_count=8192,
    seed=0,
    kernel_power=1,
    block_size=1024,
):
    ref = prepare_reference(X, metric)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(ref):
        raise ValueError("one label is required per feature row")
    i, j, exact, active = _audit_pairs(ref, metric, sample_pairs_count, seed)
    weights = paired_similarity(
        ref, i, j, metric, kernel_power=kernel_power, block_size=block_size
    )
    bounds = (-1.0, 1.0) if metric == "cosine" and kernel_power % 2 else (0.0, 1.0)
    result = ratio_summary(
        weights * (labels[i] != labels[j]),
        weights,
        numerator_bounds=bounds,
        denominator_bounds=bounds,
        exact=exact,
    )
    result.update(
        sample_pairs=len(i),
        eligible_rows=active,
        seed=int(seed),
        metric=metric,
        kernel_power=int(kernel_power),
    )
    return result
