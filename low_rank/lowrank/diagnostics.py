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


# ── Validation & Reference Preparation ───────────────────────────────────────


def positive_integer(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")


def prepare_reference(X, metric: str, device: str = "cpu"):
    import torch

    X = validate_features(X, metric)
    ref = torch.as_tensor(X, dtype=torch.float32, device=device)
    if not torch.isfinite(ref).all():
        raise ValueError("features exceed the supported float32 range")
    return standardize_for_metric(ref, metric)


# ── Support-Aware Pair Sampling & Kernel Evaluation ──────────────────────────


def sample_pairs(n: int, count: int, seed: int):
    positive_integer("sample_pairs", count)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    i = rng.integers(n, size=count)
    j = rng.integers(n - 1, size=count)
    return i, j + (j >= i)


def _audit_pairs(ref, metric: str, count: int, seed: int, B=None):
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
        i = np.repeat(np.arange(len(pool)), len(pool))
        j = np.tile(np.arange(len(pool)), len(pool))
        keep = i != j
        return pool[i[keep]], pool[j[keep]], True, len(pool)
    i, j = sample_pairs(len(pool), count, seed)
    return pool[i], pool[j], False, len(pool)


def paired_similarity(ref, i, j, metric: str, *, kernel_power: int = 1, block_size: int = 1024):
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


def _paired_factor_dot(B: np.ndarray, i: np.ndarray, j: np.ndarray, block_size: int = 1024) -> np.ndarray:
    """Evaluate aligned inner products ``sum_r B[i, r] * B[j, r]`` in bounded tiles."""
    out = np.empty(len(i), dtype=np.float64)
    for s in range(0, len(i), block_size):
        a = B[i[s : s + block_size]].astype(np.float64)
        b = B[j[s : s + block_size]].astype(np.float64)
        out[s : s + block_size] = (a * b).sum(1)
    return out


def _pair_cut_weights(labels: np.ndarray, i: np.ndarray, j: np.ndarray, fold_weights=None):
    """Return ``(pair_weights, max_weight_bound)`` for pairs ``(i, j)`` under ``labels``."""
    labels = np.asarray(labels)
    if fold_weights is None:
        return (labels[i] != labels[j]).astype(np.float64), 1.0
    from .factors import validate_fold_weights

    W = validate_fold_weights(fold_weights, len(fold_weights))
    if not np.issubdtype(labels.dtype, np.integer) or np.any(
        (labels < 0) | (labels >= len(W))
    ):
        raise ValueError("weighted audit labels must be integer indices into fold_weights")
    return W[labels[i], labels[j]].astype(np.float64), max(1.0, float(np.max(W)))


# ── Empirical-Bernstein Ratio Bounds ─────────────────────────────────────────


def _mean_interval(x: np.ndarray, bounds: Tuple[float, float], alpha: float):
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
    confidence: float = 0.95,
    exact: bool = False,
):
    x = np.asarray(numerator, dtype=np.float64)
    y = np.asarray(denominator, dtype=np.float64)
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


def _fold_to_absolute(summary: dict) -> dict:
    """Report a signed-residual ratio as an error magnitude with a folded interval."""
    if summary.get("estimate") is not None:
        summary["estimate"] = abs(summary["estimate"])
        if summary.get("ci95") is not None:
            lo, hi = summary["ci95"]
            summary["ci95"] = [
                0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)),
                max(abs(lo), abs(hi)),
            ]
    return summary


def _absolute_cut_error(signed, bounds, confidence: float, exact: bool):
    """Point and conservative upper estimate of ``|mean(signed cut residual)|``."""
    signed = np.asarray(signed, dtype=np.float64)
    if not len(signed):
        return 0.0, 0.0
    point = abs(float(signed.mean()))
    if exact or len(signed) < 2:
        return point, point
    lo, hi = _mean_interval(signed, bounds, 1.0 - confidence)
    return point, max(abs(lo), abs(hi))


def _cut_acceptance(cut_rel: dict, abs_upper: float, cut_mass: float, rel_tol: float, abs_floor: float) -> dict:
    """Decide whether the returned cut is trustworthy at the audited tolerance."""
    rel_est = cut_rel.get("estimate")
    ci = cut_rel.get("ci95")
    rel_upper = ci[1] if ci is not None else rel_est
    below_floor = abs_upper <= abs_floor
    if rel_est is None:
        accepted = below_floor
        status = "accepted_below_abs_floor" if accepted else "unresolved_low_cut_mass"
    else:
        rel_ok = rel_upper is not None and rel_upper <= rel_tol
        accepted = bool(rel_ok or below_floor)
        status = "accepted" if accepted else "rejected"
    return dict(
        accepted=bool(accepted),
        status=status,
        cut_relative_upper=None if rel_upper is None else float(rel_upper),
        cut_absolute_upper=float(abs_upper),
        cut_mass=float(cut_mass),
        rel_tol=float(rel_tol),
        abs_floor=float(abs_floor),
    )


# ── Factor, Cut & Leakage Audits ─────────────────────────────────────────────


def factor_pair_error(
    B,
    ref,
    metric: str,
    *,
    pairs: int = 2048,
    seed: int = 0,
    kernel_power: int = 1,
    block_size: int = 1024,
    confidence: float = 0.95,
    labels=None,
    fold_weights=None,
    cut_rel_tol: float = 0.1,
    cut_abs_floor: float = 1e-3,
):
    """Entrywise MAE, or returned-cut error, relative to ``|kernel|`` mass and cut mass."""
    B = np.asarray(B)
    if B.ndim != 2 or len(B) != len(ref) or not np.isfinite(B).all():
        raise ValueError("audit factor must be a finite matrix aligned to reference rows")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(ref),):
            raise ValueError("audit partition must have one fold label per reference row")

    i, j, exact_population, active_rows = _audit_pairs(ref, metric, pairs, seed, B)
    truth = paired_similarity(
        ref, i, j, metric, kernel_power=kernel_power, block_size=block_size
    )
    residual = truth - _paired_factor_dot(B, i, j, block_size=block_size)
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
        weights, weight_bound = None, 1.0
    else:
        weights, weight_bound = _pair_cut_weights(labels, i, j, fold_weights)
        numerator = residual * weights
        bound *= weight_bound
        nb = (-bound, bound)

    proven_zero = exact_population and np.all(truth == 0) and np.all(residual == 0)
    if proven_zero:
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
    result = dict(
        relative_pair_mae=summary,
        criterion="cut_error" if labels is not None else "pair_mae",
        sample_pairs=len(i),
        eligible_rows=active_rows,
        seed=int(seed),
        normalization="mean absolute off-diagonal powered kernel similarity",
        exact=exact_population,
        confidence=float(confidence),
    )
    if labels is None:
        return result

    _fold_to_absolute(summary)
    cut_denominator = np.abs(truth) * weights
    if proven_zero:
        cut_relative = dict(
            estimate=0.0,
            standard_error=0.0,
            ci95=[0.0, 0.0],
            status="proven_zero_offdiagonal",
            exact=True,
        )
    else:
        cut_relative = _fold_to_absolute(
            ratio_summary(
                numerator,
                cut_denominator,
                numerator_bounds=nb,
                denominator_bounds=(0.0, weight_bound),
                confidence=confidence,
                exact=exact_population,
            )
        )
    abs_point, abs_upper = _absolute_cut_error(
        numerator, nb, confidence, exact_population
    )
    cut_mass = float(cut_denominator.mean()) if len(cut_denominator) else 0.0
    result["cut_relative_error"] = cut_relative
    result["cut_absolute_error"] = dict(estimate=abs_point, upper=abs_upper)
    result["acceptance"] = _cut_acceptance(
        cut_relative, abs_upper, cut_mass, cut_rel_tol, cut_abs_floor
    )
    result["cut_normalization"] = "mean |kernel| mass over crossing pairs (the cut)"
    return result


def sampled_leakage(
    X,
    labels,
    metric: str = "tanimoto",
    *,
    sample_pairs_count: int = 8192,
    seed: int = 0,
    kernel_power: int = 1,
    block_size: int = 1024,
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


def compare_candidate_cuts(
    B,
    ref_or_X,
    metric: str,
    labels_a,
    labels_b,
    *,
    pairs: int = 8192,
    seed: int = 0,
    kernel_power: int = 1,
    block_size: int = 1024,
    confidence: float = 0.95,
    fold_weights=None,
):
    """Audit the true-kernel cut difference ``cut(labels_a) - cut(labels_b)`` on common pairs."""
    import torch

    B = np.asarray(B)
    ref = (
        ref_or_X
        if isinstance(ref_or_X, torch.Tensor)
        else prepare_reference(ref_or_X, metric)
    )
    la = np.asarray(labels_a)
    lb = np.asarray(labels_b)
    if la.shape != (len(ref),) or lb.shape != (len(ref),):
        raise ValueError("candidate partitions must each have one label per row")

    i, j, exact_pop, active_rows = _audit_pairs(ref, metric, pairs, seed, B)
    truth = paired_similarity(
        ref, i, j, metric, kernel_power=kernel_power, block_size=block_size
    )
    approx = _paired_factor_dot(B, i, j, block_size=block_size)
    wa, w_bound = _pair_cut_weights(la, i, j, fold_weights)
    wb, _ = _pair_cut_weights(lb, i, j, fold_weights)

    diff_true = truth * (wa - wb)
    diff_approx = float(np.mean(approx * (wa - wb))) if len(i) else 0.0
    summary = ratio_summary(
        diff_true,
        np.abs(truth),
        numerator_bounds=(-w_bound, w_bound),
        denominator_bounds=(0.0, 1.0),
        confidence=confidence,
        exact=exact_pop,
    )
    ci = summary.get("ci95")
    if ci is None:
        verdict = "unresolved_insufficient_mass"
    elif ci[1] < 0:
        verdict = "a_better"
    elif ci[0] > 0:
        verdict = "b_better"
    else:
        verdict = "unresolved_near_tie"
    factor_pref = (
        "a_better"
        if diff_approx < -1e-12
        else "b_better"
        if diff_approx > 1e-12
        else "tie"
    )
    return dict(
        paired_cut_difference=summary,
        verdict=verdict,
        factor_preference=factor_pref,
        ranking_reversed=bool(
            verdict in ("a_better", "b_better")
            and factor_pref in ("a_better", "b_better")
            and verdict != factor_pref
        ),
        sample_pairs=len(i),
        eligible_rows=active_rows,
        exact=exact_pop,
        seed=int(seed),
    )
