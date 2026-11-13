"""Target-gap API: request a difficulty, get the split that delivers it.

Step 3 (the flagship) established that the low-rank splitter's ``hardness`` dial
``alpha in [0,1]`` controls the *realized generalization gap* almost linearly
within a dataset — ``gap ≈ b + a·alpha`` with Spearman(alpha, gap) ≈ +0.9..+1.0.
That leaves the user picking ``alpha`` by hand. This module inverts the workflow:

    calibrate_gap(...)  → GapCalibrator (fits gap ≈ b + a·alpha from probe splits)
    calibrator.invert(target_gap)  → the alpha whose predicted gap is closest
    split_for_gap(..., target_gap=...)  → the actual split at that alpha

The gap is measured with the *same* estimator the Step-3 experiment used
(``PALM.benchmarks.master.model_eval.evaluate_gap``) so calibrations here are
directly comparable to ``experiments/hardness_control.py``.

The calibration slope ``a`` is dataset-specific (the honest scope of Step 3), so
calibrate per (dataset, features, model). A calibrator is cheap to reuse across
many target requests once fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# NOTE: `PALM.splitters` and `PALM.benchmarks.master.model_eval` are imported lazily
# inside the functions below. This module is re-exported from `PALM.lowrank.__init__`,
# which the splitter registry imports — a module-level `import PALM.splitters` here
# would be circular (splitters -> lowrank -> target_gap -> splitters), and pulling in
# the benchmarks model stack on every registry import is needless coupling.

DEFAULT_PROBE_ALPHAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)


# --------------------------------------------------------------------------- #
# gap measurement (shared with experiments/hardness_control.py)
# --------------------------------------------------------------------------- #
def measure_gap(feature_data: Dict, X: np.ndarray, y: np.ndarray, task_type: str,
                alpha: float, seeds: Sequence[int] = DEFAULT_SEEDS,
                splits: Sequence[float] = (8, 2),
                names: Sequence[str] = ("train", "test")) -> Dict[str, float]:
    """Mean realized generalization gap of a ``hardness=alpha`` split over ``seeds``.

    Produces a low-rank split at the given hardness for each seed, fits/scores the
    benchmark model on it (``evaluate_gap``), and averages. ``ids`` order of
    ``feature_data`` must match the rows of ``X``/``y`` (as in hardness_control.py).
    Returns ``{"alpha", "gap_mean", "gap_std", "leakage_mean", "n_ok"}``.
    """
    from PALM.splitters import SplitSpec, split
    from PALM.benchmarks.master.model_eval import evaluate_gap

    ids = list(feature_data)
    gaps: List[float] = []
    leaks: List[float] = []
    for seed in seeds:
        res = split("lowrank", feature_data,
                    SplitSpec(list(splits), list(names), seed=seed), hardness=alpha)
        tr = [j for j, i in enumerate(ids) if res.assignment[i] == names[0]]
        te = [j for j, i in enumerate(ids) if res.assignment[i] == names[1]]
        g = evaluate_gap(X, y, task_type, tr, te, seed=seed)
        gg = g.get("gen_gap")
        if gg not in (None, ""):
            gaps.append(float(gg))
        lk = res.diagnostics.get("leakage")
        if lk not in (None, ""):
            leaks.append(float(lk))
    return {
        "alpha": float(alpha),
        "gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "gap_std": float(np.std(gaps)) if gaps else float("nan"),
        "leakage_mean": float(np.mean(leaks)) if leaks else float("nan"),
        "n_ok": len(gaps),
    }


# --------------------------------------------------------------------------- #
# calibrator
# --------------------------------------------------------------------------- #
@dataclass
class GapCalibrator:
    """Calibration of realized gap vs. the hardness dial ``alpha``.

    Two engines, chosen by ``mode``:

    - ``"piecewise"`` (default when probe knots are present): a **monotone**
      piecewise-linear curve through the probe points. Step 3's alpha->gap curve
      is markedly *convex* (most of the gap opens up near alpha=1), so a straight
      line over-predicts interior gaps; the piecewise curve tracks the real shape
      and inverts far more accurately.
    - ``"linear"``: the 2-parameter fit ``gap ≈ b + a·alpha`` — kept as a compact
      summary (and the only option when no knots are supplied, e.g. unit tests).

    ``a`` may have either sign; all range logic is written sign-agnostically.
    ``achievable_range`` is the closed interval of gaps the dial can reach.
    """

    b: float                       # linear intercept = fit gap at alpha=0
    a: float                       # linear slope     = fit gap(1) - fit gap(0)
    r2: float = float("nan")
    spearman: float = float("nan")
    probe_alphas: List[float] = field(default_factory=list)
    probe_gaps: List[float] = field(default_factory=list)
    probe_gap_std: List[float] = field(default_factory=list)
    dataset: str = ""
    mode: str = "auto"             # auto -> piecewise if >=2 knots else linear

    # ---- knots (monotone, sorted by alpha) --------------------------------- #
    def _knots(self):
        """Sorted, monotone (isotonic via running max) probe knots, or None.

        The realized gap rises with hardness but is noisy, so we enforce
        non-decreasing gaps with a running max before interpolating — this makes
        the inverse well-defined without distorting the trend.
        """
        if (self.mode == "linear" or len(self.probe_alphas) < 2
                or len(self.probe_gaps) != len(self.probe_alphas)):
            return None
        aa = np.asarray(self.probe_alphas, dtype=float)
        gg = np.asarray(self.probe_gaps, dtype=float)
        order = np.argsort(aa)
        aa, gg = aa[order], gg[order]
        gg = np.maximum.accumulate(gg)             # enforce non-decreasing
        return aa, gg

    def _use_piecewise(self) -> bool:
        return self._knots() is not None

    # ---- prediction / range ------------------------------------------------ #
    def predict(self, alpha: float) -> float:
        """Predicted gap at ``alpha`` (clamped to the calibrated [0,1] domain)."""
        a = float(np.clip(alpha, 0.0, 1.0))
        k = self._knots()
        if k is not None:
            aa, gg = k
            return float(np.interp(a, aa, gg))
        return self.b + self.a * a

    @property
    def gap_at_0(self) -> float:
        return self.predict(0.0)

    @property
    def gap_at_1(self) -> float:
        return self.predict(1.0)

    @property
    def achievable_range(self) -> Tuple[float, float]:
        lo, hi = self.gap_at_0, self.gap_at_1
        return (min(lo, hi), max(lo, hi))

    @property
    def controllable(self) -> bool:
        """Whether the dial has usable range (non-degenerate span)."""
        lo, hi = self.achievable_range
        return (hi - lo) > 1e-9

    # ---- inversion --------------------------------------------------------- #
    def invert(self, target_gap: float) -> "InversionResult":
        """Find the alpha whose predicted gap is closest to ``target_gap``.

        Returns an :class:`InversionResult` with the clamped alpha, whether the
        target was inside the achievable range, the predicted gap there, and a
        human-readable message. If the calibration is degenerate (flat curve) the
        dial cannot steer the gap and we return ``alpha=1`` (the hardest, default)
        with ``controllable=False``.
        """
        lo, hi = self.achievable_range
        tgt = float(target_gap)
        if not self.controllable:
            return InversionResult(
                alpha=1.0, in_range=False, controllable=False,
                predicted_gap=self.predict(1.0), target_gap=tgt,
                achievable_range=(lo, hi),
                message=(f"gap is not controllable here (span≈0); returning "
                         f"alpha=1.0 (hardest). Gap≈{self.gap_at_0:.3f} regardless."))

        k = self._knots()
        if k is not None:
            aa, gg = k
            # gg is non-decreasing -> invert by interpolating alpha as a function
            # of gap; np.interp clamps to the endpoints outside the knot range.
            alpha = float(np.interp(tgt, gg, aa))
        else:
            alpha = float(np.clip((tgt - self.b) / self.a, 0.0, 1.0))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        in_range = (lo - 1e-9) <= tgt <= (hi + 1e-9)
        pred = self.predict(alpha)
        if in_range:
            msg = f"target gap {tgt:.3f} -> alpha={alpha:.3f} (predicted gap {pred:.3f})."
        else:
            nearest = "max" if tgt > hi else "min"
            msg = (f"target gap {tgt:.3f} is OUTSIDE the achievable range "
                   f"[{lo:.3f}, {hi:.3f}]; clamped to the {nearest} at alpha={alpha:.3f} "
                   f"(predicted gap {pred:.3f}).")
        return InversionResult(alpha=alpha, in_range=in_range, controllable=True,
                               predicted_gap=pred, target_gap=tgt,
                               achievable_range=(lo, hi), message=msg)


@dataclass
class InversionResult:
    alpha: float
    in_range: bool
    controllable: bool
    predicted_gap: float
    target_gap: float
    achievable_range: Tuple[float, float]
    message: str


def calibrate_gap(feature_data: Dict, X: np.ndarray, y: np.ndarray, task_type: str,
                  alphas: Sequence[float] = DEFAULT_PROBE_ALPHAS,
                  seeds: Sequence[int] = DEFAULT_SEEDS,
                  splits: Sequence[float] = (8, 2),
                  names: Sequence[str] = ("train", "test"),
                  dataset: str = "", verbose: bool = False) -> GapCalibrator:
    """Fit ``gap ≈ b + a·alpha`` from triplicate probe splits at ``alphas``.

    Runs ``len(alphas) × len(seeds)`` splits (default 3×3 = 9), measures the mean
    gap per alpha, and least-squares fits the line. Reports R² and Spearman of the
    (alpha, gap) probe points so the caller can see how linear/monotone the dial is
    on this dataset before trusting an inversion.
    """
    from scipy.stats import spearmanr

    probe_alphas: List[float] = []
    probe_gaps: List[float] = []
    probe_std: List[float] = []
    for alpha in alphas:
        m = measure_gap(feature_data, X, y, task_type, alpha,
                        seeds=seeds, splits=splits, names=names)
        if np.isnan(m["gap_mean"]):
            if verbose:
                print(f"  [calibrate] alpha={alpha:.2f}: no valid gap, skipped")
            continue
        probe_alphas.append(m["alpha"])
        probe_gaps.append(m["gap_mean"])
        probe_std.append(m["gap_std"])
        if verbose:
            print(f"  [calibrate] alpha={alpha:.2f}  gap={m['gap_mean']:.3f}"
                  f"±{m['gap_std']:.3f}  leakage={m['leakage_mean']:.3f}")

    if len(probe_alphas) < 2:
        raise ValueError(
            f"need >=2 valid probe alphas to calibrate; got {len(probe_alphas)} "
            f"(dataset={dataset!r}). Check that the dataset has usable targets.")

    aa = np.asarray(probe_alphas, dtype=float)
    gg = np.asarray(probe_gaps, dtype=float)
    a, b = np.polyfit(aa, gg, 1)                     # gap ≈ a·alpha + b
    pred = a * aa + b
    ss_res = float(np.sum((gg - pred) ** 2))
    ss_tot = float(np.sum((gg - gg.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rho = spearmanr(aa, gg).correlation if len(aa) >= 3 else float("nan")

    return GapCalibrator(b=float(b), a=float(a), r2=float(r2), spearman=float(rho),
                         probe_alphas=probe_alphas, probe_gaps=probe_gaps,
                         probe_gap_std=probe_std, dataset=dataset)


def invert_to_alpha(calibrator: GapCalibrator, target_gap: float) -> float:
    """Convenience: the clamped alpha for ``target_gap`` (see ``GapCalibrator.invert``)."""
    return calibrator.invert(target_gap).alpha


# --------------------------------------------------------------------------- #
# end-to-end convenience
# --------------------------------------------------------------------------- #
def split_for_gap(feature_data: Dict, target_gap: float, *,
                  X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                  task_type: Optional[str] = None,
                  calibrator: Optional[GapCalibrator] = None,
                  seed: int = 0, splits: Sequence[float] = (8, 2),
                  names: Sequence[str] = ("train", "test"),
                  alphas: Sequence[float] = DEFAULT_PROBE_ALPHAS,
                  cal_seeds: Sequence[int] = DEFAULT_SEEDS):
    """Return ``(SplitResult, InversionResult, GapCalibrator)`` for a target difficulty.

    Pass a precomputed ``calibrator`` to skip calibration (recommended when
    producing many splits for one dataset); otherwise supply ``X, y, task_type``
    and it will calibrate first. The returned split is produced at the inverted
    ``alpha`` via the public ``hardness`` knob.
    """
    from PALM.splitters import SplitSpec, split

    if calibrator is None:
        if X is None or y is None or task_type is None:
            raise ValueError("provide a `calibrator`, or X/y/task_type to fit one")
        calibrator = calibrate_gap(feature_data, X, y, task_type,
                                   alphas=alphas, seeds=cal_seeds,
                                   splits=splits, names=names)
    inv = calibrator.invert(target_gap)
    res = split("lowrank", feature_data,
                SplitSpec(list(splits), list(names), seed=seed), hardness=inv.alpha)
    return res, inv, calibrator
