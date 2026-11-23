"""Random split baseline.

Shuffles entities (or records) into blocks of the exact target sizes — the
reference every leakage-minimizing method should beat. Works on both 1-D feature
inputs and n-D record sets, and reports the same scaled ``L(pi)`` diagnostic so it
drops straight into comparisons alongside the optimization-based methods.
"""

from __future__ import annotations

import time

import numpy as np

from ..base import BaseSplitter, SplitResult, SplitSpec, register
from ..common.balanced_assignment import target_sizes
from ..common.feature_preparation import choose_metric, feature_matrix_from_dict
from ..common.leakage_metrics import macro_axis_lpi, scaled_lpi
from ..common.split_naming import assign_split_names

_LEAKAGE_MAX_N = 100_000


def _random_labels(n: int, splits, seed: int) -> np.ndarray:
    sizes = target_sizes(n, splits)
    order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sizes)])
    np.random.default_rng(seed).shuffle(order)
    return order


def _is_nd_input(data) -> bool:
    """True when ``data`` represents n-D ``(records, axis_feature_maps)`` input."""
    if hasattr(data, "records") and hasattr(data, "axis_feature_maps"):
        return True
    if isinstance(data, dict) and "records" in data and "axis_feature_maps" in data:
        return True
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
        return True
    return False


@register("random")
class RandomSplitter(BaseSplitter):
    description = "Random baseline: shuffle entities into exact target-size blocks"
    arity = "1d"  # also accepts n-D input (records, axis_feature_maps)
    Params = None

    def split(self, data, spec: SplitSpec) -> SplitResult:
        t0 = time.time()
        if _is_nd_input(data):
            from lowrank.nd import _as_nd

            records, afm = _as_nd(data)
            n = len(records)
            labels = _random_labels(n, spec.splits, spec.seed)
            assignment = assign_split_names(list(range(n)), labels, spec.splits, spec.names)
            leak = None
            if n <= _LEAKAGE_MAX_N:
                leak = round(macro_axis_lpi(records, afm, labels)[0], 6)
            return self._result(assignment, spec, time.time() - t0, leakage=leak)

        ids, X = feature_matrix_from_dict(data, min_rows=len(spec.splits))
        n = len(ids)
        labels = _random_labels(n, spec.splits, spec.seed)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        metric = choose_metric(X)
        leak = round(scaled_lpi(X, labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric, leakage=leak)
