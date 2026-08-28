"""Random split baseline.

Shuffles entities (or records) into blocks of the exact target sizes — the
reference every leakage-minimizing method should beat. Works on both 1-D feature
dicts and n-D record sets, and reports the same scaled ``L(pi)`` diagnostic so it
drops straight into the comparison alongside the real methods.
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


def _random_labels(n, splits, seed):
    sizes = target_sizes(n, splits)
    order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sizes)])
    np.random.default_rng(seed).shuffle(order)
    return order


@register("random")
class RandomSplitter(BaseSplitter):
    description = "Random baseline: shuffle entities into exact target-size blocks"
    arity = "1d"          # also accepts n-D input (records, axis_feature_maps)
    Params = None

    def split(self, data, spec: SplitSpec) -> SplitResult:
        t0 = time.time()
        # n-D input?  (records, axis_feature_maps) tuple / dict / NDInput
        from PALM.hypergraph import NDInput, _as_nd
        is_nd = isinstance(data, (NDInput, tuple)) or (isinstance(data, dict) and "records" in data)
        if is_nd:
            nd = _as_nd(data)
            n = len(nd.records)
            labels = _random_labels(n, spec.splits, spec.seed)
            assignment = assign_split_names(list(range(n)), labels, spec.splits, spec.names)
            leak = None
            if n <= _LEAKAGE_MAX_N:
                leak = round(macro_axis_lpi(nd.records, nd.axis_feature_maps, labels)[0], 6)
            return self._result(assignment, spec, time.time() - t0, leakage=leak)

        ids, X = feature_matrix_from_dict(data, min_rows=len(spec.splits))
        n = len(ids)
        labels = _random_labels(n, spec.splits, spec.seed)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        metric = choose_metric(X)
        leak = round(scaled_lpi(X, labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric, leakage=leak)
