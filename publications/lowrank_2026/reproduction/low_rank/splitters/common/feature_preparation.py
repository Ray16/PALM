"""Feature-matrix preparation and metric selection.

Canonical home for three things that were previously duplicated across the
splitters, ``metrics.py`` and ``splitting.py``:

- :func:`is_binary_fingerprint` — the "looks like an ECFP/MACCS bit vector" test.
- :func:`choose_metric` — pick tanimoto / cosine / euclidean from a feature
  matrix. Uses the internal ``"tanimoto"`` name (the splitters' convention);
  :func:`to_scipy_metric` maps it to scipy/sklearn's ``"jaccard"`` when needed.
- :func:`feature_matrix_from_dict` — the ``{id: vector} -> (ids, X)`` preamble
  every 1-D entry point repeats (sorted ids, float32, nan/inf scrubbed).
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np


def is_binary_fingerprint(X: np.ndarray) -> bool:
    """True if ``X`` looks like a binary fingerprint (>=128 dims, all 0/1)."""
    return X.shape[1] >= 128 and bool(np.all((X == 0) | (X == 1)))


def choose_metric(X: np.ndarray, zero_norm_guard: bool = True) -> str:
    """Select a similarity metric from feature characteristics.

    - binary fingerprints (all 0/1, >=128 dims) -> ``"tanimoto"``
    - sparse features (>50% zeros)               -> ``"cosine"``
    - dense features                             -> ``"euclidean"``

    Cosine is undefined for zero-norm rows; with ``zero_norm_guard`` (default)
    the chooser falls back to ``"euclidean"`` when any all-zero row is present —
    this matches the historical behaviour of ``metrics._choose_metric`` and is a
    strict superset of the splitters' old choosers (which lacked the guard).
    """
    if is_binary_fingerprint(X):
        return "tanimoto"
    sparsity = (X == 0).sum() / X.size if X.size else 0.0
    if sparsity > 0.5:
        if zero_norm_guard:
            norms = np.linalg.norm(X, axis=1)
            if (norms == 0).any():
                return "euclidean"
        return "cosine"
    return "euclidean"


def to_scipy_metric(metric: str) -> str:
    """Map the internal metric name to the scipy/sklearn distance name.

    ``"tanimoto"`` -> ``"jaccard"`` (identical for binary vectors); others pass
    through unchanged.
    """
    return "jaccard" if metric == "tanimoto" else metric


def feature_matrix_from_dict(
    feature_data: Dict[Hashable, Sequence[float]],
    min_rows: int = 1,
) -> Tuple[List[Hashable], np.ndarray]:
    """``{id: vector}`` -> ``(sorted_ids, X)`` with X float32, nan/inf scrubbed.

    Args:
        feature_data: mapping of entity id -> feature vector.
        min_rows: raise ``ValueError`` if fewer than this many entities (callers
            pass ``len(splits)`` so a split is well-defined).
    Returns:
        ``(ids, X)`` where ``ids`` is sorted and ``X`` is ``(n, d)`` float32.
    """
    ids = sorted(feature_data.keys())
    n = len(ids)
    if n < min_rows:
        raise ValueError(f"Too few entities ({n}) for the requested split ({min_rows})")
    X = np.asarray([feature_data[i] for i in ids], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return ids, X
