"""JSON-in / JSON-out wrapper — the surface a future MCP server or Agent tool wraps.

Two pure functions:

- :func:`describe_splitters_tool` — the discovery payload (name, description,
  arity, parameter JSON schema for each method). An agent reads this to decide
  which method to call and with what parameters.
- :func:`run_split_tool` — run a split from plain-JSON inputs (feature vectors as
  lists, or n-D ``records`` + ``axis_feature_maps``) and return a fully
  JSON-serializable result (assignment + diagnostics).

Nothing here registers a live tool; it just makes the library callable with
JSON, so wiring it into an MCP server / the ``Agent`` tool later is a thin shim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import SplitSpec
from .dispatch import split
from .registry import describe_splitters


def describe_splitters_tool() -> List[Dict[str, Any]]:
    """List every splitter with its description, arity and parameter schema."""
    return describe_splitters()


def _coerce_features(features: Dict[str, Any]):
    """``{id: vector-or-SMILES}`` -> usable data.

    If every value is a string the input is SMILES (for the scaffold splitter)
    and passed through unchanged; otherwise values are coerced to float vectors.
    """
    if features and all(isinstance(v, str) for v in features.values()):
        return dict(features)
    return {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}


def _coerce_axis_maps(axis_feature_maps: Dict[str, Dict[str, Any]]):
    return {ax: {k: (None if v is None else np.asarray(v, dtype=np.float32))
                 for k, v in m.items()}
            for ax, m in axis_feature_maps.items()}


def run_split_tool(method: str,
                   features: Optional[Dict[str, Any]] = None,
                   records: Optional[Sequence[dict]] = None,
                   axis_feature_maps: Optional[Dict[str, Dict[str, Any]]] = None,
                   splits: Sequence[float] = (8, 2),
                   names: Sequence[str] = ("train", "test"),
                   seed: int = 0,
                   epsilon: float = 0.05,
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run ``method`` from JSON inputs; return ``SplitResult.to_json()``.

    Provide either ``features`` (1-D: ``{id: vector}`` or ``{id: SMILES}``) or
    ``records`` + ``axis_feature_maps`` (n-D). ``params`` are the method-specific
    knobs (see :func:`describe_splitters_tool`).
    """
    params = params or {}
    spec = SplitSpec(splits=list(splits), names=list(names), seed=seed, epsilon=epsilon)
    if records is not None:
        if axis_feature_maps is None:
            raise ValueError("n-D split requires 'axis_feature_maps' alongside 'records'")
        data = (list(records), _coerce_axis_maps(axis_feature_maps))
    elif features is not None:
        data = _coerce_features(features)
    else:
        raise ValueError("provide 'features' (1-D) or 'records'+'axis_feature_maps' (n-D)")
    return split(method, data, spec, **params).to_json()
