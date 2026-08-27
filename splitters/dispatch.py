"""Single dispatch entry point: :func:`split`.

    from PALM.splitters import split, SplitSpec
    result = split("lowrank", feature_data, SplitSpec(splits=[8, 2],
                                                      names=["train", "test"]),
                   rank=256)

``params`` are the method-specific knobs (see ``describe_splitters()`` for each
method's schema); ``spec`` is the shared split geometry.
"""

from __future__ import annotations

from typing import Optional

from .base import SplitResult, SplitSpec
from .registry import get_splitter


def split(method: str, data, spec: Optional[SplitSpec] = None, **params) -> SplitResult:
    """Run ``method`` on ``data`` with ``spec`` and method ``params``."""
    if spec is None:
        spec = SplitSpec()
    elif isinstance(spec, dict):
        spec = SplitSpec(**spec)
    return get_splitter(method, **params).split(data, spec)
