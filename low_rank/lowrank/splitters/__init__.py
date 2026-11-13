"""splitters — an agent-ready library of leakage-minimizing dataset splitters.

Every splitting method is a :class:`~splitters.base.BaseSplitter` subclass
registered under a short name, so it can be selected, introspected, and driven
uniformly:

    from lowrank.splitters import split, SplitSpec

    result = split("lowrank", feature_data, SplitSpec(splits=[8, 2],
                                                      names=["train", "test"]))
    result.assignment      # {entity_id: "train" | "test"}
    result.diagnostics     # {"metric", "leakage", "split_counts", "runtime_s", ...}

Discovery (what an agent / MCP tool calls):

    from lowrank.splitters import list_splitters, describe_splitters
    list_splitters()          # ["lowrank", "lowrank_nd", "random"]
    describe_splitters()      # [{name, description, arity, params_schema}, ...]

The random baseline lives in :mod:`lowrank.splitters.methods`; low-rank
methods register when the top-level package is imported. Shared similarity,
balance, refinement and leakage helpers live in :mod:`lowrank.splitters.common`.
This standalone distribution does not include a Mt-KaHyPar partitioner.
"""

from .base import BaseSplitter, SplitResult, SplitSpec, register
from .registry import describe_splitters, get_splitter, list_splitters
from .dispatch import split

# Importing the methods package registers every built-in splitter as a side
# effect (each module calls @register at import time).
from . import methods as _methods  # noqa: F401

__all__ = [
    "BaseSplitter",
    "SplitResult",
    "SplitSpec",
    "register",
    "get_splitter",
    "list_splitters",
    "describe_splitters",
    "split",
]
