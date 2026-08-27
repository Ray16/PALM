"""PALM splitters — an agent-ready library of leakage-minimizing dataset splitters.

Every splitting method is a :class:`~PALM.splitters.base.BaseSplitter` subclass
registered under a short name, so it can be selected, introspected, and driven
uniformly:

    from PALM.splitters import split, SplitSpec

    result = split("lowrank", feature_data, SplitSpec(splits=[8, 2],
                                                      names=["train", "test"]))
    result.assignment      # {entity_id: "train" | "test"}
    result.diagnostics     # {"metric", "leakage", "imbalance", "runtime_s", ...}

Discovery (what an agent / MCP tool calls):

    from PALM.splitters import list_splitters, describe_splitters
    list_splitters()          # ["hypergraph", "graph", "lowrank", ...]
    describe_splitters()      # [{name, description, arity, params_schema}, ...]

The methods live in :mod:`PALM.splitters.methods`; shared numerical kernels
(similarity, k-NN, Nyström-free balance helpers, Mt-KaHyPar partitioning, FM
polishing, leakage scoring) live in :mod:`PALM.splitters.common`.
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
