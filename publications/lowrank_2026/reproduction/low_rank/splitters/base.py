"""Core splitter interface: ``SplitSpec``, ``SplitResult``, ``BaseSplitter``.

Every splitting method is a :class:`BaseSplitter` subclass with:

- ``name`` / ``description`` / ``arity`` class attributes (for discovery),
- a ``Params`` dataclass declaring its tunable knobs (the single source of truth
  for the JSON schema an agent/MCP tool reads), and
- a ``split(data, spec) -> SplitResult`` method.

``SplitSpec`` carries the split geometry shared by all methods (ratios, names,
seed, balance tolerance); ``SplitResult`` is the uniform, JSON-serializable
return (assignment + diagnostics), replacing the old mix of bare dicts and
``(list, info)`` tuples.
"""

from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence

from . import registry


# ── split geometry (shared by every method) ────────────────────────────────

@dataclass
class SplitSpec:
    """What to split into: ratios, names, seed, and balance tolerance.

    ``splits`` and ``names`` must be the same length (2 or 3). ``epsilon`` is the
    (1 +/- epsilon) balance corridor. ``seed`` seeds all method randomness.
    """

    splits: Sequence[float] = (8, 2)
    names: Sequence[str] = ("train", "test")
    seed: int = 0
    epsilon: float = 0.05

    def __post_init__(self):
        self.splits = list(self.splits)
        self.names = list(self.names)
        if len(self.splits) not in (2, 3):
            raise ValueError(f"'splits' must have 2 or 3 elements, got {self.splits}")
        if len(self.names) != len(self.splits):
            raise ValueError(
                f"'names' length ({len(self.names)}) must match 'splits' "
                f"length ({len(self.splits)})")
        if any(s <= 0 for s in self.splits):
            raise ValueError(f"'splits' must be positive, got {self.splits}")


# ── uniform result ─────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """A split assignment plus its diagnostics.

    ``assignment`` maps entity id (or record index, for n-D) -> split name.
    ``diagnostics`` always carries ``method``, ``n``, ``metric``, ``runtime_s``,
    ``imbalance``, ``leakage`` and the realized ``split_fractions``; individual
    methods add their own keys (``km1``, ``cut``, ``moves``, ``rank``, ...).
    """

    assignment: Dict[Hashable, str]
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def split_indices(self, ids: Optional[Sequence[Hashable]] = None) -> Dict[str, List[int]]:
        """``{split_name: [positional index, ...]}`` over ``ids`` (or sorted keys)."""
        ids = list(ids) if ids is not None else sorted(self.assignment.keys())
        out: Dict[str, List[int]] = {}
        for i, k in enumerate(ids):
            out.setdefault(self.assignment[k], []).append(i)
        return out

    def to_json(self) -> Dict[str, Any]:
        """Fully JSON-serializable view (keys stringified)."""
        return {
            "method": self.method,
            "assignment": {str(k): v for k, v in self.assignment.items()},
            "diagnostics": self.diagnostics,
        }


# ── metric/JSON helpers for param schemas ──────────────────────────────────

_JSON_TYPES = {int: "integer", float: "number", bool: "boolean", str: "string"}
_JSON_TYPES_BY_NAME = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}


def _json_type(tp) -> str:
    """JSON-schema type for a dataclass field annotation.

    Handles both real type objects and the *string* annotations produced by
    ``from __future__ import annotations`` (e.g. ``"int"``, ``"Optional[str]"``).
    Unknown / container types fall back to a permissive ``"string"``.
    """
    if isinstance(tp, type):
        return _JSON_TYPES.get(tp, "string")
    s = str(tp).replace("typing.", "")
    inner = s[len("Optional["):-1] if s.startswith("Optional[") else s
    return _JSON_TYPES_BY_NAME.get(inner, "string")


# ── the splitter base class ─────────────────────────────────────────────────

class BaseSplitter(ABC):
    """Base class for all splitters.

    Subclasses set the class attributes ``name``, ``description``, ``arity``
    ("1d" or "nd") and ``Params`` (a dataclass type), then implement
    :meth:`split`.
    """

    name: str = ""
    description: str = ""
    arity: str = "1d"
    Params: type = None  # a @dataclass type

    def __init__(self, **params):
        if self.Params is None:
            if params:
                raise TypeError(f"{self.name} takes no parameters")
            self.params = None
        else:
            self.params = self.Params(**params)

    @abstractmethod
    def split(self, data, spec: SplitSpec) -> SplitResult:
        """Split ``data`` per ``spec`` and return a :class:`SplitResult`."""

    # -- discovery --------------------------------------------------------
    @classmethod
    def param_schema(cls) -> Dict[str, Any]:
        """JSON schema for this method's params, derived from the ``Params`` dataclass."""
        props: Dict[str, Any] = {}
        if cls.Params is not None:
            for f in dataclasses.fields(cls.Params):
                default = None if f.default is dataclasses.MISSING else f.default
                props[f.name] = {"type": _json_type(f.type), "default": default}
        return {"type": "object", "properties": props}

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description.strip(),
            "arity": cls.arity,
            "params_schema": cls.param_schema(),
        }

    # -- shared result construction --------------------------------------
    def _result(self, assignment: Dict[Hashable, str], spec: SplitSpec,
                runtime_s: float, **diagnostics) -> SplitResult:
        counts: Dict[str, int] = {n: 0 for n in spec.names}
        for v in assignment.values():
            counts[v] = counts.get(v, 0) + 1
        total = sum(counts.values()) or 1
        diag = {
            "method": self.name,
            "n": len(assignment),
            "runtime_s": round(runtime_s, 4),
            "split_counts": counts,
            "split_fractions": {k: round(v / total, 4) for k, v in counts.items()},
        }
        diag.update(diagnostics)
        return SplitResult(assignment=assignment, method=self.name, diagnostics=diag)


def register(name: str):
    """Class decorator registering a :class:`BaseSplitter` subclass under ``name``."""
    def deco(cls):
        cls.name = name
        registry._REGISTRY[name] = cls
        return cls
    return deco
