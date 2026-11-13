"""Low-rank splitters from the ``lowrank-split`` package, registered into PALM.

PALM's low-rank method lives in one place: the ``lowrank-split`` package in
``PALM/low_rank`` (import name ``lowrank``). This module

* imports it — an installed copy (``pip install ./PALM/low_rank``) wins; otherwise
  the copy bundled in this checkout is used;
* registers its ``lowrank`` and ``lowrank_nd`` splitters in PALM's registry, so
  ``PALM.splitters.split("lowrank", ...)`` runs the same code as
  ``lowrank.split("lowrank", ...)`` and reports the same parameters and diagnostics;
* re-exports the factor-space helpers the benchmarks use (``balanced_lloyd``,
  ``fm_polish``, ``nystrom_features``, ``factor_leakage`` / ``lowrank_leakage``, ...).

Defaults are lowrank-split's: exact split counts unless ``allowed_size_imbalance``
(or ``balance_slack``) is passed. The former in-tree ``PALM.lowrank`` (v0.1) used
a 5% corridor and so delivered ~19% instead of 20% test sets.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PALM.splitters.base import BaseSplitter, SplitResult, SplitSpec, register

_BUNDLED = Path(__file__).resolve().parents[2] / "low_rank"      # PALM/low_rank


def _import_lowrank():
    try:
        import lowrank
    except ModuleNotFoundError as exc:
        if exc.name != "lowrank":
            raise
        if str(_BUNDLED) not in sys.path:
            sys.path.append(str(_BUNDLED))         # appended: an installed copy always wins
        import lowrank
    if not hasattr(lowrank, "LowRankSplitter"):
        raise ImportError(
            f"`lowrank` resolved to {getattr(lowrank, '__file__', '?')}, which is not the "
            "lowrank-split package; install it with `pip install ./PALM/low_rank`.")
    return lowrank


lowrank = _import_lowrank()

from lowrank import (  # noqa: E402  (re-exported for PALM benchmarks)
    balanced_lloyd,
    corridor_assign,
    factor_leakage,
    fm_polish,
    interpolate_to_random,
    lowrank_leakage,
    nystrom_features,
    realized_imbalance,
)
from lowrank.splitters.base import SplitSpec as _LowRankSpec  # noqa: E402


def _adapter(name: str, impl: type) -> type:
    """A PALM splitter that runs ``impl`` (a lowrank-split splitter class)."""

    class _LowRank(BaseSplitter):
        description = impl.description
        arity = impl.arity
        Params = impl.Params                        # same fields and defaults

        @classmethod
        def param_schema(cls):
            return impl.param_schema()               # lowrank-split's richer type mapping

        def split(self, data, spec: SplitSpec) -> SplitResult:
            kwargs = {k: getattr(self.params, k) for k in self.params.__dataclass_fields__}
            native = _LowRankSpec(splits=list(spec.splits), names=list(spec.names),
                                  seed=spec.seed, epsilon=spec.epsilon)
            res = impl(**kwargs).split(data, native)
            diagnostics = dict(res.diagnostics, method=self.name)
            return SplitResult(assignment=res.assignment, method=self.name,
                               diagnostics=diagnostics)

    _LowRank.__name__ = _LowRank.__qualname__ = impl.__name__
    _LowRank.__doc__ = impl.__doc__
    return register(name)(_LowRank)


LowRankSplitter = _adapter("lowrank", lowrank.LowRankSplitter)
LowRankNDSplitter = _adapter("lowrank_nd", lowrank.LowRankNDSplitter)

__all__ = [
    "LowRankSplitter", "LowRankNDSplitter",
    "balanced_lloyd", "corridor_assign", "factor_leakage", "fm_polish",
    "interpolate_to_random", "lowrank_leakage", "nystrom_features", "realized_imbalance",
]
