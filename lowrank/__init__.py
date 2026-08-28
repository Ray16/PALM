"""PALM low-rank splitter — standalone package for method development.

Graph-free, O(n·r) leakage-minimizing splitter: Nyström factorization of the
similarity matrix (`nystrom`) + a balanced factor-space optimizer (`optimize`)
minimizing a factor-space objective (`objective`), wrapped as the registered
`lowrank` splitter (`splitter`).

Separated from `PALM.splitters.methods` so the method can be developed on its own
(multi-objective / controllable-hardness / tighter approximation) while still
registering into the shared `PALM.splitters` registry. Importing this package (or
`PALM.splitters`) registers the `lowrank` method.
"""

from .nystrom import nystrom_features, _kmeanspp_landmarks
from .objective import factor_leakage, realized_imbalance
from .optimize import (balanced_lloyd, corridor_assign, fm_polish,
                       interpolate_to_random)
from .splitter import LowRankSplitter
from .target_gap import GapCalibrator, calibrate_gap, split_for_gap

# ``lowrank_leakage`` kept as an alias for the factor-space objective (historical
# name used by the omol25 studies + test suite).
lowrank_leakage = factor_leakage

__all__ = [
    "nystrom_features", "_kmeanspp_landmarks",
    "factor_leakage", "lowrank_leakage", "realized_imbalance",
    "balanced_lloyd", "corridor_assign", "fm_polish", "interpolate_to_random",
    "LowRankSplitter",
    "calibrate_gap", "split_for_gap", "GapCalibrator",
]
