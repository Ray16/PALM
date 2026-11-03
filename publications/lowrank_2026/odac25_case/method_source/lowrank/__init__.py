"""Capacity-controlled splitting through low-rank similarity factors.

The plain method uses compact fold sums. Optional kernel powers and
capped neighbour groups control local contamination, with additional
neighbour-construction work and conditional group preservation.
Support modules and the registry live inside lowrank.*.
"""

from .nystrom import nystrom_features, _kmeanspp_landmarks
from .objective import factor_leakage, realized_imbalance
from .optimize import (balanced_lloyd, corridor_assign, fm_polish,
                       interpolate_to_random)
from .splitter import LowRankSplitter
from .nd import LowRankNDSplitter
from .target_gap import GapCalibrator, calibrate_gap, split_for_gap

# ``lowrank_leakage`` kept as an alias for the factor-space objective (historical
# name used by the omol25 studies + test suite).
lowrank_leakage = factor_leakage

__all__ = [
    "nystrom_features", "_kmeanspp_landmarks",
    "factor_leakage", "lowrank_leakage", "realized_imbalance",
    "balanced_lloyd", "corridor_assign", "fm_polish", "interpolate_to_random",
    "LowRankSplitter", "LowRankNDSplitter",
    "calibrate_gap", "split_for_gap", "GapCalibrator",
]

# Public entry point; all support packages live inside lowrank.
from .splitters import split, SplitSpec, SplitResult, list_splitters
__all__ += ['split', 'SplitSpec', 'SplitResult', 'list_splitters']
