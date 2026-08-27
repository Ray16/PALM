"""n-D macro scaled-L(pi) — superseded, kept as a thin re-export.

The GPU, chunked macro-axis scorer now lives in
``PALM.splitters.common.leakage_metrics.macro_axis_lpi`` (identical logic and
signature). This module re-exports it under the historical
``macro_axis_lpi_gpu`` name so existing callers keep working.
"""

from __future__ import annotations

from PALM.splitters.common.leakage_metrics import macro_axis_lpi as macro_axis_lpi_gpu

__all__ = ["macro_axis_lpi_gpu"]
