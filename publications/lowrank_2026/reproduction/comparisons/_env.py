"""Put both method repositories on ``sys.path`` so comparison scripts can import
``PALM`` (hypergraph) and ``lowrank`` (low-rank) side by side.

Import this first::

    import _env  # noqa: F401  (adds ../PALM's parent and ../low_rank to sys.path)
    from PALM.splitters import split
    from lowrank import balanced_lloyd
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data_splitting/
for _p in (_BASE, os.path.join(_BASE, "low_rank")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
