"""Back-compat shim — the low-rank splitter moved to the standalone ``PALM.lowrank``.

Importing this module still registers the ``lowrank`` method and re-exports the
public helpers (``nystrom_features``, ``balanced_lloyd``, ``fm_polish``,
``lowrank_leakage``, ``LowRankSplitter``) at their historical path
``PALM.splitters.methods.lowrank`` for the test suite and the omol25 studies.
New code should import from ``PALM.lowrank`` directly.
"""

from PALM.lowrank import (  # noqa: F401
    LowRankSplitter,
    balanced_lloyd,
    factor_leakage,
    fm_polish,
    lowrank_leakage,
    nystrom_features,
)
