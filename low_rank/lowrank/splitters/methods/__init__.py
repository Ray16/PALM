"""Built-in splitter methods (standalone low-rank distribution).

Importing this package registers every built-in splitter: each module applies the
``@register`` decorator at import time. This package ships the low-rank family
(``lowrank``, ``lowrank_nd``) plus the ``random`` baseline.
"""

from . import random          # noqa: F401  (registers "random")
import lowrank                # noqa: F401  (registers "lowrank" and "lowrank_nd")
