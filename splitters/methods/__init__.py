"""Built-in splitter methods.

Importing this package registers every built-in splitter (each module applies the
``@register`` decorator at import time). New methods drop a module here and add it
to the imports below.
"""

import PALM.hypergraph        # noqa: F401  (standalone package; registers "hypergraph", "graph", "hypergraph_nd", "hypergraph_nd_knn")
import PALM.lowrank           # noqa: F401  (standalone package; registers "lowrank")
from . import adapters        # noqa: F401  (registers "datasail", "scaffold")
from . import random          # noqa: F401  (registers "random")
