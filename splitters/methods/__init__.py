"""Built-in splitter methods.

Importing this package registers every built-in splitter (each module applies the
``@register`` decorator at import time). New methods drop a module here and add it
to the imports below.
"""

from . import hypergraph      # noqa: F401  (registers "hypergraph", "graph")
from . import lowrank         # noqa: F401  (registers "lowrank")
from . import nD_hypergraph   # noqa: F401  (registers "hypergraph_nd", "hypergraph_nd_knn")
from . import adapters        # noqa: F401  (registers "datasail", "scaffold")
