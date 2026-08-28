"""PALM hypergraph splitter — standalone package for method development.

Graph-partitioning leakage minimizer: build a sparse k-NN similarity structure
over the entities (`knn`) and cut it with Mt-KaHyPar (`partition`), wrapped as
the registered 1-D splitters `hypergraph`/`graph` (`splitter`) and the n-D
(multi-component / reaction) splitters `hypergraph_nd`/`hypergraph_nd_knn`
(`nd_splitter`).

Separated from `PALM.splitters.methods` so the method can be developed on its own
(neighbourhood construction, cut objective, n-D axis handling) while still
registering into the shared `PALM.splitters` registry. Importing this package (or
`PALM.splitters`) registers the four hypergraph methods.
"""

from .knn import (build_knn_graph, build_knn_hyperedges, k_nearest_neighbors,
                  WEIGHT_SCALE)
from .partition import partition_graph, partition_hypergraph
from .splitter import GraphSplitter, HypergraphSplitter
from .nd_splitter import (HypergraphNDKnnSplitter, HypergraphNDSplitter, NDInput,
                          _as_nd)

__all__ = [
    "build_knn_hyperedges", "build_knn_graph", "k_nearest_neighbors", "WEIGHT_SCALE",
    "partition_hypergraph", "partition_graph",
    "HypergraphSplitter", "GraphSplitter",
    "HypergraphNDSplitter", "HypergraphNDKnnSplitter",
    "NDInput", "_as_nd",
]
