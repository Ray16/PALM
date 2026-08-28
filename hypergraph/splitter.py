"""Hypergraph-partitioning and weighted-graph-edge-cut splitters (1-D).

Both build a sparse k-NN similarity structure over the entities (:mod:`.knn`) and
hand it to Mt-KaHyPar (:mod:`.partition`):

- :class:`HypergraphSplitter` — one mean-weighted k-NN *hyperedge* per node, cut
  under the KM1 connectivity objective. Fast and simple; the objective is a
  coarse (count-once) proxy for pairwise leakage.
- :class:`GraphSplitter` — a weighted 2-uniform k-NN *graph* cut under the CUT
  objective, which sums each crossing pair at its true similarity (the leakage
  numerator on the retained edges), then an optional exact-leakage FM polish.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

import time

from PALM.splitters.base import BaseSplitter, SplitResult, SplitSpec, register
from PALM.splitters.common.feature_preparation import choose_metric, feature_matrix_from_dict
from PALM.splitters.common.fiduccia_mattheyses import fiduccia_mattheyses_exact
from PALM.splitters.common.leakage_metrics import scaled_lpi
from PALM.splitters.common.split_naming import assign_split_names

from .knn import build_knn_graph, build_knn_hyperedges
from .partition import partition_graph, partition_hypergraph

logger = logging.getLogger(__name__)

# Above this many entities, skip the O(n^2) exact-leakage diagnostic (the split
# itself still runs); the k-NN backends are not used at that scale anyway.
_LEAKAGE_MAX_N = 100_000


def _leakage(X, labels, metric):
    if len(labels) > _LEAKAGE_MAX_N:
        return None
    return round(scaled_lpi(X, labels, metric=metric), 6)


@register("hypergraph")
class HypergraphSplitter(BaseSplitter):
    description = "k-NN similarity hyperedges cut under Mt-KaHyPar's KM1 objective"
    arity = "1d"

    @dataclass
    class Params:
        k: int = 15
        metric: Optional[str] = None
        use_gpu: bool = True
        threads: int = 8
        preset: str = "default"

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        metric = p.metric or choose_metric(X)
        hyperedges, weights = build_knn_hyperedges(X, k=p.k, metric=metric, use_gpu=p.use_gpu)
        logger.info("  Hypergraph: %d nodes, %d hyperedges, metric=%s", n, len(hyperedges), metric)
        block_of, km1, imbalance = partition_hypergraph(
            n, hyperedges, weights, spec.splits, seed=spec.seed,
            threads=p.threads, epsilon=spec.epsilon, preset=p.preset)
        labels = np.asarray(block_of)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            imbalance=round(imbalance, 4), km1=int(km1),
                            n_hyperedges=len(hyperedges),
                            leakage=_leakage(X, labels, metric))


@register("graph")
class GraphSplitter(BaseSplitter):
    description = "Weighted k-NN edge-cut (CUT objective) + exact-leakage FM polish"
    arity = "1d"

    @dataclass
    class Params:
        k: int = 15
        metric: Optional[str] = None
        threshold: Optional[float] = None
        max_deg: int = 256
        use_gpu: bool = True
        threads: int = 8
        preset: str = "default"
        fm: bool = True
        fm_max_n: int = 40_000
        fm_max_moves: Optional[int] = None

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        metric = p.metric or choose_metric(X)
        edges, weights = build_knn_graph(X, k=p.k, metric=metric, threshold=p.threshold,
                                   max_deg=p.max_deg, use_gpu=p.use_gpu)
        logger.info("  Graph: %d nodes, %d edges, metric=%s", n, len(edges), metric)
        block_of, cut, imbalance = partition_graph(
            n, edges, weights, spec.splits, seed=spec.seed,
            threads=p.threads, epsilon=spec.epsilon, preset=p.preset)
        labels = np.asarray(block_of)
        moves = 0
        if p.fm and n <= p.fm_max_n:
            labels, moves, _ = fiduccia_mattheyses_exact(
                labels, X, spec.splits, metric=metric, epsilon=spec.epsilon,
                max_moves=p.fm_max_moves)
            logger.info("  FM polish: %d moves", moves)
        assignment = assign_split_names(ids, labels, spec.splits, spec.names)
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            imbalance=round(imbalance, 4), cut=int(cut),
                            n_edges=len(edges), fm_moves=int(moves),
                            leakage=_leakage(X, labels, metric))
