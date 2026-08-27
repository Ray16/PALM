"""Mt-KaHyPar partitioning helpers (hypergraph KM1 and graph CUT).

One place for: the process-wide initializer singleton, the preset map, and the
balanced context setup with explicit per-block capacity caps derived from the
(possibly non-uniform) split ratios. :func:`partition_hypergraph` and
:func:`partition_graph` differ only in the objective (KM1 vs CUT) and whether
they build a hypergraph or a graph.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Mt-KaHyPar must be initialized exactly once per process; re-initializing in a
# loop triggers an "already initialized" warning and non-deterministic results.
_MTK_INITIALIZER = None


def get_initializer(threads: int):
    """Return the process-wide Mt-KaHyPar initializer (created once)."""
    global _MTK_INITIALIZER
    if _MTK_INITIALIZER is None:
        import mtkahypar
        _MTK_INITIALIZER = mtkahypar.initialize(threads)
    return _MTK_INITIALIZER


def _preset(mtk, preset: str):
    preset_map = {
        "default": mtk.PresetType.DEFAULT,
        "quality": mtk.PresetType.QUALITY,
        "highest_quality": mtk.PresetType.HIGHEST_QUALITY,
        "deterministic": mtk.PresetType.DETERMINISTIC,
        "large_k": mtk.PresetType.LARGE_K,
    }
    return preset_map.get(preset, mtk.PresetType.DEFAULT)


def _make_context(mtk, init, k, epsilon, objective, n_nodes, splits, preset):
    """Balanced context with explicit per-block caps from the split ratios.

    ``set_partitioning_parameters`` fixes k and the objective; its epsilon is the
    tolerance for *uniform* target weights. We override with explicit per-block
    caps derived from the requested ratios, so those caps — not epsilon — define
    the balance constraint. epsilon is still passed for k and as the tolerance
    Mt-KaHyPar reports ``imbalance()`` against.
    """
    ctx = init.context_from_preset(_preset(mtk, preset))
    ctx.set_partitioning_parameters(k, epsilon, objective)
    ctx.logging = False
    total = sum(splits)
    block_caps = [int(np.ceil(n_nodes * s / total * (1 + epsilon))) + 1 for s in splits]
    ctx.set_individual_target_block_weights(block_caps)
    return ctx


def partition_hypergraph(n_nodes: int, hyperedges: List[List[int]],
                         edge_weights: List[int], splits: Sequence[float],
                         seed: int = 42, threads: int = 8, epsilon: float = 0.05,
                         preset: str = "default") -> Tuple[list, int, float]:
    """KM1-minimizing balanced hypergraph partition. Returns ``(block_of, km1, imbalance)``."""
    import mtkahypar as mtk

    k = len(splits)
    init = get_initializer(threads)
    mtk.set_seed(seed)
    ctx = _make_context(mtk, init, k, epsilon, mtk.Objective.KM1, n_nodes, splits, preset)

    node_weights = [1] * n_nodes
    hg = init.create_hypergraph(ctx, n_nodes, len(hyperedges), hyperedges,
                                node_weights, edge_weights)
    phg = hg.partition(ctx)
    return list(phg.get_partition()), phg.km1(), phg.imbalance(ctx)


def partition_graph(n_nodes: int, edges: List[tuple], edge_weights: List[int],
                    splits: Sequence[float], seed: int = 42, threads: int = 8,
                    epsilon: float = 0.05, preset: str = "default") -> Tuple[list, int, float]:
    """CUT-minimizing balanced weighted-edge partition. Returns ``(block_of, cut, imbalance)``."""
    import mtkahypar as mtk

    k = len(splits)
    init = get_initializer(threads)
    mtk.set_seed(seed)
    ctx = _make_context(mtk, init, k, epsilon, mtk.Objective.CUT, n_nodes, splits, preset)

    node_weights = [1] * n_nodes
    g = init.create_graph(ctx, n_nodes, len(edges), edges, node_weights, edge_weights)
    pg = g.partition(ctx)
    return list(pg.get_partition()), pg.cut(), pg.imbalance(ctx)
