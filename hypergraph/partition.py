"""Mt-KaHyPar partitioning helpers (hypergraph KM1 and graph CUT).

One place for: the process-wide initializer singleton, the preset map, and the
balanced context setup with explicit per-block capacity caps derived from the
(possibly non-uniform) split ratios. :func:`partition_hypergraph` and
:func:`partition_graph` differ only in the objective (KM1 vs CUT) and whether
they build a hypergraph or a graph — the *cut* half of the method, where the
partitioning objective and balance constraint are tuned.

Mt-KaHyPar only enforces per-block *upper* caps, so a KM1/CUT optimum pushes the
large block to its cap and can starve the small ones below the corridor floor.
Both entry points therefore validate the result against the two-sided corridor
(:func:`balance_corridor`), repair it with objective-aware moves
(:func:`rebalance`) and recompute the objective and imbalance from the FINAL
labels (:func:`connectivity_objective`, :func:`imbalance_of`).
"""

from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np

from PALM.splitters.common.balanced_assignment import target_sizes

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


def balance_corridor(n: int, splits: Sequence[float], epsilon: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-block integer size bounds ``(lo, hi)`` of the ``(1 +/- epsilon)`` corridor.

    With target ``t = n * s / total``: ``lo = ceil((1-eps) t)``, ``hi = floor((1+eps) t)``,
    each widened only as far as needed to contain the exact largest-remainder
    target size (:func:`target_sizes`) — so tiny blocks never get an empty
    corridor, and ``sum(lo) <= n <= sum(hi)`` always holds (feasible).
    """
    total = float(sum(splits))
    t = np.array([n * s / total for s in splits])
    exact = target_sizes(n, splits)
    lo = np.minimum(np.ceil(t * (1 - epsilon) - 1e-9).astype(int), exact)
    hi = np.maximum(np.floor(t * (1 + epsilon) + 1e-9).astype(int), exact)
    return lo, hi


def imbalance_of(labels, splits: Sequence[float]) -> float:
    """Two-sided imbalance ``max_b |size_b / t_b - 1|`` of ``labels`` vs the targets."""
    labels = np.asarray(labels)
    n, total = len(labels), float(sum(splits))
    t = np.array([n * s / total for s in splits])
    sizes = np.bincount(labels, minlength=len(splits))[:len(splits)]
    return float(np.max(np.abs(sizes / t - 1.0)))


def _incidence(hyperedges):
    """Flattened ``(pins, edge_id)`` incidence arrays of a hyperedge/edge list."""
    lens = np.fromiter((len(e) for e in hyperedges), dtype=np.int64, count=len(hyperedges))
    pins = np.fromiter(itertools.chain.from_iterable(hyperedges), dtype=np.int64,
                       count=int(lens.sum()))
    return pins, np.repeat(np.arange(len(hyperedges)), lens)


def connectivity_objective(hyperedges, edge_weights, labels) -> int:
    """KM1 ``sum_e w_e (lambda_e - 1)`` of ``labels``; equals CUT on 2-pin edges."""
    if not len(hyperedges):
        return 0
    labels = np.asarray(labels, dtype=np.int64)
    k = int(labels.max()) + 1
    pins, eid = _incidence(hyperedges)
    lam = np.bincount(np.unique(eid * k + labels[pins]) // k, minlength=len(hyperedges))
    return int((np.asarray(edge_weights, dtype=np.int64) * np.maximum(lam - 1, 0)).sum())


def rebalance(n: int, hyperedges, edge_weights, labels, splits: Sequence[float],
              epsilon: float) -> Tuple[np.ndarray, int]:
    """Move nodes until every block size is inside :func:`balance_corridor`.

    Repeatedly pairs an over-cap (or surplus) donor block with an under-floor
    (or spare-capacity) receiver and moves the donor nodes whose move raises the
    KM1/CUT objective least (batched per pair; ties by node index, so it is
    deterministic). Always terminates: the corridor contains the exact target
    sizes. Returns ``(labels, n_moved)``; a no-op when already balanced.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    k = len(splits)
    lo, hi = balance_corridor(n, splits, epsilon)
    t = np.array([n * s / float(sum(splits)) for s in splits])
    sizes = np.bincount(labels, minlength=k)
    if ((sizes <= hi) & (sizes >= lo)).all():
        return labels, 0

    if len(hyperedges):
        pins, eid = _incidence(hyperedges)
        w = np.asarray(edge_weights, dtype=np.float64)[eid]
    else:
        pins = eid = np.zeros(0, dtype=np.int64); w = np.zeros(0)
    phi = np.zeros((len(hyperedges), k), dtype=np.int64)     # pins of edge e in block b
    np.add.at(phi, (eid, labels[pins]), 1)

    moved = 0
    rel = lambda s: (s - t) / np.maximum(t, 1e-12)             # noqa: E731
    while True:
        sizes = np.bincount(labels, minlength=k)
        over, under = sizes - hi, lo - sizes
        if (over <= 0).all() and (under <= 0).all():
            break
        if (over > 0).any():
            a = int(np.argmax(over))
            ok = sizes < hi
            b = int(np.argmin(np.where(ok, rel(sizes), np.inf)))
            need = int(min(over[a], hi[b] - sizes[b]))
        else:
            b = int(np.argmax(under))
            ok = sizes > lo
            a = int(np.argmax(np.where(ok, rel(sizes), -np.inf)))
            need = int(min(under[b], sizes[a] - lo[a]))
        # objective delta of moving each node a -> b: new blocks joined minus blocks left
        contrib = w * ((phi[eid, b] == 0).astype(np.float64) - (phi[eid, a] == 1))
        delta = np.bincount(pins, weights=contrib, minlength=n)
        delta[labels != a] = np.inf
        take = np.argsort(delta, kind="stable")[:need]
        labels[take] = b
        sel = np.isin(pins, take)
        np.add.at(phi, (eid[sel], a), -1)
        np.add.at(phi, (eid[sel], b), 1)
        moved += need
    return labels, moved


def _finalize(n, hyperedges, edge_weights, block_of, splits, epsilon, return_info):
    """Repair balance, then recompute objective + imbalance from the final labels."""
    labels, moved = rebalance(n, hyperedges, edge_weights, block_of, splits, epsilon)
    obj = connectivity_objective(hyperedges, edge_weights, labels)
    imb = imbalance_of(labels, splits)
    if not return_info:
        return list(labels), obj, imb
    lo, hi = balance_corridor(n, splits, epsilon)
    sizes = np.bincount(labels, minlength=len(splits))
    return list(labels), obj, imb, {
        "balance_repair_moves": int(moved),
        "within_corridor": bool(((sizes >= lo) & (sizes <= hi)).all())}


def _make_context(mtk, init, k, epsilon, objective, n_nodes, splits, preset):
    """Balanced context with explicit per-block caps from the split ratios.

    ``set_partitioning_parameters`` fixes k and the objective; its epsilon is the
    tolerance for *uniform* target weights. We override with explicit per-block
    caps — the upper side of :func:`balance_corridor` (no slack beyond it) — so
    those caps, not epsilon, define the balance constraint Mt-KaHyPar sees. The
    lower side is enforced afterwards by :func:`rebalance`.
    """
    ctx = init.context_from_preset(_preset(mtk, preset))
    ctx.set_partitioning_parameters(k, epsilon, objective)
    ctx.logging = False
    _, hi = balance_corridor(n_nodes, splits, epsilon)
    ctx.set_individual_target_block_weights([int(c) for c in hi])
    return ctx


def partition_hypergraph(n_nodes: int, hyperedges: List[List[int]],
                         edge_weights: List[int], splits: Sequence[float],
                         seed: int = 42, threads: int = 8, epsilon: float = 0.05,
                         preset: str = "default", return_info: bool = False):
    """KM1-minimizing balanced hypergraph partition. Returns ``(block_of, km1, imbalance)``.

    All three are post-repair (see :func:`rebalance`); ``imbalance`` is
    :func:`imbalance_of`. ``return_info=True`` appends a dict with
    ``balance_repair_moves`` / ``within_corridor``.
    """
    import mtkahypar as mtk

    k = len(splits)
    init = get_initializer(threads)
    mtk.set_seed(seed)
    ctx = _make_context(mtk, init, k, epsilon, mtk.Objective.KM1, n_nodes, splits, preset)

    node_weights = [1] * n_nodes
    hg = init.create_hypergraph(ctx, n_nodes, len(hyperedges), hyperedges,
                                node_weights, edge_weights)
    phg = hg.partition(ctx)
    return _finalize(n_nodes, hyperedges, edge_weights, phg.get_partition(), splits,
                     epsilon, return_info)


def partition_graph(n_nodes: int, edges: List[tuple], edge_weights: List[int],
                    splits: Sequence[float], seed: int = 42, threads: int = 8,
                    epsilon: float = 0.05, preset: str = "default",
                    return_info: bool = False):
    """CUT-minimizing balanced weighted-edge partition. Returns ``(block_of, cut, imbalance)``.

    Post-repair, as :func:`partition_hypergraph`.
    """
    import mtkahypar as mtk

    k = len(splits)
    init = get_initializer(threads)
    mtk.set_seed(seed)
    ctx = _make_context(mtk, init, k, epsilon, mtk.Objective.CUT, n_nodes, splits, preset)

    node_weights = [1] * n_nodes
    g = init.create_graph(ctx, n_nodes, len(edges), edges, node_weights, edge_weights)
    pg = g.partition(ctx)
    return _finalize(n_nodes, edges, edge_weights, pg.get_partition(), splits,
                     epsilon, return_info)
