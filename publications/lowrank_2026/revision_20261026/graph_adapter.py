"""Published-method adapters for the low-rank revision. No hypergraph baseline.

The sparse comparator uses Mt-KaHyPar's Graph API and CUT objective. Native
partition output and any subsequent lower-bound repair are retained separately.
Lo-Hi uses its official API with published defaults, without converting its
discarded molecules into a full-coverage partition.
"""

from __future__ import annotations
import hashlib
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix


def integer_graph_weights(values, maximum_scale=1_000_000):
    """Quantize within the installed partitioner's 32-bit accumulation range.

    Reserve an eightfold margin for intermediate gain/weight sums. Rounding
    and minimum-one promotion contribute at most one per edge, accounted for
    before selecting the scale. Return weights and the actual scale.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.array([], dtype=np.int64), maximum_scale
    budget = np.iinfo(np.int32).max // 8
    if len(values) >= budget or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            "invalid positive edge weights or too many edges for safe integer encoding"
        )
    scale = min(
        maximum_scale, max(1, int((budget - len(values)) / max(values.sum(), 1e-300)))
    )
    weights = np.maximum(1, np.rint(values * scale)).astype(np.int64)
    if int(weights.sum()) > budget:
        raise ValueError("graph weights exceed safe 32-bit accumulation budget")
    return weights, scale


def repair_graph_counts(adjacency, labels, floors, caps):
    """Move individual nodes using sparse-cut gains until both bounds hold."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    k = len(caps)
    counts = np.bincount(labels, minlength=k)
    connection = np.asarray(adjacency @ np.eye(k)[labels])
    moved = 0
    while np.any(counts > caps) or np.any(counts < floors):
        best = None
        for a in np.flatnonzero(counts > floors):
            nodes = np.flatnonzero(labels == a)
            for b in np.flatnonzero(counts < caps):
                if a == b or (counts[a] <= caps[a] and counts[b] >= floors[b]):
                    continue
                gains = connection[nodes, b] - connection[nodes, a]
                j = int(np.argmax(gains))
                if best is None or gains[j] > best[0]:
                    best = (gains[j], int(nodes[j]), a, b)
        if best is None:
            raise RuntimeError(
                "no feasible point move for the requested capacity corridor"
            )
        _, u, a, b = best
        lo, hi = adjacency.indptr[u : u + 2]
        idx, weight = adjacency.indices[lo:hi], adjacency.data[lo:hi]
        connection[idx, a] -= weight
        connection[idx, b] += weight
        labels[u] = b
        counts[a] -= 1
        counts[b] += 1
        moved += 1
    return labels, moved


def graph_cut(X, metric, splits=(8, 1, 1), epsilon=0.01, seed=0, k=32, threads=2):
    """Symmetrized union top-k graph, S weights, QUALITY partitioner, CUT.

    Top-k construction is exact on CPU; no thresholding other than S>0.
    Similarity weights are rounded to integers at scale up to 10^6, reduced
    to keep the total within a safe 32-bit accumulation budget. Each undirected
    edge is stored once in Mt-KaHyPar. The returned labels satisfy the two-sided
    corridor; diagnostics include the partitioner's unmodified labels.
    """
    import mtkahypar as mtk
    from lowrank.neighbours import tiled_neighbours
    from lowrank.splitters.common.balanced_assignment import capacity_corridor

    start = time.perf_counter()
    n = len(X)
    idx, similarity = tiled_neighbours(
        X, metric, k=k, row_block=128, reference_block=4096, device="cpu"
    )
    src = np.repeat(np.arange(n), idx.shape[1])
    dst, values = idx.ravel(), similarity.ravel()
    keep = (src != dst) & (values > 0)
    endpoints = np.sort(np.column_stack([src[keep], dst[keep]]), axis=1)
    endpoints, first = np.unique(endpoints, axis=0, return_index=True)
    edge_similarity = values[keep][first].astype(np.float64)
    weights, scale = integer_graph_weights(edge_similarity)
    graph_build_s = time.perf_counter() - start
    t = time.perf_counter()
    initializer = mtk.initialize(threads, print_warnings=False)
    context = initializer.context_from_preset(mtk.PresetType.QUALITY)
    context.set_partitioning_parameters(len(splits), float(epsilon), mtk.Objective.CUT)
    caps, floors = capacity_corridor(n, splits, epsilon)
    context.set_individual_target_block_weights(caps.tolist())
    context.logging = False
    mtk.set_seed(seed)
    graph = initializer.create_graph(
        context,
        n,
        len(endpoints),
        [tuple(map(int, edge)) for edge in endpoints],
        [1] * n,
        weights.tolist(),
    )
    partition = graph.partition(context)
    native = np.asarray(partition.get_partition(), dtype=np.int64)
    exact_native_cut = int(
        weights[native[endpoints[:, 0]] != native[endpoints[:, 1]]].sum()
    )
    if int(partition.cut()) != exact_native_cut:
        raise RuntimeError(
            "native graph cut differs from exact integer sum; possible overflow"
        )
    partition_s = time.perf_counter() - t
    # Use precisely the same integer-weight graph for any necessary repair.
    t = time.perf_counter()
    a, b = endpoints.T if len(endpoints) else (np.array([], int), np.array([], int))
    adjacency = coo_matrix(
        (np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])), shape=(n, n)
    ).tocsr()
    labels, moved = repair_graph_counts(adjacency, native, floors, caps)
    repair_s = time.perf_counter() - t
    try:
        version = importlib.metadata.version("mtkahypar")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(mtk, "__version__", "unknown")
    diag = dict(
        graph_build_s=graph_build_s,
        partition_s=partition_s,
        repair_s=repair_s,
        split_s=time.perf_counter() - start,
        graph_k=k,
        graph_edges=int(len(endpoints)),
        graph_type="ordinary undirected pairwise graph",
        objective="CUT",
        preset="QUALITY",
        edge_rule="union top-k; one edge per unordered pair; positive similarities only",
        weight_scale=scale,
        sum_integer_edge_weights=int(weights.sum()),
        integer_weight_budget=np.iinfo(np.int32).max // 8,
        max_weight_quantization_error=float(
            np.max(np.abs(weights / scale - edge_similarity))
        )
        if len(weights)
        else 0.0,
        native_integer_cut=exact_native_cut,
        threads=threads,
        seed=seed,
        native_labels=native.tolist(),
        native_counts=np.bincount(native, minlength=len(splits)).tolist(),
        repaired_counts=np.bincount(labels, minlength=len(splits)).tolist(),
        repair_moves=moved,
        mtkahypar_version=version,
        mtkahypar_module_sha256=hashlib.sha256(
            Path(mtk.__file__).read_bytes()
        ).hexdigest(),
    )
    return labels, diag
