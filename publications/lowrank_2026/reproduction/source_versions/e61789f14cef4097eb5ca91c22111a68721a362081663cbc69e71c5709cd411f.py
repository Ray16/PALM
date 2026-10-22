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
            raise RuntimeError("no feasible point move for the requested capacity corridor")
        _, u, a, b = best
        lo, hi = adjacency.indptr[u:u + 2]
        idx, weight = adjacency.indices[lo:hi], adjacency.data[lo:hi]
        connection[idx, a] -= weight
        connection[idx, b] += weight
        labels[u] = b
        counts[a] -= 1
        counts[b] += 1
        moved += 1
    return labels, moved


def graph_cut(X, metric, splits=(8, 1, 1), epsilon=.01, seed=0, k=32, threads=2):
    """Symmetrized union top-k graph, S weights, QUALITY partitioner, CUT.

    Top-k construction is exact on CPU; no thresholding other than S>0.
    Similarity weights are rounded to integers at scale 10^6. Each undirected
    edge is stored once in Mt-KaHyPar. The returned labels satisfy the two-sided
    corridor; diagnostics include the partitioner's unmodified labels.
    """
    import mtkahypar as mtk
    from lowrank.neighbours import tiled_neighbours
    from splitters.common.balanced_assignment import capacity_corridor
    start = time.perf_counter()
    n = len(X)
    idx, similarity = tiled_neighbours(X, metric, k=k, row_block=128,
                                      reference_block=4096, device="cpu")
    src = np.repeat(np.arange(n), idx.shape[1])
    dst, values = idx.ravel(), similarity.ravel()
    keep = (src != dst) & (values > 0)
    endpoints = np.sort(np.column_stack([src[keep], dst[keep]]), axis=1)
    endpoints, first = np.unique(endpoints, axis=0, return_index=True)
    weights = np.maximum(1, np.rint(values[keep][first].astype(np.float64) * 1_000_000)).astype(np.int64)
    graph_build_s = time.perf_counter() - start
    t = time.perf_counter()
    initializer = mtk.initialize(threads, print_warnings=False)
    context = initializer.context_from_preset(mtk.PresetType.QUALITY)
    context.set_partitioning_parameters(len(splits), float(epsilon), mtk.Objective.CUT)
    caps, floors = capacity_corridor(n, splits, epsilon)
    context.set_individual_target_block_weights(caps.tolist())
    context.logging = False
    mtk.set_seed(seed)
    graph = initializer.create_graph(context, n, len(endpoints),
        [tuple(map(int, edge)) for edge in endpoints], [1] * n, weights.tolist())
    partition = graph.partition(context)
    native = np.asarray(partition.get_partition(), dtype=np.int64)
    partition_s = time.perf_counter() - t
    # Use precisely the same integer-weight graph for any necessary repair.
    t = time.perf_counter()
    a, b = endpoints.T if len(endpoints) else (np.array([], int), np.array([], int))
    adjacency = coo_matrix((np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    labels, moved = repair_graph_counts(adjacency, native, floors, caps)
    repair_s = time.perf_counter() - t
    try:
        version = importlib.metadata.version("mtkahypar")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(mtk, "__version__", "unknown")
    diag = dict(graph_build_s=graph_build_s, partition_s=partition_s, repair_s=repair_s,
        split_s=time.perf_counter() - start, graph_k=k, graph_edges=int(len(endpoints)),
        graph_type="ordinary undirected pairwise graph", objective="CUT", preset="QUALITY",
        edge_rule="union top-k; one edge per unordered pair; positive similarities only",
        weight_scale=1_000_000, threads=threads, seed=seed,
        native_labels=native.tolist(), native_counts=np.bincount(native, minlength=len(splits)).tolist(),
        repaired_counts=np.bincount(labels, minlength=len(splits)).tolist(), repair_moves=moved,
        mtkahypar_version=version, mtkahypar_module_sha256=hashlib.sha256(Path(mtk.__file__).read_bytes()).hexdigest())
    return labels, diag


def lohi_hi(smiles):
    """Official Hi defaults. Preserve row indices and explicitly expose discards."""
    import lohi_splitter as lohi
    start = time.perf_counter()
    train, test = lohi.hi_train_test_split(smiles=np.asarray(smiles),
        similarity_threshold=.4, train_min_frac=.7, test_min_frac=.1,
        coarsening_threshold=None, verbose=False, max_mip_gap=.1)
    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
    both = np.r_[train, test]
    if len(np.unique(both)) != len(both) or np.any(both < 0) or np.any(both >= len(smiles)):
        raise ValueError("Lo-Hi returned invalid or overlapping row indices")
    labels = np.full(len(smiles), "discard", dtype="U7")
    labels[train], labels[test] = "train", "test"
    return labels, dict(split_s=time.perf_counter() - start, similarity_threshold=.4,
        train_min_frac=.7, test_min_frac=.1, max_mip_gap=.1, coarsening_threshold=None,
        discard_fraction=float(np.mean(labels == "discard")), train_indices=train.tolist(),
        test_indices=test.tolist(),
        solver_status="assignment returned; API supplies no optimality certificate",
        module_path=str(Path(lohi.__file__)),
        module_sha256=hashlib.sha256(Path(lohi.__file__).read_bytes()).hexdigest())
