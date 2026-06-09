"""Hypergraph-partitioning splitting backend — a DataSAIL alternative.

Design (from the deep-research recommendation):
  GPU (torch) k-NN similarity-graph construction  ->  Mt-KaHyPar (CPU) cut.

Each entity is a vertex. For each entity we add one *similarity hyperedge*
covering the entity and its k nearest neighbours, weighted by their mean
similarity. Partitioning into balanced blocks while minimizing the connectivity
(KM1) / cut-net objective therefore minimizes the total weight of similarity
relationships that straddle two splits — i.e. data leakage.

The partition is mapped back to ``{entity_id: split_name}`` so it is a drop-in
for the rest of the PALM pipeline (metrics, visualization, ML exports).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Similarity weights are floats in [0, 1]; Mt-KaHyPar needs positive integer
# edge weights, so we scale before rounding.
_WEIGHT_SCALE = 1000


# ── k-NN similarity-graph construction ─────────────────────────────────────

def _is_binary_fingerprint(X):
    return X.shape[1] >= 128 and np.all((X == 0) | (X == 1))


def _knn_torch(X, k, metric, block=4096):
    """k-NN on GPU via torch, chunked over query rows to bound memory.

    Computes top-k per row in blocks of `block` queries against all n
    references, so peak memory is O(block * n) instead of O(n^2). Scales to
    hundreds of thousands of entities on a single GPU. Returns
    (neighbors[n,k] int, sims[n,k] float).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    k = min(k, n - 1)

    if metric == "tanimoto":
        ref = Xt
        card_all = Xt.sum(1)                       # |b| for every reference
    elif metric == "cosine":
        ref = torch.nn.functional.normalize(Xt, dim=1)
    else:  # euclidean (standardized) -> similarity = 1/(1+dist)
        ref = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)

    all_idx = torch.empty((n, k), dtype=torch.long, device=device)
    all_sim = torch.empty((n, k), dtype=torch.float32, device=device)
    for s in range(0, n, block):
        e = min(s + block, n)
        q = ref[s:e]
        if metric == "tanimoto":
            inter = q @ ref.T
            union = card_all[s:e][:, None] + card_all[None, :] - inter
            sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        elif metric == "cosine":
            sim = q @ ref.T
        else:
            sim = 1.0 / (1.0 + torch.cdist(q, ref))
        # exclude self (global column index == row index)
        rows = torch.arange(s, e, device=device)
        sim[torch.arange(e - s, device=device), rows] = -1.0
        ts, ti = torch.topk(sim, k, dim=1)
        all_sim[s:e], all_idx[s:e] = ts, ti

    return all_idx.cpu().numpy(), all_sim.clamp_min(0).cpu().numpy()


def _knn_sklearn(X, k, metric):
    """CPU fallback k-NN. Returns (neighbors[n,k], sims[n,k])."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    k = min(k, n - 1)
    if metric == "tanimoto":
        nn = NearestNeighbors(n_neighbors=k + 1, metric="jaccard")
        Xf = X
    elif metric == "cosine":
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        Xf = X
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        Xf = StandardScaler().fit_transform(X)
    nn.fit(Xf)
    dist, idx = nn.kneighbors(Xf)
    # drop the self column (first neighbour is the point itself)
    idx, dist = idx[:, 1:], dist[:, 1:]
    sim = 1.0 - dist if metric in ("tanimoto", "cosine") else 1.0 / (1.0 + dist)
    return idx, np.clip(sim, 0, None)


def _choose_metric(X):
    if _is_binary_fingerprint(X):
        return "tanimoto"
    sparsity = (X == 0).sum() / X.size if X.size else 0.0
    return "cosine" if sparsity > 0.5 else "euclidean"


def build_knn_hyperedges(X, k=15, metric=None, use_gpu=True):
    """Build weighted k-NN similarity hyperedges from a feature matrix.

    Returns (hyperedges: list[list[int]], weights: list[int], metric: str).
    One hyperedge per node: {node} U {its k nearest neighbours}.
    """
    if metric is None:
        metric = _choose_metric(X)

    if use_gpu:
        try:
            neigh, sims = _knn_torch(X, k, metric)
        except Exception as exc:
            logger.warning("  GPU k-NN failed (%s); falling back to sklearn", exc)
            neigh, sims = _knn_sklearn(X, k, metric)
    else:
        neigh, sims = _knn_sklearn(X, k, metric)

    hyperedges, weights = [], []
    for i in range(neigh.shape[0]):
        members = [i] + [int(j) for j in neigh[i]]
        w = int(round(float(sims[i].mean()) * _WEIGHT_SCALE)) + 1  # >=1
        hyperedges.append(members)
        weights.append(w)
    return hyperedges, weights, metric


# ── Mt-KaHyPar partitioning ────────────────────────────────────────────────

# Mt-KaHyPar must be initialized exactly once per process; re-initializing in a
# loop triggers an "already initialized" warning and non-deterministic results.
_MTK_INITIALIZER = None


def _get_initializer(threads):
    global _MTK_INITIALIZER
    if _MTK_INITIALIZER is None:
        import mtkahypar
        _MTK_INITIALIZER = mtkahypar.initialize(threads)
    return _MTK_INITIALIZER

def partition_hypergraph(n_nodes, hyperedges, edge_weights, splits, seed=42,
                         threads=8, epsilon=0.05, preset="default"):
    """Partition into len(splits) blocks sized by `splits`, minimizing KM1.

    Returns a list `block_of[node] -> block_index`.
    """
    import mtkahypar as mtk

    k = len(splits)
    init = _get_initializer(threads)
    mtk.set_seed(seed)

    preset_map = {
        "default": mtk.PresetType.DEFAULT,
        "quality": mtk.PresetType.QUALITY,
        "highest_quality": mtk.PresetType.HIGHEST_QUALITY,
        "deterministic": mtk.PresetType.DETERMINISTIC,
        "large_k": mtk.PresetType.LARGE_K,
    }
    ctx = init.context_from_preset(preset_map.get(preset, mtk.PresetType.DEFAULT))
    ctx.set_partitioning_parameters(k, epsilon, mtk.Objective.KM1)
    ctx.logging = False

    # Target block max-weights from the split ratios (unit node weights).
    total = sum(splits)
    block_caps = [int(np.ceil(n_nodes * s / total * (1 + epsilon))) + 1 for s in splits]
    ctx.set_individual_target_block_weights(block_caps)

    node_weights = [1] * n_nodes
    hg = init.create_hypergraph(ctx, n_nodes, len(hyperedges), hyperedges,
                                node_weights, edge_weights)
    phg = hg.partition(ctx)
    return list(phg.get_partition()), phg.km1(), phg.imbalance(ctx)


# ── Public entry point ─────────────────────────────────────────────────────

def run_hypergraph_split(feature_data, splits, names, k=15, metric=None,
                         use_gpu=True, seed=42, threads=8, epsilon=0.05,
                         preset="default"):
    """Split entities via hypergraph partitioning.

    Args:
        feature_data: dict {entity_id: feature_vector}
        splits: list of relative split sizes, e.g. [8, 2]
        names:  list of split names, e.g. ["train", "test"]
    Returns:
        dict {entity_id: split_name}
    """
    ids = sorted(feature_data.keys())
    X = np.asarray([feature_data[i] for i in ids], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(ids)
    if n < len(splits):
        raise ValueError(f"Too few entities ({n}) for {len(splits)} splits")

    hyperedges, weights, used_metric = build_knn_hyperedges(X, k=k, metric=metric, use_gpu=use_gpu)
    logger.info("  Hypergraph: %d nodes, %d hyperedges, metric=%s", n, len(hyperedges), used_metric)

    block_of, km1, imbalance = partition_hypergraph(
        n, hyperedges, weights, splits, seed=seed, threads=threads,
        epsilon=epsilon, preset=preset,
    )
    logger.info("  Partition: KM1=%d imbalance=%.3f", km1, imbalance)

    # Map block index -> split name. Largest block -> first split name (train),
    # so block ordering follows the requested split sizes.
    from collections import Counter
    sizes = Counter(block_of)
    blocks_by_size = [b for b, _ in sizes.most_common()]
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    block_to_name = {}
    for rank, block in enumerate(blocks_by_size):
        split_idx = order_by_split[rank] if rank < len(order_by_split) else order_by_split[-1]
        block_to_name[block] = names[split_idx]

    return {ids[i]: block_to_name.get(block_of[i], names[-1]) for i in range(n)}


# ── n-D (reaction / multi-component) splitting ─────────────────────────────

def _cluster_axis(values, feat, sim_threshold):
    """Cluster a single axis's unique values by similarity.

    Returns {value: cluster_label}. Values with no feature vector (or an
    all-zero one) fall back to their own identity cluster. ``sim_threshold`` is
    the minimum similarity to merge; 1.0 -> pure identity (no merging).
    """
    import numpy as _np

    feats = [feat.get(v) for v in values]
    has_feat = [f is not None and _np.any(f) for f in feats]
    labels = {}
    idx = [i for i, h in enumerate(has_feat) if h]
    if sim_threshold >= 1.0 or len(idx) < 2:
        return {v: f"id::{v}" for v in values}

    X = _np.asarray([feats[i] for i in idx], dtype=float)
    metric = _choose_metric(X)
    # distance threshold = 1 - similarity threshold (cosine/tanimoto are in [0,1])
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    if metric in ("tanimoto", "cosine"):
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(X, metric="jaccard" if metric == "tanimoto" else "cosine"))
        cl = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                     linkage="average", distance_threshold=1.0 - sim_threshold)
        lab = cl.fit_predict(D)
    else:
        Xs = StandardScaler().fit_transform(X)
        cl = AgglomerativeClustering(n_clusters=None, linkage="ward",
                                     distance_threshold=float(sim_threshold))
        lab = cl.fit_predict(Xs)

    for j, i in enumerate(idx):
        labels[values[i]] = f"cl::{int(lab[j])}"
    for i, h in enumerate(has_feat):
        if not h:
            labels[values[i]] = f"id::{values[i]}"
    return labels


def run_hypergraph_split_nd(records, axis_feature_maps, splits, names,
                            sim_threshold=1.0, axis_weights=None, seed=42,
                            threads=8, epsilon=0.05, preset="quality"):
    """Split multi-component records (e.g. reactions) via hypergraph partitioning.

    Each record is a vertex. For every component axis we group records that share
    the same (optionally similarity-clustered) component value into one
    hyperedge — cutting it means that component (or a similar one) straddles two
    splits, i.e. leakage on that axis. Minimizing the connectivity objective
    therefore minimizes total cross-split component leakage across all axes at
    once (the n-D generalization DataSAIL's 2-D engine cannot express).

    Args:
        records: list of dicts {axis_name: component_value} (one per record)
        axis_feature_maps: {axis_name: {value: feature_vector or None}}.
            None / all-zero features -> identity hyperedges only (categorical).
        splits, names: e.g. [8, 2], ["train", "test"]
        sim_threshold: min similarity to merge component values into a cluster
            (1.0 = pure identity; <1.0 enables similarity-aware grouping).
        axis_weights: optional {axis_name: float} relative leakage weight.
    Returns:
        (assignment: list[str] per record, info: dict)
    """
    from collections import defaultdict

    n = len(records)
    axes = list(axis_feature_maps.keys())
    aw = axis_weights or {}

    # 1. per-axis value -> cluster label (featurize uniques once, done by caller)
    axis_clusters = {}
    for axis in axes:
        values = sorted({str(r[axis]) for r in records})
        axis_clusters[axis] = _cluster_axis(values, axis_feature_maps[axis], sim_threshold)

    # 2. one hyperedge per (axis, cluster): records sharing that clustered value
    hyperedges, weights = [], []
    per_axis_groups = {}
    for axis in axes:
        v2c = axis_clusters[axis]
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[v2c[str(r[axis])]].append(i)
        per_axis_groups[axis] = groups
        w = max(1, int(round(aw.get(axis, 1.0) * _WEIGHT_SCALE)))
        for members in groups.values():
            if 2 <= len(members) < n:        # singletons/all-spanning add nothing
                hyperedges.append(members)
                weights.append(w)

    if not hyperedges:
        raise ValueError("No non-trivial hyperedges; cannot split (check component variety)")

    # 3. partition records, minimizing connectivity (KM1)
    block_of, km1, imbalance = partition_hypergraph(
        n, hyperedges, weights, splits, seed=seed, threads=threads,
        epsilon=epsilon, preset=preset,
    )

    # 4. map block -> split name by size
    from collections import Counter
    sizes = Counter(block_of)
    blocks_by_size = [b for b, _ in sizes.most_common()]
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    block_to_name = {}
    for rank, block in enumerate(blocks_by_size):
        block_to_name[block] = names[order_by_split[min(rank, len(order_by_split) - 1)]]
    assignment = [block_to_name.get(block_of[i], names[-1]) for i in range(n)]

    # per-axis identity overlap (values appearing in >1 split) for reporting
    overlap = {}
    for axis in axes:
        by_split = defaultdict(set)
        for i, r in enumerate(records):
            by_split[assignment[i]].add(str(r[axis]))
        shared = set.intersection(*by_split.values()) if len(by_split) > 1 else set()
        overlap[axis] = {"n_values": len({str(r[axis]) for r in records}),
                         "shared_across_splits": len(shared)}

    info = {"km1": km1, "imbalance": round(imbalance, 4),
            "n_hyperedges": len(hyperedges), "axis_overlap": overlap}
    return assignment, info
