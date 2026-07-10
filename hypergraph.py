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
    # set_partitioning_parameters fixes k and the KM1 objective; its epsilon is
    # the balance tolerance for *uniform* target block weights. We immediately
    # override the targets below with explicit per-block caps derived from the
    # requested (non-uniform) split ratios, so those caps — not epsilon — define
    # the actual balance constraint. epsilon is still passed for k and as the
    # tolerance Mt-KaHyPar reports `imbalance()` against.
    ctx.set_partitioning_parameters(k, epsilon, mtk.Objective.KM1)
    ctx.logging = False

    # Per-block max-weights from the split ratios (unit node weights), with an
    # (1+epsilon) slack so the partitioner has room to honour the ratios while
    # minimizing the cut. The +1 avoids an infeasible cap from integer rounding.
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
    # float32 throughout: the GPU k-NN casts to float32 anyway, so this halves
    # peak host memory (e.g. ~0.7 GB vs 1.5 GB for 93k x 2048) with no change
    # to the computed neighbours.
    X = np.asarray([feature_data[i] for i in ids], dtype=np.float32)
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

    # Map block index -> split name by descending block size: the largest block
    # becomes the largest-ratio split (e.g. train). Ties are broken by block
    # index for determinism. NOTE: this assumes the requested split ratios are
    # distinct (e.g. [8, 2]); for (near-)equal ratios the size->name assignment
    # is arbitrary among the tied splits, which is harmless since they are
    # interchangeable by construction.
    from collections import Counter
    sizes = Counter(block_of)
    blocks_by_size = sorted(sizes, key=lambda b: (-sizes[b], b))
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    block_to_name = {}
    for rank, block in enumerate(blocks_by_size):
        split_idx = order_by_split[rank] if rank < len(order_by_split) else order_by_split[-1]
        block_to_name[block] = names[split_idx]

    return {ids[i]: block_to_name.get(block_of[i], names[-1]) for i in range(n)}


# ── Weighted graph edge-cut backend (Tier-1 refinement) ────────────────────
#
# The hyperedge/KM1 construction above charges one *mean-weighted* net per node,
# counted once if the neighbourhood straddles the cut. For a 2-block split that
# degenerates to a per-node "is my neighbourhood split?" indicator, discarding
# both the magnitude and the additive long tail of the true leakage metric
# L = sum over train x test pairs of sim(i,j). A weighted 2-uniform *graph*
# edge-cut sums each crossing neighbour pair at its true similarity, exactly
# once — the true objective restricted to the retained edges. This backend
# builds that graph, cuts it with Mt-KaHyPar's graph mode (CUT objective), and
# optionally polishes with a Fiduccia-Mattheyses pass on the *exact* pairwise L.

def _knn_graph_edges_torch(X, k, metric, threshold=None, max_deg=256, block=4096):
    """Directed (row -> col) similarity edges on GPU.

    Keeps, per node, its top-k neighbours plus (if ``threshold`` is set) any
    neighbour with sim >= threshold, capped at ``max_deg`` per node so dense
    congeneric regions contribute their full leakage mass without blowing up.
    Returns (rows, cols, sims) numpy arrays for edges with sim > 0.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    kk = min(k, n - 1)
    K = min(max(kk, max_deg) if threshold is not None else kk, n - 1)

    if metric == "tanimoto":
        ref = Xt
        card = Xt.sum(1)
    elif metric == "cosine":
        ref = torch.nn.functional.normalize(Xt, dim=1)
    else:  # euclidean (standardized) -> similarity = 1/(1+dist)
        ref = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)

    rows_all, cols_all, sims_all = [], [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        q = ref[s:e]
        if metric == "tanimoto":
            inter = q @ ref.T
            union = card[s:e][:, None] + card[None, :] - inter
            sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        elif metric == "cosine":
            sim = q @ ref.T
        else:
            sim = 1.0 / (1.0 + torch.cdist(q, ref))
        rows_g = torch.arange(s, e, device=device)
        sim[torch.arange(e - s, device=device), rows_g] = -1.0   # mask self
        vals, idx = torch.topk(sim, K, dim=1)
        ranks = torch.arange(K, device=device)[None, :]
        keep = ranks < kk
        if threshold is not None:
            keep = keep | (vals >= threshold)
        keep = keep & (vals > 0)
        bi, ki = torch.nonzero(keep, as_tuple=True)
        rows_all.append(rows_g[bi])
        cols_all.append(idx[bi, ki])
        sims_all.append(vals[bi, ki])

    rows = torch.cat(rows_all).cpu().numpy()
    cols = torch.cat(cols_all).cpu().numpy()
    sims = torch.cat(sims_all).clamp_min(0).cpu().numpy() if len(sims_all) else np.array([])
    return rows, cols, sims


def build_knn_graph(X, k=15, metric=None, threshold=None, max_deg=256, use_gpu=True):
    """Build a symmetrized weighted k-NN graph from a feature matrix.

    Returns (edges: list[(u, v)], weights: list[int], metric). Each undirected
    edge weight is round(sim * SCALE)+1 so Mt-KaHyPar's integer CUT objective
    sums to the (scaled) cross-split similarity — i.e. the leakage numerator.
    """
    if metric is None:
        metric = _choose_metric(X)
    n = X.shape[0]

    if use_gpu:
        try:
            rows, cols, sims = _knn_graph_edges_torch(X, k, metric, threshold, max_deg)
        except Exception as exc:
            logger.warning("  GPU k-NN graph failed (%s); falling back to sklearn", exc)
            neigh, s = _knn_sklearn(X, k, metric)
            rows = np.repeat(np.arange(n), neigh.shape[1])
            cols = neigh.ravel()
            sims = s.ravel()
    else:
        neigh, s = _knn_sklearn(X, k, metric)
        rows = np.repeat(np.arange(n), neigh.shape[1])
        cols = neigh.ravel()
        sims = s.ravel()

    if len(rows) == 0:
        return [], [], metric

    # collapse directed -> undirected, keeping the max similarity per pair
    a = np.minimum(rows, cols).astype(np.int64)
    b = np.maximum(rows, cols).astype(np.int64)
    key = a * n + b
    uniq, inv = np.unique(key, return_inverse=True)
    w = np.zeros(len(uniq), dtype=np.float64)
    np.maximum.at(w, inv, sims)
    ua = (uniq // n).astype(int)
    ub = (uniq % n).astype(int)
    edges = list(zip(ua.tolist(), ub.tolist()))
    weights = [int(round(float(x) * _WEIGHT_SCALE)) + 1 for x in w]
    return edges, weights, metric


def partition_graph(n_nodes, edges, edge_weights, splits, seed=42, threads=8,
                    epsilon=0.05, preset="default"):
    """Balanced weighted-edge-cut partition via Mt-KaHyPar graph mode.

    Returns (block_of[node] -> block_index, cut, imbalance).
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
    ctx.set_partitioning_parameters(k, epsilon, mtk.Objective.CUT)
    ctx.logging = False
    total = sum(splits)
    block_caps = [int(np.ceil(n_nodes * s / total * (1 + epsilon))) + 1 for s in splits]
    ctx.set_individual_target_block_weights(block_caps)

    node_weights = [1] * n_nodes
    g = init.create_graph(ctx, n_nodes, len(edges), edges, node_weights, edge_weights)
    pg = g.partition(ctx)
    return list(pg.get_partition()), pg.cut(), pg.imbalance(ctx)


def fm_refine(labels, X, splits, metric=None, epsilon=0.05, max_moves=None,
              block=4096, tol_frac=1e-4):
    """Fiduccia-Mattheyses polish on the EXACT pairwise leakage objective.

    Greedily moves the single node with the largest true-L reduction across the
    cut, subject to per-block capacity/floor from ``splits`` (+/- epsilon), and
    updates the exact per-node/per-block similarity sums incrementally. Monotone:
    every applied move has a non-positive exact delta on
    L = sum_{i,j in different blocks} sim(i,j). Returns (labels, n_moves, info).

    ``labels``: np.int array of block indices (0..k-1).
    """
    import torch

    if metric is None:
        metric = _choose_metric(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n, k = Xt.shape[0], len(splits)
    if metric == "tanimoto":
        ref = Xt
        card = Xt.sum(1)
    elif metric == "cosine":
        ref = torch.nn.functional.normalize(Xt, dim=1)
    else:
        ref = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)

    def sim_col(v):
        """sim(v, :) with self zeroed."""
        if metric == "tanimoto":
            inter = ref @ ref[v]
            union = card + card[v] - inter
            s = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        elif metric == "cosine":
            s = ref @ ref[v]
        else:
            s = 1.0 / (1.0 + torch.cdist(ref, ref[v][None, :]).squeeze(1))
        s[v] = 0.0
        return s

    lab = torch.as_tensor(labels, dtype=torch.long, device=device)
    onehot = torch.zeros(n, k, device=device)
    onehot[torch.arange(n, device=device), lab] = 1.0
    # S[i, c] = sum_{u in block c} sim(i, u), self excluded (diag zeroed per block)
    S = torch.zeros(n, k, device=device)
    for s0 in range(0, n, block):
        e0 = min(s0 + block, n)
        q = ref[s0:e0]
        if metric == "tanimoto":
            inter = q @ ref.T
            union = card[s0:e0][:, None] + card[None, :] - inter
            sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        elif metric == "cosine":
            sim = q @ ref.T
        else:
            sim = 1.0 / (1.0 + torch.cdist(q, ref))
        rg = torch.arange(s0, e0, device=device)
        sim[torch.arange(e0 - s0, device=device), rg] = 0.0
        S[s0:e0] = sim @ onehot

    sizes = torch.bincount(lab, minlength=k).float()
    total = sum(splits)
    caps = torch.tensor([np.ceil(n * s / total * (1 + epsilon)) + 1 for s in splits], device=device)
    floors = torch.tensor([np.floor(n * s / total * (1 - epsilon)) for s in splits], device=device)
    tol = tol_frac  # minimum reduction (in raw similarity units) to bother moving

    max_moves = max_moves if max_moves is not None else n
    moves = 0
    L0 = None
    while moves < max_moves:
        can_recv = sizes < caps                      # (k,) blocks with room
        can_leave = sizes > floors                   # (k,) blocks that can shrink
        Smask = S.clone()
        Smask[:, ~can_recv] = -float("inf")
        best_val, best_b = Smask.max(dim=1)
        cur = S.gather(1, lab[:, None]).squeeze(1)
        red = best_val - cur                          # reduction in true L
        red[best_b == lab] = -float("inf")
        red[~can_leave[lab]] = -float("inf")
        v = int(torch.argmax(red).item())
        if not torch.isfinite(red[v]) or red[v].item() <= tol:
            break
        a, b = int(lab[v].item()), int(best_b[v].item())
        col = sim_col(v)
        S[:, a] -= col
        S[:, b] += col
        lab[v] = b
        sizes[a] -= 1
        sizes[b] += 1
        moves += 1

    labels_out = lab.cpu().numpy()
    return labels_out, moves, {}


def run_graph_split(feature_data, splits, names, k=15, metric=None, threshold=None,
                    max_deg=256, use_gpu=True, seed=42, threads=8, epsilon=0.05,
                    preset="default", fm=True, fm_max_n=40000, fm_max_moves=None):
    """Split entities via weighted graph edge-cut (+ optional FM polish).

    Drop-in analogue of :func:`run_hypergraph_split` that minimizes the true
    pairwise leakage far more faithfully. Returns {entity_id: split_name}.
    """
    ids = sorted(feature_data.keys())
    X = np.asarray([feature_data[i] for i in ids], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(ids)
    if n < len(splits):
        raise ValueError(f"Too few entities ({n}) for {len(splits)} splits")

    edges, weights, used_metric = build_knn_graph(
        X, k=k, metric=metric, threshold=threshold, max_deg=max_deg, use_gpu=use_gpu)
    logger.info("  Graph: %d nodes, %d edges, metric=%s", n, len(edges), used_metric)
    block_of, cut, imbalance = partition_graph(
        n, edges, weights, splits, seed=seed, threads=threads,
        epsilon=epsilon, preset=preset)
    logger.info("  Partition: CUT=%d imbalance=%.3f", cut, imbalance)

    labels = np.asarray(block_of)
    if fm and n <= fm_max_n:
        labels, n_moves, _ = fm_refine(
            labels, X, splits, metric=used_metric, epsilon=epsilon,
            max_moves=fm_max_moves)
        logger.info("  FM polish: %d moves", n_moves)

    from collections import Counter
    sizes = Counter(labels.tolist())
    blocks_by_size = sorted(sizes, key=lambda b: (-sizes[b], b))
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    block_to_name = {}
    for rank, blk in enumerate(blocks_by_size):
        block_to_name[blk] = names[order_by_split[min(rank, len(order_by_split) - 1)]]
    return {ids[i]: block_to_name.get(int(labels[i]), names[-1]) for i in range(n)}


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
        sim_threshold: min similarity to merge component values into a cluster.
            Default 1.0 = pure identity grouping (no merging); set <1.0 to enable
            similarity-aware grouping, which is the n-D generalization's main
            advantage over identity-only leakage control.
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
        # Report how many unique values had no usable feature and therefore fell
        # back to identity-only hyperedges (no similarity grouping on those).
        n_identity = sum(1 for lab in axis_clusters[axis].values() if lab.startswith("id::"))
        if n_identity:
            logger.info("  Axis %r: %d/%d values fall back to identity (no features)",
                        axis, n_identity, len(values))

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

    # 4. map block -> split name by descending size (ties broken by block index)
    from collections import Counter
    sizes = Counter(block_of)
    blocks_by_size = sorted(sizes, key=lambda b: (-sizes[b], b))
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


def run_hypergraph_split_nd_knn(records, axis_feature_maps, splits, names,
                                k=25, seed=42, threads=8, epsilon=0.05,
                                preset="quality"):
    """Multi-component split via per-axis *k-NN* hyperedges (record-level).

    Unlike :func:`run_hypergraph_split_nd`, which groups each axis's component
    values by identity/similarity *clusters*, this builds, for every axis, one
    similarity hyperedge per record covering that record and its ``k`` nearest
    neighbours *on that axis* (Tanimoto/cosine over the component feature),
    weighted by mean similarity — exactly the 1D construction of
    :func:`run_hypergraph_split`, applied per axis and unioned. Minimizing KM1
    then minimizes the full pairwise similarity that straddles the split on
    every axis at once, which tracks the scaled ``L(pi)`` metric far better than
    the cluster construction on high-cardinality, near-unique axes (e.g. diverse
    reaction reactants), where few exact/near-exact value groups exist.

    Records whose component lacks a feature on an axis are excluded from that
    axis's k-NN graph (they contribute no similarity edge there).
    """
    from collections import Counter

    n = len(records)
    axes = list(axis_feature_maps.keys())
    all_edges, all_w = [], []
    for axis in axes:
        vals = [str(r[axis]) for r in records]
        feat_map = axis_feature_maps[axis]
        # per-record feature matrix on this axis; keep only records with a feature
        dim = 0
        for v in vals:
            f = feat_map.get(v)
            if f is not None and np.any(f):
                dim = len(np.asarray(f).ravel()); break
        if not dim:
            continue
        idx_keep, rows = [], []
        for i, v in enumerate(vals):
            f = feat_map.get(v)
            if f is not None and np.any(f):
                idx_keep.append(i); rows.append(np.asarray(f, dtype=np.float32).ravel())
        if len(idx_keep) < 3:
            continue
        X = np.vstack(rows)
        edges, w, _ = build_knn_hyperedges(X, k=min(k, len(idx_keep) - 1),
                                           metric=None, use_gpu=True)
        remap = np.asarray(idx_keep)
        for e, ww in zip(edges, w):
            all_edges.append([int(remap[j]) for j in e]); all_w.append(ww)
    if not all_edges:
        raise ValueError("No k-NN hyperedges built (no axis had usable features)")

    block_of, km1, imbalance = partition_hypergraph(
        n, all_edges, all_w, splits, seed=seed, threads=threads,
        epsilon=epsilon, preset=preset)

    sizes = Counter(block_of)
    blocks_by_size = sorted(sizes, key=lambda b: (-sizes[b], b))
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    block_to_name = {}
    for rank, block in enumerate(blocks_by_size):
        block_to_name[block] = names[order_by_split[min(rank, len(order_by_split) - 1)]]
    assignment = [block_to_name.get(block_of[i], names[-1]) for i in range(n)]
    info = {"km1": km1, "imbalance": round(imbalance, 4), "n_hyperedges": len(all_edges)}
    return assignment, info
