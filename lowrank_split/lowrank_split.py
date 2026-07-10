"""Low-rank factorized leakage-minimizing splitter — a graph-free alternative.

Motivation
----------
The leakage metric PALM cares about is the *full* pairwise similarity that
straddles the train/test boundary,

    L(pi) = sum_{i in train, j in test} sim(i, j),

which is an O(n^2) object. The hypergraph/graph backends approximate it with a
sparse k-NN graph (which truncates similarity in dense/congeneric regions) and
hand it to a multilevel partitioner. This module takes a different route: it
*factorizes* the similarity matrix,

    S ~= B B^T ,   B in R^{n x r}   (r << n),

using a Nystrom approximation (Tanimoto/Jaccard is a valid positive-definite
kernel, so this is well posed). With the per-block feature sums
``p_c = sum_{i in block c} B_i``, the leakage decomposes exactly in the factor
space as

    cross-leakage  =  sum_{c < c'} p_c . p_{c'}
                   =  0.5 ( ||sum_c p_c||^2 - sum_c ||p_c||^2 ) ,

which is evaluated in O(n r) and never materializes S. Minimizing it is
equivalent to a *balanced k-means / max-diversity* partition in the B-space, so
we optimize it with a balanced-Lloyd sweep (batched, O(n r k) per iteration)
plus an optional Fiduccia-Mattheyses single-move polish. No graph, no ILP, no
k-NN truncation — the factorization captures the whole similarity matrix.

Empirically (see ``derisk_lowrank.py``) the factor-space objective correlates
0.995-1.000 with the exact ECFP ``scaled_lpi`` at rank 128-256, so optimizing it
optimizes the real metric.

Public entry point: :func:`run_lowrank_split`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Metrics whose similarity lies in [0, 1] and is a valid PD kernel for Nystrom.
_SIM_METRICS = ("tanimoto", "cosine")


# ── metric / feature helpers ───────────────────────────────────────────────

def _is_binary_fingerprint(X: np.ndarray) -> bool:
    """True if X looks like a binary fingerprint (>=128 dims, all 0/1)."""
    return X.shape[1] >= 128 and bool(np.all((X == 0) | (X == 1)))


def choose_metric(X: np.ndarray) -> str:
    """Pick a similarity metric from feature characteristics (matches PALM)."""
    if _is_binary_fingerprint(X):
        return "tanimoto"
    sparsity = (X == 0).sum() / X.size if X.size else 0.0
    return "cosine" if sparsity > 0.5 else "euclidean"


def _get_torch():
    import torch
    return torch


def _similarity(A, B, metric: str):
    """Pairwise similarity between rows of A (a x d) and B (b x d) -> (a x b).

    tanimoto : Jaccard over binary vectors, inter / (|A| + |B| - inter)
    cosine   : inner product of L2-normalized rows
    euclidean: 1 / (1 + distance) on standardized features (a bounded similarity)
    """
    torch = _get_torch()
    if metric == "tanimoto":
        inter = A @ B.T
        card_a = A.sum(1)[:, None]
        card_b = B.sum(1)[None, :]
        union = card_a + card_b - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(inter))
    if metric == "cosine":
        An = torch.nn.functional.normalize(A, dim=1)
        Bn = torch.nn.functional.normalize(B, dim=1)
        return An @ Bn.T
    # euclidean -> bounded similarity
    return 1.0 / (1.0 + torch.cdist(A, B))


# ── Nystrom low-rank factorization  S ~= B B^T ─────────────────────────────

def _kmeanspp_landmarks(Xt, n_landmarks: int, metric: str, seed: int) -> "np.ndarray":
    """Pick landmark rows by k-means++ (D^2-weighted) sampling on the GPU.

    Gives better coverage of dense regions than uniform sampling while avoiding
    the pure-outlier bias of farthest-point sampling — the right trade-off for a
    Nystrom basis. Returns landmark row indices (length ``n_landmarks``).
    """
    torch = _get_torch()
    n = Xt.shape[0]
    n_landmarks = min(n_landmarks, n)
    gen = torch.Generator(device=Xt.device).manual_seed(seed)

    first = int(torch.randint(0, n, (1,), generator=gen, device=Xt.device).item())
    chosen = [first]
    # min "distance" (= 1 - similarity) of every point to the chosen set
    sim_to_set = _similarity(Xt, Xt[first][None, :], metric).squeeze(1)
    min_dist = 1.0 - sim_to_set
    for _ in range(n_landmarks - 1):
        weights = torch.clamp(min_dist, min=0.0) ** 2
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        nxt = int(torch.multinomial(weights, 1, generator=gen).item())
        chosen.append(nxt)
        sim_new = _similarity(Xt, Xt[nxt][None, :], metric).squeeze(1)
        min_dist = torch.minimum(min_dist, 1.0 - sim_new)
    return np.asarray(chosen, dtype=np.int64)


def nystrom_features(
    X: np.ndarray,
    rank: int = 256,
    metric: Optional[str] = None,
    landmark: str = "kmeans++",
    seed: int = 0,
) -> Tuple[np.ndarray, str]:
    """Nystrom low-rank factor B such that ``B @ B.T ~= S`` (the similarity matrix).

    Args:
        X: (n, d) feature matrix.
        rank: number of Nystrom landmarks r (the factor dimension).
        metric: similarity metric; inferred from X if None.
        landmark: 'kmeans++' (default, density-aware) or 'uniform'.
        seed: RNG seed for landmark selection.
    Returns:
        (B, metric) with B of shape (n, r), float32.
    """
    torch = _get_torch()
    if metric is None:
        metric = choose_metric(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = Xt.shape[0]
    rank = min(rank, n)

    if landmark == "uniform":
        gen = torch.Generator(device=device).manual_seed(seed)
        idx = torch.randperm(n, generator=gen, device=device)[:rank].cpu().numpy()
    else:
        idx = _kmeanspp_landmarks(Xt, rank, metric, seed)
    L = Xt[torch.as_tensor(idx, device=device)]           # (r, d) landmarks

    C = _similarity(Xt, L, metric)                        # (n, r)  point-landmark
    W = _similarity(L, L, metric)                         # (r, r)  landmark-landmark
    # B = C W^{-1/2}, so B B^T = C W^{-1} C^T (the Nystrom approximation of S).
    evals, evecs = torch.linalg.eigh(W)
    evals = torch.clamp(evals, min=1e-6)
    W_inv_sqrt = (evecs * (1.0 / torch.sqrt(evals))[None, :]) @ evecs.T
    B = C @ W_inv_sqrt
    return B.cpu().numpy().astype(np.float32), metric


# ── low-rank leakage objective + balanced optimization ─────────────────────

def _block_sums(Bt, labels_t, n_blocks: int):
    """P[c] = sum_{i in block c} B_i, shape (k, r)."""
    torch = _get_torch()
    r = Bt.shape[1]
    P = torch.zeros(n_blocks, r, device=Bt.device, dtype=Bt.dtype)
    P.index_add_(0, labels_t, Bt)
    return P


def lowrank_leakage(B: np.ndarray, labels: Sequence[int], n_blocks: int) -> float:
    """Cross-block leakage in factor space: 0.5(||s||^2 - sum_c ||p_c||^2).

    Equals sum_{c<c'} p_c . p_{c'} ~= the exact cross-split similarity, self
    pairs excluded. This is the quantity the optimizer minimizes and the score
    used to pick among restarts.
    """
    torch = _get_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    P = _block_sums(Bt, lab, n_blocks)
    s = P.sum(0)
    return 0.5 * float((s @ s) - (P * P).sum())


def _capacities(n: int, splits: Sequence[float], epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-block max/min sizes for the (1 +/- epsilon) balance corridor (FM slack)."""
    total = float(sum(splits))
    caps = np.array([int(np.ceil(n * s / total * (1 + epsilon))) + 1 for s in splits])
    floors = np.array([int(np.floor(n * s / total * (1 - epsilon))) for s in splits])
    return caps, floors


def _target_sizes(n: int, splits: Sequence[float]) -> np.ndarray:
    """Exact per-block sizes at the requested ratio, summing to n (largest remainder)."""
    total = float(sum(splits))
    raw = np.array([n * s / total for s in splits])
    sizes = np.floor(raw).astype(int)
    remainder = n - int(sizes.sum())
    # hand the leftover units to the blocks with the largest fractional parts
    for c in np.argsort(-(raw - sizes))[:remainder]:
        sizes[c] += 1
    return sizes


def _balanced_assign(scores, sizes: np.ndarray):
    """Assign n points to k blocks of *exactly* ``sizes`` maximizing chosen scores.

    ``scores``: (n, k) torch tensor, higher = better fit of point i to block c.
    ``sizes``: exact target block sizes summing to n. Exact/optimal for k == 2
    (sort by block-preference difference); greedy regret-ordered for k > 2.
    Returns a (n,) long tensor of labels.
    """
    torch = _get_torch()
    n, k = scores.shape
    if k == 2:
        diff = scores[:, 1] - scores[:, 0]                # >0 prefers block 1
        order = torch.argsort(diff, descending=True)
        lab = torch.zeros(n, dtype=torch.long, device=scores.device)
        lab[order[:int(sizes[1])]] = 1                    # top-size1 -> block 1
        return lab
    # k > 2: exact sizes via sequential "peel" — fully vectorized (k-1 sorts), so
    # it scales to millions of rows. Peel block 0 as the top-sizes[0] rows by
    # (score for block 0) - (best score among still-competing blocks), then peel
    # block 1 from the remainder, etc. Each block gets exactly its target size.
    dev = scores.device
    remaining = torch.arange(n, device=dev)
    lab = torch.full((n,), k - 1, dtype=torch.long, device=dev)
    for c in range(k - 1):
        sub = scores[remaining]
        other = sub[:, c + 1:].max(dim=1).values          # best not-yet-peeled block
        pref = sub[:, c] - other
        order = torch.argsort(pref, descending=True)
        take = order[:int(sizes[c])]
        lab[remaining[take]] = c
        keep = torch.ones(remaining.numel(), dtype=torch.bool, device=dev)
        keep[take] = False
        remaining = remaining[keep]
    return lab


def balanced_lloyd(
    B: np.ndarray,
    splits: Sequence[float],
    epsilon: float = 0.05,
    n_iter: int = 25,
    init_labels: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Balanced-Lloyd minimization of the low-rank leakage in B-space.

    Alternates: (1) block sums P from the current labels, (2) reassign every
    point to the most-similar block subject to capacity. Each iteration is
    O(n r k). Returns the final labels (numpy int array).
    """
    torch = _get_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    n, _ = Bt.shape
    k = len(splits)
    sizes = _target_sizes(n, splits)                      # exact target ratio
    self_sim = (Bt * Bt).sum(1)                           # ||B_i||^2, self term

    if init_labels is None:
        gen = np.random.default_rng(seed)
        order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sizes)])
        gen.shuffle(order)
        lab = torch.as_tensor(order, dtype=torch.long, device=device)
    else:
        lab = torch.as_tensor(np.asarray(init_labels), dtype=torch.long, device=device)

    for _ in range(n_iter):
        P = _block_sums(Bt, lab, k)                       # (k, r)
        scores = Bt @ P.T                                 # (n, k): sim of i to block c
        # exclude self similarity from a point's own current block
        scores[torch.arange(n, device=device), lab] -= self_sim
        new_lab = _balanced_assign(scores, sizes)
        if torch.equal(new_lab, lab):
            break
        lab = new_lab
    return lab.cpu().numpy()


def fm_polish(
    B: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[float],
    epsilon: float = 0.05,
    max_moves: Optional[int] = None,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """Fiduccia-Mattheyses single-move polish on the low-rank leakage objective.

    Repeatedly moves the one point with the largest exact leakage reduction
    across the cut, subject to the capacity/floor corridor, updating the factor
    sums incrementally. Monotone: every applied move strictly lowers the
    low-rank leakage. Returns (labels, n_moves).
    """
    torch = _get_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    n, _ = Bt.shape
    k = len(splits)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    caps, floors = _capacities(n, splits, epsilon)
    caps_t = torch.as_tensor(caps, device=device)
    floors_t = torch.as_tensor(floors, device=device)
    self_sim = (Bt * Bt).sum(1)

    P = _block_sums(Bt, lab, k)                            # (k, r)
    S = Bt @ P.T                                           # (n, k)
    sizes = torch.bincount(lab, minlength=k).to(torch.float32)

    max_moves = max_moves if max_moves is not None else n
    moves = 0
    while moves < max_moves:
        can_recv = sizes < caps_t
        can_leave = sizes > floors_t
        S_recv = S.clone()
        S_recv[:, ~can_recv] = -float("inf")
        best_val, best_b = S_recv.max(dim=1)
        cur = S.gather(1, lab[:, None]).squeeze(1) - self_sim      # self excluded
        reduction = best_val - cur
        reduction[best_b == lab] = -float("inf")
        reduction[~can_leave[lab]] = -float("inf")
        v = int(torch.argmax(reduction).item())
        if not torch.isfinite(reduction[v]) or reduction[v].item() <= tol:
            break
        a, b = int(lab[v].item()), int(best_b[v].item())
        col = Bt @ Bt[v]                                   # (n,) factor-space sim to v
        S[:, a] -= col
        S[:, b] += col
        lab[v] = b
        sizes[a] -= 1
        sizes[b] += 1
        moves += 1
    return lab.cpu().numpy(), moves


# ── public entry point ─────────────────────────────────────────────────────

def _blocks_to_names(labels: np.ndarray, splits: Sequence[float], names: Sequence[str]) -> Dict[int, str]:
    """Map block index -> split name by descending block size (largest -> train)."""
    from collections import Counter
    sizes = Counter(labels.tolist())
    blocks_by_size = sorted(sizes, key=lambda b: (-sizes[b], b))
    order_by_split = sorted(range(len(splits)), key=lambda i: splits[i], reverse=True)
    return {blk: names[order_by_split[min(rank, len(order_by_split) - 1)]]
            for rank, blk in enumerate(blocks_by_size)}


def run_lowrank_split(
    feature_data: Dict[Hashable, np.ndarray],
    splits: Sequence[float],
    names: Sequence[str],
    rank: int = 256,
    metric: Optional[str] = None,
    landmark: str = "kmeans++",
    n_restarts: int = 4,
    n_iter: int = 25,
    fm: bool = True,
    fm_max_n: int = 200_000,
    epsilon: float = 0.05,
    seed: int = 0,
) -> Dict[Hashable, str]:
    """Split entities by minimizing the low-rank factorized leakage objective.

    Drop-in analogue of ``hypergraph.run_hypergraph_split`` / ``run_graph_split``.

    Args:
        feature_data: {entity_id: feature_vector}.
        splits, names: e.g. [8, 2], ["train", "test"].
        rank: Nystrom rank r (factor dimension). 256 is a strong default.
        metric: similarity metric; inferred if None.
        landmark: Nystrom landmark sampling ('kmeans++' or 'uniform').
        n_restarts: independent balanced-Lloyd restarts; the lowest-leakage one
            is kept (selection is free — it uses the O(n r) factor objective).
        n_iter: max Lloyd iterations per restart.
        fm: run the monotone FM polish on the best restart.
        fm_max_n: skip FM above this n (its single-move loop is O(n r)/move).
        epsilon: balance-corridor tolerance.
        seed: base RNG seed.
    Returns:
        {entity_id: split_name}.
    """
    ids = sorted(feature_data.keys())
    X = np.asarray([feature_data[i] for i in ids], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(ids)
    if n < len(splits):
        raise ValueError(f"Too few entities ({n}) for {len(splits)} splits")

    B, used_metric = nystrom_features(X, rank=rank, metric=metric, landmark=landmark, seed=seed)
    logger.info("  Low-rank: n=%d rank=%d metric=%s", n, B.shape[1], used_metric)

    best_labels, best_obj = None, np.inf
    for r in range(n_restarts):
        labels = balanced_lloyd(B, splits, epsilon=epsilon, n_iter=n_iter, seed=seed + r)
        obj = lowrank_leakage(B, labels, len(splits))
        if obj < best_obj:
            best_obj, best_labels = obj, labels
    logger.info("  Best-of-%d Lloyd leakage=%.1f", n_restarts, best_obj)

    if fm and n <= fm_max_n:
        best_labels, n_moves = fm_polish(B, best_labels, splits, epsilon=epsilon)
        logger.info("  FM polish: %d moves, leakage=%.1f", n_moves,
                    lowrank_leakage(B, best_labels, len(splits)))

    block_to_name = _blocks_to_names(best_labels, splits, names)
    return {ids[i]: block_to_name.get(int(best_labels[i]), names[-1]) for i in range(n)}
