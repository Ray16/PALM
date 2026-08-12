"""Low-rank factorized leakage-minimizing splitter (1-D), graph-free.

Factorizes the similarity matrix ``S ~= B B^T`` (Nyström; Tanimoto/cosine are
valid PD kernels), so the cross-split leakage decomposes exactly in the r-dim
factor space and is evaluated in O(n·r) without ever materializing S. Minimizing
it is a balanced k-means / max-diversity partition in B-space, optimized with
balanced-Lloyd restarts + a monotone FM polish. Scales to millions of rows.

Public helpers ``nystrom_features``, ``balanced_lloyd``, ``fm_polish`` and
``lowrank_leakage`` are re-exported for the benchmark/omol25 studies that drive
the pieces directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..base import BaseSplitter, SplitResult, SplitSpec, register
from ..common.balanced_assignment import balanced_assign, target_sizes
from ..common.feature_preparation import choose_metric, feature_matrix_from_dict
from ..common.fiduccia_mattheyses import fiduccia_mattheyses_lowrank as fm_polish
from ..common.leakage_metrics import factor_leakage as lowrank_leakage
from ..common.leakage_metrics import scaled_lpi
from ..common.split_naming import assign_split_names
from ..common.pairwise_similarity import pairwise_similarity

logger = logging.getLogger(__name__)

_LEAKAGE_MAX_N = 100_000


# ── Nyström low-rank factorization  S ~= B B^T ─────────────────────────────

def _kmeanspp_landmarks(Xt, n_landmarks, metric, seed):
    """Pick landmark rows by k-means++ (D^2-weighted) sampling on the GPU."""
    import torch

    n = Xt.shape[0]
    n_landmarks = min(n_landmarks, n)
    gen = torch.Generator(device=Xt.device).manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=gen, device=Xt.device).item())
    chosen = [first]
    sim_to_set = pairwise_similarity(Xt, Xt[first][None, :], metric).squeeze(1)
    min_dist = 1.0 - sim_to_set
    for _ in range(n_landmarks - 1):
        weights = torch.clamp(min_dist, min=0.0) ** 2
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        nxt = int(torch.multinomial(weights, 1, generator=gen).item())
        chosen.append(nxt)
        sim_new = pairwise_similarity(Xt, Xt[nxt][None, :], metric).squeeze(1)
        min_dist = torch.minimum(min_dist, 1.0 - sim_new)
    return np.asarray(chosen, dtype=np.int64)


def nystrom_features(X, rank=256, metric=None, landmark="kmeans++", seed=0):
    """Nyström low-rank factor ``B`` with ``B @ B.T ~= S``. Returns ``(B, metric)``."""
    import torch

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
    L = Xt[torch.as_tensor(idx, device=device)]

    C = pairwise_similarity(Xt, L, metric)                  # (n, r) point-landmark
    W = pairwise_similarity(L, L, metric)                   # (r, r) landmark-landmark
    evals, evecs = torch.linalg.eigh(W)
    evals = torch.clamp(evals, min=1e-6)
    W_inv_sqrt = (evecs * (1.0 / torch.sqrt(evals))[None, :]) @ evecs.T
    B = C @ W_inv_sqrt
    return B.cpu().numpy().astype(np.float32), metric


# ── balanced optimization in factor space ──────────────────────────────────

def balanced_lloyd(B, splits, epsilon=0.05, n_iter=25, init_labels=None, seed=0):
    """Balanced-Lloyd minimization of the low-rank leakage in B-space.

    Alternates: block sums from current labels, then reassign every point to the
    most-similar block subject to the exact target ratio. O(n·r·k) per iteration.
    Returns the final labels (numpy int array).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    n, r = Bt.shape
    k = len(splits)
    sizes = target_sizes(n, splits)
    self_sim = (Bt * Bt).sum(1)

    if init_labels is None:
        gen = np.random.default_rng(seed)
        order = np.concatenate([np.full(int(sz), c) for c, sz in enumerate(sizes)])
        gen.shuffle(order)
        lab = torch.as_tensor(order, dtype=torch.long, device=device)
    else:
        lab = torch.as_tensor(np.asarray(init_labels), dtype=torch.long, device=device)

    for _ in range(n_iter):
        P = torch.zeros(k, r, device=device, dtype=Bt.dtype)
        P.index_add_(0, lab, Bt)
        scores = Bt @ P.T
        scores[torch.arange(n, device=device), lab] -= self_sim   # exclude self
        new_lab = balanced_assign(scores, sizes)
        if torch.equal(new_lab, lab):
            break
        lab = new_lab
    return lab.cpu().numpy()


# ── the splitter ────────────────────────────────────────────────────────────

@register("lowrank")
class LowRankSplitter(BaseSplitter):
    description = "Nyström low-rank factorization + balanced-Lloyd + FM (graph-free, O(n·r))"
    arity = "1d"

    @dataclass
    class Params:
        rank: int = 256
        metric: Optional[str] = None
        landmark: str = "kmeans++"
        n_restarts: int = 4
        n_iter: int = 25
        fm: bool = True
        fm_max_n: int = 200_000

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        B, metric = nystrom_features(X, rank=p.rank, metric=p.metric,
                                     landmark=p.landmark, seed=spec.seed)
        logger.info("  Low-rank: n=%d rank=%d metric=%s", n, B.shape[1], metric)

        best_labels, best_obj = None, np.inf
        for r in range(p.n_restarts):
            labels = balanced_lloyd(B, spec.splits, epsilon=spec.epsilon,
                                    n_iter=p.n_iter, seed=spec.seed + r)
            obj = lowrank_leakage(B, labels, len(spec.splits))
            if obj < best_obj:
                best_obj, best_labels = obj, labels
        logger.info("  Best-of-%d Lloyd leakage=%.1f", p.n_restarts, best_obj)

        moves = 0
        if p.fm and n <= p.fm_max_n:
            best_labels, moves = fm_polish(B, best_labels, spec.splits, epsilon=spec.epsilon)
            best_obj = lowrank_leakage(B, best_labels, len(spec.splits))
            logger.info("  FM polish: %d moves, leakage=%.1f", moves, best_obj)

        assignment = assign_split_names(ids, best_labels, spec.splits, spec.names)
        leak = round(scaled_lpi(X, best_labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            rank=int(B.shape[1]), factor_leakage=round(float(best_obj), 3),
                            fm_moves=int(moves), leakage=leak)
