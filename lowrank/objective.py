"""Low-rank split objectives (factor space).

``factor_leakage`` is the quantity the low-rank optimizer minimizes: the exact
cross-split similarity, computed in the r-dim Nyström factor space in O(n·r)
without ever forming the full similarity matrix. This module is the home for the
*objective* — where the multi-objective / controllable-hardness extensions land.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def factor_leakage(B: np.ndarray, labels: Sequence[int], n_blocks: int) -> float:
    """Cross-block leakage in Nyström factor space: ``0.5(||s||^2 - sum_c ||p_c||^2)``.

    Equals ``sum_{c<c'} p_c . p_c'`` ~= the exact cross-split similarity (self
    pairs excluded), where ``p_c`` is the sum of B-rows in block ``c``.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    P = torch.zeros(n_blocks, Bt.shape[1], device=device, dtype=Bt.dtype)
    P.index_add_(0, lab, Bt)
    s = P.sum(0)
    return 0.5 * float((s @ s) - (P * P).sum())


def realized_imbalance(labels: Sequence[int], splits: Sequence[float]) -> float:
    """Max relative deviation of realized block fractions from the target ratio.

    ``0.0`` = exactly balanced; ``0.1`` = some block is 10% off its target share.
    The balance axis of the leakage↔balance tradeoff.
    """
    labels = np.asarray(labels)
    n = len(labels)
    total = float(sum(splits))
    tgt = np.array([s / total for s in splits])
    real = np.array([(labels == c).sum() / n for c in range(len(splits))])
    return float(np.max(np.abs(real - tgt) / tgt))
