"""GPU, chunked macro scaled L(pi) for multi-component (n-D) records.

The CPU scorer in ``benchmark_reactions.axis_scaled_lpi`` materializes the full
per-axis n x n similarity and is O(n^2) on numpy -- fine at a few thousand
records, far too slow past ~20-30k. This is a torch reimplementation, chunked
over query rows (peak memory O(block * n)), numerically identical to the CPU
version: per axis, similarity is the feature Tanimoto when both records carry a
component feature on that axis and the identity indicator otherwise; leakage is
the cross-split share of total similarity (diagonal removed). Used by the n-D
scaling study so every dataset size can be scored on the GPU.
"""

import numpy as np


def macro_axis_lpi_gpu(records, axis_feature_maps, labels, block=2048):
    """Macro-average scaled L(pi) over all component axes, on the GPU.

    Matches ``benchmark_reactions.evaluate``'s macro_lpi to <1e-3 but scales to
    1e5 records. Returns (macro_lpi: float, per_axis: {axis: lpi}).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = np.asarray(labels)
    _, lab = np.unique(labels, return_inverse=True)
    lab_t = torch.as_tensor(lab, device=device)
    n = len(records)

    per_axis = {}
    for axis in axis_feature_maps:
        vals = [str(r[axis]) for r in records]
        fmap = axis_feature_maps[axis]
        _, codes = np.unique(np.asarray(vals), return_inverse=True)
        codes_t = torch.as_tensor(codes, device=device)

        # per-record feature matrix (0/1 Morgan); has[i] flags a usable feature
        dim = 0
        for v in vals:
            f = fmap.get(v)
            if f is not None and np.any(f):
                dim = int(np.asarray(f).ravel().shape[0]); break
        has = np.zeros(n, dtype=bool)
        if dim:
            F = np.zeros((n, dim), dtype=np.float32)
            for i, v in enumerate(vals):
                f = fmap.get(v)
                if f is not None and np.any(f):
                    F[i] = np.asarray(f, np.float32).ravel(); has[i] = True
            X = torch.as_tensor(F, device=device)
            card = X.sum(1)
        has_t = torch.as_tensor(has, device=device)

        total = torch.zeros((), device=device)
        leak = torch.zeros((), device=device)
        for s in range(0, n, block):
            e = min(s + block, n)
            if dim:
                inter = X[s:e] @ X.T
                union = card[s:e][:, None] + card[None, :] - inter
                fsim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
                both = has_t[s:e][:, None] & has_t[None, :]
                idsim = (codes_t[s:e][:, None] == codes_t[None, :]).float()
                sim = torch.where(both, fsim, idsim)
            else:
                sim = (codes_t[s:e][:, None] == codes_t[None, :]).float()
            cross = (lab_t[s:e][:, None] != lab_t[None, :]).float()
            total += sim.sum()
            leak += (sim * cross).sum()
        total = total - n                      # drop diagonal (self-similarity == 1)
        per_axis[axis] = float((leak / total).item()) if float(total) > 0 else 0.0

    macro = float(np.mean(list(per_axis.values())))
    return macro, per_axis
