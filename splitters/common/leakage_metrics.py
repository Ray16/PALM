"""Canonical scaled ``L(pi)`` leakage scorers.

The scaled leakage of a split is the cross-split share of total pairwise
similarity (diagonal excluded):

    L(pi) = ( sum sim(i,j) over i,j in different splits ) / ( sum over all pairs )

Lower is better; 0 = no cross-split similarity. This matches DataSAIL's
``eval_split`` when scored over ECFP/Tanimoto.

- :func:`scaled_lpi` — generic, on a feature matrix + integer labels + metric.
  Used both for benchmark scoring and for ``SplitResult`` diagnostics.
- :func:`scaled_lpi_smiles` — the ECFP-1024/Tanimoto convenience (DataSAIL-exact).
- :func:`macro_axis_lpi` — macro-average over the axes of an n-D record set.
- :func:`factor_leakage` — the low-rank factor-space proxy (the optimizer's cheap
  objective; correlates ~0.995–1.0 with the ECFP metric at rank 128–256).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .pairwise_similarity import pairwise_similarity, standardize_for_metric


def scaled_lpi(X, labels: Sequence[int], metric: str = "tanimoto",
               block: int = 4096) -> float:
    """Scaled L(pi) over the pairwise ``metric`` similarity of feature rows ``X``.

    ``labels`` is any per-row split id (ints or strings); the diagonal is
    excluded. Returns the leakage ratio in [0, 1].
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    Xt = torch.nan_to_num(Xt)
    ref = standardize_for_metric(Xt, metric)
    _, lab = np.unique(np.asarray(labels), return_inverse=True)
    lab_t = torch.as_tensor(lab, device=device)
    n = Xt.shape[0]

    total = torch.zeros((), device=device)
    leak = torch.zeros((), device=device)
    diag = torch.zeros((), device=device)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = pairwise_similarity(ref[s:e], ref, metric)
        rg = torch.arange(s, e, device=device)
        diag += sim[torch.arange(e - s, device=device), rg].sum()
        total += sim.sum()
        cross = (lab_t[s:e][:, None] != lab_t[None, :]).float()
        leak += (sim * cross).sum()
    denom = total - diag                          # drop self-similarity
    return float((leak / denom).item()) if float(denom) > 0 else 0.0


def scaled_lpi_smiles(smiles: Sequence[str], split: Dict[str, str],
                      block: int = 4096) -> Tuple[float, int]:
    """DataSAIL-exact scaled L(pi) over whole-molecule ECFP-1024/Tanimoto.

    ``split`` maps SMILES -> split name. Unparseable SMILES are dropped. Returns
    ``(leakage_ratio, n_scored)``.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    def _ecfp1024(s):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        a = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), a)
        return a

    fps, labels = [], []
    for s in smiles:
        fp = _ecfp1024(s)
        if fp is None:
            continue
        fps.append(fp)
        labels.append(split[s])
    if not fps:
        return 0.0, 0
    return scaled_lpi(np.asarray(fps), labels, metric="tanimoto", block=block), len(fps)


def macro_axis_lpi(records, axis_feature_maps, labels, block: int = 2048
                   ) -> Tuple[float, Dict[str, float]]:
    """Macro-average scaled L(pi) over the component axes of an n-D record set.

    Per axis, similarity is the feature Tanimoto when both records carry a
    component feature on that axis and the identity indicator otherwise; leakage
    is the cross-split share of total similarity (diagonal removed). Returns
    ``(macro_lpi, {axis: lpi})``.
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
        total = total - n                          # drop diagonal (self-sim == 1)
        per_axis[axis] = float((leak / total).item()) if float(total) > 0 else 0.0

    macro = float(np.mean(list(per_axis.values()))) if per_axis else 0.0
    return macro, per_axis


def factor_leakage(B: np.ndarray, labels: Sequence[int], n_blocks: int) -> float:
    """Cross-block leakage in Nyström factor space: ``0.5(||s||^2 - sum_c ||p_c||^2)``.

    Equals ``sum_{c<c'} p_c . p_c'`` ~= the exact cross-split similarity (self
    pairs excluded). This is the quantity the low-rank optimizer minimizes.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Bt = torch.as_tensor(B, dtype=torch.float32, device=device)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    P = torch.zeros(n_blocks, Bt.shape[1], device=device, dtype=Bt.dtype)
    P.index_add_(0, lab, Bt)
    s = P.sum(0)
    return 0.5 * float((s @ s) - (P * P).sum())
