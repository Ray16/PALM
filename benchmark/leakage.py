"""GPU scaled-L(pi) leakage, matching DataSAIL's eval_split exactly.

DataSAIL's molecular L(pi) (uniform weights) is:
    L = ( sum of ECFP-Tanimoto over train x test pairs ) / ( sum over ALL pairs )
with ECFP = Morgan radius 2, 1024 bits, whole molecule (datasail/cluster/ecfp.py).

eval_split builds the full n x n similarity on CPU, so it OOMs/timeouts past
~20-40k and crashes on datasets with unparseable SMILES. This reimplementation
computes the same quantity on GPU in chunks, so it scales to ~100k+ and simply
drops unparseable SMILES (as DataSAIL intends to).
"""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def _ecfp1024(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    a = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), a)
    return a


def scaled_lpi(smiles, split, block=4096):
    """Scaled L(pi) for a split assignment (dict {smiles: split_name}).

    Returns leakage ratio in [0,1] (lower = less leakage), matching eval_split.
    Unparseable SMILES are dropped.
    """
    import torch

    fps, labels = [], []
    lab_ids, next_id = {}, 0
    for s in smiles:
        fp = _ecfp1024(s)
        if fp is None:
            continue
        sp = split[s]
        if sp not in lab_ids:
            lab_ids[sp] = next_id; next_id += 1
        fps.append(fp); labels.append(lab_ids[sp])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(np.asarray(fps), dtype=torch.float32, device=device)
    lab = torch.as_tensor(labels, dtype=torch.long, device=device)
    card = X.sum(1)
    n = X.shape[0]

    total = torch.zeros((), device=device)
    leak = torch.zeros((), device=device)
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = X[s:e] @ X.T                       # (b, n)
        union = card[s:e][:, None] + card[None, :] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        total += sim.sum()
        cross = (lab[s:e][:, None] != lab[None, :]).float()
        leak += (sim * cross).sum()
    return float((leak / total).item()), n


def validate_against_eval_split(smiles, split, tol=1e-3):
    """Sanity-check that scaled_lpi matches DataSAIL's eval_split on this split.

    Both quantities are the scaled L(pi) over ECFP/Tanimoto pairs; scaled_lpi is
    just the GPU, chunked reimplementation. Returns (ours, theirs, abs_diff,
    ok). Only feasible on small datasets, since eval_split builds the full n x n
    matrix on CPU. Raises if DataSAIL is unavailable.
    """
    from datasail.eval import eval_split

    data = {s: s for s in smiles}
    theirs, _, _ = eval_split("M", data, None, "ecfp", None, None, split)
    ours, _ = scaled_lpi(list(smiles), split)
    diff = abs(ours - theirs)
    return ours, theirs, diff, diff <= tol
