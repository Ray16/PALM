"""Trustworthiness audit for the low-rank L(pi) result.

Checks three ways the "low-rank beats hypergraph" result could be an artifact:

  (1) BALANCE CONFOUND. L(pi) = cross / total and cross ~ |train|*|test|, so a
      smaller test set lowers L(pi) for free. We report the actual test fraction
      of every method; a fair comparison needs them equal.

  (2) CIRCULARITY. Low-rank minimizes ECFP-Tanimoto cross similarity, which is
      exactly what scaled_lpi scores. We therefore RE-SCORE every split with an
      INDEPENDENT fingerprint (MACCS keys, 167 structural bits) that no method
      optimized. If low-rank still has the lowest leakage on MACCS, the win is
      real; if it collapses, it was overfitting the ECFP metric.

  (3) LIVE DataSAIL. DataSAIL is run live (not its published number) so it is
      scored on the exact same molecules and both fingerprints.

All methods use epsilon=0 (exact 80/20) so balance cannot confound. Hypergraph
is averaged over 3 seeds (it is nondeterministic); low-rank is deterministic.

Run (palm env):  python -m PALM.lowrank_split.validate_lowrank
"""

import sys
import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import logging
logging.disable(logging.CRITICAL)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
RDLogger.DisableLog("rdApp.*")

from PALM.hypergraph import run_hypergraph_split
from PALM.lowrank_split.lowrank_split import run_lowrank_split
from PALM.benchmark.benchmark_moleculenet1d import load_smiles

DATASETS = ["esol", "sider", "bace", "tox21"]
DATASAIL_MAX_SEC = 400
EPS = 0.0            # force exact 80/20 -> no balance confound


def features(smiles):
    """Return (ECFP-1024, MACCS-167) binary matrices; MACCS is the independent metric."""
    ecfp = np.zeros((len(smiles), 1024), dtype=np.float32)
    maccs = np.zeros((len(smiles), 167), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), ecfp[i])
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), maccs[i])
    return ecfp, maccs


def tanimoto_lpi(Xbin, labels, block=4096):
    """scaled L(pi) on arbitrary binary features (same formula as leakage.scaled_lpi)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(Xbin, dtype=torch.float32, device=dev)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev)
    card = X.sum(1)
    n = X.shape[0]
    total = torch.zeros((), device=dev)
    leak = torch.zeros((), device=dev)
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = X[s:e] @ X.T
        union = card[s:e][:, None] + card[None, :] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        total += sim.sum()
        cross = (lab[s:e][:, None] != lab[None, :]).float()
        leak += (sim * cross).sum()
    return float((leak / total).item())


def datasail_split(smiles):
    from datasail.sail import datasail
    try:
        e_s, _, _ = datasail(techniques=["C1e"], splits=[8, 2], names=["train", "test"],
                             e_type="M", e_data={s: s for s in smiles}, max_sec=DATASAIL_MAX_SEC)
        return e_s["C1e"][0]
    except Exception as exc:
        print(f"    DataSAIL failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def main():
    print(f"{'dataset':<8}{'method':<12}{'test_frac':>10}{'lpi_ECFP':>10}{'lpi_MACCS':>11}")
    print("-" * 51)
    for ds in DATASETS:
        smiles = load_smiles(ds)
        n = len(smiles)
        ecfp, maccs = features(smiles)
        fd = {smiles[i]: ecfp[i] for i in range(n)}

        def as_labels(split):
            return np.array([0 if split[smiles[i]] == "train" else 1 for i in range(n)])

        # hypergraph: average metrics over 3 seeds (nondeterministic)
        hg_frac, hg_e, hg_m = [], [], []
        for seed in range(3):
            lab = as_labels(run_hypergraph_split(fd, [8, 2], ["train", "test"],
                                                 k=15, preset="quality", epsilon=EPS, seed=seed))
            hg_frac.append(lab.mean()); hg_e.append(tanimoto_lpi(ecfp, lab)); hg_m.append(tanimoto_lpi(maccs, lab))
        # low-rank (deterministic)
        lr = as_labels(run_lowrank_split(fd, [8, 2], ["train", "test"], rank=256, epsilon=EPS, seed=0))
        # DataSAIL live
        ds_split = datasail_split(smiles)

        rows = [("hypergraph", np.mean(hg_frac), np.mean(hg_e), np.mean(hg_m)),
                ("lowrank", lr.mean(), tanimoto_lpi(ecfp, lr), tanimoto_lpi(maccs, lr))]
        if ds_split is not None:
            dl = as_labels({s: ds_split.get(s, "train") for s in smiles})
            rows.append(("DataSAIL", dl.mean(), tanimoto_lpi(ecfp, dl), tanimoto_lpi(maccs, dl)))
        for name, frac, le, lm in rows:
            print(f"{ds:<8}{name:<12}{frac:>10.3f}{le:>10.4f}{lm:>11.4f}", flush=True)
        print()


if __name__ == "__main__":
    main()
