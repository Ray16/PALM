"""Self-contained featurizers for the dataset loaders.

- :func:`ecfp_matrix` — SMILES -> ECFP-1024 bit vectors (Tanimoto space) for
  organic small molecules / polymers.
- :func:`composition_matrix` — chemical formula -> MAGPIE elemental descriptors
  (via PALM's material featurizer) for inorganic crystals / MOFs.

Kept independent of the benchmark tree so the dataset layer has no moving deps.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def ecfp_matrix(smiles: Sequence[str], n_bits: int = 1024, radius: int = 2
                ) -> Tuple[List[int], np.ndarray]:
    """SMILES -> ``(kept_indices, X)`` with X an (m, n_bits) 0/1 float32 matrix.

    Unparseable SMILES are dropped; ``kept_indices`` are the positions in the
    input that were kept (so callers can align ids/labels).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    kept, rows = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        a = np.zeros(n_bits, dtype=np.float32)
        a[list(fp.GetOnBits())] = 1.0        # avoids ConvertToNumpyArray (numpy-2 ABI)
        kept.append(i)
        rows.append(a)
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, n_bits), np.float32)
    return kept, X


def composition_matrix(formulas: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    """``{id: formula}`` -> ``(ids, X)`` MAGPIE composition descriptors (dense).

    Uses PALM's material featurizer. ids/rows with an unparseable formula are
    dropped.
    """
    from PALM.features.material_features import compute_material_features

    feat = compute_material_features(
        formulas, ["magpie_properties", "electronic", "bonding", "stoichiometry"])
    ids, rows = [], []
    for k in formulas:
        if k in feat.index:
            v = np.asarray(feat.loc[k].values, dtype=np.float32)
            if np.all(np.isfinite(v)):
                ids.append(k)
                rows.append(v)
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 1), np.float32)
    return ids, X
