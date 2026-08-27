"""Morgan / ECFP fingerprint matrices — the one copy.

Extracted from the ~10 duplicated ``ecfp1024`` / ``morgan_matrix`` helpers that
lived in the benchmark scripts. Unparseable SMILES yield an all-zero row.
"""

from __future__ import annotations

import numpy as np


def morgan_matrix(smiles, n_bits: int = 2048, radius: int = 2, dtype=np.int8):
    """Dense (len(smiles), n_bits) Morgan fingerprint matrix.

    Memory-light vs the DataFrame path. Unparseable SMILES -> zero row.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    X = np.zeros((len(smiles), n_bits), dtype=dtype)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def ecfp1024(smiles):
    """(len(smiles), 1024) float32 ECFP (Morgan r=2) matrix — the leakage-metric space."""
    return morgan_matrix(smiles, n_bits=1024, radius=2, dtype=np.float32)
