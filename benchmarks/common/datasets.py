"""MoleculeNet 1-D dataset loader — the one copy.

Extracted from ``benchmark_moleculenet1d.py``. The ``DATA`` dir and the
``SMILES_COL`` overrides live here so every benchmark reads the same molecules.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
# benchmarks/common/ -> PALM/ -> data/DataSAIL_data/1D/moleculenet
DATA = os.path.join(HERE, "..", "..", "data", "DataSAIL_data", "1D", "moleculenet")

# datasets whose SMILES column is not literally "smiles"
SMILES_COL = {"bace": "mol"}


def load_smiles(ds):
    """De-duplicated, non-null SMILES strings for MoleculeNet dataset ``ds``."""
    import pandas as pd

    col = SMILES_COL.get(ds, "smiles")
    df = pd.read_csv(os.path.join(DATA, f"{ds}.csv"))
    df = df.dropna(subset=[col]).drop_duplicates(subset=col).reset_index(drop=True)
    return [s for s in df[col].astype(str) if s and s != "nan"]
