"""Download a handful of Therapeutics Data Commons (TDC) single-prediction
datasets into ``PALM/data/tdc/`` as tidy ``{name}.csv`` (columns: id, smiles, y).

TDC is the easiest source to configure — ``pip install PyTDC`` and it pulls from
TDC's own dataverse (no API key). Each dataset here is a small-molecule ADMET
regression/classification set, capped at ``LIMIT`` rows.

Run (dedicated env, or boltz-2 with the conda lib on LD_LIBRARY_PATH):
    python -m PALM.data.prepare_tdc
"""

import os

import pandas as pd

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "tdc", "_raw")
OUT = os.path.join(HERE, "tdc")
LIMIT = 10_000

# (module, class, dataset-name) — small, diverse ADMET small-molecule sets.
DATASETS = [
    ("single_pred", "ADME", "Lipophilicity_AstraZeneca"),
    ("single_pred", "ADME", "Solubility_AqSolDB"),
    ("single_pred", "ADME", "Caco2_Wang"),
    ("single_pred", "ADME", "BBB_Martins"),
    ("single_pred", "Tox", "LD50_Zhu"),
]


def prepare():
    os.makedirs(RAW, exist_ok=True)
    import importlib
    written = []
    for module, cls_name, name in DATASETS:
        try:
            mod = importlib.import_module(f"tdc.{module}")
            cls = getattr(mod, cls_name)
            data = cls(name=name, path=RAW)
            df = data.get_data()                       # columns: Drug_ID, Drug, Y
            df = df.rename(columns={"Drug_ID": "id", "Drug": "smiles", "Y": "y"})
            df = df[["id", "smiles", "y"]].dropna(subset=["smiles"])
            if len(df) > LIMIT:
                df = df.sample(LIMIT, random_state=0).reset_index(drop=True)
            path = os.path.join(OUT, f"{name}.csv")
            df.to_csv(path, index=False)
            written.append((name, len(df), path))
            print(f"[TDC] {name}: {len(df)} rows -> {path}")
        except BaseException as exc:
            # TDC raises SystemExit when its Harvard Dataverse host is under
            # maintenance — catch BaseException so one unreachable dataset does
            # not abort the rest (they resume when the host is back).
            print(f"[TDC] {name}: SKIPPED ({type(exc).__name__}: {str(exc)[:80]})")
    return written


if __name__ == "__main__":
    prepare()
