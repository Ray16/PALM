"""Prepare a LARGE USPTO 3-reactant multicomponent-reaction (MCR) master set.

Same construction as ``prepare_uspto_mcr.py`` (exactly three reactants, drop
mega-reagents appearing in > MAX_FREQ reactions, validity-filter, canonicalize
the unordered triple), but (i) reads the already-downloaded full USPTO directly
from the TDC cache CSV (no ``tdc`` import needed at runtime) and (ii) keeps the
whole filtered pool instead of subsampling to 6k. The full USPTO has ~1.94M
reactions, ~405k with exactly three reactants, ~75k surviving the MAX_FREQ=200
reagent filter -- a ~17x larger, near-unique-but-congeneric set used for the
n-D scaling study (``benchmark_mcr_scaling.py``).

Writes 3D+/uspto_mcr/records_large.csv (rA, rB, rC, product). The small
``records.csv`` (n=4,300) is left untouched.

    python -m PALM.benchmark.prepare_uspto_mcr_large            # full ~75k pool
    N_SAMPLE=120000 MAX_FREQ=1000 python -m ...prepare_uspto_mcr_large
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "DataSAIL_data", "3D+")
OUT = os.path.join(_BASE, "uspto_mcr")
CSV = os.path.join(_BASE, "tdc_cache", "uspto.csv")   # full USPTO from TDC RetroSyn cache
MAX_FREQ = int(os.environ.get("MAX_FREQ", 200))
N_SAMPLE = int(os.environ.get("N_SAMPLE", 0))         # 0 -> keep the whole filtered pool
SEED = 7


def _ok(smiles):
    m = Chem.MolFromSmiles(smiles)
    return m is not None and 5 <= m.GetNumHeavyAtoms() <= 60


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV)                              # columns: reactant, product
    reac = df["reactant"].astype(str)
    three = df[reac.str.split(".").apply(len) == 3].reset_index(drop=True)
    r = three["reactant"].astype(str).str.split(".", expand=True)

    freq = pd.concat([r[0], r[1], r[2]]).value_counts()
    keep = r.apply(lambda row: all(freq[row[c]] <= MAX_FREQ for c in range(3)), axis=1)
    sub = three[keep].reset_index(drop=True)
    rk = r[keep].reset_index(drop=True)
    print(f"3-reactant={len(three)}  after MAX_FREQ={MAX_FREQ}: {len(sub)}")

    if N_SAMPLE and N_SAMPLE < len(sub):
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(sub), size=N_SAMPLE, replace=False)
        sub, rk = sub.iloc[idx].reset_index(drop=True), rk.iloc[idx].reset_index(drop=True)

    rows = []
    for i in range(len(sub)):
        mols = [rk[c].iloc[i] for c in range(3)]
        if not all(_ok(m) for m in mols):
            continue
        a, b, c = sorted(mols)                         # canonicalize the unordered triple
        rows.append((a, b, c, sub["product"].iloc[i]))
    rec = pd.DataFrame(rows, columns=["rA", "rB", "rC", "product"]).drop_duplicates()
    path = os.path.join(OUT, "records_large.csv")
    rec.to_csv(path, index=False)

    vc = pd.concat([rec.rA, rec.rB, rec.rC]).value_counts()
    print(f"wrote {path}  n={len(rec)}")
    print(f"unique reactants per axis: rA={rec.rA.nunique()} rB={rec.rB.nunique()} rC={rec.rC.nunique()}")
    print(f"records per reactant: median={int(vc.median())} mean={vc.mean():.2f} max={vc.max()} "
          f"frac_appearing_once={(vc == 1).mean():.3f}")


if __name__ == "__main__":
    main()
