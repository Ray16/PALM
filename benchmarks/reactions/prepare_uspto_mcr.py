"""Prepare the USPTO 3-reactant multicomponent-reaction (MCR) dataset.

Downloads USPTO via TDC (tdc.generation.RetroSyn 'uspto', ~1.1M reactions),
keeps reactions with exactly three reactants (A + B + C -> product) -- three
genuinely independent reactant axes (only the product is derived) -- drops
common reagents/solvents (a reactant appearing in > MAX_FREQ reactions) and
tiny/oversized fragments, subsamples, canonicalizes by sorting the three
reactant SMILES, and writes 3D+/uspto_mcr/records.csv (rA, rB, rC, product).

The point is a HIGH-cardinality, near-unique-but-congeneric multi-axis set:
~91% of reactant molecules appear once, yet analog families remain (mega-reagents
removed, not the families), so a similarity-aware split has structure to exploit
while identity grouping does not. Run in the boltz-2 env from the PALM parent:

    python -m PALM.benchmarks.reactions.prepare_uspto_mcr
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "DataSAIL_data", "3D+", "uspto_mcr")
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "DataSAIL_data", "3D+", "tdc_cache")
MAX_FREQ = 200      # drop reactants appearing in more than this many reactions (solvents/bases)
N_SAMPLE = 6000
SEED = 7


def _ok(smiles):
    m = Chem.MolFromSmiles(smiles)
    return m is not None and 5 <= m.GetNumHeavyAtoms() <= 60


def main():
    os.makedirs(OUT, exist_ok=True)
    from tdc.generation import RetroSyn
    df = RetroSyn(name="uspto", path=CACHE).get_data(format="df")
    reac = df["output"].astype(str)
    three = df[reac.str.split(".").apply(len) == 3].reset_index(drop=True)
    r = three["output"].astype(str).str.split(".", expand=True)

    freq = pd.concat([r[0], r[1], r[2]]).value_counts()
    keep = r.apply(lambda row: all(freq[row[c]] <= MAX_FREQ for c in range(3)), axis=1)
    sub = three[keep].reset_index(drop=True)
    rk = r[keep].reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(sub), size=min(N_SAMPLE, len(sub)), replace=False)
    sub, rk = sub.iloc[idx].reset_index(drop=True), rk.iloc[idx].reset_index(drop=True)

    rows = []
    for i in range(len(sub)):
        mols = [rk[c].iloc[i] for c in range(3)]
        if not all(_ok(m) for m in mols):
            continue
        a, b, c = sorted(mols)                      # canonicalize the unordered triple
        rows.append((a, b, c, sub["input"].iloc[i]))
    rec = pd.DataFrame(rows, columns=["rA", "rB", "rC", "product"]).drop_duplicates()
    rec.to_csv(os.path.join(OUT, "records.csv"), index=False)

    vc = pd.concat([rec.rA, rec.rB, rec.rC]).value_counts()
    print(f"wrote {OUT}/records.csv  n={len(rec)}")
    print(f"unique reactants per axis: rA={rec.rA.nunique()} rB={rec.rB.nunique()} rC={rec.rC.nunique()}")
    print(f"records per reactant: median={int(vc.median())} mean={vc.mean():.2f} max={vc.max()} "
          f"frac_appearing_once={(vc == 1).mean():.3f}")


if __name__ == "__main__":
    main()
