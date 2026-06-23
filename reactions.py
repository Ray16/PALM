"""Reaction (n-D / multi-component) dataset loading + per-axis featurization.

Each reaction is a record with several component axes (reactant, ligand, base,
solvent, ...). Different axes need different featurizers:
  - molecule  : SMILES -> Morgan fingerprint (Tanimoto similarity)
  - composition: formula -> MAGPIE elemental descriptors (PALM material featurizer)
                 — the right choice for ionic/inorganic bases (and works for
                 organic ones too, since they have formulas).
  - solvent   : curated physicochemical descriptor vector
  - drop      : constant axis (no information) — removed

The output ({records}, {axis_feature_maps}) plugs directly into
``hypergraph.run_hypergraph_split_nd``.
"""

import logging
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

_DATA = os.path.join(os.path.dirname(__file__), "data", "DataSAIL_data", "3D+")


# ── per-axis featurizers (featurize UNIQUE values once) ────────────────────

def _morgan(smiles, n_bits=2048, radius=2):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    a = np.zeros(n_bits, dtype=np.int8)
    a[list(fp.GetOnBits())] = 1          # avoids ConvertToNumpyArray (numpy-2.x ABI issue)
    return a


def morgan_map(values):
    """{value(SMILES): morgan vector}; None for unparseable."""
    out, bad = {}, []
    for v in values:
        fp = _morgan(v)
        out[v] = fp
        if fp is None:
            bad.append(v)
    if bad:
        logger.warning("  %d/%d SMILES unparseable on axis (identity fallback): %s",
                       len(bad), len(values), bad[:5])
    return out


def composition_map(values, formula_of=None):
    """{value: MAGPIE composition vector} via PALM's material featurizer.

    `formula_of` maps a value to its chemical formula (defaults to the value
    itself, i.e. it is already a formula like 'K3PO4'). Works for ionic salts
    and organics alike.
    """
    from .features.material_features import compute_material_features
    formula_of = formula_of or {v: str(v) for v in values}
    feat = compute_material_features({v: formula_of[v] for v in values},
                                     ["magpie_properties", "electronic", "bonding", "stoichiometry"])
    return {v: feat.loc[v].values.astype(float) for v in values}


# Minimal curated solvent descriptors: [dielectric eps, dipole (D), logP, H-bond
# donor alpha, H-bond acceptor beta]. Extend as needed.
_SOLVENT_DESCRIPTORS = {
    "MeCN":  [37.5, 3.92, -0.34, 0.19, 0.40],
    "THF":   [7.58, 1.75,  0.46, 0.00, 0.55],
    "DMF":   [36.7, 3.82, -1.01, 0.00, 0.69],
    "MeOH":  [32.7, 1.70, -0.77, 0.98, 0.66],
    "H2O":   [80.1, 1.85, -1.38, 1.17, 0.47],
    "DMAc":  [37.8, 3.72, -0.77, 0.00, 0.78],
    "DMSO":  [46.7, 3.96, -1.35, 0.00, 0.76],
    "EtOH":  [24.5, 1.69, -0.31, 0.86, 0.75],
}


def solvent_map(values):
    """{value: solvent descriptor vector}. Mixtures averaged; replicate tags normalized."""
    out = {}
    for v in values:
        base = _normalize_solvent(v)
        if "/" in base:                      # mixture, e.g. MeOH/H2O 9:1
            parts = [p.strip() for p in base.split("/")]
            vecs = [_SOLVENT_DESCRIPTORS[p] for p in parts if p in _SOLVENT_DESCRIPTORS]
            out[v] = np.mean(vecs, axis=0) if vecs else None
        else:
            d = _SOLVENT_DESCRIPTORS.get(base)
            out[v] = np.asarray(d, float) if d else None
    return out


def _normalize_solvent(tag):
    """Strip replicate/version annotations: 'THF_V2' -> 'THF', 'MeOH/H2O_V2 9:1' -> 'MeOH/H2O'."""
    t = str(tag).split("_V")[0].strip()
    t = t.split()[0] if " " in t and "/" in t else t   # drop ratio suffix like '9:1'
    return t


# ── dataset loaders -> (records, axis_feature_maps, target) ────────────────

def load_buchwald_hartwig(sheet="FullCV_01"):
    """All 4 axes are organic SMILES already in the file -> Morgan FP each."""
    path = os.path.join(_DATA, "Buchwald–Hartwig dataset.xlsx")
    df = pd.read_excel(path, sheet_name=sheet)
    axes = ["Ligand", "Additive", "Base", "Aryl halide"]
    records = [{a: str(row[a]) for a in axes} for _, row in df.iterrows()]
    axis_feature_maps = {a: morgan_map(sorted({r[a] for r in records})) for a in axes}
    return records, axis_feature_maps, df["Output"].values


# Suzuki structural SMILES (curated). Entries marked TODO are unverified and
# fall back to identity (no similarity) until confirmed.
_SUZUKI_REACTANT1_SMILES = {
    "6-chloroquinoline": "Clc1ccc2ncccc2c1",
    "6-Bromoquinoline": "Brc1ccc2ncccc2c1",
    "6-Iodoquinoline": "Ic1ccc2ncccc2c1",
    "6-triflatequinoline": "O=S(=O)(Oc1ccc2ncccc2c1)C(F)(F)F",
    "6-quinoline-boronic acid hydrochloride": "OB(O)c1ccc2ncccc2c1",
    "Potassium quinoline-6-trifluoroborate": "[K+].[B-](F)(F)(F)c1ccc2ncccc2c1",
    "6-Quinolineboronic acid pinacol ester": "CC1(C)OB(OC1(C)C)c1ccc2ncccc2c1",
}
_SUZUKI_LIGAND_SMILES = {
    "P(tBu)3": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "P(Ph)3": "c1ccc(cc1)P(c2ccccc2)c3ccccc3",
    "P(Cy)3": "C1CCCCC1P(C2CCCCC2)C3CCCCC3",
    "P(o-Tol)3": "Cc1ccccc1P(c2ccccc2C)c3ccccc3C",
    # TODO verify: AmPhos, CataCXium A, SPhos, dtbpf, Xantphos, None
}
# Suzuki bases -> formulas for MAGPIE (ionic + organic handled uniformly)
_SUZUKI_BASE_FORMULA = {
    "NaOH": "NaOH", "NaHCO3": "NaHCO3", "CsF": "CsF", "K3PO4": "K3PO4",
    "KOH": "KOH", "LiOtBu": "C4H9LiO", "Et3N": "C6H15N", "Et3N ": "C6H15N",
}


def load_suzuki_miyaura(sheet="Sheet1"):
    """Heterogeneous axes: structural (SMILES), base (MAGPIE), solvent (descriptors).

    Catalyst is constant (Pd(OAc)2) -> dropped. Structural values without a
    verified SMILES fall back to identity hyperedges automatically.
    """
    path = os.path.join(_DATA, "Suzuki–Miyaura dataset.xlsx")
    df = pd.read_excel(path, sheet_name=sheet)
    colmap = {
        "reactant1": "Reactant_1_Name", "reactant2": "Reactant_2_Name",
        "ligand": "Ligand_Short_Hand", "base": "Reagent_1_Short_Hand",
        "solvent": "Solvent_1_Short_Hand",
    }  # Catalyst_1_Short_Hand dropped (constant)
    records = [{k: str(row[c]).strip() for k, c in colmap.items()} for _, row in df.iterrows()]

    r1 = sorted({r["reactant1"] for r in records})
    lig = sorted({r["ligand"] for r in records})
    base = sorted({r["base"] for r in records})
    solv = sorted({r["solvent"] for r in records})
    r2 = sorted({r["reactant2"] for r in records})

    axis_feature_maps = {
        "reactant1": {v: _morgan(_SUZUKI_REACTANT1_SMILES[v]) if v in _SUZUKI_REACTANT1_SMILES else None for v in r1},
        "ligand":    {v: _morgan(_SUZUKI_LIGAND_SMILES[v]) if v in _SUZUKI_LIGAND_SMILES else None for v in lig},
        "reactant2": {v: None for v in r2},          # identity (4 activation forms; structures TODO)
        "base":      composition_map(base, formula_of={v: _SUZUKI_BASE_FORMULA.get(v, v) for v in base}),
        "solvent":   solvent_map(solv),
    }
    yield_col = "Product_Yield_PCT_Area_UV"
    return records, axis_feature_maps, df[yield_col].values


def load_uspto_mcr():
    """USPTO 3-reactant reactions: a genuine 3D, near-unique, structured case.

    Each record is rA + rB + rC -> product, three *independent* reactant axes
    (only the product is derived, and it is not used as a split axis). After
    dropping common reagents (global freq > 5) and tiny fragments, ~96% of
    reactant molecules are unique -- so identity grouping forms no useful groups
    (identity split ~ random) and only SIMILARITY-aware grouping (scaffold
    clustering, sim_threshold < 1) can reduce scaled L(pi). This is the n-D
    analogue of the 1D molecular scaffold split, and the regime where the
    hypergraph beats both random and a trivial identity group-split.

    Files in 3D+/uspto_mcr/records.csv; needs no TDC at runtime.
    """
    base = os.path.join(_DATA, "uspto_mcr")
    df = pd.read_csv(os.path.join(base, "records.csv"))
    axes = ["rA", "rB", "rC"]
    records = [{a: str(getattr(r, a)) for a in axes} for r in df.itertuples()]
    axis_feature_maps = {a: morgan_map(sorted({rec[a] for rec in records})) for a in axes}
    return records, axis_feature_maps, df["product"].values
