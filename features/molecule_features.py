"""SMILES-based molecule featurization (3 feature sets).

Generalized from OC22 adsorbate featurization — works with any SMILES string.
"""

import numpy as np
import pandas as pd

from .elemental_data import ELEM_PROPS, PROP_NAMES
from .utils import parse_formula


def rdkit_descriptors(smiles):
    """Compute RDKit molecular descriptors from a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: 0.0 for k in [
            "MolWt", "NumHBondDonors", "NumHBondAcceptors", "NumLonePairs",
            "TPSA", "MolLogP", "NumHeavyAtoms",
            "NumValenceElectrons", "NumRadicalElectrons",
        ]}

    mol_h = Chem.AddHs(mol)

    num_hbond_donors = 0
    num_hbond_acceptors = 0
    num_lone_pairs = 0
    for atom in mol_h.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum in (7, 8):  # N, O
            h_neighbors = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
            if h_neighbors > 0:
                num_hbond_donors += h_neighbors
            valence = 5 if anum == 7 else 6
            bond_order = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
            radical = atom.GetNumRadicalElectrons()
            lone_pair_electrons = valence - bond_order - radical
            lp = max(0, int(lone_pair_electrons)) // 2
            num_lone_pairs += lp
            if lp > 0:
                num_hbond_acceptors += 1

    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumHBondDonors": num_hbond_donors,
        "NumHBondAcceptors": num_hbond_acceptors,
        "NumLonePairs": num_lone_pairs,
        "TPSA": Descriptors.TPSA(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
    }


def composition(formula):
    """Composition features from a molecular formula: element counts + weighted property means."""
    comp = parse_formula(formula)
    total = sum(comp.values())

    feats = {}
    for elem, cnt in sorted(comp.items()):
        feats[f"count_{elem}"] = cnt

    for pidx, pname in enumerate(PROP_NAMES):
        weighted_sum = 0.0
        for elem, cnt in comp.items():
            if elem in ELEM_PROPS:
                weighted_sum += cnt * ELEM_PROPS[elem][pidx]
        feats[f"wtd_mean_{pname}"] = weighted_sum / total if total > 0 else 0.0

    return feats


def physicochemical(smiles):
    """Physicochemical features computed from SMILES via RDKit.

    Replaces hardcoded property tables by computing everything from structure.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: 0.0 for k in [
            "mol_weight", "num_atoms", "num_heavy_atoms", "num_H",
            "num_rotatable_bonds", "is_radical", "unpaired_electrons",
            "total_valence_electrons", "num_rings", "num_aromatic_rings",
            "fraction_sp3",
        ]}

    mol_h = Chem.AddHs(mol)
    num_H = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    num_radical = Descriptors.NumRadicalElectrons(mol)

    return {
        "mol_weight": Descriptors.MolWt(mol),
        "num_atoms": mol_h.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_H": num_H,
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "is_radical": int(num_radical > 0),
        "unpaired_electrons": num_radical,
        "total_valence_electrons": Descriptors.NumValenceElectrons(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_sp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


# Registry of all molecule feature sets
MOLECULE_FEATURE_SETS = {
    "rdkit_descriptors": rdkit_descriptors,
    "composition": composition,
    "physicochemical": physicochemical,
}


def compute_molecule_features(entities, feature_sets=None, smiles_map=None):
    """Compute molecule features for a dict of {entity_id: identifier}.

    Args:
        entities: dict mapping entity ID to identifier (SMILES or name)
        feature_sets: list of feature set names, or None for all
        smiles_map: optional dict mapping identifier to SMILES string.
                    If None, identifiers are treated as SMILES directly.

    Returns:
        DataFrame with entity_id as index, feature columns
    """
    if feature_sets is None:
        feature_sets = list(MOLECULE_FEATURE_SETS.keys())

    rows = {}
    for entity_id, identifier in entities.items():
        smiles = smiles_map.get(identifier, identifier) if smiles_map else identifier
        feats = {}
        for fs_name in feature_sets:
            fn = MOLECULE_FEATURE_SETS[fs_name]
            if fs_name == "composition":
                # Composition uses molecular formula, derive from SMILES
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    formula = Chem.rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))
                else:
                    formula = identifier
                feats.update(fn(formula))
            else:
                feats.update(fn(smiles))
        rows[entity_id] = feats

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "entity_id"
    # Fill NaN from composition columns (different molecules have different elements)
    df = df.fillna(0)
    return df
