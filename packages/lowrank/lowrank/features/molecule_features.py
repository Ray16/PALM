"""SMILES-based molecule featurization (4 feature sets).

Generalized from OC22 adsorbate featurization — works with any SMILES string.
"""

import logging

import numpy as np
import pandas as pd

from functools import partial

from .elemental_data import ELEM_PROPS, PROP_NAMES
from .utils import parse_formula, parallel_map

logger = logging.getLogger(__name__)


def rdkit_descriptors(smiles_or_mol):
    """Compute RDKit molecular descriptors from a SMILES string or mol object."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = smiles_or_mol if hasattr(smiles_or_mol, 'GetNumAtoms') else Chem.MolFromSmiles(smiles_or_mol)
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


def physicochemical(smiles_or_mol):
    """Physicochemical features computed from SMILES or mol object via RDKit.

    Replaces hardcoded property tables by computing everything from structure.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = smiles_or_mol if hasattr(smiles_or_mol, 'GetNumAtoms') else Chem.MolFromSmiles(smiles_or_mol)
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


# Cache of bit-column names per n_bits so the 2048 f-strings are built once,
# not once per molecule.
_MORGAN_KEYS = {}


def _morgan_keys(n_bits):
    keys = _MORGAN_KEYS.get(n_bits)
    if keys is None:
        keys = [f"morgan_{i}" for i in range(n_bits)]
        _MORGAN_KEYS[n_bits] = keys
    return keys


def morgan_fingerprint(smiles_or_mol, radius=2, n_bits=2048):
    """Morgan (ECFP) circular fingerprint as a bit vector.

    Produces a 2048-bit fingerprint that captures molecular substructure
    topology. Best used with Tanimoto distance for clustering.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    keys = _morgan_keys(n_bits)
    mol = smiles_or_mol if hasattr(smiles_or_mol, 'GetNumAtoms') else Chem.MolFromSmiles(smiles_or_mol)
    if mol is None:
        return {k: 0 for k in keys}

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    # Extract all bits in one C++ call instead of n_bits Python-level fp[i] reads.
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return dict(zip(keys, arr.tolist()))


def maccs_keys(smiles_or_mol):
    """MACCS structural keys — 167-bit substructure fingerprint."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys

    keys = _maccs_key_names()
    mol = smiles_or_mol if hasattr(smiles_or_mol, 'GetNumAtoms') else Chem.MolFromSmiles(smiles_or_mol)
    if mol is None:
        return {k: 0 for k in keys}
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((len(keys),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return dict(zip(keys, arr.tolist()))


_MACCS_KEYS = None


def _maccs_key_names():
    global _MACCS_KEYS
    if _MACCS_KEYS is None:
        # GenMACCSKeys returns a 167-bit vector (index 0 is unused but present).
        _MACCS_KEYS = [f"maccs_{i}" for i in range(167)]
    return _MACCS_KEYS


_RDKIT_FULL_NAMES = None


def _rdkit_full_names():
    global _RDKIT_FULL_NAMES
    if _RDKIT_FULL_NAMES is None:
        from rdkit.Chem import Descriptors
        _RDKIT_FULL_NAMES = [name for name, _ in Descriptors.descList]
    return _RDKIT_FULL_NAMES


def rdkit_descriptors_full(smiles_or_mol):
    """The full RDKit 2D descriptor set (~200 descriptors).

    Columns are prefixed ``rdkitfull_`` so they never collide with the curated
    ``rdkit_descriptors`` set when both are requested.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    names = _rdkit_full_names()
    mol = smiles_or_mol if hasattr(smiles_or_mol, 'GetNumAtoms') else Chem.MolFromSmiles(smiles_or_mol)
    if mol is None:
        return {f"rdkitfull_{n}": 0.0 for n in names}
    vals = Descriptors.CalcMolDescriptors(mol)
    out = {}
    for n in names:
        v = vals.get(n, 0.0)
        out[f"rdkitfull_{n}"] = float(v) if v is not None else 0.0
    return out


# ── ChemBERTa molecular language-model embeddings ──────────────────────────

# Model registry: name -> (HuggingFace hub path, embedding dim)
CHEMBERTA_MODELS = {
    "chemberta_zinc":  ("seyonec/ChemBERTa-zinc-base-v1", 768),
    "chemberta_77m":   ("DeepChem/ChemBERTa-77M-MLM", 384),
    "chemberta_10m":   ("DeepChem/ChemBERTa-10M-MLM", 384),
}


class ChemBERTaEmbedder:
    """Lazy-loaded ChemBERTa model producing mean-pooled SMILES embeddings."""

    def __init__(self, model_name="chemberta_zinc", batch_size=16, max_length=512):
        if model_name not in CHEMBERTA_MODELS:
            raise ValueError(
                f"Unknown ChemBERTa model '{model_name}'. "
                f"Available: {list(CHEMBERTA_MODELS.keys())}"
            )
        self.hub_name, self.embed_dim = CHEMBERTA_MODELS[model_name]
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"  Loading ChemBERTa model: {self.hub_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.hub_name)
        self._model = AutoModel.from_pretrained(self.hub_name)
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            logger.info(f"  Using GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("  Using CPU (no GPU detected)")
        self._model.eval()

    def embed_smiles(self, smiles_list):
        """Mean-pooled embeddings for a list of SMILES (np.ndarray)."""
        import torch

        self._load_model()
        device = next(self._model.parameters()).device

        all_embeddings = []
        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state          # (B, L, D)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            all_embeddings.append((summed / counts).cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


_chemberta_embedder = None


def _get_chemberta_embedder(model_name="chemberta_zinc", batch_size=16):
    global _chemberta_embedder
    if _chemberta_embedder is None or _chemberta_embedder.model_name != model_name:
        _chemberta_embedder = ChemBERTaEmbedder(model_name=model_name, batch_size=batch_size)
    return _chemberta_embedder


def chemberta_embedding(smiles):
    """Sentinel — ChemBERTa embeddings are computed in batch by compute_molecule_features."""
    raise RuntimeError(
        "chemberta_embedding should not be called directly. "
        "It is computed in batch by compute_molecule_features."
    )


# Registry of all molecule feature sets
MOLECULE_FEATURE_SETS = {
    "rdkit_descriptors": rdkit_descriptors,
    "rdkit_descriptors_full": rdkit_descriptors_full,
    "composition": composition,
    "physicochemical": physicochemical,
    "morgan_fingerprint": morgan_fingerprint,
    "maccs_keys": maccs_keys,
    "chemberta_embedding": chemberta_embedding,
}


# Embedding sets are computed in batch (model inference), not per-entity.
_MOLECULE_EMBEDDING_SETS = {"chemberta_embedding"}

# Default when no feature_sets are specified: the dependency-light curated sets
# (excludes the full descriptor list and the language-model embedding).
_MOLECULE_DEFAULT_SETS = [
    "rdkit_descriptors", "composition", "physicochemical", "morgan_fingerprint",
]


def _featurize_one_molecule(item, simple_sets, smiles_map):
    """Featurize a single (entity_id, identifier). Module-level so it is
    picklable for parallel execution. Returns (entity_id, feats, failed)."""
    from rdkit import Chem

    entity_id, identifier = item
    smiles = smiles_map.get(identifier, identifier) if smiles_map else identifier
    mol = Chem.MolFromSmiles(smiles)
    feats = {}
    for fs_name in simple_sets:
        fn = MOLECULE_FEATURE_SETS[fs_name]
        if fs_name == "composition":
            # Composition uses molecular formula, derive from SMILES
            if mol is not None:
                formula = Chem.rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))
            else:
                formula = identifier
            feats.update(fn(formula))
        else:
            feats.update(fn(mol if mol is not None else smiles))
    return entity_id, feats, mol is None


def compute_molecule_features(entities, feature_sets=None, smiles_map=None,
                              chemberta_model="chemberta_zinc",
                              chemberta_batch_size=16, n_jobs=1):
    """Compute molecule features for a dict of {entity_id: identifier}.

    Args:
        entities: dict mapping entity ID to identifier (SMILES or name)
        feature_sets: list of feature set names, or None for the default sets
        smiles_map: optional dict mapping identifier to SMILES string.
                    If None, identifiers are treated as SMILES directly.
        chemberta_model: ChemBERTa variant for the "chemberta_embedding" set.
        chemberta_batch_size: batch size for ChemBERTa inference.
        n_jobs: parallel workers for per-molecule featurization (1 = serial).

    Returns:
        DataFrame with entity_id as index, feature columns
    """
    if feature_sets is None:
        feature_sets = list(_MOLECULE_DEFAULT_SETS)

    simple_sets = [fs for fs in feature_sets if fs not in _MOLECULE_EMBEDDING_SETS]

    worker = partial(_featurize_one_molecule, simple_sets=simple_sets, smiles_map=smiles_map)
    results = parallel_map(worker, entities.items(), n_jobs=n_jobs)

    rows = {}
    failed = []
    for entity_id, feats, is_failed in results:
        rows[entity_id] = feats
        if is_failed:
            failed.append(entity_id)

    if failed:
        logger.warning(
            "  %d of %d molecules failed to parse and were zero-filled "
            "(they will look identical in feature space): e.g. %s",
            len(failed), len(entities), failed[:5],
        )

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "entity_id"
    # Fill NaN from composition columns (different molecules have different elements)
    df = df.fillna(0)

    # ChemBERTa embedding (batch model inference)
    if "chemberta_embedding" in feature_sets:
        entity_ids = list(entities.keys())
        smiles_list = [
            (smiles_map.get(entities[eid], entities[eid]) if smiles_map else entities[eid])
            for eid in entity_ids
        ]
        embedder = _get_chemberta_embedder(model_name=chemberta_model,
                                           batch_size=chemberta_batch_size)
        logger.info(f"  Computing ChemBERTa embeddings ({chemberta_model}) for {len(smiles_list)} molecules...")
        emb_matrix = embedder.embed_smiles(smiles_list)
        emb_cols = [f"chemberta_{i}" for i in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, index=entity_ids, columns=emb_cols)
        emb_df.index.name = "entity_id"
        df = pd.concat([df, emb_df], axis=1)
        logger.info(f"  ChemBERTa embeddings: {emb_matrix.shape[1]} dimensions")

    return df
