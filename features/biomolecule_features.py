"""Protein/biomolecule featurization (3 feature sets).

Supports featurization from amino acid sequences using:
  - Physicochemical descriptors (no dependencies)
  - ESM2 protein language model embeddings (requires torch, transformers)
  - Pre-computed embeddings from file (requires numpy)
"""

import numpy as np
import pandas as pd

# Amino acid properties: MW, pI, hydrophobicity (Kyte-Doolittle), charge at pH 7,
#                         volume (A^3), surface_area (A^2), flexibility
AA_PROPERTIES = {
    "A": [89.09,  6.00,  1.8,   0, 88.6,  115.0, 0.360],
    "R": [174.20, 10.76, -4.5,  1, 173.4, 225.0, 0.530],
    "N": [132.12, 5.41,  -3.5,  0, 114.1, 160.0, 0.460],
    "D": [133.10, 2.77,  -3.5, -1, 111.1, 150.0, 0.510],
    "C": [121.16, 5.07,  2.5,   0, 108.5, 135.0, 0.350],
    "E": [147.13, 3.22,  -3.5, -1, 138.4, 190.0, 0.500],
    "Q": [146.15, 5.65,  -3.5,  0, 143.8, 180.0, 0.490],
    "G": [75.03,  5.97,  -0.4,  0, 60.1,  75.0,  0.540],
    "H": [155.16, 7.59,  -3.2,  0, 153.2, 195.0, 0.320],
    "I": [131.17, 6.02,  4.5,   0, 166.7, 175.0, 0.460],
    "L": [131.17, 5.98,  3.8,   0, 166.7, 170.0, 0.400],
    "K": [146.19, 9.74,  -3.9,  1, 168.6, 200.0, 0.470],
    "M": [149.21, 5.74,  1.9,   0, 162.9, 185.0, 0.410],
    "F": [165.19, 5.48,  2.8,   0, 189.9, 210.0, 0.310],
    "P": [115.13, 6.30,  -1.6,  0, 122.7, 145.0, 0.510],
    "S": [105.09, 5.68,  -0.8,  0, 89.0,  115.0, 0.510],
    "T": [119.12, 5.60,  -0.7,  0, 116.1, 140.0, 0.440],
    "W": [204.23, 5.89,  -0.9,  0, 227.8, 255.0, 0.310],
    "Y": [181.19, 5.66,  -1.3,  0, 193.6, 230.0, 0.420],
    "V": [117.15, 5.96,  4.2,   0, 140.0, 155.0, 0.390],
}
AA_PROP_NAMES = ["MW", "pI", "hydrophobicity", "charge_pH7",
                 "volume", "surface_area", "flexibility"]

# Amino acid categories
HYDROPHOBIC_AA = set("AILMFWV")
POLAR_AA = set("STNQYC")
CHARGED_POS_AA = set("RKH")
CHARGED_NEG_AA = set("DE")
AROMATIC_AA = set("FWY")
TINY_AA = set("AGS")
SMALL_AA = set("AGSCTDNPV")

# ESM2 model variants: name -> (hub_name, embedding_dim)
ESM2_MODELS = {
    "esm2_t6":  ("facebook/esm2_t6_8M_UR50D", 320),
    "esm2_t12": ("facebook/esm2_t12_35M_UR50D", 480),
    "esm2_t30": ("facebook/esm2_t30_150M_UR50D", 640),
    "esm2_t33": ("facebook/esm2_t33_650M_UR50D", 1280),
    "esm2_t36": ("facebook/esm2_t36_3B_UR50D", 2560),
    "esm2_t48": ("facebook/esm2_t48_15B_UR50D", 5120),
}


def sequence_properties(sequence):
    """Global physicochemical properties computed from sequence."""
    seq = sequence.upper()
    total = len(seq)
    if total == 0:
        return {k: 0.0 for k in [
            "length", "molecular_weight", "avg_hydrophobicity", "charge_pH7",
            "isoelectric_point_approx", "fraction_hydrophobic", "fraction_polar",
            "fraction_charged_pos", "fraction_charged_neg", "fraction_aromatic",
            "fraction_tiny", "fraction_small", "gravy",
            "avg_flexibility", "avg_volume", "avg_surface_area",
        ]}

    mw_total = 0.0
    hydro_sum = 0.0
    charge_sum = 0
    flex_sum = 0.0
    vol_sum = 0.0
    sa_sum = 0.0
    pi_sum = 0.0

    n_hydrophobic = 0
    n_polar = 0
    n_pos = 0
    n_neg = 0
    n_aromatic = 0
    n_tiny = 0
    n_small = 0

    for aa in seq:
        if aa not in AA_PROPERTIES:
            continue
        props = AA_PROPERTIES[aa]
        mw_total += props[0]
        pi_sum += props[1]
        hydro_sum += props[2]
        charge_sum += props[3]
        vol_sum += props[4]
        sa_sum += props[5]
        flex_sum += props[6]

        if aa in HYDROPHOBIC_AA:
            n_hydrophobic += 1
        if aa in POLAR_AA:
            n_polar += 1
        if aa in CHARGED_POS_AA:
            n_pos += 1
        if aa in CHARGED_NEG_AA:
            n_neg += 1
        if aa in AROMATIC_AA:
            n_aromatic += 1
        if aa in TINY_AA:
            n_tiny += 1
        if aa in SMALL_AA:
            n_small += 1

    # Water loss during peptide bond formation
    mw_total -= (total - 1) * 18.015

    return {
        "length": total,
        "molecular_weight": mw_total,
        "avg_hydrophobicity": hydro_sum / total,
        "charge_pH7": charge_sum,
        "isoelectric_point_approx": pi_sum / total,
        "fraction_hydrophobic": n_hydrophobic / total,
        "fraction_polar": n_polar / total,
        "fraction_charged_pos": n_pos / total,
        "fraction_charged_neg": n_neg / total,
        "fraction_aromatic": n_aromatic / total,
        "fraction_tiny": n_tiny / total,
        "fraction_small": n_small / total,
        "gravy": hydro_sum / total,
        "avg_flexibility": flex_sum / total,
        "avg_volume": vol_sum / total,
        "avg_surface_area": sa_sum / total,
    }


# ── ESM2 embedding support ───────────────────────────────────────────────

class ESM2Embedder:
    """Lazy-loaded ESM2 model for generating protein embeddings.

    Mean-pools per-residue representations from the last hidden layer
    to produce a fixed-size embedding per sequence.
    """

    def __init__(self, model_name="esm2_t33", batch_size=8, max_length=1022):
        if model_name not in ESM2_MODELS:
            raise ValueError(
                f"Unknown ESM2 model '{model_name}'. "
                f"Available: {list(ESM2_MODELS.keys())}"
            )
        self.hub_name, self.embed_dim = ESM2_MODELS[model_name]
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"  Loading ESM2 model: {self.hub_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.hub_name)
        self._model = AutoModel.from_pretrained(self.hub_name)

        if torch.cuda.is_available():
            self._model = self._model.cuda()
            print(f"  Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("  Using CPU (no GPU detected)")

        self._model.eval()

    def embed_sequences(self, sequences):
        """Generate mean-pooled embeddings for a list of sequences.

        Args:
            sequences: list of amino acid sequence strings

        Returns:
            np.ndarray of shape (len(sequences), embed_dim)
        """
        import torch

        self._load_model()
        device = next(self._model.parameters()).device

        # Truncate sequences exceeding max_length
        truncated = [seq[:self.max_length] for seq in sequences]

        all_embeddings = []
        for i in range(0, len(truncated), self.batch_size):
            batch_seqs = truncated[i:i + self.batch_size]
            inputs = self._tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length + 2,  # +2 for BOS/EOS tokens
            ).to(device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean pool over sequence positions, excluding BOS/EOS tokens
            hidden = outputs.last_hidden_state  # (B, L, D)
            mask = inputs["attention_mask"]      # (B, L)

            # Zero out BOS (pos 0) and EOS (last non-pad) tokens
            mask = mask.clone()
            mask[:, 0] = 0
            for j in range(mask.shape[0]):
                last_pos = mask[j].sum().item() - 1
                if last_pos > 0:
                    mask[j, int(last_pos)] = 0

            # Mean pool
            mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
            summed = (hidden * mask_expanded).sum(dim=1)  # (B, D)
            counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            embeddings = (summed / counts).cpu().numpy()

            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)


# Module-level embedder instance (lazy, shared across calls)
_esm2_embedder = None


def _get_esm2_embedder(model_name="esm2_t33", batch_size=8):
    """Get or create the shared ESM2 embedder instance."""
    global _esm2_embedder
    if _esm2_embedder is None or _esm2_embedder.model_name != model_name:
        _esm2_embedder = ESM2Embedder(model_name=model_name, batch_size=batch_size)
    return _esm2_embedder


def esm_embedding(sequence, _embedder_cache=None):
    """Placeholder — ESM2 embeddings are computed in batch by compute_biomolecule_features.

    This function is registered in BIOMOLECULE_FEATURE_SETS as a sentinel;
    the actual computation is handled specially in compute_biomolecule_features.
    """
    raise RuntimeError(
        "esm_embedding should not be called directly. "
        "It is computed in batch by compute_biomolecule_features."
    )


def precomputed_embedding(sequence):
    """Placeholder — pre-computed embeddings are loaded by compute_biomolecule_features.

    This function is registered in BIOMOLECULE_FEATURE_SETS as a sentinel;
    the actual loading is handled specially in compute_biomolecule_features.
    """
    raise RuntimeError(
        "precomputed_embedding should not be called directly. "
        "It is loaded from file by compute_biomolecule_features."
    )


# Registry of biomolecule feature sets
BIOMOLECULE_FEATURE_SETS = {
    "sequence_properties": sequence_properties,
    "esm_embedding": esm_embedding,
    "precomputed_embedding": precomputed_embedding,
}


def compute_biomolecule_features(
    entities,
    feature_sets=None,
    esm_model="esm2_t33",
    esm_batch_size=8,
    embedding_file=None,
):
    """Compute biomolecule features for a dict of {entity_id: sequence}.

    Args:
        entities: dict mapping entity ID to amino acid sequence
        feature_sets: list of feature set names, or None for all non-embedding sets.
                      Include "esm_embedding" to compute ESM2 embeddings.
                      Include "precomputed_embedding" to load from embedding_file.
        esm_model: ESM2 model variant (default: "esm2_t33" = 650M param).
                   Options: esm2_t6 (8M), esm2_t12 (35M), esm2_t30 (150M),
                            esm2_t33 (650M), esm2_t36 (3B), esm2_t48 (15B)
        esm_batch_size: batch size for ESM2 inference
        embedding_file: path to pre-computed embeddings (CSV or .npy/.npz).
                        CSV must have entity_id as first column.
                        NPZ must have 'ids' and 'embeddings' arrays.

    Returns:
        DataFrame with entity_id as index, feature columns
    """
    if feature_sets is None:
        # Default: sequence_properties (no heavy dependencies)
        feature_sets = ["sequence_properties"]

    entity_ids = list(entities.keys())

    # Compute per-sequence features (non-batch)
    simple_sets = [fs for fs in feature_sets
                   if fs not in ("esm_embedding", "precomputed_embedding")]
    rows = {}
    for entity_id in entity_ids:
        sequence = entities[entity_id]
        feats = {}
        for fs_name in simple_sets:
            fn = BIOMOLECULE_FEATURE_SETS[fs_name]
            feats.update(fn(sequence))
        rows[entity_id] = feats

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "entity_id"
    df = df.fillna(0)

    # ESM2 embedding (batch)
    if "esm_embedding" in feature_sets:
        embedder = _get_esm2_embedder(model_name=esm_model, batch_size=esm_batch_size)
        sequences = [entities[eid] for eid in entity_ids]
        print(f"  Computing ESM2 embeddings ({esm_model}) for {len(sequences)} sequences...")
        emb_matrix = embedder.embed_sequences(sequences)

        emb_cols = [f"esm_{i}" for i in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, index=entity_ids, columns=emb_cols)
        emb_df.index.name = "entity_id"
        df = pd.concat([df, emb_df], axis=1)
        print(f"  ESM2 embeddings: {emb_matrix.shape[1]} dimensions")

    # Pre-computed embeddings from file
    if "precomputed_embedding" in feature_sets:
        if embedding_file is None:
            raise ValueError(
                "embedding_file must be specified when using 'precomputed_embedding' feature set"
            )
        print(f"  Loading pre-computed embeddings from {embedding_file}")

        from .utils import load_precomputed_embeddings
        emb_aligned = load_precomputed_embeddings(embedding_file, entity_ids)
        emb_cols = [f"emb_{i}" for i in range(emb_aligned.shape[1])]
        emb_df = pd.DataFrame(emb_aligned, index=entity_ids, columns=emb_cols)
        emb_df.index.name = "entity_id"
        df = pd.concat([df, emb_df], axis=1)
        print(f"  Pre-computed embeddings: {emb_aligned.shape[1]} dimensions")

    return df
