"""Utility functions for feature computation."""

import re
import math
import numpy as np

from .elemental_data import ELEM_PROPS, PROP_NAMES


def parse_formula(formula):
    """Parse formula like 'Ta8O20' -> {'Ta': 8, 'O': 20}."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    comp = {}
    for elem, count in pairs:
        if elem == "":
            continue
        comp[elem] = comp.get(elem, 0) + (int(count) if count else 1)
    return comp


def magpie_stats(values):
    """Compute Magpie-style statistics: mean, std, min, max, range."""
    arr = np.array(values, dtype=float) if values else np.array([0.0])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }


def composition_weighted_mean(comp, prop_table, prop_index=None):
    """Compute composition-weighted mean of a property.

    Args:
        comp: dict of {element: count}
        prop_table: dict of {element: value} or {element: [values...]}
        prop_index: if prop_table values are lists, index into them
    """
    total = sum(comp.values())
    if total == 0:
        return 0.0
    weighted_sum = 0.0
    for elem, cnt in comp.items():
        if elem not in prop_table:
            continue
        val = prop_table[elem]
        if prop_index is not None:
            val = val[prop_index]
        weighted_sum += cnt * val
    return weighted_sum / total


def load_precomputed_embeddings(embedding_file, entity_ids):
    """Load pre-computed embeddings from CSV, npy, npz, or pt/pth files.

    Args:
        embedding_file: path to the embedding file
        entity_ids: list of entity IDs to align embeddings to

    Returns:
        np.ndarray of shape (len(entity_ids), embedding_dim)
    """
    import pandas as pd

    if embedding_file.endswith(".npz"):
        data = np.load(embedding_file)
        emb_ids = list(data["ids"])
        emb_matrix = data["embeddings"]
    elif embedding_file.endswith(".npy"):
        emb_matrix = np.load(embedding_file)
        emb_ids = entity_ids
    elif embedding_file.endswith(".csv"):
        emb_raw = pd.read_csv(embedding_file, index_col=0)
        emb_ids = list(emb_raw.index.astype(str))
        emb_matrix = emb_raw.values
    elif embedding_file.endswith((".pt", ".pth")):
        import torch
        data = torch.load(embedding_file, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            emb_ids = list(data.keys())
            emb_matrix = np.stack([
                v.numpy() if hasattr(v, "numpy") else np.array(v)
                for v in data.values()
            ])
        else:
            emb_matrix = data.numpy() if hasattr(data, "numpy") else np.array(data)
            emb_ids = entity_ids
    else:
        raise ValueError(f"Unsupported embedding file format: {embedding_file}")

    # Align embeddings to entity_ids
    id_to_idx = {str(eid): i for i, eid in enumerate(emb_ids)}
    aligned_embs = []
    for eid in entity_ids:
        if str(eid) in id_to_idx:
            aligned_embs.append(emb_matrix[id_to_idx[str(eid)]])
        else:
            print(f"    WARNING: No embedding found for entity '{eid}', using zeros")
            aligned_embs.append(np.zeros(emb_matrix.shape[1]))

    return np.stack(aligned_embs)


def p_norm(vec, p):
    """Compute the L-p norm of a vector."""
    if not vec:
        return 0.0
    return sum(abs(x) ** p for x in vec) ** (1.0 / p)
