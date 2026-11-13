"""2-D t-SNE maps of the entity embeddings the splitter works on.

``compute_map`` featurizes one entity column exactly as the pipeline does
(``_extract_entities`` + ``_featurize``, sharing PALM's feature cache, so a map
computed before a run makes the run's featurization a cache hit and vice
versa), reduces it with the pipeline's own t-SNE helper, and attaches the other
columns aggregated per entity so the browser can colour points by them.

``split_labels`` then reads a finished run's row-level assignments and maps
them onto the same points (in 2D an entity whose rows land in several splits
is labelled "multiple").

Maps are cached on disk keyed by file content + featurization spec, so the
same dataset/settings are only reduced once and always give the same layout.
"""

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path

# Embedding used for each entity type when the caller does not choose one:
# the pipeline's own SMILES default for molecules, ESM2-650M for proteins.
DEFAULT_FEATURE_SETS = {
    # Kept equal to PALM's dependency-light defaults by a regression check.
    "molecule": ["rdkit_descriptors", "composition", "physicochemical", "morgan_fingerprint"],
    "biomolecule": ["esm_embedding"],
    "gene": ["nucleotide_composition", "kmer_frequencies"],
    "material": ["magpie_properties", "stoichiometry", "electronic",
                 "bonding", "thermodynamic", "classification"],
}

FEATURE_LABELS = {
    "esm_embedding": "ESM2-650M",
    "morgan_fingerprint": "Morgan FP",
    "rdkit_descriptors": "RDKit descriptors",
    "nt_embedding": "Nucleotide Transformer",
    "chemberta_embedding": "ChemBERTa",
    "physicochemical": "physchem",
    "magpie_properties": "Magpie",
    "nucleotide_composition": "nucleotide composition",
    "kmer_frequencies": "k-mers",
}

MAX_POINTS = int(os.environ.get("WEBAPP_TSNE_MAX_POINTS", "5000"))
MAX_COLOR_COLUMNS = 12
MAX_CATEGORIES = 30
SEED = 42

MAP_CACHE_DIR = Path(os.environ.get("PALM_CACHE_DIR", "~/.palm_cache")).expanduser() / "webapp_maps"
MAP_VERSION = "2"        # shared dimensionality-reduction cache, same mathematical recipe

# One map at a time: featurization (ESM, ChemBERTa, ...) and t-SNE are both
# heavy, and the language-model embedders are not guaranteed thread-safe.
_COMPUTE_LOCK = threading.Lock()


def resolve_feature_sets(etype, feature_sets):
    return list(feature_sets) if feature_sets else list(DEFAULT_FEATURE_SETS.get(etype, []))


def embedding_label(feature_sets):
    names = [FEATURE_LABELS.get(f, f.replace("_", " ")) for f in feature_sets]
    return " + ".join(names) if len(names) <= 4 else f"{names[0]} + {len(names) - 1} more feature sets"


def map_key(file_path, column, etype, feature_sets):
    from PALM.benchmarks.datasail_pipeline.cache import file_fingerprint
    from .reduction import signature_stamp
    spec = {"v": MAP_VERSION, "backend": signature_stamp(), "file": file_fingerprint(file_path), "column": column,
            "type": etype, "feature_sets": sorted(feature_sets),
            "max_points": MAX_POINTS, "seed": SEED}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def cached_map(key):
    path = MAP_CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    m = json.loads(path.read_text())
    m["embedding"] = embedding_label(m["feature_sets"])     # labels may change; data does not
    return m


def _attributes(df, column, ids):
    """Other columns aggregated per entity: mean if numeric, mode if categorical."""
    import pandas as pd
    out = {}
    for col in df.columns:
        if col == column or col.startswith("_") or len(out) >= MAX_COLOR_COLUMNS:
            continue
        sub = df[[column, col]].dropna()
        if sub.empty:
            continue
        keys = sub[column].astype(str)
        if pd.api.types.is_bool_dtype(sub[col]):
            sub = sub.assign(**{col: sub[col].astype(str)})
        if pd.api.types.is_numeric_dtype(sub[col]):
            agg = sub[col].groupby(keys).mean()
            vals = [None if pd.isna(v) else round(float(v), 6)
                    for v in agg.reindex(ids).tolist()]
            out[col] = {"kind": "numeric", "values": vals}
        else:
            s = sub[col].astype(str)
            if s.nunique() > MAX_CATEGORIES:
                continue                              # free text / IDs: not a colour
            agg = s.groupby(keys).agg(lambda v: Counter(v).most_common(1)[0][0])
            vals = [None if pd.isna(v) else v for v in agg.reindex(ids).tolist()]
            out[col] = {"kind": "categorical", "values": vals}
    return out


def _hover_labels(df, column, ids):
    """A short human-readable label per entity: a name/ID column if one exists."""
    import pandas as pd
    n = len(ids)
    for col in df.columns:
        if col == column or col.startswith("_") or df[col].dtype != object:
            continue
        sub = df[[column, col]].dropna()
        first = sub.groupby(sub[column].astype(str))[col].first()
        if first.nunique() >= 0.9 * n and first.astype(str).str.len().max() <= 60:
            return col, [None if pd.isna(v) else str(v) for v in first.reindex(ids)]
    return None, None


def compute_map(file_path, column, etype, feature_sets=None, progress=None):
    """Featurize ``column`` like the splitter does and embed it in 2-D with t-SNE."""
    import numpy as np
    from PALM.benchmarks.datasail_pipeline.config import EntityConfig
    from PALM.benchmarks.datasail_pipeline.loaders import load_data
    from PALM.benchmarks.datasail_pipeline.pipeline import _extract_entities, _featurize
    from .reduction import cached_reduce
    feature_sets = resolve_feature_sets(etype, feature_sets)
    key = map_key(file_path, column, etype, feature_sets)
    hit = cached_map(key)
    if hit is not None:
        return hit

    def _say(msg):
        if progress:
            progress(msg)

    with _COMPUTE_LOCK:
        hit = cached_map(key)              # computed while we waited
        if hit is not None:
            return hit
        df = load_data(str(file_path))
        if column not in df.columns:
            raise KeyError(column)
        cfg = EntityConfig(name="entity", type=etype, extract_column=column,
                           feature_sets=feature_sets)
        entities = _extract_entities(df, cfg)
        if len(entities) < 5:
            raise ValueError(f"Column '{column}' has only {len(entities)} unique values")
        _say(f"Computing {embedding_label(feature_sets)} features "
             f"for {len(entities):,} entities...")
        feats = _featurize(entities, cfg)
        feats.index = feats.index.astype(str)

        ids = sorted(feats.index)
        n_total = len(ids)
        if n_total > MAX_POINTS:
            rng = np.random.default_rng(SEED)
            ids = sorted(rng.choice(ids, MAX_POINTS, replace=False).tolist())
        X = (feats.loc[ids].replace([np.inf, -np.inf], np.nan).fillna(0)
             .to_numpy(dtype=float))
        _say(f"Running t-SNE on {len(ids):,} points ({X.shape[1]:,}-d)...")
        coords = cached_reduce(X, "tsne", seed=SEED)
        if coords is None:
            raise ValueError("All features are constant - nothing to visualise")

        label_col, labels = _hover_labels(df, column, ids)
        result = {
            "key": key,
            "column": column, "type": etype, "feature_sets": feature_sets,
            "embedding": embedding_label(feature_sets), "dims": int(X.shape[1]),
            "n_entities": n_total, "n_shown": len(ids),
            "ids": ids,
            "x": [round(float(v), 4) for v in coords[:, 0]],
            "y": [round(float(v), 4) for v in coords[:, 1]],
            "label_column": label_col, "labels": labels,
            "attributes": _attributes(df, column, ids),
        }
        MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MAP_CACHE_DIR / f"{key}.json.tmp"
        tmp.write_text(json.dumps(result))
        tmp.replace(MAP_CACHE_DIR / f"{key}.json")
        return result


def split_labels(file_path, column, ids, output_dir, dataset_name, techniques):
    """Per technique, the split of each map point (``multiple`` if it spans splits)."""
    import pandas as pd
    from PALM.benchmarks.datasail_pipeline.loaders import load_data
    df = load_data(str(file_path))
    row_entity = pd.Series(df[column].astype(str).values,
                           index=df["_row_id"].astype(str).values)
    split_dir = Path(output_dir) / "split_result"
    out = {}
    for tech in techniques:
        path = split_dir / f"datasail_split_{tech}_{dataset_name}.csv"
        if not path.is_file():
            continue
        assign = pd.read_csv(path, dtype={"_row_id": str})
        ent = row_entity.reindex(assign["_row_id"].values).values
        per_entity = {}
        for e, lab in zip(ent, assign["split"].astype(str)):
            if isinstance(e, str) and lab != "not selected":
                per_entity.setdefault(e, set()).add(lab)
        out[tech] = [(next(iter(s)) if len(s) == 1 else "multiple") if (s := per_entity.get(i))
                     else None for i in ids]
    return out
