"""Orchestrator: load -> featurize -> split -> save."""

import os
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .loaders import load_data
from .features.material_features import compute_material_features
from .features.molecule_features import compute_molecule_features
from .features.biomolecule_features import compute_biomolecule_features
from .features.gene_features import compute_gene_features
from .splitting import run_splitting


def _extract_entities(df, entity_config):
    """Extract unique entities from a column, return {entity_id: value}."""
    col = entity_config.extract_column
    entities = {}
    for val in df[col].dropna().unique():
        entities[str(val)] = str(val)
    return entities


def _featurize(entities, entity_config):
    """Generate features for entities based on their type."""
    feature_sets = entity_config.feature_sets or None
    if entity_config.type == "material":
        return compute_material_features(entities, feature_sets=feature_sets)
    elif entity_config.type == "molecule":
        return compute_molecule_features(
            entities,
            feature_sets=feature_sets,
            smiles_map=entity_config.smiles_map,
        )
    elif entity_config.type == "biomolecule":
        return compute_biomolecule_features(
            entities,
            feature_sets=feature_sets,
            esm_model=entity_config.esm_model,
            esm_batch_size=entity_config.esm_batch_size,
            embedding_file=entity_config.embedding_file,
        )
    elif entity_config.type == "gene":
        return compute_gene_features(
            entities,
            feature_sets=feature_sets,
            nt_model=entity_config.nt_model,
            nt_batch_size=entity_config.nt_batch_size,
            embedding_file=entity_config.embedding_file,
        )
    else:
        raise ValueError(f"Unknown entity type: {entity_config.type}")


def run_pipeline(config: PipelineConfig, progress_callback=None):
    """Run the full pipeline: load -> featurize -> split -> save.

    Args:
        config: PipelineConfig object.
        progress_callback: optional callable(percent: int, message: str).
            Called at each major step so callers (e.g. the web UI) can track
            progress. The pipeline still prints to stdout regardless.
    """

    def _report(pct: int, msg: str):
        print(msg)
        if progress_callback is not None:
            progress_callback(pct, msg)

    _report(0,  "=" * 70)
    _report(0,  f"Data Splitting Agent: {config.dataset_name}")
    _report(0,  "=" * 70)

    # 1. Load data
    _report(5,  f"\n[1/5] Loading data from {config.input_file}")
    df = load_data(config.input_file, filters=config.filters)
    _report(15, f"  Loaded {len(df)} rows")

    # 2. Extract entities
    _report(20, f"\n[2/5] Extracting entities")
    e_entities = _extract_entities(df, config.e)
    f_entities = _extract_entities(df, config.f)
    _report(25, f"  {config.e.name} (e): {len(e_entities)} unique entities")
    _report(28, f"  {config.f.name} (f): {len(f_entities)} unique entities")

    # 3. Generate features
    _report(30, f"\n[3/5] Generating features")
    _report(32, f"  Featurizing {config.e.name}...")
    e_feat_df = _featurize(e_entities, config.e)
    _report(50, f"  {config.e.name} features: {e_feat_df.shape}")
    _report(52, f"  Featurizing {config.f.name}...")
    f_feat_df = _featurize(f_entities, config.f)
    _report(65, f"  {config.f.name} features: {f_feat_df.shape}")

    # Save features
    feat_dir = os.path.join(config.output_dir, "features", config.dataset_name)
    os.makedirs(os.path.join(feat_dir, config.e.name), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, config.f.name), exist_ok=True)

    e_feat_path = os.path.join(feat_dir, config.e.name, "features.csv")
    f_feat_path = os.path.join(feat_dir, config.f.name, "features.csv")
    e_feat_df.to_csv(e_feat_path)
    f_feat_df.to_csv(f_feat_path)
    _report(68, f"  Saved: {e_feat_path}")
    _report(70, f"  Saved: {f_feat_path}")

    # 4. Build interaction list and entity data dicts
    _report(72, f"\n[4/5] Running DataSAIL splitting")
    e_col = config.e.extract_column
    f_col = config.f.extract_column

    interactions = []
    row_to_inter = {}
    for idx, row in df.iterrows():
        e_id = str(row[e_col])
        f_id = str(row[f_col])
        interactions.append((e_id, f_id))
        row_to_inter[str(row["_row_id"])] = (e_id, f_id)

    _report(75, f"  Interactions: {len(interactions)} (unique pairs: {len(set(interactions))})")

    # Convert feature DataFrames to dicts for DataSAIL
    # Replace NaN/Inf with 0 to prevent downstream clustering errors
    e_feat_df = e_feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    f_feat_df = f_feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    e_data = {name: e_feat_df.loc[name].values.astype(float) for name in e_feat_df.index}
    f_data = {name: f_feat_df.loc[name].values.astype(float) for name in f_feat_df.index}

    split_results = run_splitting(e_data, f_data, interactions, config.splitting)
    _report(90, "  DataSAIL splitting complete")

    # 5. Map splits back to row IDs and save
    _report(92, f"\n[5/5] Saving split results")
    split_dir = os.path.join(config.output_dir, "split_result")
    os.makedirs(split_dir, exist_ok=True)

    for technique in sorted(split_results.keys()):
        result = split_results[technique]

        records = []
        for row_id, inter_key in row_to_inter.items():
            split = result.get(inter_key, "not selected")
            records.append({"_row_id": row_id, "split": split})

        out_df = pd.DataFrame(records)
        assigned = out_df[out_df["split"].isin(config.splitting.names)]

        summary = [f"\n  --- {technique} ---"]
        for name in config.splitting.names:
            n = (assigned["split"] == name).sum()
            pct = 100 * n / len(df)
            summary.append(f"    {name.capitalize()}: {n:,} ({pct:.1f}%)")

        n_not = (out_df["split"] == "not selected").sum()
        if n_not > 0:
            summary.append(f"    Not selected: {n_not:,}")

        # Entity overlap analysis
        for name in config.splitting.names:
            mask = assigned["split"] == name
            sids = assigned[mask]["_row_id"].values
            e_set = set(str(df.loc[df["_row_id"].isin(sids), e_col].values[i])
                        for i in range(len(sids)) if len(sids) > 0)
            f_set = set(str(df.loc[df["_row_id"].isin(sids), f_col].values[i])
                        for i in range(len(sids)) if len(sids) > 0)

        out_path = os.path.join(split_dir, f"datasail_split_{technique}_{config.dataset_name}.csv")
        out_df.to_csv(out_path, index=False)
        summary.append(f"    Saved to {out_path}")
        for line in summary:
            _report(95, line)

    _report(100, f"\nDone! Results saved to {config.output_dir}/")
