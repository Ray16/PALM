"""Orchestrator: load -> featurize -> split -> save."""

import json
import os
import shutil
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .loaders import load_data, _detect_format
from .features.material_features import compute_material_features
from .features.molecule_features import compute_molecule_features
from .features.biomolecule_features import compute_biomolecule_features
from .features.gene_features import compute_gene_features
from .splitting import run_splitting, run_splitting_1d


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


def _save_split_data(df, out_df, split_dir, technique, dataset_name, input_file,
                     fmt, split_names):
    """Save actual split data files matching the input format.

    For CSV → train.csv / test.csv
    For JSON → train.json / test.json
    For structure directories → train/ and test/ folders with copied files
    """
    technique_dir = os.path.join(split_dir, f"{technique}_{dataset_name}")
    os.makedirs(technique_dir, exist_ok=True)

    # Merge split assignments with original data
    merged = df.merge(out_df[["_row_id", "split"]], on="_row_id", how="left")

    if fmt == "csv":
        for name in split_names:
            subset = merged[merged["split"] == name].drop(columns=["split", "_row_id"])
            subset.to_csv(os.path.join(technique_dir, f"{name}.csv"), index=False)

    elif fmt == "json":
        # Read original JSON to preserve structure
        with open(input_file) as fh:
            original = json.load(fh)

        if isinstance(original, dict):
            # Dict-of-dicts: keys are row IDs
            for name in split_names:
                row_ids = set(merged.loc[merged["split"] == name, "_row_id"])
                subset = {k: v for k, v in original.items() if str(k) in row_ids}
                with open(os.path.join(technique_dir, f"{name}.json"), "w") as fh:
                    json.dump(subset, fh, indent=2)
        else:
            # List-of-dicts: use index as row ID
            for name in split_names:
                row_ids = set(merged.loc[merged["split"] == name, "_row_id"])
                subset = [item for i, item in enumerate(original)
                          if str(i) in row_ids]
                with open(os.path.join(technique_dir, f"{name}.json"), "w") as fh:
                    json.dump(subset, fh, indent=2)

    elif fmt in ("pdb_dir", "mmcif_dir"):
        # Map row IDs back to source structure files via structure_id column
        ext_map = {"pdb_dir": ".pdb", "mmcif_dir": (".cif", ".mmcif")}
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            mask = merged["split"] == name
            structure_ids = merged.loc[mask, "structure_id"].unique()
            for sid in structure_ids:
                # Find the source file
                for fname in os.listdir(input_file):
                    stem = os.path.splitext(fname)[0]
                    if stem == sid and fname.endswith(ext_map[fmt]):
                        src = os.path.join(input_file, fname)
                        shutil.copy2(src, os.path.join(dest, fname))
                        break

    elif fmt in ("cif_dir", "xyz_dir"):
        # _row_id is the filename stem
        ext = ".cif" if fmt == "cif_dir" else ".xyz"
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            row_ids = merged.loc[merged["split"] == name, "_row_id"].unique()
            for rid in row_ids:
                fname = rid + ext
                src = os.path.join(input_file, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(dest, fname))

    elif fmt in ("pdb", "mmcif"):
        # Single structure file — copy it into train or test based on majority
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            if (merged["split"] == name).any():
                shutil.copy2(input_file, os.path.join(dest,
                             os.path.basename(input_file)))

    elif fmt in ("sdf", "smi", "smiles", "mol", "mol2", "fasta", "ase_db"):
        # For these formats, save as CSV since splitting individual
        # entries back into the original format is lossy
        for name in split_names:
            subset = merged[merged["split"] == name].drop(columns=["split", "_row_id"])
            subset.to_csv(os.path.join(technique_dir, f"{name}.csv"), index=False)


def run_pipeline(config: PipelineConfig, progress_callback=None):
    """Run the full pipeline: load -> featurize -> split -> save.

    Supports both 1D (single entity) and 2D (interaction) splitting.
    When config.e2 is None, runs in 1D mode.

    Args:
        config: PipelineConfig object.
        progress_callback: optional callable(percent: int, message: str).
            Called at each major step so callers (e.g. the web UI) can track
            progress. The pipeline still prints to stdout regardless.
    """
    is_1d = config.e2 is None

    def _report(pct: int, msg: str):
        print(msg)
        if progress_callback is not None:
            progress_callback(pct, msg)

    _report(0,  "=" * 70)
    mode_label = "1D" if is_1d else "2D"
    _report(0,  f"Data Splitting Agent ({mode_label}): {config.dataset_name}")
    _report(0,  "=" * 70)

    # 1. Load data
    _report(5,  f"\n[1/5] Loading data from {config.input_file}")
    df = load_data(config.input_file, filters=config.filters)
    _report(15, f"  Loaded {len(df)} rows")

    # 2. Extract entities
    _report(20, f"\n[2/5] Extracting entities")
    e1_entities = _extract_entities(df, config.e1)
    _report(25, f"  {config.e1.name}: {len(e1_entities)} unique entities")

    if not is_1d:
        e2_entities = _extract_entities(df, config.e2)
        _report(28, f"  {config.e2.name}: {len(e2_entities)} unique entities")

    # 3. Generate features
    _report(30, f"\n[3/5] Generating features")
    _report(32, f"  Featurizing {config.e1.name}...")
    e1_feat_df = _featurize(e1_entities, config.e1)
    _report(50, f"  {config.e1.name} features: {e1_feat_df.shape}")

    # Save e1 features
    feat_dir = os.path.join(config.output_dir, "features", config.dataset_name)
    os.makedirs(os.path.join(feat_dir, config.e1.name), exist_ok=True)
    e1_feat_path = os.path.join(feat_dir, config.e1.name, "features.csv")
    e1_feat_df.to_csv(e1_feat_path)
    _report(55, f"  Saved: {e1_feat_path}")

    if not is_1d:
        _report(57, f"  Featurizing {config.e2.name}...")
        e2_feat_df = _featurize(e2_entities, config.e2)
        _report(65, f"  {config.e2.name} features: {e2_feat_df.shape}")

        os.makedirs(os.path.join(feat_dir, config.e2.name), exist_ok=True)
        e2_feat_path = os.path.join(feat_dir, config.e2.name, "features.csv")
        e2_feat_df.to_csv(e2_feat_path)
        _report(70, f"  Saved: {e2_feat_path}")

    # 4. Run DataSAIL splitting
    _report(72, f"\n[4/5] Running DataSAIL splitting")
    e1_col = config.e1.extract_column

    # Replace NaN/Inf with 0 to prevent downstream clustering errors
    e1_feat_df = e1_feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    e1_data = {name: e1_feat_df.loc[name].values.astype(float) for name in e1_feat_df.index}

    if is_1d:
        # 1D mode: split by entity only
        row_ids = df["_row_id"].astype(str)
        e1_ids = df[e1_col].astype(str)
        row_to_entity = dict(zip(row_ids, e1_ids))

        _report(75, f"  Entities: {len(e1_data)} unique")
        split_results = run_splitting_1d(e1_data, config.splitting)
    else:
        # 2D mode: split by interactions
        e2_col = config.e2.extract_column
        e1_ids = df[e1_col].astype(str)
        e2_ids = df[e2_col].astype(str)
        row_ids = df["_row_id"].astype(str)
        interactions = list(zip(e1_ids, e2_ids))
        row_to_inter = dict(zip(row_ids, interactions))

        _report(75, f"  Interactions: {len(interactions)} (unique pairs: {len(set(interactions))})")

        e2_feat_df = e2_feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        e2_data = {name: e2_feat_df.loc[name].values.astype(float) for name in e2_feat_df.index}

        split_results = run_splitting(e1_data, e2_data, interactions, config.splitting)

    _report(90, "  DataSAIL splitting complete")

    # 5. Map splits back to row IDs and save
    _report(92, f"\n[5/5] Saving split results")
    split_dir = os.path.join(config.output_dir, "split_result")
    os.makedirs(split_dir, exist_ok=True)

    # Detect input format for producing format-specific outputs
    fmt = _detect_format(config.input_file)

    for technique in sorted(split_results.keys()):
        result = split_results[technique]

        records = []
        if is_1d:
            for row_id, entity_id in row_to_entity.items():
                split = result.get(entity_id, "not selected")
                records.append({"_row_id": row_id, "split": split})
        else:
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

        # Entity overlap analysis between splits
        if len(config.splitting.names) >= 2:
            split_entities = {}
            for name in config.splitting.names:
                mask = assigned["split"] == name
                sids = assigned.loc[mask, "_row_id"].values
                rows_in_split = df[df["_row_id"].isin(sids)]
                split_entities[name] = {
                    "e1": set(rows_in_split[e1_col].astype(str)),
                }
                if not is_1d:
                    split_entities[name]["e2"] = set(rows_in_split[e2_col].astype(str))

            n0, n1 = config.splitting.names[0], config.splitting.names[1]
            if n0 in split_entities and n1 in split_entities:
                e1_overlap = split_entities[n0]["e1"] & split_entities[n1]["e1"]
                summary.append(f"    {config.e1.name} overlap ({n0}∩{n1}): {len(e1_overlap)}")
                if not is_1d:
                    e2_overlap = split_entities[n0]["e2"] & split_entities[n1]["e2"]
                    summary.append(f"    {config.e2.name} overlap ({n0}∩{n1}): {len(e2_overlap)}")

        out_path = os.path.join(split_dir, f"datasail_split_{technique}_{config.dataset_name}.csv")
        out_df.to_csv(out_path, index=False)
        summary.append(f"    Saved to {out_path}")

        # Save format-specific split data (train/test CSV, JSON, or structure folders)
        try:
            _save_split_data(df, out_df, split_dir, technique,
                             config.dataset_name, config.input_file,
                             fmt, config.splitting.names)
            data_dir = os.path.join(split_dir, f"{technique}_{config.dataset_name}")
            summary.append(f"    Split data saved to {data_dir}/")
        except Exception as exc:
            summary.append(f"    Warning: could not save split data files: {exc}")

        for line in summary:
            _report(95, line)

    _report(100, f"\nDone! Results saved to {config.output_dir}/")
