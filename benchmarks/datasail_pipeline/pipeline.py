"""Orchestrator: load -> featurize -> split -> save."""

import json
import logging
import math
import os
import random
import shutil
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .config import EXTERNAL_TECHNIQUES, PipelineConfig, SplitContext
from .loaders import load_data, _detect_format
from PALM.features.material_features import compute_material_features
from PALM.features.molecule_features import compute_molecule_features
from PALM.features.biomolecule_features import compute_biomolecule_features
from PALM.features.gene_features import compute_gene_features
from .cache import get_cached_features, save_cached_features, file_fingerprint
from .metrics import compute_row_split_metrics, save_metrics
from .splitting import run_splitting, run_splitting_1d, run_scaffold_splitting
from .visualization import generate_split_plots, generate_comparison_chart, _reduce


def _extract_entities(df, entity_config):
    """Extract unique entities from a column, return {entity_id: value}."""
    col = entity_config.extract_column
    entities = {}
    for val in df[col].dropna().unique():
        entities[str(val)] = str(val)
    return entities


def _feature_cache_spec(entity_config):
    """Every EntityConfig setting that changes the feature matrix, plus content
    fingerprints of the files/dirs features are read from (batch sizes and
    n_jobs only affect speed and are left out)."""
    return {
        "type": entity_config.type,
        "feature_sets": sorted(entity_config.feature_sets or []),
        "smiles_map": (json.dumps(entity_config.smiles_map, sort_keys=True, default=str)
                       if entity_config.smiles_map else None),
        "esm_model": entity_config.esm_model,
        "nt_model": entity_config.nt_model,
        "chemberta_model": entity_config.chemberta_model,
        "embedding_file": file_fingerprint(entity_config.embedding_file),
        "structure_dir": file_fingerprint(entity_config.structure_dir),
    }


def _featurize(entities, entity_config):
    """Generate features for entities based on their type (with caching)."""
    feature_sets = entity_config.feature_sets or None
    spec = _feature_cache_spec(entity_config)

    # Check cache
    cached = get_cached_features(entities, entity_config.type, feature_sets, spec)
    if cached is not None:
        return cached

    result = _featurize_uncached(entities, entity_config)

    # Save to cache
    save_cached_features(entities, entity_config.type, feature_sets, result, spec)
    return result


def _featurize_uncached(entities, entity_config):
    """Generate features for entities based on their type."""
    feature_sets = entity_config.feature_sets or None
    if entity_config.type == "material":
        return compute_material_features(
            entities, feature_sets=feature_sets,
            structure_dir=entity_config.structure_dir,
        )
    elif entity_config.type == "molecule":
        return compute_molecule_features(
            entities,
            feature_sets=feature_sets,
            smiles_map=entity_config.smiles_map,
            chemberta_model=entity_config.chemberta_model,
            chemberta_batch_size=entity_config.chemberta_batch_size,
            n_jobs=entity_config.feature_n_jobs,
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
        # Build lookup map once: {stem: filename}
        file_lookup = {}
        for fname in os.listdir(input_file):
            if fname.endswith(ext_map[fmt]):
                file_lookup[os.path.splitext(fname)[0]] = fname
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            mask = merged["split"] == name
            structure_ids = merged.loc[mask, "structure_id"].unique()
            for sid in structure_ids:
                fname = file_lookup.get(sid)
                if fname:
                    src = os.path.join(input_file, fname)
                    shutil.copy2(src, os.path.join(dest, fname))

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

    elif fmt in ("pdb", "mmcif", "cif"):
        # Single structure file — copy it into each split folder
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            if (merged["split"] == name).any():
                shutil.copy2(input_file, os.path.join(dest,
                             os.path.basename(input_file)))

    elif fmt == "sdf":
        # Write split molecules back as SDF files using RDKit
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        mol_map = {}
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            mol_name = (mol.GetProp("_Name")
                        if mol.HasProp("_Name") and mol.GetProp("_Name").strip()
                        else str(i))
            mol_map[mol_name] = mol

        for name in split_names:
            row_ids = set(merged.loc[merged["split"] == name, "_row_id"])
            out_path = os.path.join(technique_dir, f"{name}.sdf")
            writer = Chem.SDWriter(out_path)
            for rid in row_ids:
                if rid in mol_map:
                    writer.write(mol_map[rid])
            writer.close()

    elif fmt == "sdf_dir":
        # Copy source SDF files into train/test folders
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            subset = merged[merged["split"] == name]
            if "_source_file" in subset.columns:
                src_files = subset["_source_file"].dropna().unique()
            else:
                # Fallback: try _row_id + .sdf
                src_files = [rid + ".sdf" for rid in subset["_row_id"].unique()]
            for fname in src_files:
                src = os.path.join(input_file, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(dest, fname))

    elif fmt == "fasta":
        # Write split sequences back as FASTA files
        seqs = {}  # _row_id → (header_line, sequence)
        cur_id, cur_header, cur_seq = None, None, []
        with open(input_file) as fh:
            for line in fh:
                line_s = line.strip()
                if line_s.startswith(">"):
                    if cur_id is not None:
                        seqs[cur_id] = (cur_header, "".join(cur_seq))
                    parts = line_s[1:].split()
                    cur_id = parts[0] if parts else str(len(seqs))
                    cur_header = line_s
                    cur_seq = []
                else:
                    cur_seq.append(line_s)
            if cur_id is not None:
                seqs[cur_id] = (cur_header, "".join(cur_seq))

        for name in split_names:
            row_ids = merged.loc[merged["split"] == name, "_row_id"]
            out_path = os.path.join(technique_dir, f"{name}.fasta")
            with open(out_path, "w") as fh:
                for rid in row_ids:
                    if rid in seqs:
                        header, seq = seqs[rid]
                        fh.write(header + "\n")
                        for j in range(0, len(seq), 80):
                            fh.write(seq[j:j + 80] + "\n")

    elif fmt in ("smiles",):
        # Write split SMILES back as .smi files
        for name in split_names:
            subset = merged[merged["split"] == name]
            out_path = os.path.join(technique_dir, f"{name}.smi")
            with open(out_path, "w") as fh:
                for _, row in subset.iterrows():
                    fh.write(f"{row['smiles']} {row['_row_id']}\n")

    elif fmt in ("mol", "mol2"):
        # Single-molecule file — copy into each split folder
        for name in split_names:
            dest = os.path.join(technique_dir, name)
            os.makedirs(dest, exist_ok=True)
            if (merged["split"] == name).any():
                shutil.copy2(input_file, os.path.join(dest,
                             os.path.basename(input_file)))

    elif fmt == "ase_db":
        # ASE database — save split subsets as new .db files
        try:
            from ase.db import connect
            src_db = connect(input_file)
            for name in split_names:
                row_ids = set(merged.loc[merged["split"] == name, "_row_id"])
                out_path = os.path.join(technique_dir, f"{name}.db")
                dst_db = connect(out_path)
                for row in src_db.select():
                    if str(row.id - 1) in row_ids or row.get("formula", "") in row_ids:
                        dst_db.write(row.toatoms(), key_value_pairs=row.key_value_pairs)
        except (ImportError, OSError) as exc:
            # Fallback to CSV if ASE write fails
            for name in split_names:
                subset = merged[merged["split"] == name].drop(columns=["split", "_row_id"])
                subset.to_csv(os.path.join(technique_dir, f"{name}.csv"), index=False)


def _save_ml_exports(df, out_df, output_dir, dataset_name, split_names):
    """Save ML-framework-friendly export files.

    Creates:
    - indices.json: split indices for PyTorch/sklearn
    - fold_column.csv: original data with a 'split' column added
    """
    export_dir = os.path.join(output_dir, "ml_exports")
    os.makedirs(export_dir, exist_ok=True)

    merged = df.merge(out_df[["_row_id", "split"]], on="_row_id", how="left")

    # 1. Split indices (for PyTorch Dataset / sklearn)
    indices = {}
    for name in split_names:
        mask = merged["split"] == name
        indices[name] = sorted(merged.index[mask].tolist())

    indices_path = os.path.join(export_dir, f"{dataset_name}_indices.json")
    with open(indices_path, "w") as f:
        json.dump(indices, f, indent=2)

    # 2. Fold column CSV (original data + split assignment)
    display_cols = [c for c in merged.columns if not c.startswith("_")]
    fold_df = merged[display_cols]
    fold_path = os.path.join(export_dir, f"{dataset_name}_with_splits.csv")
    fold_df.to_csv(fold_path, index=False)

    return export_dir


def _target_fractions(splitting):
    total = float(sum(splitting.splits))
    return {n: s / total for n, s in zip(splitting.names, splitting.splits)}


def _validate_split(split_labels, splitting):
    """Check a row-level split against the requested configuration.

    Returns a list of human-readable failure reasons (empty list = valid):
      - rows left unassigned ("not selected"), unless allow_incomplete is set
      - any requested split empty
      - any split's row fraction (among assigned rows) further than
        size_tolerance from its target fraction
    """
    names = list(splitting.names)
    labels = list(split_labels)
    n_rows = len(labels)
    counts = {n: sum(1 for l in labels if l == n) for n in names}
    n_assigned = sum(counts.values())
    reasons = []

    if n_rows == 0:
        return ["no rows to split"]
    n_unassigned = n_rows - n_assigned
    if n_unassigned and not splitting.allow_incomplete:
        reasons.append(
            f"incomplete assignment: {n_unassigned}/{n_rows} rows "
            f"({100 * n_unassigned / n_rows:.1f}%) not assigned to any split "
            f"(set splitting.allow_incomplete to accept partial splits)"
        )
    for n in names:
        if counts[n] == 0:
            reasons.append(f"split '{n}' is empty")
    if n_assigned:
        for n, target in _target_fractions(splitting).items():
            frac = counts[n] / n_assigned
            if abs(frac - target) > splitting.size_tolerance:
                reasons.append(
                    f"split '{n}' has {frac:.1%} of assigned rows, target "
                    f"{target:.1%} (tolerance ±{splitting.size_tolerance:.0%})"
                )
    return reasons


def _random_row_labels(n_rows, splitting, seed):
    """Seeded random row split at the target proportions (largest remainder)."""
    fracs = _target_fractions(splitting)
    raw = {n: fracs[n] * n_rows for n in splitting.names}
    sizes = {n: int(math.floor(v)) for n, v in raw.items()}
    leftover = n_rows - sum(sizes.values())
    for n in sorted(raw, key=lambda k: raw[k] - sizes[k], reverse=True)[:leftover]:
        sizes[n] += 1
    labels = [n for n in splitting.names for _ in range(sizes[n])]
    rng = np.random.default_rng(seed)
    return list(rng.permutation(labels))


def _nan_to_none(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


# Pareto criteria: (name suffix, direction). "max" = higher is better.
_PARETO_DOC = (
    "Non-dominated techniques among VALID splits only. Criteria: coverage (max); "
    "size_dev = max |split fraction - target| (min); per entity axis: "
    "overlap_frac = entities appearing in >1 split / entities seen (min) and "
    "nn_sep = primary-pair mean NN distance divided by that of a seeded random "
    "split of the same rows (max; falls back to the raw mean NN distance, which "
    "is comparable within one dataset, when the random baseline is 0). "
    "Distribution shift is reported but not used for dominance because its "
    "desired direction is task-dependent. Missing metrics are null and that "
    "criterion is skipped for the pair being compared, never treated as 0."
)
_RECOMMENDATION_BASIS = (
    "Among the Pareto set: best mean within-dataset rank of nn_sep over entity "
    "axes (rank 1 = largest separation among valid techniques; ranks keep axes "
    "with relative and raw nn_sep from being mixed); ties broken by lower mean "
    "overlap_frac, then lower size_dev, then name."
)


def _technique_criteria(metrics, targets, baseline):
    """Size-normalized criteria for one technique (NaN when unavailable)."""
    nan = float("nan")
    crit = {"coverage": metrics.get("coverage", nan)}
    fracs = metrics.get("split_fractions") or {}
    devs = [abs(fracs[n] - t) for n, t in targets.items() if n in fracs]
    crit["size_dev"] = max(devs) if devs else nan
    report = {}
    for ax, am in (metrics.get("axes") or {}).items():
        of = am.get("multi_split_frac")
        crit[f"overlap_frac_{ax}"] = nan if of is None else of
        nn = (am.get("nn_leakage") or {}).get("mean_nn_dist")
        base = (baseline or {}).get(ax)
        if nn is None:
            crit[f"nn_sep_{ax}"] = nan
            report[f"nn_rel_{ax}"] = nan
        elif base:
            crit[f"nn_sep_{ax}"] = nn / base
            report[f"nn_rel_{ax}"] = nn / base
        else:
            crit[f"nn_sep_{ax}"] = nn
            report[f"nn_rel_{ax}"] = nan
        report[f"nn_mean_dist_{ax}"] = nan if nn is None else nn
        ds = (am.get("distribution_shift") or {}).get("mean_normalized_shift")
        report[f"dist_shift_{ax}"] = nan if ds is None else ds
    return crit, report


def _direction(criterion):
    return "max" if criterion == "coverage" or criterion.startswith("nn_sep_") else "min"


def _dominates(a, b):
    """True if criteria dict ``a`` Pareto-dominates ``b`` (NaNs skipped)."""
    comparable = False
    strictly = False
    for c, va in a.items():
        vb = b.get(c, float("nan"))
        if math.isnan(va) or math.isnan(vb):
            continue
        comparable = True
        better, worse = (va > vb, va < vb) if _direction(c) == "max" else (va < vb, va > vb)
        if worse:
            return False
        strictly = strictly or better
    return comparable and strictly


def _build_comparison_summary(all_metrics, split_names, failed=None,
                              targets=None, baseline=None):
    """Build a comparison summary across all techniques.

    Invalid techniques (``failed``: {technique: [reasons]}) are listed but never
    ranked. Valid techniques get a Pareto table over size-normalized criteria
    (see ``pareto_criteria`` in the output); a single ``recommendation`` is
    kept for API compatibility and derived from the Pareto set by the
    documented ``recommendation_basis``.
    """
    failed = failed or {}
    targets = targets or {}
    summary = {
        "status": {**{t: "valid" for t in all_metrics},
                   **{t: "failed" for t in failed}},
        "failed": failed,
        "random_baseline_nn_mean_dist": baseline,
        "pareto_criteria": _PARETO_DOC,
        "techniques": {},
        "pareto_set": [],
        "recommendation": None,
        "recommendation_basis": _RECOMMENDATION_BASIS,
    }

    criteria = {}
    for technique, metrics in all_metrics.items():
        crit, report = _technique_criteria(metrics, targets, baseline)
        criteria[technique] = crit
        summary["techniques"][technique] = {
            k: _nan_to_none(round(v, 6) if isinstance(v, float) else v)
            for k, v in {**crit, **report}.items()
        }

    pareto = [t for t in criteria
              if not any(_dominates(criteria[o], criteria[t]) for o in criteria if o != t)]
    for t in summary["techniques"]:
        summary["techniques"][t]["pareto_optimal"] = t in pareto
    summary["pareto_set"] = sorted(pareto)

    def _mean(vals):
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    # Within-dataset rank of nn_sep per axis (1 = best; ties share the best rank)
    nn_rank = {t: [] for t in criteria}
    nn_axes = {c for crit in criteria.values() for c in crit if c.startswith("nn_sep_")}
    for c in nn_axes:
        vals = {t: criteria[t].get(c, float("nan")) for t in criteria}
        for t, v in vals.items():
            if not math.isnan(v):
                nn_rank[t].append(1 + sum(1 for o in vals.values()
                                          if not math.isnan(o) and o > v))

    def _rank_key(t):
        rk = _mean(nn_rank[t])
        ov = _mean([v for c, v in criteria[t].items() if c.startswith("overlap_frac_")])
        sd = criteria[t]["size_dev"]
        return (rk if not math.isnan(rk) else math.inf,
                ov if not math.isnan(ov) else math.inf,
                sd if not math.isnan(sd) else math.inf,
                t)

    if pareto:
        summary["recommendation"] = min(pareto, key=_rank_key)

    return summary


def _run_external(names, ctx, failures, report):
    """Run registered non-DataSAIL techniques; failures go into ``failures``."""
    results = {}
    for name in names:
        t0 = time.time()
        report(f"  Running {name}...")
        try:
            results[name] = EXTERNAL_TECHNIQUES[name].fn(ctx)
            report(f"  {name} finished in {time.time() - t0:.1f}s")
        except Exception as exc:
            logger.warning(f"{name} failed: {exc}", exc_info=True)
            report(f"  Warning: {name} failed: {exc}")
            failures[name] = f"{name} error: {exc}"
    return results


def run_pipeline(config: PipelineConfig, progress_callback=None):
    """Run the full pipeline: load -> featurize -> split -> save.

    Supports both 1D (single entity) and 2D (interaction) splitting.
    When config.e2 is None, runs in 1D mode.

    Args:
        config: PipelineConfig object.
        progress_callback: optional callable(percent: int, message: str).
            Called at each major step so callers (e.g. the web UI) can track
            progress. The pipeline still prints to stdout regardless.

    Returns:
        dict with "valid" (list of techniques whose split passed validation),
        "failed" ({technique: [reasons]}) and "comparison" (the summary saved
        to metrics/comparison_<dataset>.json).

    Raises:
        RuntimeError if no technique produced a valid split.
    """
    is_1d = config.e2 is None

    # Seed global RNGs (see SplittingConfig.seed for what this does and does
    # not control — DataSAIL reseeds internally).
    random.seed(config.splitting.seed)
    np.random.seed(config.splitting.seed)

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

    # Auto-populate structure_dir for material entities when using structure dirs
    fmt = _detect_format(config.input_file)
    if fmt in ("cif_dir", "xyz_dir"):
        if config.e1.type == "material" and config.e1.structure_dir is None:
            config.e1.structure_dir = config.input_file
        if config.e2 is not None and config.e2.type == "material" and config.e2.structure_dir is None:
            config.e2.structure_dir = config.input_file

    # 2. Extract entities
    _report(20, f"\n[2/5] Extracting entities")
    e1_entities = _extract_entities(df, config.e1)
    _report(25, f"  {config.e1.name}: {len(e1_entities)} unique entities")

    if not is_1d:
        e2_entities = _extract_entities(df, config.e2)
        _report(28, f"  {config.e2.name}: {len(e2_entities)} unique entities")

    # 3. Generate features
    _report(30, f"\n[3/5] Generating features")
    _report(32, f"  Featurizing {config.e1.name} ({len(e1_entities)} entities)...")
    _report(33, f"  Feature sets: {', '.join(config.e1.feature_sets or ['default'])}")
    e1_feat_df = _featurize(e1_entities, config.e1)
    _report(50, f"  {config.e1.name} features: {e1_feat_df.shape[0]} entities × {e1_feat_df.shape[1]} features")

    # Save e1 features
    feat_dir = os.path.join(config.output_dir, "features", config.dataset_name)
    os.makedirs(os.path.join(feat_dir, config.e1.name), exist_ok=True)
    e1_feat_path = os.path.join(feat_dir, config.e1.name, "features.csv")
    e1_feat_df.to_csv(e1_feat_path)
    _report(55, f"  Saved: {e1_feat_path}")

    if not is_1d:
        _report(57, f"  Featurizing {config.e2.name} ({len(e2_entities)} entities)...")
        _report(58, f"  Feature sets: {', '.join(config.e2.feature_sets or ['default'])}")
        e2_feat_df = _featurize(e2_entities, config.e2)
        _report(65, f"  {config.e2.name} features: {e2_feat_df.shape[0]} entities × {e2_feat_df.shape[1]} features")

        os.makedirs(os.path.join(feat_dir, config.e2.name), exist_ok=True)
        e2_feat_path = os.path.join(feat_dir, config.e2.name, "features.csv")
        e2_feat_df.to_csv(e2_feat_path)
        _report(70, f"  Saved: {e2_feat_path}")

    # 4. Run DataSAIL splitting
    _report(72, f"\n[4/5] Running splitting")
    external = [t for t in config.splitting.techniques if t in EXTERNAL_TECHNIQUES]
    e1_col = config.e1.extract_column

    # Replace NaN/Inf with 0 to prevent downstream clustering errors
    e1_feat_df = e1_feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Convert the whole frame once instead of a per-row .loc lookup.
    _e1_matrix = e1_feat_df.to_numpy(dtype=float)
    e1_data = {name: row for name, row in zip(e1_feat_df.index, _e1_matrix)}

    solver_failures = {}
    row_ids = df["_row_id"].astype(str).tolist()
    e1_ids = df[e1_col].astype(str).tolist()
    if is_1d:
        # 1D mode: split by entity only
        _report(75, f"  Entities: {len(e1_data)} unique")

        # Separate scaffold technique from DataSAIL techniques
        datasail_techniques = [t for t in config.splitting.techniques
                               if t != "scaffold" and t not in EXTERNAL_TECHNIQUES]
        has_scaffold = "scaffold" in config.splitting.techniques

        split_results = {}
        if datasail_techniques:
            from copy import copy
            ds_config = copy(config.splitting)
            ds_config.techniques = datasail_techniques
            try:
                split_results = run_splitting_1d(e1_data, ds_config, failures=solver_failures)
            except RuntimeError:
                # every DataSAIL technique failed (each recorded in solver_failures);
                # still run the other techniques before deciding the run failed
                if not (external or has_scaffold):
                    raise

        if has_scaffold and config.e1.type == "molecule":
            try:
                scaffold_result = run_scaffold_splitting(e1_entities, config.splitting)
                split_results["scaffold"] = scaffold_result
                _report(78, "  Scaffold splitting complete")
            except Exception as exc:
                logger.warning(f"Scaffold splitting failed: {exc}")
                _report(78, f"  Warning: scaffold splitting failed: {exc}")
                solver_failures["scaffold"] = f"scaffold splitting error: {exc}"
        if external:
            ctx = SplitContext(True, e1_data, None, None, e1_entities,
                               config.e1, None, config.splitting)
            split_results.update(_run_external(
                external, ctx, solver_failures, lambda m: _report(80, m)))
        row_keys = e1_ids
        axes_spec = [("e1", config.e1.name, e1_data, e1_ids)]
    else:
        # 2D mode: split by interactions
        e2_col = config.e2.extract_column
        e2_ids = df[e2_col].astype(str).tolist()
        interactions = list(zip(e1_ids, e2_ids))

        _report(75, f"  Interactions: {len(interactions)} (unique pairs: {len(set(interactions))})")

        e2_feat_df = e2_feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        _e2_matrix = e2_feat_df.to_numpy(dtype=float)
        e2_data = {name: row for name, row in zip(e2_feat_df.index, _e2_matrix)}

        datasail_techniques = [t for t in config.splitting.techniques
                               if t not in EXTERNAL_TECHNIQUES]
        split_results = {}
        if datasail_techniques:
            from copy import copy
            ds_config = copy(config.splitting)
            ds_config.techniques = datasail_techniques
            try:
                split_results = run_splitting(e1_data, e2_data, interactions, ds_config,
                                              failures=solver_failures)
            except RuntimeError:
                if not external:
                    raise
        if external:
            ctx = SplitContext(False, e1_data, e2_data, interactions, e1_entities,
                               config.e1, config.e2, config.splitting)
            split_results.update(_run_external(
                external, ctx, solver_failures, lambda m: _report(80, m)))
        row_keys = interactions
        axes_spec = [("e1", config.e1.name, e1_data, e1_ids),
                     ("e2", config.e2.name, e2_data, e2_ids)]

    _report(90, "  Splitting complete")

    # 5. Map splits back to row IDs, validate, and save
    _report(92, f"\n[5/5] Saving split results")
    split_dir = os.path.join(config.output_dir, "split_result")
    os.makedirs(split_dir, exist_ok=True)
    failed_dir = os.path.join(split_dir, "failed")

    all_technique_metrics = {}
    failed = {t: [msg] for t, msg in solver_failures.items()}
    split_names = config.splitting.names
    axis_names = {"e1": config.e1.name}
    if not is_1d:
        axis_names["e2"] = config.e2.name

    # Random-split baseline on the same rows, for size-normalized NN separation
    baseline = None
    try:
        rand_labels = _random_row_labels(len(row_ids), config.splitting,
                                         config.splitting.seed)
        rand_metrics = compute_row_split_metrics(axes_spec, rand_labels, split_names)
        baseline = {ax: (am.get("nn_leakage") or {}).get("mean_nn_dist")
                    for ax, am in rand_metrics["axes"].items()}
    except Exception as exc:
        logger.warning(f"Random baseline computation failed: {exc}")

    # Pre-compute dimensionality reduction once for all techniques
    _viz_coords = None
    try:
        _sorted_names = sorted(e1_data.keys())
        _X_viz = np.array([e1_data[n] for n in _sorted_names])
        if _X_viz.shape[0] >= 3:
            _reduced = _reduce(_X_viz, "tsne", seed=config.splitting.seed)
            if _reduced is not None:
                _viz_coords = (_sorted_names, _reduced)
    except (ValueError, ImportError):
        pass

    for technique in sorted(split_results.keys()):
        result = split_results[technique]

        labels = [result.get(k, "not selected") for k in row_keys]
        out_df = pd.DataFrame({"_row_id": row_ids, "split": labels})

        summary = [f"\n  --- {technique} ---"]
        for name in split_names:
            n = labels.count(name)
            pct = 100 * n / len(df)
            summary.append(f"    {name.capitalize()}: {n:,} ({pct:.1f}%)")

        n_not = labels.count("not selected")
        if n_not > 0:
            summary.append(f"    Not selected: {n_not:,}")

        # Validate before publishing anything: incomplete / empty / off-size
        # splits are marked failed, kept only under split_result/failed/ for
        # inspection, and excluded from exports, metrics and recommendation.
        reasons = _validate_split(labels, config.splitting)
        if reasons:
            failed[technique] = reasons
            os.makedirs(failed_dir, exist_ok=True)
            fail_csv = os.path.join(
                failed_dir, f"datasail_split_{technique}_{config.dataset_name}.csv")
            out_df.to_csv(fail_csv, index=False)
            with open(os.path.join(failed_dir,
                                   f"{technique}_{config.dataset_name}.json"), "w") as fh:
                json.dump({"technique": technique, "status": "failed",
                           "reasons": reasons,
                           "split_counts": {n: labels.count(n) for n in split_names},
                           "not_selected": n_not, "n_rows": len(labels)}, fh, indent=2)
            summary.append(f"    FAILED (invalid split, not exported):")
            summary.extend(f"      - {r}" for r in reasons)
            summary.append(f"    Raw assignment kept for inspection: {fail_csv}")
            for line in summary:
                _report(95, line)
            continue

        out_path = os.path.join(split_dir, f"datasail_split_{technique}_{config.dataset_name}.csv")
        out_df.to_csv(out_path, index=False)
        summary.append(f"    Saved to {out_path}")

        # Save format-specific split data (train/test CSV, JSON, or structure folders)
        try:
            _save_split_data(df, out_df, split_dir, technique,
                             config.dataset_name, config.input_file,
                             fmt, split_names)
            data_dir = os.path.join(split_dir, f"{technique}_{config.dataset_name}")
            summary.append(f"    Split data saved to {data_dir}/")
        except Exception as exc:
            logger.warning(f"Could not save split data files: {exc}")
            summary.append(f"    Warning: could not save split data files: {exc}")

        # Save ML-friendly exports (indices + fold column)
        try:
            ml_dir = _save_ml_exports(df, out_df, config.output_dir,
                                      f"{technique}_{config.dataset_name}",
                                      split_names)
            summary.append(f"    ML exports saved to {ml_dir}/")
        except Exception as exc:
            logger.warning(f"ML export failed: {exc}")
            summary.append(f"    Warning: ML export failed: {exc}")

        # Compute split quality metrics directly on the row-level assignment,
        # per entity axis (no per-entity majority vote).
        split_metrics = None
        try:
            split_metrics = compute_row_split_metrics(axes_spec, labels, split_names)
            split_metrics["status"] = "valid"
            save_metrics(split_metrics, config.output_dir, technique, config.dataset_name)
            all_technique_metrics[technique] = split_metrics
            for ax, am in split_metrics["axes"].items():
                pair_txt = ", ".join(f"{p}={v['count']}" for p, v in am["overlap_pairs"].items())
                summary.append(f"    {axis_names[ax]} in >1 split: {am['multi_split_count']}"
                               f"/{am['n_entities']} ({pair_txt})")
                nn = am.get("nn_leakage", {})
                if nn:
                    summary.append(f"    {axis_names[ax]} NN leakage: {nn.get('zero_dist_count', 0)} "
                                   f"zero-dist (mean dist: {nn.get('mean_nn_dist', 'N/A')})")
                ds = am.get("distribution_shift", {})
                if ds:
                    summary.append(f"    {axis_names[ax]} distribution shift: "
                                   f"{ds.get('mean_normalized_shift', 'N/A')} "
                                   f"(max: {ds.get('max_normalized_shift', 'N/A')})")
            summary.append(f"    Coverage: {split_metrics['coverage'] * 100:.1f}%")
        except Exception as exc:
            logger.warning(f"Metrics computation failed: {exc}")
            summary.append(f"    Warning: metrics computation failed: {exc}")
            # Still a valid split; its criteria are reported as missing (null).
            all_technique_metrics[technique] = {"status": "valid"}

        # Visualization of e1 entities. In 2D an entity whose interactions
        # span several splits is labelled "multiple" rather than hidden.
        try:
            if is_1d:
                entity_splits = result
            else:
                seen = {}
                for eid, lab in zip(e1_ids, labels):
                    if lab in split_names:
                        seen.setdefault(eid, set()).add(lab)
                entity_splits = {eid: (next(iter(labs)) if len(labs) == 1 else "multiple")
                                 for eid, labs in seen.items()}
            plot_path = generate_split_plots(
                e1_data, entity_splits, config.output_dir,
                config.dataset_name, config.e1.name, technique,
                split_names=split_names,
                precomputed_coords=_viz_coords,
                seed=config.splitting.seed,
            )
            if plot_path:
                summary.append(f"    Plot: {plot_path}")
        except Exception as exc:
            logger.warning(f"Visualization failed: {exc}")
            summary.append(f"    Warning: visualization failed: {exc}")

        for line in summary:
            _report(95, line)

    # Generate comparison chart across valid techniques
    if all_technique_metrics:
        try:
            chart_path = generate_comparison_chart(
                all_technique_metrics, config.output_dir, config.dataset_name,
            )
            if chart_path:
                _report(98, f"  Comparison chart: {chart_path}")
        except Exception as exc:
            logger.warning(f"Comparison chart failed: {exc}")
            _report(98, f"  Warning: comparison chart failed: {exc}")

    # Comparison summary: validity status for every technique + Pareto table
    summary_data = _build_comparison_summary(
        all_technique_metrics, split_names, failed=failed,
        targets=_target_fractions(config.splitting), baseline=baseline,
    )
    summary_dir = os.path.join(config.output_dir, "metrics")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"comparison_{config.dataset_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    def _fmt(v, pct=False):
        if v is None:
            return "N/A"
        return f"{v * 100:.1f}%" if pct else f"{v:.4f}"

    _report(99, "\n  === Technique Comparison (valid splits) ===")
    axes_keys = list(axis_names)
    header = f"  {'Technique':<12} {'Pareto':>6} {'SizeDev':>8}"
    for ax in axes_keys:
        header += f" {ax + ' ovl':>9} {ax + ' nnRel':>9}"
    _report(99, header)
    for t_name, t_data in summary_data["techniques"].items():
        line = (f"  {t_name:<12} {'*' if t_data['pareto_optimal'] else '':>6} "
                f"{_fmt(t_data.get('size_dev'), pct=True):>8}")
        for ax in axes_keys:
            line += (f" {_fmt(t_data.get(f'overlap_frac_{ax}'), pct=True):>9}"
                     f" {_fmt(t_data.get(f'nn_rel_{ax}')):>9}")
        _report(99, line)
    for t_name, reasons in failed.items():
        _report(99, f"  {t_name:<12} FAILED: {'; '.join(reasons)}")
    if summary_data.get("recommendation"):
        _report(99, f"\n  Recommended: {summary_data['recommendation']} "
                    f"({summary_data['recommendation_basis']})")
    _report(99, f"  Comparison saved to {summary_path}")

    if not all_technique_metrics:
        raise RuntimeError(
            "No technique produced a valid split: "
            + "; ".join(f"{t}: {' / '.join(r)}" for t, r in failed.items())
        )

    _report(100, f"\nDone! Results saved to {config.output_dir}/")
    return {"valid": sorted(all_technique_metrics), "failed": failed,
            "comparison": summary_data}
