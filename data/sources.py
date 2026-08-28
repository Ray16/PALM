"""Dataset registry: name -> loader returning a featurized ``DatasetBundle``.

Each loader is self-describing about availability: if the raw data or a required
credential is missing it returns ``available=False`` with a ``reason`` instead of
raising, so the splitter-wiring driver can skip it cleanly and report why.

Categories: ``organic`` (small molecules), ``inorganic`` (crystals / MOFs),
``reaction`` (multi-component), ``polymer``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .featurize import composition_matrix, ecfp_matrix

HERE = os.path.dirname(__file__)
DSAIL = os.path.join(HERE, "DataSAIL_data")

# Cap for very large sources (per the project's disk budget). Small sets keep all.
DEFAULT_LIMIT = 10_000


@dataclass
class DatasetBundle:
    name: str
    category: str                 # organic | inorganic | reaction | polymer
    kind: str                     # "1d" (feature vectors) | "nd" (records)
    available: bool
    reason: str = ""
    feature_data: Optional[dict] = None          # 1d: {id: vector}
    smiles: Optional[dict] = None                # 1d optional: {id: SMILES} (scaffold)
    records: Optional[list] = None               # nd
    axis_feature_maps: Optional[dict] = None     # nd
    targets: Optional[dict] = None               # {id: y} for the generalization-gap layer
    task_type: Optional[str] = None              # "regression" | "classification" | None
    target_name: Optional[str] = None            # source column of the target
    identifiers: Optional[dict] = None           # {id: raw str} — SMILES/formula/sequence for re-featurization + routing
    identifier_kind: Optional[str] = None        # "smiles" | "formula" | "protein" | "nucleotide"
    entity_type: Optional[str] = None            # explicit routing type: molecule|material|mof|polymer|protein|gene
    meta: dict = field(default_factory=dict)


def _unavailable(name, category, kind, reason):
    return DatasetBundle(name, category, kind, available=False, reason=reason)


def _subsample_indices(n, limit, seed=0):
    if limit is None or n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


# ── organic small molecules (SMILES -> ECFP) ───────────────────────────────

def _smiles_csv_bundle(name, category, path, smiles_col, limit,
                       target_col=None, task_type=None):
    if not os.path.exists(path):
        return _unavailable(name, category, "1d", f"file not found: {path}")
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        cand = [c for c in df.columns if c.lower() in ("smiles", "drug", "canonical_smiles", "mol")]
        if not cand:
            return _unavailable(name, category, "1d", f"no SMILES column in {path}")
        smiles_col = cand[0]
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    idx = _subsample_indices(len(df), limit)
    smi = [str(df[smiles_col].iloc[i]).strip() for i in idx]
    kept, X = ecfp_matrix(smi)
    ids = [int(idx[k]) for k in kept]
    feature_data = {i: X[j] for j, i in enumerate(ids)}
    smiles = {i: smi[kept[j]] for j, i in enumerate(ids)}
    # optional target for the generalization-gap layer; NaN-coerced (missing
    # labels are kept as NaN and masked out by the model step, not dropped here,
    # so the split still sees every entity).
    targets = None
    if target_col and target_col in df.columns:
        yv = pd.to_numeric(df[target_col], errors="coerce")
        targets = {i: float(yv.iloc[i]) for i in ids}
    return DatasetBundle(name, category, "1d", True, feature_data=feature_data,
                         smiles=smiles, identifiers=dict(smiles), identifier_kind="smiles",
                         targets=targets,
                         task_type=task_type if targets else None,
                         target_name=target_col if targets else None,
                         meta={"n": len(ids), "source_rows": len(df)})


# canonical (target column, task type) per MoleculeNet set; multitask sets
# (tox21/sider/muv/qm8) use one representative column so the suite stays 1-target.
_MNET_TARGETS = {
    "bace":          ("Class", "classification"),
    "bbbp":          ("p_np", "classification"),
    "esol":          ("measured log solubility in mols per litre", "regression"),
    "freesolv":      ("expt", "regression"),
    "lipophilicity": ("exp", "regression"),
    "clintox":       ("CT_TOX", "classification"),
    "hiv":           ("HIV_active", "classification"),
    "tox21":         ("NR-AR", "classification"),
    "sider":         ("Hepatobiliary disorders", "classification"),
    "muv":           ("MUV-466", "classification"),
    "qm8":           ("E1-CC2", "regression"),
}


def load_moleculenet(sub, limit=DEFAULT_LIMIT):
    path = os.path.join(DSAIL, "1D", "moleculenet", f"{sub}.csv")
    tcol, ttype = _MNET_TARGETS.get(sub, (None, None))
    return _smiles_csv_bundle(f"moleculenet_{sub}", "organic", path, "smiles", limit,
                              target_col=tcol, task_type=ttype)


def load_tdc(sub, limit=DEFAULT_LIMIT):
    path = os.path.join(HERE, "tdc", f"{sub}.csv")
    if not os.path.exists(path):
        return _unavailable(
            f"tdc_{sub}", "organic", "1d",
            "TDC data not downloaded (Harvard Dataverse was under maintenance); "
            "run `python -m PALM.data.prepare_tdc` when the host is back")
    return _smiles_csv_bundle(f"tdc_{sub}", "organic", path, "smiles", limit)


# ── inorganic crystals / MOFs (formula -> MAGPIE composition) ───────────────

def load_qmof(limit=None):
    """QMOF: all ~20k MOFs, MAGPIE composition features from the reduced formula."""
    path = os.path.join(HERE, "qmof", "qmof.csv")
    if not os.path.exists(path):
        return _unavailable("qmof", "inorganic", "1d", "qmof.csv not found")
    df = pd.read_csv(path, low_memory=False)
    fcol = "info.formula_reduced" if "info.formula_reduced" in df.columns else "info.formula"
    idcol = "qmof_id" if "qmof_id" in df.columns else df.columns[0]
    df = df.dropna(subset=[fcol]).reset_index(drop=True)
    idx = _subsample_indices(len(df), limit)
    formulas = {str(df[idcol].iloc[i]): str(df[fcol].iloc[i]) for i in idx}
    row_of_id = {str(df[idcol].iloc[i]): i for i in idx}
    ids, X = composition_matrix(formulas)
    feature_data = {qid: X[j] for j, qid in enumerate(ids)}
    # PBE band gap -> regression target for the generalization-gap layer
    targets = None
    tcol = "outputs.pbe.bandgap"
    if tcol in df.columns:
        yv = pd.to_numeric(df[tcol], errors="coerce")
        targets = {qid: float(yv.iloc[row_of_id[qid]]) for qid in ids}
    # linker SMILES (stored as a stringified list, e.g. "['smi1','smi2']") — parse
    # and take the first linker; enables the scaffold splitter + the mof linker_ecfp
    # featurizer candidate.
    smiles = None
    scol = "info.mofid.smiles_linkers"
    if scol in df.columns:
        import ast
        by_id = {str(df[idcol].iloc[i]): df[scol].iloc[i] for i in idx}

        def _first_linker(v):
            if not isinstance(v, str) or not v.strip():
                return None
            try:
                lst = ast.literal_eval(v)
                v = lst[0] if isinstance(lst, (list, tuple)) and lst else v
            except (ValueError, SyntaxError):
                pass
            return str(v).split(".")[0] or None

        smiles = {qid: _first_linker(by_id.get(qid)) for qid in ids}
        smiles = {k: v for k, v in smiles.items() if v} or None
    return DatasetBundle("qmof", "inorganic", "1d", True, feature_data=feature_data,
                         smiles=smiles, targets=targets,
                         identifiers={qid: formulas[qid] for qid in ids}, identifier_kind="formula",
                         entity_type="mof",
                         task_type="regression" if targets else None,
                         target_name=tcol if targets else None,
                         meta={"n": len(ids), "formula_col": fcol,
                               "linker_smiles": smiles or {}})   # for the mof linker_ecfp candidate


def load_omol25(limit=DEFAULT_LIMIT):
    """OMol25: subsample of the cached 115-d structural descriptors (9.55M total)."""
    cache = os.path.join(DSAIL, "1D", "omol25", "_cache")
    feat_path = os.path.join(cache, "features.npy")
    meta_path = os.path.join(cache, "meta.parquet")
    if not (os.path.exists(feat_path) and os.path.exists(meta_path)):
        return _unavailable("omol25", "inorganic", "1d",
                            "OMol25 descriptor cache not found (_cache/features.npy)")
    feats = np.load(feat_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path, columns=["db_id"]) if os.path.exists(meta_path) else None
    idx = _subsample_indices(feats.shape[0], limit)
    X = np.asarray(feats[idx], dtype=np.float32)
    if meta is not None:
        ids = [str(meta["db_id"].iloc[int(i)]) for i in idx]
        # de-dup ids defensively
        seen = {}
        ids = [seen.setdefault(x, x) if x not in seen else f"{x}__{i}" for i, x in enumerate(ids)]
    else:
        ids = [f"omol_{int(i)}" for i in idx]
    feature_data = {ids[j]: X[j] for j in range(len(ids))}
    return DatasetBundle("omol25", "inorganic", "1d", True, feature_data=feature_data,
                         meta={"n": len(ids), "dim": int(X.shape[1]), "total": int(feats.shape[0])})


def _mp_bundle_from_formulas(formulas, energy):
    """Build the materials_project DatasetBundle from formula/energy dicts."""
    ids, X = composition_matrix(formulas)
    feature_data = {mid: X[j] for j, mid in enumerate(ids)}
    # formation energy per atom -> regression target
    targets = {mid: float(energy[mid]) for mid in ids if energy.get(mid) is not None}
    targets = targets or None
    return DatasetBundle("materials_project", "inorganic", "1d", True,
                         feature_data=feature_data, targets=targets,
                         identifiers={mid: formulas[mid] for mid in ids}, identifier_kind="formula",
                         task_type="regression" if targets else None,
                         target_name="formation_energy_per_atom" if targets else None,
                         meta={"n": len(ids)})


def load_materials_project(limit=DEFAULT_LIMIT):
    """Materials Project composition dataset -> MAGPIE composition features.

    Prefers the on-disk snapshot at ``materials_project/summary.csv`` (reproducible,
    offline; created by ``download_materials_project.py``). Falls back to a live
    ``mp-api`` pull (needs ``MP_API_KEY``) when the snapshot is absent.
    """
    csv_path = os.path.join(HERE, "materials_project", "summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if limit and len(df) > limit:
            df = df.sample(limit, random_state=0).reset_index(drop=True)
        formulas = {str(r.material_id): str(r.formula_pretty) for r in df.itertuples()}
        energy = {str(r.material_id): getattr(r, "formation_energy_per_atom", None)
                  for r in df.itertuples()}
        return _mp_bundle_from_formulas(formulas, energy)

    key = os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        return _unavailable(
            "materials_project", "inorganic", "1d",
            "MP_API_KEY not set — get a free key at https://materialsproject.org/api "
            "and export MP_API_KEY, then this loader pulls <=%d summaries" % (limit or DEFAULT_LIMIT))
    try:
        from mp_api.client import MPRester
    except Exception:
        return _unavailable("materials_project", "inorganic", "1d",
                            "mp-api not installed (`pip install mp-api`)")
    import math
    want = limit or DEFAULT_LIMIT
    per = 1000                                   # MP caps chunk_size at 1000
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            fields=["material_id", "formula_pretty", "formation_energy_per_atom"],
            num_chunks=max(1, math.ceil(want / per)), chunk_size=per)
    docs = docs[:want]
    formulas = {str(d.material_id): str(d.formula_pretty) for d in docs}
    energy = {str(d.material_id): getattr(d, "formation_energy_per_atom", None) for d in docs}
    return _mp_bundle_from_formulas(formulas, energy)


# ── reactions (n-D) ─────────────────────────────────────────────────────────

def load_uspto_mcr(limit=DEFAULT_LIMIT):
    """USPTO 3-reactant reactions -> (records, per-axis Morgan feature maps)."""
    recs_path = os.path.join(DSAIL, "3D+", "uspto_mcr", "records.csv")
    if not os.path.exists(recs_path):
        return _unavailable("uspto_mcr", "reaction", "nd", "uspto_mcr/records.csv not found")
    from PALM import reactions as R
    records, afm, _target = R.load_uspto_mcr()
    idx = _subsample_indices(len(records), limit)
    records = [records[int(i)] for i in idx]
    return DatasetBundle("uspto_mcr", "reaction", "nd", True, records=records,
                         axis_feature_maps=afm, meta={"n": len(records), "axes": list(afm)})


# ── polymers ────────────────────────────────────────────────────────────────

def load_openpolymer26(limit=DEFAULT_LIMIT):
    """Open Polymers 2026 (OPoly26): 10k polymer-cluster subsample (train split).

    Atomistic records (no SMILES) featurized by ``formula`` -> MAGPIE composition.
    Prepared by ``python -m PALM.data.prepare_openpolymer26`` (streams 10k rows
    from the colabfit/OPoly26-train parquet shard).
    """
    path = os.path.join(HERE, "openpolymer26", "records.csv")
    if not os.path.exists(path):
        return _unavailable(
            "openpolymer26", "polymer", "1d",
            "OPoly26 not prepared — run `python -m PALM.data.prepare_openpolymer26` "
            "(streams from colabfit/OPoly26-train; arXiv:2512.23117)")
    df = pd.read_csv(path)
    if "smiles" in df.columns:
        return _smiles_csv_bundle("openpolymer26", "polymer", path, "smiles", limit)
    df = df.dropna(subset=["formula"]).reset_index(drop=True)
    idx = _subsample_indices(len(df), limit)
    formulas = {str(df["id"].iloc[i]): str(df["formula"].iloc[i]) for i in idx}
    row_of_id = {str(df["id"].iloc[i]): i for i in idx}
    ids, X = composition_matrix(formulas)
    feature_data = {pid: X[j] for j, pid in enumerate(ids)}
    # per-cluster DFT energy (records column ``y``) -> regression target
    targets = None
    if "y" in df.columns:
        yv = pd.to_numeric(df["y"], errors="coerce")
        targets = {pid: float(yv.iloc[row_of_id[pid]]) for pid in ids}
    return DatasetBundle("openpolymer26", "polymer", "1d", True,
                         feature_data=feature_data, targets=targets,
                         identifiers={pid: formulas[pid] for pid in ids}, identifier_kind="formula",
                         entity_type="polymer",
                         task_type="regression" if targets else None,
                         target_name="y" if targets else None,
                         meta={"n": len(ids)})


# ── proteins (sequence -> ESM2 / sequence properties) ──────────────────────

def _feat_df_to_map(df):
    """compute_*_features DataFrame -> {id: float32 vector} (NaN-filled)."""
    df = df.fillna(0.0)
    return {idx: np.asarray(df.loc[idx].values, dtype=np.float32) for idx in df.index}


def load_lp_pdbbind(limit=DEFAULT_LIMIT):
    """LP-PDBBind proteins: unique sequences -> mean binding affinity (regression).

    Entity = protein sequence (deduped; affinity averaged over its ligands, so the
    target carries residual ligand-dependent noise). Load-time features are the
    cheap ``sequence_properties`` set; the router re-featurizes with ESM2 when asked.
    """
    path = os.path.join(DSAIL, "2D", "lp_pdbbind", "LP_PDBBind.csv")
    if not os.path.exists(path):
        return _unavailable("lp_pdbbind", "protein", "1d", f"file not found: {path}")
    df = pd.read_csv(path)
    if "seq" not in df.columns:
        return _unavailable("lp_pdbbind", "protein", "1d", "no 'seq' column")
    df = df.dropna(subset=["seq"]).copy()
    df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
    agg = df.groupby("seq")["value"].mean().reset_index().dropna(subset=["value"])
    idx = _subsample_indices(len(agg), limit)
    seqs = {f"p{int(i)}": str(agg["seq"].iloc[int(i)]) for i in idx}
    tgt = {f"p{int(i)}": float(agg["value"].iloc[int(i)]) for i in idx}
    from PALM.features.biomolecule_features import compute_biomolecule_features
    fmap = _feat_df_to_map(compute_biomolecule_features(seqs, feature_sets=["sequence_properties"]))
    ids = [i for i in seqs if i in fmap]
    return DatasetBundle("lp_pdbbind", "protein", "1d", True,
                         feature_data={i: fmap[i] for i in ids},
                         identifiers={i: seqs[i] for i in ids}, identifier_kind="protein",
                         entity_type="protein",
                         targets={i: tgt[i] for i in ids}, task_type="regression",
                         target_name="binding_affinity", meta={"n": len(ids)})


# ── RNA / nucleotide (sequence -> k-mer / Nucleotide Transformer) ───────────

def _parse_fasta(path):
    recs, hdr, seq = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    recs.append((hdr, "".join(seq)))
                hdr, seq = line[1:], []
            elif line:
                seq.append(line)
    if hdr is not None:
        recs.append((hdr, "".join(seq)))
    return recs


def load_rfam(limit=DEFAULT_LIMIT):
    """Rfam RNA: sequence -> family class (13-way classification).

    Class is the leading ``RFxxxxx`` accession in the FASTA header. Load-time
    features are cheap canonical k-mer frequencies; router can swap in the
    Nucleotide Transformer.
    """
    path = os.path.join(DSAIL, "1D", "rfam_rna", "dataset_Rfam_6320_13classes.fasta")
    if not os.path.exists(path):
        return _unavailable("rfam", "gene", "1d", f"file not found: {path}")
    recs = _parse_fasta(path)
    if not recs:
        return _unavailable("rfam", "gene", "1d", "no sequences parsed")
    fams = sorted({h.split("_")[0] for h, _ in recs})
    fam_idx = {f: i for i, f in enumerate(fams)}
    idx = _subsample_indices(len(recs), limit)
    seqs = {f"r{int(i)}": recs[int(i)][1].replace("U", "T") for i in idx}
    tgt = {f"r{int(i)}": fam_idx[recs[int(i)][0].split("_")[0]] for i in idx}
    from PALM.features.gene_features import compute_gene_features
    fmap = _feat_df_to_map(compute_gene_features(seqs, feature_sets=["canonical_kmer_frequencies"]))
    ids = [i for i in seqs if i in fmap]
    return DatasetBundle("rfam", "gene", "1d", True,
                         feature_data={i: fmap[i] for i in ids},
                         identifiers={i: seqs[i] for i in ids}, identifier_kind="nucleotide",
                         entity_type="gene",
                         targets={i: tgt[i] for i in ids}, task_type="classification",
                         target_name="rfam_family", meta={"n": len(ids), "n_classes": len(fams)})


# ── DNA / genomic (sequence -> k-mer / genomic LM) ──────────────────────────

def load_genomic(limit=DEFAULT_LIMIT):
    """Genomic Benchmarks (human enhancers): DNA sequence -> binary label.

    DNA modality, distinct from RNA (Rfam). Cheap canonical k-mer frequencies at
    load time (ACGT); router can swap in a genomic LM. Prepared by
    ``python -m PALM.data.prepare_genomic`` (or PALM.data.download_all).
    """
    path = os.path.join(HERE, "genomic", "records.csv")
    if not os.path.exists(path):
        return _unavailable("genomic", "gene", "1d",
                            "not prepared — run `python -m PALM.data.prepare_genomic`")
    df = pd.read_csv(path).dropna(subset=["sequence"]).reset_index(drop=True)
    idx = _subsample_indices(len(df), limit)
    seqs = {f"g{int(i)}": str(df["sequence"].iloc[int(i)]).upper() for i in idx}
    tgt = {f"g{int(i)}": int(df["label"].iloc[int(i)]) for i in idx}
    from PALM.features.gene_features import compute_gene_features
    fmap = _feat_df_to_map(compute_gene_features(seqs, feature_sets=["canonical_kmer_frequencies"]))
    ids = [i for i in seqs if i in fmap]
    return DatasetBundle("genomic", "gene", "1d", True,
                         feature_data={i: fmap[i] for i in ids},
                         identifiers={i: seqs[i] for i in ids}, identifier_kind="nucleotide",
                         entity_type="gene",
                         targets={i: tgt[i] for i in ids}, task_type="classification",
                         target_name="enhancer", meta={"n": len(ids), "n_classes": 2})


# ── catalysis (OC22 composition -> MAGPIE) ──────────────────────────────────

def load_oc22(limit=DEFAULT_LIMIT):
    """OC22 catalytic systems: composition (formula) -> DFT relaxed energy.

    Interfacial/adsorption materials modality; featurized by formula -> MAGPIE
    composition like materials_project. Prepared by
    ``python -m PALM.data.prepare_oc22`` (self-downloads the 114 MB IS2RE LMDBs).
    """
    path = os.path.join(HERE, "oc22", "records.csv")
    if not os.path.exists(path):
        return _unavailable("oc22", "inorganic", "1d",
                            "not prepared — run `python -m PALM.data.prepare_oc22`")
    df = pd.read_csv(path).dropna(subset=["formula"]).reset_index(drop=True)
    idx = _subsample_indices(len(df), limit)
    formulas = {str(df["id"].iloc[int(i)]): str(df["formula"].iloc[int(i)]) for i in idx}
    row_of_id = {str(df["id"].iloc[int(i)]): int(i) for i in idx}
    ids, X = composition_matrix(formulas)
    feature_data = {mid: X[j] for j, mid in enumerate(ids)}
    yv = pd.to_numeric(df["energy"], errors="coerce")
    targets = {mid: float(yv.iloc[row_of_id[mid]]) for mid in ids}
    return DatasetBundle("oc22", "inorganic", "1d", True,
                         feature_data=feature_data, targets=targets,
                         identifiers={mid: formulas[mid] for mid in ids}, identifier_kind="formula",
                         entity_type="material",
                         task_type="regression", target_name="relaxed_energy",
                         meta={"n": len(ids)})


# ── omics / perturbation (LINCS L1000; precomputed expression features) ──────

def load_lincs_l1000(limit=DEFAULT_LIMIT):
    """LINCS L1000: compound-perturbation signatures (978 landmark genes).

    Precomputed-feature modality — the 978-d expression vector IS the feature
    (loaded from expression.npy). Each signature also carries the perturbagen's
    SMILES, so a split can be compared in expression space vs structure space
    (route by ``identifier_kind='smiles'``, or use the scaffold splitter). No
    natural regression target -> split-quality only. Prepared by
    ``python -m PALM.data.prepare_lincs_l1000``.
    """
    d = os.path.join(HERE, "lincs_l1000")
    recs, npy = os.path.join(d, "records.csv"), os.path.join(d, "expression.npy")
    if not (os.path.exists(recs) and os.path.exists(npy)):
        return _unavailable("lincs_l1000", "omics", "1d",
                            "not prepared — run `python -m PALM.data.prepare_lincs_l1000`")
    df = pd.read_csv(recs)
    X = np.load(npy)
    n = min(len(df), len(X))                              # guard row alignment
    idx = _subsample_indices(n, limit)
    ids = [f"l{int(i)}" for i in idx]
    feature_data = {ids[k]: X[int(i)].astype(np.float32) for k, i in enumerate(idx)}
    smiles = {ids[k]: str(df["smiles"].iloc[int(i)]) for k, i in enumerate(idx)}
    return DatasetBundle("lincs_l1000", "omics", "1d", True,
                         feature_data=feature_data, smiles=smiles,
                         identifiers=dict(smiles), identifier_kind="smiles",
                         entity_type="molecule",
                         task_type=None, target_name=None,
                         meta={"n": len(ids), "dim": int(X.shape[1])})


# ── registry ────────────────────────────────────────────────────────────────

# name -> (category, callable(limit) -> DatasetBundle)
REGISTRY: Dict[str, Callable] = {
    # organic small molecules (TDC primary; MoleculeNet as always-available proxies)
    "tdc_Lipophilicity_AstraZeneca": lambda limit=DEFAULT_LIMIT: load_tdc("Lipophilicity_AstraZeneca", limit),
    "tdc_Solubility_AqSolDB": lambda limit=DEFAULT_LIMIT: load_tdc("Solubility_AqSolDB", limit),
    "tdc_BBB_Martins": lambda limit=DEFAULT_LIMIT: load_tdc("BBB_Martins", limit),
    "moleculenet_bace": lambda limit=DEFAULT_LIMIT: load_moleculenet("bace", limit),
    "moleculenet_bbbp": lambda limit=DEFAULT_LIMIT: load_moleculenet("bbbp", limit),
    "moleculenet_esol": lambda limit=DEFAULT_LIMIT: load_moleculenet("esol", limit),
    "moleculenet_freesolv": lambda limit=DEFAULT_LIMIT: load_moleculenet("freesolv", limit),
    "moleculenet_lipophilicity": lambda limit=DEFAULT_LIMIT: load_moleculenet("lipophilicity", limit),
    "moleculenet_clintox": lambda limit=DEFAULT_LIMIT: load_moleculenet("clintox", limit),
    "moleculenet_hiv": lambda limit=DEFAULT_LIMIT: load_moleculenet("hiv", limit),
    "moleculenet_tox21": lambda limit=DEFAULT_LIMIT: load_moleculenet("tox21", limit),
    "moleculenet_sider": lambda limit=DEFAULT_LIMIT: load_moleculenet("sider", limit),
    "moleculenet_muv": lambda limit=DEFAULT_LIMIT: load_moleculenet("muv", limit),
    "moleculenet_qm8": lambda limit=DEFAULT_LIMIT: load_moleculenet("qm8", limit),
    # inorganic crystals / MOFs
    "qmof": lambda limit=None: load_qmof(limit),
    "omol25": lambda limit=DEFAULT_LIMIT: load_omol25(limit),
    "materials_project": lambda limit=DEFAULT_LIMIT: load_materials_project(limit),
    # reactions (n-D)
    "uspto_mcr": lambda limit=DEFAULT_LIMIT: load_uspto_mcr(limit),
    # polymers
    "openpolymer26": lambda limit=DEFAULT_LIMIT: load_openpolymer26(limit),
    # proteins
    "lp_pdbbind": lambda limit=DEFAULT_LIMIT: load_lp_pdbbind(limit),
    # RNA / nucleotide
    "rfam": lambda limit=DEFAULT_LIMIT: load_rfam(limit),
    # DNA / genomic
    "genomic": lambda limit=DEFAULT_LIMIT: load_genomic(limit),
    # catalysis (interfacial materials)
    "oc22": lambda limit=DEFAULT_LIMIT: load_oc22(limit),
    # omics / perturbation (precomputed expression features)
    "lincs_l1000": lambda limit=DEFAULT_LIMIT: load_lincs_l1000(limit),
}


def list_datasets() -> List[str]:
    return list(REGISTRY)


def load_dataset(name: str, limit: Optional[int] = DEFAULT_LIMIT,
                 route: bool = False, feature_override: Optional[dict] = None) -> DatasetBundle:
    """Load a featurized :class:`DatasetBundle`.

    By default the loader's canonical featurizer is used (ECFP for molecules,
    MAGPIE composition for materials) — this keeps the committed master benchmark
    reproducible. Set ``route=True`` to instead featurize via the agent router
    (``PALM.data.routing``): entity type is detected and the feature set is chosen
    from the learned ``feature_heuristics.json`` (``per_dataset`` → per-type →
    default). ``feature_override`` (e.g. ``{"feature_set": "maccs", "reason": ...}``)
    forces a representation and is logged. No-op for n-D / identifier-less bundles.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    bundle = REGISTRY[name](limit)
    if (route or feature_override) and bundle.available and bundle.identifiers:
        from .routing import apply_featurizer, route as route_features
        # prefer the loader's explicit entity_type (protein/gene/mof/polymer); fall
        # back to identifier-kind for molecule/material where it isn't set.
        etype = bundle.entity_type or {"smiles": "molecule",
                                       "formula": "material"}.get(bundle.identifier_kind)
        r = route_features(name, identifiers=bundle.identifiers,
                           entity_type=etype, override=feature_override)
        Xmap = apply_featurizer(r.entity_type, r.feature_set, bundle.identifiers)
        bundle.feature_data = Xmap
        if bundle.targets:
            bundle.targets = {i: bundle.targets[i] for i in Xmap if i in bundle.targets}
        bundle.meta.update(feature_set=r.feature_set, feature_source=r.source,
                           feature_reason=r.reason)
    return bundle
