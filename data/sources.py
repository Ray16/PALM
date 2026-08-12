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
    meta: dict = field(default_factory=dict)


def _unavailable(name, category, kind, reason):
    return DatasetBundle(name, category, kind, available=False, reason=reason)


def _subsample_indices(n, limit, seed=0):
    if limit is None or n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, limit, replace=False))


# ── organic small molecules (SMILES -> ECFP) ───────────────────────────────

def _smiles_csv_bundle(name, category, path, smiles_col, limit):
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
    return DatasetBundle(name, category, "1d", True, feature_data=feature_data,
                         smiles=smiles, meta={"n": len(ids), "source_rows": len(df)})


def load_moleculenet(sub, limit=DEFAULT_LIMIT):
    path = os.path.join(DSAIL, "1D", "moleculenet", f"{sub}.csv")
    return _smiles_csv_bundle(f"moleculenet_{sub}", "organic", path, "smiles", limit)


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
    ids, X = composition_matrix(formulas)
    feature_data = {qid: X[j] for j, qid in enumerate(ids)}
    # linker SMILES (if present) enable the scaffold splitter on MOF organic linkers
    smiles = None
    scol = "info.mofid.smiles_linkers"
    if scol in df.columns:
        by_id = {str(df[idcol].iloc[i]): df[scol].iloc[i] for i in idx}
        smiles = {qid: str(by_id[qid]).split(".")[0] for qid in ids
                  if isinstance(by_id.get(qid), str) and by_id[qid]}
        smiles = smiles or None
    return DatasetBundle("qmof", "inorganic", "1d", True, feature_data=feature_data,
                         smiles=smiles, meta={"n": len(ids), "formula_col": fcol})


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


def load_materials_project(limit=DEFAULT_LIMIT):
    """Materials Project: <=limit summary docs -> MAGPIE composition (needs MP_API_KEY)."""
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
            fields=["material_id", "formula_pretty"],
            num_chunks=max(1, math.ceil(want / per)), chunk_size=per)
    formulas = {str(d.material_id): str(d.formula_pretty) for d in docs[:want]}
    ids, X = composition_matrix(formulas)
    feature_data = {mid: X[j] for j, mid in enumerate(ids)}
    return DatasetBundle("materials_project", "inorganic", "1d", True,
                         feature_data=feature_data, meta={"n": len(ids)})


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
    ids, X = composition_matrix(formulas)
    feature_data = {pid: X[j] for j, pid in enumerate(ids)}
    return DatasetBundle("openpolymer26", "polymer", "1d", True,
                         feature_data=feature_data, meta={"n": len(ids)})


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
    # inorganic crystals / MOFs
    "qmof": lambda limit=None: load_qmof(limit),
    "omol25": lambda limit=DEFAULT_LIMIT: load_omol25(limit),
    "materials_project": lambda limit=DEFAULT_LIMIT: load_materials_project(limit),
    # reactions (n-D)
    "uspto_mcr": lambda limit=DEFAULT_LIMIT: load_uspto_mcr(limit),
    # polymers
    "openpolymer26": lambda limit=DEFAULT_LIMIT: load_openpolymer26(limit),
}


def list_datasets() -> List[str]:
    return list(REGISTRY)


def load_dataset(name: str, limit: Optional[int] = DEFAULT_LIMIT) -> DatasetBundle:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](limit)
