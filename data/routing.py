"""Featurization router — decide *how* to featurize a dataset, automatically.

Replaces the hand-picked ``ecfp_matrix`` / ``composition_matrix`` calls in the
loaders with a single decision point:

1. **detect** the entity type from the raw identifiers (SMILES / formula /
   protein / nucleotide) — so a new dataset needs no manual wiring;
2. **choose** the feature representation, in this precedence:
   ``LLM override``  →  learned ``feature_heuristics.json``  →  type default;
3. **apply** the chosen featurizer, returning ``{id: vector}`` ready for
   ``PALM.splitters``.

The heuristics table is produced by ``benchmarks/master/derive_heuristics.py``
from the feature sweep (which representation gives the cleanest *and* still
predictive splits per dataset). LLM overrides are logged to
``routing_overrides.jsonl`` so they can feed the next heuristics update.

Agent-ready: ``describe_router()`` mirrors ``describe_splitters()`` so an
MCP/agent can introspect detectable types, candidate features, and the current
heuristics, and can pass an ``override`` with a reason.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(__file__)
HEURISTICS_PATH = os.path.join(HERE, "feature_heuristics.json")
OVERRIDE_LOG = os.path.join(HERE, "routing_overrides.jsonl")

# Candidate representations per entity type. FIRST ENTRY IS THE DEMONSTRATED
# DEFAULT (via the feature sweep + predictive-validity gate; see
# benchmarks/master/README.md "Default featurizers per entity type" and insight.md
# §5/§5b). Validated per-dataset exceptions live in data/feature_heuristics.json
# (currently: moleculenet_bace→chemberta, qmof→mat2vec).
#   molecule→ecfp1024  material→magpie  mof→magpie  polymer→magpie
#   protein→esm2 (150M) gene→canonical_kmer
FEATURE_CANDIDATES: Dict[str, List[str]] = {
    "molecule": ["ecfp1024", "maccs", "rdkit_descriptors", "chemberta"],
    "material": ["magpie", "mat2vec"],   # matminer available after `pip install matminer pymatgen`
    "mof": ["magpie", "mat2vec", "linker_ecfp"],
    "polymer": ["magpie", "mat2vec"],
    "protein": ["esm2", "sequence_properties"],
    "gene": ["canonical_kmer", "kmer", "nucleotide_composition"],  # + "nt" once local disk frees for the ~2GB download
}
FEATURE_DEFAULTS = {t: c[0] for t, c in FEATURE_CANDIDATES.items()}

# below this detection confidence, the router flags for agent verification rather
# than silently trusting the sniffed entity type.
CONF_THRESHOLD = 0.6

# feature_set name -> the underlying compute_*_features set name (None = special path)
_MOL_SET = {"maccs": "maccs_keys", "rdkit_descriptors": "rdkit_descriptors",
            "chemberta": "chemberta_embedding", "physicochemical": "physicochemical"}
_MAT_SET = {"mat2vec": "mat2vec_embedding", "matminer": "matminer_elementproperty",
            "thermodynamic": "thermodynamic"}
_PROT_SET = {"esm2": "esm_embedding", "sequence_properties": "sequence_properties"}
_GENE_SET = {"canonical_kmer": "canonical_kmer_frequencies", "kmer": "kmer_frequencies",
             "nucleotide_composition": "nucleotide_composition", "nt": "nt_embedding"}

_AA = set("ACDEFGHIKLMNPQRSTVWY")
_NT = set("ACGTU")
_SMILES_HINT = re.compile(r"[a-z\[\]()=#@+\-\\/]|[0-9]")          # lowercase aromatics, brackets, bonds, ring digits
_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*\.?\d*)+$")           # e.g. Fe2O3, LiMn2O4, H2.0O


# ── entity-type detection ───────────────────────────────────────────────────

def detect_entity_type(values: Sequence, sample: int = 200) -> tuple:
    """Sniff the entity type of a column of raw identifiers.

    Returns ``(entity_type, confidence, evidence)``. Uses cheap string tests
    first (protein/nucleotide alphabets, formula regex), then RDKit as the
    arbiter for molecules. Falls back to ``("unknown", 0.0, ...)``.
    """
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()][:sample]
    if not vals:
        return "unknown", 0.0, "no values"

    def frac(pred):
        return sum(1 for v in vals if pred(v)) / len(vals)

    # nucleotide: only ACGTU, reasonably long
    if frac(lambda v: len(v) >= 12 and set(v.upper()) <= _NT) >= 0.8:
        return "gene", frac(lambda v: set(v.upper()) <= _NT), "ACGTU alphabet"
    # protein: only amino-acid letters, no SMILES punctuation/lowercase, longish
    if frac(lambda v: len(v) >= 20 and set(v.upper()) <= _AA
            and not _SMILES_HINT.search(v)) >= 0.8:
        return "biomolecule", 0.9, "amino-acid alphabet"
    # molecule vs material: RDKit is the arbiter for SMILES
    mol_frac = _rdkit_frac(vals)
    form_frac = frac(lambda v: bool(_FORMULA_RE.match(v)))
    if mol_frac >= 0.7 and mol_frac >= form_frac:
        return "molecule", mol_frac, f"RDKit parses {mol_frac:.0%} as SMILES"
    if form_frac >= 0.7:
        return "material", form_frac, f"{form_frac:.0%} match a chemical-formula pattern"
    if mol_frac >= 0.5:
        return "molecule", mol_frac, f"RDKit parses {mol_frac:.0%} as SMILES (weak)"
    return "unknown", 0.0, f"mol={mol_frac:.0%} formula={form_frac:.0%}"


def _rdkit_frac(vals):
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return 0.0
    ok = 0
    for v in vals:
        try:
            if Chem.MolFromSmiles(v) is not None:
                ok += 1
        except Exception:
            pass
    return ok / len(vals) if vals else 0.0


# ── featurizer application ──────────────────────────────────────────────────

def apply_featurizer(entity_type: str, feature_set: str, identifiers: Dict) -> Dict:
    """``{id: raw_str}`` -> ``{id: np.float32 vector}`` for the chosen representation.

    Dispatches to the fast fingerprint paths (ecfp / magpie composition) or to the
    ``PALM.features.compute_*_features`` batteries for everything else.
    """
    from .featurize import composition_matrix, ecfp_matrix

    if entity_type == "molecule" or (entity_type == "mof" and feature_set == "linker_ecfp"):
        if entity_type == "molecule" and feature_set not in ("ecfp1024", "ecfp", "morgan1024"):
            from PALM.features.molecule_features import compute_molecule_features
            df = compute_molecule_features(identifiers, feature_sets=[_MOL_SET[feature_set]])
        else:                                          # ecfp1024, or mof linker_ecfp on linker SMILES
            ids = list(identifiers)
            kept, X = ecfp_matrix([identifiers[i] for i in ids])
            return {ids[k]: X[j] for j, k in enumerate(kept)}
    elif entity_type in ("material", "mof", "polymer"):
        if feature_set in ("magpie", "magpie_composition"):
            ids, X = composition_matrix(identifiers)
            return {ids[j]: X[j] for j in range(len(ids))}
        from PALM.features.material_features import compute_material_features
        df = compute_material_features(identifiers, feature_sets=[_MAT_SET[feature_set]])
    elif entity_type == "protein":
        from PALM.features.biomolecule_features import compute_biomolecule_features
        # esm2_t30 (150M) is cached locally; avoids the 650M esm2_t33 default download
        df = compute_biomolecule_features(identifiers, feature_sets=[_PROT_SET[feature_set]],
                                          esm_model="esm2_t30")
    elif entity_type == "gene":
        from PALM.features.gene_features import compute_gene_features
        df = compute_gene_features(identifiers, feature_sets=[_GENE_SET[feature_set]])
    else:
        raise ValueError(f"apply_featurizer: unsupported entity_type={entity_type!r}")

    df = df.fillna(0.0)
    return {idx: np.asarray(df.loc[idx].values, dtype=np.float32) for idx in df.index}


# ── routing decision ────────────────────────────────────────────────────────

@dataclass
class Routing:
    entity_type: str
    feature_set: str
    source: str                 # "override" | "heuristic" | "default"
    reason: str = ""
    confidence: float = 1.0


def _load_heuristics(path=HEURISTICS_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _lookup(heur: dict, name: Optional[str], entity_type: str,
            target_property: Optional[str]) -> Optional[str]:
    if not heur:
        return None
    per_ds = heur.get("per_dataset", {})
    if name and name in per_ds:
        return per_ds[name]
    if name and target_property and f"{name}/{target_property}" in per_ds:
        return per_ds[f"{name}/{target_property}"]
    return heur.get("per_entity_type", {}).get(entity_type)


def _log_override(name, routing: Routing, override: dict):
    try:
        with open(OVERRIDE_LOG, "a") as fh:
            fh.write(json.dumps({"dataset": name, **asdict(routing),
                                 "override": override}) + "\n")
    except Exception:
        pass


def route(name: Optional[str] = None, identifiers: Optional[Dict] = None,
          entity_type: Optional[str] = None, target_property: Optional[str] = None,
          override: Optional[dict] = None, heuristics: Optional[dict] = None,
          log: bool = True, strict: bool = False) -> Routing:
    """Decide (entity_type, feature_set) for a dataset.

    Precedence: ``override`` (LLM, logged) → learned heuristics → type default.
    ``entity_type`` may be passed directly (e.g. from a known ``identifier_kind``)
    to skip detection; otherwise it is sniffed from ``identifiers``. When detection
    is low-confidence the returned ``reason`` is flagged for agent verification;
    ``strict=True`` raises instead of guessing.
    """
    low_conf = ""
    if entity_type is None:
        if not identifiers:
            raise ValueError("route: need identifiers or an explicit entity_type")
        entity_type, conf, evidence = detect_entity_type(list(identifiers.values()))
        if entity_type == "unknown" or conf < CONF_THRESHOLD:
            msg = f"low-confidence detection (type={entity_type}, conf={conf:.2f}, {evidence})"
            if strict:
                raise ValueError(f"route: {msg} — pass entity_type explicitly or an override")
            low_conf = f"[VERIFY: {msg}] "
            if entity_type == "unknown" and not (override and override.get("feature_set")):
                # can't pick a featurizer for an unknown type — return flagged, unresolved
                return Routing("unknown", "", "unresolved", low_conf + "agent must resolve", conf)
    else:
        conf = 1.0

    if override and override.get("feature_set"):
        r = Routing(override.get("entity_type", entity_type), override["feature_set"],
                    "override", low_conf + override.get("reason", "LLM override"), conf)
        if log:
            _log_override(name, r, override)
        return r

    fs = _lookup(heuristics if heuristics is not None else _load_heuristics(),
                 name, entity_type, target_property)
    if fs:
        return Routing(entity_type, fs, "heuristic",
                       low_conf + f"learned best for {name or entity_type}", conf)

    default = FEATURE_DEFAULTS.get(entity_type)
    if not default:
        raise ValueError(f"route: no default featurizer for entity_type={entity_type!r}")
    return Routing(entity_type, default, "default", low_conf + "type default", conf)


def describe_router() -> dict:
    """Agent/MCP-facing snapshot: detectable types, candidates, live heuristics."""
    heur = _load_heuristics()
    return {
        "detectable_types": list(FEATURE_CANDIDATES),
        "feature_candidates": FEATURE_CANDIDATES,
        "defaults": FEATURE_DEFAULTS,
        "heuristics_loaded": bool(heur),
        "heuristics": heur,
        "precedence": ["override", "heuristic", "default"],
    }
