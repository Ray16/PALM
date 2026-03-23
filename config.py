"""YAML config loading and validation."""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from .features.material_features import MATERIAL_FEATURE_SETS
from .features.molecule_features import MOLECULE_FEATURE_SETS
from .features.biomolecule_features import BIOMOLECULE_FEATURE_SETS
from .features.gene_features import GENE_FEATURE_SETS

ENTITY_FEATURE_SETS = {
    "molecule": MOLECULE_FEATURE_SETS,
    "material": MATERIAL_FEATURE_SETS,
    "biomolecule": BIOMOLECULE_FEATURE_SETS,
    "gene": GENE_FEATURE_SETS,
}


@dataclass
class EntityConfig:
    name: str
    type: str  # "molecule", "material", "biomolecule", or "gene"
    extract_column: str
    feature_sets: list = field(default_factory=list)
    smiles_map: Optional[dict] = None
    esm_model: str = "esm2_t33"
    esm_batch_size: int = 8
    nt_model: str = "nt_500m_human_ref"
    nt_batch_size: int = 8
    embedding_file: Optional[str] = None
    structure_dir: Optional[str] = None

    def __post_init__(self):
        if self.type not in ENTITY_FEATURE_SETS:
            raise ValueError(
                f"Entity type must be one of {list(ENTITY_FEATURE_SETS.keys())}, got '{self.type}'"
            )

        valid_sets = ENTITY_FEATURE_SETS[self.type]
        for fs in self.feature_sets:
            if fs not in valid_sets:
                raise ValueError(
                    f"Feature set '{fs}' not valid for {self.type} entity. "
                    f"Valid: {list(valid_sets.keys())}"
                )


VALID_TECHNIQUES_2D = {"R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"}
VALID_TECHNIQUES_1D = {"R", "I1e", "C1e", "scaffold"}


@dataclass
class SplittingConfig:
    techniques: list = field(default_factory=lambda: ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"])
    splits: list = field(default_factory=lambda: [8, 2])
    names: list = field(default_factory=lambda: ["train", "test"])
    e2_clusters: int = 30
    max_sec: int = 300
    solver: str = "SCIP"

    def __post_init__(self):
        if not isinstance(self.splits, list) or len(self.splits) not in (2, 3):
            raise ValueError(
                f"'splits' must be a list of 2 or 3 elements, got {self.splits}"
            )
        if not isinstance(self.names, list) or len(self.names) != len(self.splits):
            raise ValueError(
                f"'names' list length ({len(self.names)}) must match "
                f"'splits' list length ({len(self.splits)})"
            )
        if not isinstance(self.techniques, list) or not self.techniques:
            raise ValueError("'techniques' must be a non-empty list of strings")
        all_valid = VALID_TECHNIQUES_2D | VALID_TECHNIQUES_1D
        for t in self.techniques:
            if t not in all_valid:
                raise ValueError(
                    f"Invalid splitting technique '{t}'. "
                    f"Valid 2D techniques: {sorted(VALID_TECHNIQUES_2D)}. "
                    f"Valid 1D techniques: {sorted(VALID_TECHNIQUES_1D)}."
                )


@dataclass
class FilterConfig:
    column: str
    not_equal: Optional[str] = None
    not_empty: bool = False


@dataclass
class PipelineConfig:
    input_file: str
    output_dir: str
    dataset_name: str
    e1: EntityConfig
    e2: Optional[EntityConfig] = None
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    filters: list = field(default_factory=list)


def _parse_entity(raw: dict) -> EntityConfig:
    """Parse an entity config block from YAML."""
    return EntityConfig(
        name=raw["name"],
        type=raw["type"],
        extract_column=raw["extract"]["column"],
        feature_sets=raw.get("feature_sets", []),
        smiles_map=raw.get("smiles_map"),
        esm_model=raw.get("esm_model", "esm2_t33"),
        esm_batch_size=raw.get("esm_batch_size", 8),
        nt_model=raw.get("nt_model", "nt_500m_human_ref"),
        nt_batch_size=raw.get("nt_batch_size", 8),
        embedding_file=raw.get("embedding_file"),
        structure_dir=raw.get("structure_dir"),
    )


def load_config(path):
    """Load and validate a YAML config file."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    # --- Validate required top-level keys ---
    required_top_level = ["input_file", "output_dir", "dataset_name"]
    missing = [k for k in required_top_level if k not in raw]
    if missing:
        raise ValueError(
            f"Config is missing required top-level keys: {missing}"
        )

    # At least one entity key must be present
    has_e1 = "e1" in raw or "e" in raw
    if not has_e1:
        raise ValueError(
            "Config must contain at least one entity key ('e1' or 'e')"
        )

    # --- Validate entity config blocks ---
    e1_key = "e1" if "e1" in raw else "e"
    e2_key = "e2" if "e2" in raw else "f"

    def _validate_entity_block(block, key_name):
        """Validate that an entity block has required sub-keys."""
        if not isinstance(block, dict):
            raise ValueError(
                f"Entity '{key_name}' must be a mapping, got {type(block).__name__}"
            )
        if "name" not in block:
            raise ValueError(
                f"Entity '{key_name}' is missing required key 'name'"
            )
        if "type" not in block:
            raise ValueError(
                f"Entity '{key_name}' is missing required key 'type'"
            )
        if "extract" not in block or not isinstance(block.get("extract"), dict):
            raise ValueError(
                f"Entity '{key_name}' is missing required key 'extract' "
                f"(must be a mapping with a 'column' sub-key)"
            )
        if "column" not in block["extract"]:
            raise ValueError(
                f"Entity '{key_name}.extract' is missing required key 'column'"
            )

    _validate_entity_block(raw[e1_key], e1_key)
    e1_cfg = _parse_entity(raw[e1_key])

    e2_cfg = None
    if e2_key in raw:
        _validate_entity_block(raw[e2_key], e2_key)
        e2_cfg = _parse_entity(raw[e2_key])

    # --- Validate structure_dir paths exist if specified ---
    for entity_key, entity_cfg in [(e1_key, e1_cfg), (e2_key, e2_cfg)]:
        if entity_cfg is not None and entity_cfg.structure_dir is not None:
            if not os.path.isdir(entity_cfg.structure_dir):
                raise FileNotFoundError(
                    f"structure_dir for entity '{entity_key}' does not exist: "
                    f"{entity_cfg.structure_dir}"
                )

    # --- Parse filters ---
    filters = []
    for filt in raw.get("filters", []):
        filters.append(FilterConfig(
            column=filt["column"],
            not_equal=filt.get("not_equal"),
            not_empty=filt.get("not_empty", False),
        ))

    # --- Parse splitting (SplittingConfig.__post_init__ handles validation) ---
    split_raw = raw.get("splitting", {})
    _defaults = SplittingConfig.__dataclass_fields__
    splitting = SplittingConfig(
        techniques=split_raw.get("techniques", _defaults["techniques"].default_factory()),
        splits=split_raw.get("splits", _defaults["splits"].default_factory()),
        names=split_raw.get("names", _defaults["names"].default_factory()),
        e2_clusters=split_raw.get("e2_clusters", split_raw.get("f_clusters", 30)),
        max_sec=split_raw.get("max_sec", 300),
        solver=split_raw.get("solver", "SCIP"),
    )

    # Cross-validate techniques against dimensionality (1D vs 2D)
    is_1d = e2_cfg is None
    if is_1d:
        invalid = [t for t in splitting.techniques if t not in VALID_TECHNIQUES_1D]
        if invalid:
            raise ValueError(
                f"Techniques {invalid} are not valid for 1D (single-entity) datasets. "
                f"Valid 1D techniques: {sorted(VALID_TECHNIQUES_1D)}"
            )

    cfg = PipelineConfig(
        input_file=raw["input_file"],
        output_dir=raw["output_dir"],
        dataset_name=raw["dataset_name"],
        e1=e1_cfg,
        e2=e2_cfg,
        splitting=splitting,
        filters=filters,
    )

    # Validate input file exists
    if not os.path.exists(cfg.input_file):
        raise FileNotFoundError(f"Input file not found: {cfg.input_file}")

    return cfg
