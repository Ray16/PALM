"""YAML config loading and validation."""

import os
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class SplittingConfig:
    techniques: list = field(default_factory=lambda: ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"])
    splits: list = field(default_factory=lambda: [8, 2])
    names: list = field(default_factory=lambda: ["train", "test"])
    f_clusters: int = 30
    max_sec: int = 300
    solver: str = "SCIP"


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
    e: EntityConfig
    f: EntityConfig
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    filters: list = field(default_factory=list)


def load_config(path):
    """Load and validate a YAML config file."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    # Parse entities
    e_raw = raw["e"]
    e_cfg = EntityConfig(
        name=e_raw["name"],
        type=e_raw["type"],
        extract_column=e_raw["extract"]["column"],
        feature_sets=e_raw.get("feature_sets", []),
        smiles_map=e_raw.get("smiles_map"),
        esm_model=e_raw.get("esm_model", "esm2_t33"),
        esm_batch_size=e_raw.get("esm_batch_size", 8),
        nt_model=e_raw.get("nt_model", "nt_500m_human_ref"),
        nt_batch_size=e_raw.get("nt_batch_size", 8),
        embedding_file=e_raw.get("embedding_file"),
    )

    f_raw = raw["f"]
    f_cfg = EntityConfig(
        name=f_raw["name"],
        type=f_raw["type"],
        extract_column=f_raw["extract"]["column"],
        feature_sets=f_raw.get("feature_sets", []),
        smiles_map=f_raw.get("smiles_map"),
        esm_model=f_raw.get("esm_model", "esm2_t33"),
        esm_batch_size=f_raw.get("esm_batch_size", 8),
        nt_model=f_raw.get("nt_model", "nt_500m_human_ref"),
        nt_batch_size=f_raw.get("nt_batch_size", 8),
        embedding_file=f_raw.get("embedding_file"),
    )

    # Parse filters
    filters = []
    for filt in raw.get("filters", []):
        filters.append(FilterConfig(
            column=filt["column"],
            not_equal=filt.get("not_equal"),
            not_empty=filt.get("not_empty", False),
        ))

    # Parse splitting
    split_raw = raw.get("splitting", {})
    _defaults = SplittingConfig()
    splitting = SplittingConfig(
        techniques=split_raw.get("techniques", _defaults.techniques),
        splits=split_raw.get("splits", _defaults.splits),
        names=split_raw.get("names", _defaults.names),
        f_clusters=split_raw.get("f_clusters", 30),
        max_sec=split_raw.get("max_sec", 300),
        solver=split_raw.get("solver", "SCIP"),
    )

    cfg = PipelineConfig(
        input_file=raw["input_file"],
        output_dir=raw["output_dir"],
        dataset_name=raw["dataset_name"],
        e=e_cfg,
        f=f_cfg,
        splitting=splitting,
        filters=filters,
    )

    # Validate input file exists
    if not os.path.exists(cfg.input_file):
        raise FileNotFoundError(f"Input file not found: {cfg.input_file}")

    return cfg
