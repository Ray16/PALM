"""YAML config loading and validation."""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from PALM.features.material_features import MATERIAL_FEATURE_SETS
from PALM.features.molecule_features import MOLECULE_FEATURE_SETS
from PALM.features.biomolecule_features import BIOMOLECULE_FEATURE_SETS
from PALM.features.gene_features import GENE_FEATURE_SETS

ENTITY_FEATURE_SETS = {
    "molecule": MOLECULE_FEATURE_SETS,
    "material": MATERIAL_FEATURE_SETS,
    "biomolecule": BIOMOLECULE_FEATURE_SETS,
    "gene": GENE_FEATURE_SETS,
}


def check_path_component(value, field_name: str) -> str:
    """Validate a name that becomes one path component of the output tree.

    ``dataset_name`` and entity names are joined into output paths, so they must
    not be able to escape ``output_dir``: no separators, no ``.``/``..``, no NUL,
    not absolute. Anything else (spaces, unicode, dots inside the name) is fine.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    if len(value) > 100:
        raise ValueError(f"{field_name} must be at most 100 characters, got {len(value)}")
    bad = [c for c in ("/", "\\", "\0") if c in value]
    if bad or value in (".", "..") or os.path.isabs(value):
        raise ValueError(
            f"{field_name} {value!r} is used as a directory/file name and must not contain "
            "path separators, be '.' or '..', or be an absolute path")
    return value


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
    chemberta_model: str = "chemberta_zinc"
    chemberta_batch_size: int = 16
    embedding_file: Optional[str] = None
    structure_dir: Optional[str] = None
    # Parallel workers for per-entity featurization (1 = serial).
    feature_n_jobs: int = 1

    def __post_init__(self):
        check_path_component(self.name, "entity name")
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
class ExternalTechnique:
    """A non-DataSAIL splitter plugged into the pipeline by name.

    ``fn(ctx)`` receives a :class:`SplitContext` and returns the assignment in
    the same shape DataSAIL results use: ``{entity_id: split_name}`` in 1D,
    ``{(e1_id, e2_id): split_name}`` in 2D. Its output then goes through the
    same validation, metrics, plots and exports as every DataSAIL technique.
    """
    fn: object
    supports_1d: bool = True
    supports_2d: bool = True
    description: str = ""


@dataclass
class SplitContext:
    """Everything an external technique needs to produce a split.

    ``e1_data``/``e2_data`` are the featurized entities ({id: vector}, NaN/Inf
    already zeroed); ``interactions`` is the per-row (e1_id, e2_id) list in 2D
    (None in 1D); ``e1_entities`` is the raw entity map (e.g. id -> SMILES).
    """
    is_1d: bool
    e1_data: dict
    e2_data: Optional[dict]
    interactions: Optional[list]
    e1_entities: dict
    e1_config: "EntityConfig"
    e2_config: Optional["EntityConfig"]
    splitting: "SplittingConfig"


# name -> ExternalTechnique. Populated by register_technique(); this package
# itself registers nothing, so it carries no dependency on other splitters.
EXTERNAL_TECHNIQUES = {}


def register_technique(name, fn, supports_1d=True, supports_2d=True, description=""):
    """Make ``name`` a valid technique that the pipeline runs via ``fn(ctx)``."""
    if name in VALID_TECHNIQUES_1D | VALID_TECHNIQUES_2D:
        raise ValueError(f"'{name}' is a built-in technique and cannot be re-registered")
    EXTERNAL_TECHNIQUES[name] = ExternalTechnique(fn, supports_1d, supports_2d, description)


def valid_techniques(is_1d):
    """All technique names valid for a 1D (single-entity) or 2D dataset."""
    builtin = VALID_TECHNIQUES_1D if is_1d else VALID_TECHNIQUES_2D
    external = {n for n, t in EXTERNAL_TECHNIQUES.items()
                if (t.supports_1d if is_1d else t.supports_2d)}
    return builtin | external


@dataclass
class SplittingConfig:
    techniques: list = field(default_factory=lambda: ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"])
    splits: list = field(default_factory=lambda: [8, 2])
    names: list = field(default_factory=lambda: ["train", "test"])
    e2_clusters: int = 30
    max_sec: int = 300
    solver: str = "SCIP"
    # Reproducibility: seeds Python's `random` and numpy's global RNG at
    # pipeline start, the random-split baseline used by the comparison, and
    # the t-SNE/UMAP layouts. It does NOT change DataSAIL's output: the
    # installed datasail() has no seed argument and its main routine
    # unconditionally calls random.seed(42) / np.random.seed(42) on every
    # call, so DataSAIL splits (including its R technique) are deterministic
    # but fixed at DataSAIL's internal seed regardless of this value.
    seed: int = 42
    # Forwarded to datasail(threads=...), which DataSAIL uses for its external
    # clustering tools (CD-HIT, MMseqs2, DIAMOND, ...). With the precomputed
    # distance matrices this pipeline passes, those tools are not invoked, and
    # DataSAIL's MILP solver thread count is hard-coded inside datasail
    # (solver/utils.py, num_threads=14) — it cannot be set from here.
    threads: int = 1
    # Run multiple techniques concurrently via a thread pool. Off by default:
    # the SCIP solver can spawn child processes that don't always shut down
    # cleanly, so sequential is the safe default. Set >1 to overlap techniques.
    # Concurrent techniques share numpy's global RNG (which DataSAIL reseeds
    # per call), so n_jobs > 1 is not guaranteed to be bit-reproducible.
    n_jobs: int = 1
    # Split validation. By default every row must be assigned to a named
    # split; techniques that leave rows "not selected" are marked failed and
    # excluded from exports and the recommendation. Note that double-cold 2D
    # techniques (I2, C2) inherently drop cross-split interactions, so they
    # only pass when allow_incomplete is set.
    allow_incomplete: bool = False
    # Max absolute deviation of each split's row fraction (among assigned
    # rows) from its target fraction splits[i] / sum(splits).
    size_tolerance: float = 0.1

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
        all_valid = valid_techniques(True) | valid_techniques(False)
        for t in self.techniques:
            if t not in all_valid:
                raise ValueError(
                    f"Invalid splitting technique '{t}'. "
                    f"Valid 2D techniques: {sorted(valid_techniques(False))}. "
                    f"Valid 1D techniques: {sorted(valid_techniques(True))}."
                )
        if not (0 < self.size_tolerance <= 1):
            raise ValueError(
                f"'size_tolerance' must be in (0, 1], got {self.size_tolerance}"
            )
        if self.threads < 1 or self.n_jobs < 1:
            raise ValueError("'threads' and 'n_jobs' must be >= 1")


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

    def __post_init__(self):
        check_path_component(self.dataset_name, "dataset_name")
        if self.e2 is not None and self.e2.name == self.e1.name:
            raise ValueError(
                f"e1 and e2 must have different names (both {self.e1.name!r}); each "
                "entity's features are written to a directory named after it")


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
        chemberta_model=raw.get("chemberta_model", "chemberta_zinc"),
        chemberta_batch_size=raw.get("chemberta_batch_size", 16),
        embedding_file=raw.get("embedding_file"),
        structure_dir=raw.get("structure_dir"),
        feature_n_jobs=raw.get("feature_n_jobs", 1),
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
        seed=split_raw.get("seed", 42),
        threads=split_raw.get("threads", 1),
        n_jobs=split_raw.get("n_jobs", 1),
        allow_incomplete=bool(split_raw.get("allow_incomplete", False)),
        size_tolerance=float(split_raw.get("size_tolerance", 0.1)),
    )

    # Cross-validate techniques against dimensionality (1D vs 2D)
    is_1d = e2_cfg is None
    if is_1d:
        invalid = [t for t in splitting.techniques if t not in valid_techniques(True)]
        if invalid:
            raise ValueError(
                f"Techniques {invalid} are not valid for 1D (single-entity) datasets. "
                f"Valid 1D techniques: {sorted(valid_techniques(True))}"
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
