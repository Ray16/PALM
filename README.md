# PALM — Physics-Aware Leakage Minimizer

PALM creates train/test splits for scientific ML datasets with reduced data leakage using [DataSAIL](https://github.com/kalininalab/DataSAIL). It supports **molecules, biomolecules, materials, and genes** through a single configuration interface and an easy-to-use web app.

## Why PALM?

Random train/test splits allow structurally similar entities into both sets, inflating model performance. PALM uses cluster-based splitting to ensure test samples are genuinely novel:

| Dataset | Random R² | Cluster R² | Inflation |
|---------|-----------|------------|-----------|
| ESOL (1,128 molecules) | 0.667 | 0.333 | +0.334 |
| Lipophilicity (1,000 molecules) | 0.362 | 0.131 | +0.231 |
| Materials Project (500 crystals) | 0.923 | 0.412 | +0.511 |

## Quick Start

### Installation

```bash
conda create -n palm python=3.12 -y
conda activate palm

# Core
pip install datasail pandas numpy pyyaml scipy rdkit scikit-learn matplotlib

# Web app
pip install fastapi uvicorn pydantic python-multipart

# Structure/material parsing
pip install biopython ase

# Language-model embeddings (optional)
pip install torch transformers
```

### CLI Usage

```bash
python -m PALM config.yaml
```

### Web App

```bash
uvicorn PALM.webapp.app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` → Upload dataset → Configure entities → Run splitting → Download results.

The web app includes guided tooltips, technique recommendations, data preview, real-time progress streaming, interactive metrics dashboards, and t-SNE visualizations.

#### Web App Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASAIL_JOBS_DIR` | `/tmp/datasail_webapp` | Directory for job storage |
| `PALM_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `PALM_MAX_UPLOAD_MB` | `500` | Maximum upload file size in MB |
| `PALM_RATE_LIMIT` | `30` | Max requests per IP per 60-second window |
| `PALM_JOB_MAX_AGE_HOURS` | `24` | Auto-delete jobs older than this |

## Configuration

### 1D Dataset (single entity)

```yaml
input_file: "molecules.csv"
output_dir: "output"
dataset_name: "my_dataset"

e1:
  name: "compound"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets: [rdkit_descriptors, morgan_fingerprint]

splitting:
  techniques: [R, C1e]
  splits: [8, 2]
  names: ["train", "test"]
```

### 2D Dataset (entity pair interactions)

```yaml
input_file: "interactions.csv"
output_dir: "output"
dataset_name: "dti"

e1:
  name: "drug"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets: [rdkit_descriptors]

e2:
  name: "protein"
  type: "biomolecule"
  extract:
    column: "sequence"
  feature_sets: [sequence_properties]

splitting:
  techniques: [R, I2, C1e, C2]
  splits: [8, 2]
  names: ["train", "test"]
```

## Supported Input Formats

PALM auto-detects format from file extension or directory contents:

| Category | Formats |
|----------|---------|
| Tabular | `.csv`, `.tsv`, `.json`, `.parquet` |
| Molecules | `.smi`, `.smiles`, `.sdf`, `.mol`, `.mol2`, directory of `.sdf` |
| Structures | `.pdb`, `.cif`, `.mmcif`, directory of these |
| Sequences | `.fasta`, `.fa`, `.faa` |
| Materials | ASE `.db`, directory of `.cif` or `.xyz` |

**Format preservation**: Output splits are saved in the same format as input (e.g., SDF → `train.sdf`/`test.sdf`, FASTA → `train.fasta`/`test.fasta`).

## Splitting Techniques

| Technique | Description | Use case |
|-----------|-------------|----------|
| `R` | Random baseline | Comparison benchmark |
| `I1e` / `I1f` | No identity overlap on one axis | Remove exact duplicates |
| `I2` | No identity overlap on either axis | Strict deduplication for 2D data |
| `C1e` / `C1f` | Cluster-based split on one axis | Prevent near-duplicate leakage |
| `C2` | Cluster-based split on both axes | Most stringent (2D data) |
| `scaffold` | Bemis-Murcko scaffold grouping | Chemistry-aware splitting (molecules only) |

For 1D datasets, `R`, `I1e`, `C1e`, and `scaffold` apply. 2D techniques are automatically mapped to their 1D equivalents. Scaffold splitting groups molecules by their generic ring/linker framework and ensures no scaffold appears in multiple splits.

**Benchmark mode**: When multiple techniques are run, PALM automatically generates a comparison summary with per-technique metrics and a recommendation for the best technique.

## Feature Sets

### Molecule
- `rdkit_descriptors` — MW, HBond donors/acceptors, TPSA, LogP, etc. (9 features)
- `physicochemical` — atom counts, rotatable bonds, ring counts, sp3 fraction (11 features)
- `composition` — element counts + weighted elemental properties
- `morgan_fingerprint` — 2048-bit ECFP4 circular fingerprint

### Biomolecule
- `sequence_properties` — length, MW, hydrophobicity, charge, pI, GRAVY (16 features)
- `esm_embedding` — ESM2 protein language model (320–5120 dim)
- `precomputed_embedding` — load from CSV/npy/npz/pt file

### Material
- `magpie_properties` — elemental statistics (120+ features)
- `stoichiometry` — element diversity, entropy, L-norm descriptors (8 features)
- `electronic` — d-band, valence electrons, oxidation states (12 features)
- `bonding` — electronegativity, ionic radii, polarizability (9 features)
- `thermodynamic` — melting points, enthalpies, reducibility (8 features)
- `classification` — metal type flags, d-band filling (8 features)
- `matminer_elementproperty` — MAGPIE via matminer (requires `matminer`)
- `mat2vec_embedding` — composition-weighted mat2vec embeddings (200 features)
- `crystalnn_fingerprint` — CrystalNN site fingerprints (requires structure files + `matminer`)
- `soap_descriptor` — SOAP 3D descriptors (requires structure files + `dscribe`)

### Gene
- `nucleotide_composition` — GC%, AT skew, CpG O/E, melting temp (11 features)
- `kmer_frequencies` — dinucleotide + trinucleotide frequencies (80 features)
- `nt_embedding` — Nucleotide Transformer or DNABERT-2 embeddings
- `precomputed_embedding` — load from file

## Adaptive Distance Metrics

PALM automatically selects the distance metric based on feature characteristics:

| Feature type | Metric | Condition |
|-------------|--------|-----------|
| Binary fingerprints | Tanimoto | All values 0/1, ≥128 dims |
| Sparse features | Cosine | 50–90% zeros |
| Dense features | Euclidean | <50% zeros |
| Very sparse | PCA → Euclidean | >90% zeros |

## Outputs

```
output/
├── features/<dataset>/<entity>/features.csv
├── split_result/
│   ├── datasail_split_<technique>_<dataset>.csv
│   └── <technique>_<dataset>/
│       ├── train.<format>
│       └── test.<format>
├── ml_exports/
│   ├── <technique>_<dataset>_indices.json      # split indices for PyTorch/sklearn
│   └── <technique>_<dataset>_with_splits.csv   # original data + split column
├── metrics/
│   ├── <technique>_<dataset>.json
│   └── comparison_<dataset>.json               # cross-technique comparison + recommendation
└── plots/
    ├── <technique>_<entity>_tsne.png
    └── comparison_<dataset>.png
```

### Quality Metrics (per technique)

Each `metrics/*.json` file contains:
- **Coverage** — fraction of entities assigned to splits
- **NN leakage** — nearest-neighbor distances between test and train (mean, median, min, max, zero-distance count)
- **Distribution shift** — per-feature mean/max normalized shift between splits
- **Entity overlap** — count of shared entities across splits (2D datasets)

### ML Export Formats

Each technique generates ML-framework-friendly exports in `ml_exports/`:

- **`_indices.json`** — Split indices keyed by split name (`train`, `test`, `val`). Use directly with PyTorch `Subset` or sklearn indexing.
- **`_with_splits.csv`** — Original data with a `split` column appended. Drop-in for pandas/sklearn workflows.

### Comparison Summary

When multiple techniques are run, `metrics/comparison_<dataset>.json` contains:
- Per-technique coverage, NN separation, distribution shift, and entity overlap
- A weighted recommendation for the best technique

## Caching

Feature vectors and distance matrices are cached in `~/.palm_cache` (configurable via `PALM_CACHE_DIR`). Cache keys include a version number — when feature computation code changes, increment `CACHE_VERSION` in `cache.py` to invalidate stale caches.

## Running Tests

```bash
# Prepare data + run all integration tests
bash PALM/tests/run_tests.sh

# Improvement tests (config validation, cache, security, edge cases)
python PALM/tests/test_improvements.py

# ML leakage demonstration (ESOL, Lipophilicity, MP formation energy)
python PALM/tests/run_ml_experiment.py
```
