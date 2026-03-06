# Data Splitting Agent

A pipeline for scientifically rigorous train/test splitting of interaction datasets using [DataSAIL](https://github.com/kalininalab/DataSAIL). It featurizes entities (molecules, materials, proteins, genes) and applies DataSAIL's cluster-aware splitting algorithms to minimise data leakage across splits.

---

## Motivation

Random splitting of interaction data (e.g. drug–protein, adsorbate–adsorbent, gene–phenotype) leads to overly optimistic evaluation because similar entities appear in both train and test. This agent produces splits where the test set contains entities that are dissimilar to those in the train set, giving a more realistic estimate of generalisation.

---

## Quick Start

```bash
python -m PALM config.yaml
```

### Web App

PALM also includes a browser-based UI for interactive use. To start the web server:

```bash
# From the parent directory of PALM (i.e. the directory containing the PALM/ package)
cd /path/to/parent-of-PALM

# Ensure web app deps are installed first (see Installation -> Optional packages -> Web app UI)

# Run directly with uvicorn
uvicorn PALM.webapp.app:app --host 0.0.0.0 --port 8080 --reload

# Or use the helper script (uses `conda run -n palm`)
bash PALM/webapp/run.sh [port]
```

Then open `http://localhost:8080` in your browser. The web interface lets you:

1. **Upload** a data file (CSV, JSON, etc.)
2. **Configure** entity types, feature sets, and splitting techniques
3. **Run** the pipeline with real-time progress streaming
4. **Download** results as a ZIP file

---

## Input

### 1. Data file

Any of the supported formats below. The file is loaded into a table where each row represents one interaction pair (an e-entity interacting with an f-entity).

| Format | Extension(s) | Description |
|--------|-------------|-------------|
| CSV | `.csv` | Tabular data; any columns |
| JSON | `.json` | Dict-of-dicts (OC22-style) or list-of-dicts |
| ASE database | `.db` | ASE SQLite DB; extracts formula + key-value pairs |
| SMILES | `.smi`, `.smiles` | One SMILES per line, optional ID in second column |
| SDF | `.sdf` | Multi-molecule SD file; extracts SMILES, formula, all SDF properties |
| MOL | `.mol` | Single molecule MOL file; extracts SMILES and formula |
| MOL2 | `.mol2` | Single molecule MOL2 file; extracts SMILES and formula |
| PDB | `.pdb` | Single PDB file; one row per chain (or per chain–ligand pair) |
| PDB directory | directory of `.pdb` | All PDB files in a directory |
| mmCIF | `.cif`, `.mmcif` | Single mmCIF file; one row per chain (or chain–ligand pair) |
| mmCIF directory | directory of `.cif` | All mmCIF files in a directory |
| FASTA | `.fasta`, `.fa`, `.faa` | Protein or nucleotide sequences; one row per sequence |
| CIF directory | directory of `.cif` | Material CIF files (via ASE); extracts formula |
| XYZ directory | directory of `.xyz` | Material XYZ files (via ASE); extracts formula |

The format is auto-detected from the file extension or directory contents. You can override it with `fmt:` in the config.

All loaders produce a `_row_id` column used to track which split each row is assigned to.

### 2. YAML config

```yaml
input_file: "path/to/data.csv"
output_dir: "output"
dataset_name: "my_dataset"

# Optional row filters applied before splitting
filters:
  - column: "some_col"
    not_equal: ""       # exclude rows where some_col == ""
  - column: "other_col"
    not_empty: true     # exclude rows where other_col is blank

# e-entity: one axis of the interaction matrix (e.g. drug, adsorbate, gene)
e:
  name: "drug"
  type: "molecule"          # molecule | material | biomolecule | gene
  extract:
    column: "smiles"        # column whose values identify each entity
  feature_sets:
    - rdkit_descriptors
    - physicochemical

# f-entity: the other axis (e.g. protein, surface, phenotype)
f:
  name: "protein"
  type: "biomolecule"
  extract:
    column: "sequence"
  feature_sets:
    - sequence_properties

# DataSAIL splitting parameters
splitting:
  techniques: [R, I1e, I1f, I2, C1e, C1f, C2]
  splits: [8, 2]            # ratio: 80% train, 20% test
  names: ["train", "test"]
  f_clusters: 30
  max_sec: 300
  solver: "SCIP"
```

---

## Entity Types and Feature Sets

### `molecule` — small molecules

Input: SMILES strings (directly in a column, or via `smiles_map` for non-standard identifiers).

| Feature set | Dimensions | Dependencies | Description |
|-------------|-----------|--------------|-------------|
| `rdkit_descriptors` | 9 | RDKit | MW, HBD, HBA, lone pairs, TPSA, LogP, heavy atoms, valence/radical electrons |
| `physicochemical` | 11 | RDKit | MW, atom counts, rotatable bonds, ring counts, aromaticity, fraction sp3 |
| `composition` | variable | RDKit | Per-element counts + weighted elemental property means (electronegativity, radius, etc.) |

Optional config keys: `smiles_map` (dict mapping identifier → SMILES, for non-SMILES columns).

### `material` — bulk inorganic materials

Input: chemical formula strings (e.g. `Fe2O3`, `Cu3Au`).

| Feature set | Dimensions | Dependencies | Description |
|-------------|-----------|--------------|-------------|
| `magpie_properties` | ~130 | none | MAGPIE elemental statistics (mean, range, min, max, std) over the composition |
| `stoichiometry` | ~10 | none | L1/L2/L3/L5/L7/L10 norms of the stoichiometric vector, element count |
| `electronic` | ~20 | none | Electronegativity, electron affinity, ionisation energy stats |
| `bonding` | ~10 | none | Atomic radius, covalent radius stats |
| `thermodynamic` | ~10 | none | Melting point, cohesive energy stats |
| `classification` | ~10 | none | Fraction of each block (s/p/d/f), metal/nonmetal/metalloid fractions |

### `biomolecule` — proteins

Input: amino acid sequences (single-letter code).

| Feature set | Dimensions | Dependencies | Description |
|-------------|-----------|--------------|-------------|
| `sequence_properties` | 16 | none | Length, MW, hydrophobicity, charge, pI estimate, fraction hydrophobic/polar/aromatic/charged, flexibility, volume, surface area |
| `esm_embedding` | 320–5120 | torch, transformers | ESM2 protein language model mean-pooled embeddings |
| `precomputed_embedding` | any | numpy | Load pre-computed embeddings from file (see below) |

Optional config keys:
- `esm_model`: `esm2_t6` (8M, 320d), `esm2_t12` (35M, 480d), `esm2_t30` (150M, 640d), `esm2_t33` (650M, 1280d, default), `esm2_t36` (3B, 2560d), `esm2_t48` (15B, 5120d)
- `esm_batch_size`: batch size for GPU inference (default: 8)
- `embedding_file`: path for `precomputed_embedding`

### `gene` — DNA/RNA sequences

Input: nucleotide sequences (DNA or RNA; U is automatically normalised to T).

| Feature set | Dimensions | Dependencies | Description |
|-------------|-----------|--------------|-------------|
| `nucleotide_composition` | 11 | none | Length, per-base frequencies (A/T/G/C), GC%, AT%, GC skew, AT skew, CpG O/E ratio, melting temperature estimate |
| `kmer_frequencies` | 80 | none | Sliding-window dinucleotide (16) and trinucleotide/codon (64) relative frequencies |
| `nt_embedding` | 1024–2560 | torch, transformers | DNA language model mean-pooled embeddings (Nucleotide Transformer or DNABERT-2) |
| `precomputed_embedding` | any | numpy | Load pre-computed embeddings from file |

Optional config keys:
- `nt_model`: `nt_500m_human_ref` (default), `nt_500m_1000g`, `nt_2500m_multi_species`, `nt_2500m_1000g`, `dnabert2`
- `nt_batch_size`: batch size for GPU inference (default: 8)
- `embedding_file`: path for `precomputed_embedding`

#### Precomputed embedding file formats (biomolecule and gene)

| Extension | Expected structure |
|-----------|-------------------|
| `.csv` | First column = entity ID (index), remaining columns = embedding dimensions |
| `.npy` | 2-D array; rows assumed to be in the same order as entities |
| `.npz` | Must contain `ids` (array of entity IDs) and `embeddings` (2-D float array) |
| `.pt` / `.pth` | Dict `{entity_id: tensor}` or a bare 2-D tensor (same order as entities) |

---

## Splitting Techniques

DataSAIL provides the following splitting strategies (configure under `splitting.techniques`):

| Code | Name | Description |
|------|------|-------------|
| `R` | Random | Baseline random split; no structural constraints |
| `I1e` | Identity 1 (e-axis) | Split by e-entity identity; no e-entity appears in both sets |
| `I1f` | Identity 1 (f-axis) | Split by f-entity identity; no f-entity appears in both sets |
| `I2` | Identity 2 | No e- or f-entity appears in both sets |
| `C1e` | Cluster 1 (e-axis) | Cluster e-entities by feature similarity; split at cluster boundaries |
| `C1f` | Cluster 1 (f-axis) | Cluster f-entities by feature similarity; split at cluster boundaries |
| `C2` | Cluster 2 | Cluster both axes; split at both cluster boundaries simultaneously |

`C1e`, `C1f`, and `C2` are the most stringent and best reflect out-of-distribution generalisation.

Additional splitting parameters:
- `splits`: list of integers defining split ratios (e.g. `[8, 2]` → 80/20)
- `names`: split labels (e.g. `["train", "test"]` or `["train", "val", "test"]`)
- `f_clusters`: number of clusters for f-entity clustering (default: 30)
- `max_sec`: solver time limit in seconds (default: 300)
- `solver`: ILP solver (`SCIP` default; `GLPK`, `CPLEX` also supported)

---

## Output

All outputs are written to `<output_dir>/`.

### Feature files

```
<output_dir>/features/<dataset_name>/<entity_name>/features.csv
```

One row per unique entity, columns are feature dimensions. These can be reused or inspected independently.

### Split assignment files

```
<output_dir>/split_result/datasail_split_<technique>_<dataset_name>.csv
```

One file per splitting technique. Each file has two columns:

| Column | Description |
|--------|-------------|
| `_row_id` | Row identifier from the input file |
| `split` | Assigned split name (`train`, `test`, etc.) or `not selected` |

A summary is printed to stdout for each technique:

```
--- C2 ---
  Train: 8,432 (80.1%)
  Test:  2,098 (19.9%)
  Saved to output/split_result/datasail_split_C2_my_dataset.csv
```

---

## Example Configs

### Drug–Protein (molecule + biomolecule)

```yaml
input_file: "data/drug_protein.csv"
output_dir: "output"
dataset_name: "dti"

e:
  name: "drug"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets: [rdkit_descriptors, physicochemical]

f:
  name: "protein"
  type: "biomolecule"
  extract:
    column: "sequence"
  feature_sets: [sequence_properties]

splitting:
  techniques: [R, I1e, I1f, I2, C1e, C1f, C2]
  splits: [8, 2]
  names: ["train", "test"]
```

### Gene–Drug (gene + molecule)

```yaml
input_file: "data/gene_drug.csv"
output_dir: "output"
dataset_name: "gene_drug"

e:
  name: "gene"
  type: "gene"
  extract:
    column: "nucleotide_sequence"
  feature_sets: [nucleotide_composition, kmer_frequencies]

f:
  name: "drug"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets: [rdkit_descriptors, physicochemical]

splitting:
  techniques: [R, I1e, I1f, I2, C1e, C1f, C2]
  splits: [8, 2]
  names: ["train", "test"]
```

### Adsorbate–Surface from OC22 (molecule + material)

```yaml
input_file: "data/oc22/metadata.json"
output_dir: "output"
dataset_name: "oc22"

filters:
  - column: "ads_symbols"
    not_equal: ""

e:
  name: "adsorbate"
  type: "molecule"
  extract:
    column: "ads_symbols"
  feature_sets: [rdkit_descriptors, composition, physicochemical]
  smiles_map:
    "CO": "[C-]#[O+]"
    "H2O": "O"
    "O2": "O=O"

f:
  name: "adsorbent"
  type: "material"
  extract:
    column: "bulk_symbols"
  feature_sets: [magpie_properties, stoichiometry, electronic, bonding]

splitting:
  techniques: [R, I1e, I1f, I2, C1e, C1f, C2]
  splits: [8, 2]
  names: ["train", "test"]
  f_clusters: 30
  solver: "SCIP"
```

### PDB Protein–Ligand (biomolecule from structure file)

For PDB/mmCIF inputs the loader automatically extracts chains as sequences and detects ligands. The resulting table has one row per (chain, ligand) pair.

```yaml
input_file: "data/structures/"    # directory of .pdb files
output_dir: "output"
dataset_name: "plip"

e:
  name: "protein"
  type: "biomolecule"
  extract:
    column: "sequence"
  feature_sets: [sequence_properties]

f:
  name: "ligand"
  type: "molecule"
  extract:
    column: "ligand_id"           # 3-letter residue name from PDB HETATM
  feature_sets: [rdkit_descriptors]

splitting:
  techniques: [R, I1e, I1f, C1e, C1f, C2]
  splits: [8, 2]
  names: ["train", "test"]
```

---

## Installation

### 1. Create a conda environment

```bash
conda create -n palm python=3.12 -y
conda activate palm
```

### 2. Essential packages

These are required for the pipeline to run at all.

```bash
# Core pipeline (splitting engine, config, data loading)
pip install datasail pandas numpy pyyaml scipy

# Molecule featurisation and SDF/MOL/MOL2 file loading
pip install rdkit
```

### 3. Optional packages

Install only what you need for your data type.

#### Web app UI (FastAPI server)

Mandatory for the current web app phase.

```bash
# FastAPI app server + ASGI runtime
pip install fastapi uvicorn pydantic python-multipart
```

#### Biomolecule / material entity types

```bash
# PDB and mmCIF structure loading — falls back to built-in parser if absent
pip install biopython

# CIF, XYZ, and ASE .db file loading (material entity type)
pip install ase
```

#### Language model embeddings

Only needed if you include `esm_embedding` (proteins) or `nt_embedding` (genes)
in your feature sets.

```bash
# PyTorch — match your CUDA version; see https://pytorch.org
pip install torch

# ESM2 (proteins), Nucleotide Transformer / DNABERT-2 (genes)
pip install transformers
```

#### Cluster-based splitting (`C1e`, `C1f`, `C2` techniques)

DataSAIL uses external bioinformatics tools for sequence/structure clustering.
These are only needed when using `C1e`, `C1f`, or `C2` with biomolecule or gene
entities. For molecule and material entities, clustering uses built-in
feature-vector methods and none of these are required.

```bash
# Protein and nucleotide sequence clustering
mamba install -c bioconda mmseqs2 cd-hit -y

# Protein homology search
mamba install -c bioconda diamond -y

# Structure-based clustering and alignment
mamba install -c bioconda foldseek -y
mamba install -c bioconda tmalign -y

# k-mer sketch similarity (genomic data)
mamba install -c bioconda mash -y
```

### 4. Verify installation

```bash
# Essential
python -c "import datasail, rdkit, pandas, numpy; print('Essential deps OK')"

# Mandatory for current web app phase
python -c "import fastapi, uvicorn, pydantic, multipart; print('web app deps OK')"

# Optional (run whichever apply to your use case)
python -c "import ase; from Bio import SeqIO; print('biomolecule/material deps OK')"
python -c "import torch, transformers; print('LM embedding deps OK')"
```

---

## Dependencies

### Essential

| Package | Purpose |
|---------|---------|
| `datasail` | Splitting engine (ILP-based train/test assignment) |
| `pandas`, `numpy` | Data loading and feature matrices |
| `pyyaml` | Config file parsing |
| `scipy` | Distance matrix computation for clustering |
| `rdkit` | Molecule featurisation; SDF/MOL/MOL2 file loading |

### Optional

| Package | Purpose | Required for |
|---------|---------|-------------|
| `biopython` | PDB/mmCIF parsing | Biomolecule entity from structure files; falls back to built-in parser if absent |
| `ase` | CIF/XYZ/.db file loading | Material entity from structure files |
| `fastapi`, `uvicorn`, `pydantic`, `python-multipart` | Web server, request models, file upload handling | Web app UI (`PALM.webapp.app`) |
| `torch` | GPU tensor computation | `esm_embedding` or `nt_embedding` feature sets |
| `transformers` | ESM2, Nucleotide Transformer, DNABERT-2 | `esm_embedding` or `nt_embedding` feature sets |

### Optional external binaries (cluster-based splitting only)

Only needed for `C1e`, `C1f`, `C2` techniques with biomolecule or gene entities.
Molecule and material clustering uses built-in feature-vector methods.

| Tool | Install via | Purpose |
|------|------------|---------|
| `mmseqs2` | `mamba install -c bioconda mmseqs2` | Sequence clustering (proteins, genes) |
| `cd-hit` | `mamba install -c bioconda cd-hit` | Protein sequence clustering |
| `diamond` | `mamba install -c bioconda diamond` | Protein homology search |
| `foldseek` | `mamba install -c bioconda foldseek` | Structure-based clustering |
| `TMalign` | `mamba install -c bioconda tmalign` | Structure alignment |
| `mash` | `mamba install -c bioconda mash` | k-mer sketch similarity (genomic) |
