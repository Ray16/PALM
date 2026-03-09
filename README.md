# PALM - Physics-Aware Leakage Minimizer

PALM creates train/test splits for interaction datasets with reduced data leakage using [DataSAIL](https://github.com/kalininalab/DataSAIL).

It supports molecule, material, biomolecule, and gene entities.

## To run the web app Locally

### 1. Create conda environment

```bash
conda create -n palm python=3.12 -y
conda activate palm
```

### 2. Install required packages

```bash
# Core + web app + structure/material parsing + LM embeddings
pip install datasail pandas numpy pyyaml scipy rdkit
pip install fastapi uvicorn pydantic python-multipart
pip install biopython ase

# PyTorch and Language-model backends (ESM, Nucleotide Transformer, DNABERT-2)
pip install torch transformers

#clustering tools
mamba install -c bioconda mmseqs2 cd-hit diamond foldseek tmalign mash -y
```

### 4. Start the web app

```bash
cd <path_to_PALM>

# Option A
uvicorn PALM.webapp.app:app --host 0.0.0.0 --port 8080 --reload

# Option B
bash PALM/webapp/run.sh [port]
```

Open `http://localhost:8080`.

Web app flow:
1. Upload a dataset
2. Configure entity columns/types/features
3. Run splitting
4. Download results

## Optional: Run by CLI

```bash
python -m PALM config.yaml
```

## Minimal Config Example

### 2D (interaction) dataset

```yaml
input_file: "data.csv"
output_dir: "output"
dataset_name: "my_dataset"

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
  techniques: [R, I1e, I1f, I2, C1e, C1f, C2]
  splits: [8, 2]
  names: ["train", "test"]
```

### 1D (single entity) dataset

```yaml
input_file: "data.csv"
output_dir: "output"
dataset_name: "my_dataset"

e1:
  name: "compound"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets: [rdkit_descriptors]

splitting:
  techniques: [R, I1e, C1e]
  splits: [8, 2]
  names: ["train", "test"]
```

> **Note:** The old `e`/`f` config keys are still accepted for backwards compatibility.

## Input Data (Supported)

PALM auto-detects format from file extension (or directory contents):

- Table: `.csv`, `.json`
- Molecules: `.smi`, `.smiles`, `.sdf`, `.mol`, `.mol2`
- Structures: `.pdb`, `.cif`, `.mmcif`, directory of these files
- Sequences: `.fasta`, `.fa`, `.faa`
- Materials: ASE `.db`, directory of `.cif` or `.xyz`

## Splitting Techniques

- `R`: random baseline
- `I1e` / `I1f`: no overlap of entity identities on one axis (entity1 or entity2)
- `I2`: no overlap on either axis
- `C1e` / `C1f`: cluster-based split on one axis (entity1 or entity2)
- `C2`: cluster-based split on both axes (most stringent)

For 1D datasets (single entity), only `R`, `I1e`, and `C1e` apply; 2D techniques are automatically mapped to their 1D equivalents.

## Outputs

Generated under `output`:

- `features/<dataset_name>/<entity_name>/features.csv`
- `split_result/datasail_split_<technique>_<dataset_name>.csv`

Split files contain:
- `_row_id`: original row identifier
- `split`: assigned label (`train`, `test`, etc.)

## Dependencies by Use Case

All dependencies below are required for this project setup:
- `datasail`, `pandas`, `numpy`, `pyyaml`, `scipy`, `rdkit`
- `fastapi`, `uvicorn`, `pydantic`, `python-multipart`
- `biopython` for PDB/mmCIF parsing
- `ase` for CIF/XYZ/.db material loading
- `torch`, `transformers` for LM embeddings (`esm_embedding`, `nt_embedding`)
- `mmseqs2`, `cd-hit`, `diamond`, `foldseek`, `tmalign`, `mash`
