# PALM - Physics-Aware Leakage Minimizer

PALM creates train/test splits for interaction datasets with reduced data leakage using [DataSAIL](https://github.com/kalininalab/DataSAIL).

It supports molecule, material, biomolecule, and gene entities.

## Fastest Path: Run Web App Locally

### 1. Create environment

```bash
conda create -n palm python=3.12 -y
conda activate palm
```

### 2. Install required packages

```bash
pip install datasail pandas numpy pyyaml scipy rdkit
pip install fastapi uvicorn pydantic python-multipart
```

### 3. Start the web app

```bash
# From the parent directory containing PALM/
cd /path/to/parent-of-PALM

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

## If you refer to run it locally

```bash
python -m PALM config.yaml
```

## Minimal Config Example

```yaml
input_file: "data.csv"
output_dir: "output"
dataset_name: "my_dataset"

e:
  name: "drug"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets: [rdkit_descriptors]

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

## Input Data (Supported)

PALM auto-detects format from file extension (or directory contents):

- Table: `.csv`, `.json`
- Molecules: `.smi`, `.smiles`, `.sdf`, `.mol`, `.mol2`
- Structures: `.pdb`, `.cif`, `.mmcif`, directory of these files
- Sequences: `.fasta`, `.fa`, `.faa`
- Materials: ASE `.db`, directory of `.cif` or `.xyz`

## Splitting Techniques

- `R`: random baseline
- `I1e` / `I1f`: no overlap of entity identities on one axis
- `I2`: no overlap on either axis
- `C1e` / `C1f`: cluster-based split on one axis
- `C2`: cluster-based split on both axes (most stringent)

## Outputs

Generated under `output`:

- `features/<dataset_name>/<entity_name>/features.csv`
- `split_result/datasail_split_<technique>_<dataset_name>.csv`

Split files contain:
- `_row_id`: original row identifier
- `split`: assigned label (`train`, `test`, etc.)

## Dependencies by Use Case

Required:
- `datasail`, `pandas`, `numpy`, `pyyaml`, `scipy`, `rdkit`

Web app:
- `fastapi`, `uvicorn`, `pydantic`, `python-multipart`

Optional:
- `biopython` for PDB/mmCIF parsing
- `ase` for CIF/XYZ/.db material loading
- `torch`, `transformers` for LM embeddings (`esm_embedding`, `nt_embedding`)

Optional external clustering tools (mainly for biomolecule/gene `C1e/C1f/C2`):
- `mmseqs2`, `cd-hit`, `diamond`, `foldseek`, `tmalign`, `mash`