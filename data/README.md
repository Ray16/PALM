# PALM datasets — fetch & reproduce

Everything needed to go from a fresh clone to running the master benchmark.

## Data model

- **Committed to the repo** (a fresh clone already has them): the small/medium
  DataSAIL sets under `DataSAIL_data/` — MoleculeNet, Rfam, NASA, LP-PDBBind, the
  gold-standard PPI set, PDBBind-core, USPTO-MCR records — plus the 21 MB
  `qmof/qmof.csv` derived table.
- **Fetched on demand** (git-ignored, regenerable): the large / API / streamed
  sources — TDC, Materials Project, OpenPolymer26, and the new modalities
  (genomic, OC22, LINCS L1000). One command fetches them all (below).
- **Opt-in** (git-ignored, large + gated): OMol25 only — excluded from the default
  fetch; run `python -m PALM.data.download_all --only omol25` (see Notes).

## 1. Environment (single env)

```bash
conda create -n palm python=3.12 -y && conda activate palm
pip install -e ".[benchmark]"     # splitters + all dataset loaders + matplotlib
```

`[benchmark]` pulls the splitter stack (torch, mtkahypar, rdkit) and every loader
dependency (PyTDC, mp-api, ase, lmdb, datasets, h5py). Materials Project also
needs a free API key:

```bash
export MP_API_KEY=...             # https://materialsproject.org/api
```

## 2. Fetch everything

```bash
python -m PALM.data.download_all
```

Idempotent (skips what's already present). Useful flags:

- `--cache-dir /scratch/palm_cache` — put HuggingFace + tmp scratch here. **Use
  this if your home disk is small**: downloads and HF caches otherwise land in
  `~/.cache` and can fill the disk.
- `--only tdc oc22` / `--skip lincs_l1000` — target subsets.
- `--mp-n 100000` — pull the full 100k Materials Project snapshot (default 10k).
- `--limit N` — sample cap for openpolymer26 / lincs_l1000 (default 10k / 20k).

### Disk budget

| Dataset | Download | On disk after prep |
|---|---|---|
| TDC | small | ~5 CSVs |
| Materials Project | API | ~10 MB (100k) |
| OpenPolymer26 | streamed | few MB |
| genomic | ~10 MB | ~40 MB |
| OC22 | 114 MB tar | ~record CSV + LMDBs |
| LINCS L1000 | 5 GB gz | ~12 GB GCTX + 78 MB features |
| OMol25 *(opt-in)* | 28 GB gated | ~48 GB raw + 4.4 GB feature cache |

LINCS is the heavy one in the default run — skip it with `--skip lincs_l1000` for a
light setup. OMol25 is heavier still but opt-in (`--only omol25`), so it never runs
unless you ask for it.

## 3. Run the benchmark

```bash
python -m PALM.benchmarks.master.run_benchmark --seeds 0 1 2 --limit 10000
# -> benchmarks/results/master_benchmark.csv
```

Runs every dataset in `PALM.data.sources.REGISTRY` × every applicable splitter ×
seeds, recording split quality (leakage/imbalance/runtime) and, where a target
exists, the generalization gap. Unavailable datasets are recorded with a reason,
never dropped.

## Notes

- **OMol25** (28 GB gated download + a 9.5M×115 feature cache) is the one
  **opt-in** dataset — excluded from the default `download_all` run because it is
  too large/gated for a routine setup, but reachable by one command:

  ```bash
  # one-time: accept the license at https://huggingface.co/facebook/OMol25
  pip install huggingface_hub && huggingface-cli login   # or export HF_TOKEN=...
  python -m PALM.data.download_all --only omol25         # ~28 GB + featurization
  ```

  `prepare_omol25` downloads the `{train_4M,val,test}.tar.gz` tarballs, extracts
  the `*.aselmdb` shards, and featurizes them (ase-only, no GPU) into
  `_cache/features.npy` + `meta.parquet` — the files the loader reads. Every stage
  is idempotent, and if the license/login is missing it prints the exact steps.
- **New modalities** — `genomic`, `oc22`, and `lincs_l1000` are prepared by
  `download_all` but only enter the benchmark once their loaders are registered in
  `PALM/data/sources.py`. Formats:
  - `genomic/records.csv` — id, sequence, label, split (DNA, binary)
  - `oc22/records.csv` — id, formula, energy, natoms, nads, split (composition + relaxed energy)
  - `lincs_l1000/` — records.csv (id, smiles, pert_iname, cell_id, dose, time) +
    `expression.npy` (N×978, row-aligned) + `landmark_genes.txt`
- Per-source provenance (URLs, citations, access) is in `data/data_source.csv` and
  `data/DataSAIL_data/source.csv`.
