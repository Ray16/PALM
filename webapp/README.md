# webapp — leakage-aware splitting in the browser

Upload a dataset (drag-and-drop) or pick a bundled demo, see how its entities are
distributed in embedding space, split it with several methods side by side, and
download the splits.

## Setup

The app is the `PALM.webapp` package, so the repository folder must be named `PALM`
(the default for `git clone`). Commands below run from the directory that
*contains* the checkout.

```bash
git clone https://github.com/Ray16/PALM.git
conda create -n palm python=3.11 -y && conda activate palm
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r PALM/webapp/deployment/requirements-cpu.txt          # pinned, tested set
pip install -e PALM/low_rank                                        # low-rank splitter (optional:
                                                                    #  the bundled copy is used otherwise)
```

## Run

```bash
bash PALM/webapp/run.sh               # CPU only (GPUs hidden), http://localhost:8080
bash PALM/webapp/run.sh --dev         # auto-reload on PALM/webapp/ changes
bash PALM/webapp/run.sh --gpu 8090    # reserve one free GPU via a gpu_reserve gate
```

`run.sh` works from any directory: it `cd`s to the folder containing `PALM/`, so
`PALM` and `lowrank` import correctly. It runs in the `palm` conda env; set
`PALM_CONDA_ENV=""` to use your active environment instead. Equivalent manual start:
`uvicorn PALM.webapp.app:app --port 8080`, run from the folder that contains `PALM/`.
Do not start it from *inside* `PALM/`: PALM's own `lowrank/` folder would shadow the
`lowrank-split` package.

Tests: `pip install pytest`, then `CUDA_VISIBLE_DEVICES= python -m pytest PALM/webapp/tests`
(same directory).

## Flow

1. **Data** — upload CSV/JSON/SDF/SMI/FASTA/PDB/CIF/XYZ/… (or a folder/ZIP), or a demo.
2. **Configure** — entity column(s) + type + feature sets, split ratio, methods.
   The **data map** is an interactive t-SNE of the exact features the splitters use
   (molecules: PALM's default SMILES sets; proteins: ESM2-650M; DNA/RNA and materials:
   their defaults). Colour by any column: numeric → one-hue ramp; ≤3 classes →
   categorical; >3 classes → highlight one class against the rest.
3. **Run** — live progress (SSE, polling fallback) and log. Submitting queues a
   *run*; you can leave, tweak the settings and submit more ("Submit another run").
   **Runs** (header) lists every run from this browser — status, progress, timing —
   with Open / Edit & rerun / Cancel / Delete, and a toast appears when one finishes.
4. **Results** — recommendation, comparison table (size deviation, per-axis overlap,
   NN separation vs random, distribution shift, Pareto set), the same t-SNE coloured
   by each method's split, per-method and full ZIP downloads, pipeline PNGs.

Maps: scroll / trackpad-pinch to zoom, drag to pan, double-click to reset; zoom is
kept while switching colouring or method.

## Runs

`jobs.py` keeps datasets (uploads/demos) and runs on disk
(`<jobs>/<dataset>/dataset.json`, `<jobs>/<dataset>/runs/<run>/{run.json,output/}`),
so the run list survives restarts — including `--dev` reloads, which **interrupt
any executing run** (marked *interrupted*; queued runs are re-queued). At most
`WEBAPP_MAX_CONCURRENT_RUNS` (2) execute at once; the rest wait. Cancelling a
running run stops it at the next pipeline step. Runs are listed per browser via an
anonymous `X-Client-Id` (separation, not authentication).

## Methods

DataSAIL techniques (`R`, `I1e`, `I1f`, `I2`, `C1e`, `C1f`, `C2`) and `scaffold` come
from `PALM/benchmarks/datasail_pipeline`. `methods.py` adds two more through that
pipeline's `register_technique` hook, so they get identical validation, metrics,
plots and exports:

| technique | 1D | 2D (interaction data) |
|---|---|---|
| `hypergraph` | PALM `hypergraph` (k-NN hyperedges, Mt-KaHyPar) | `hypergraph_nd_knn` |
| `lowrank` | low_rank `lowrank` (Nyström + balanced Lloyd + FM) | `lowrank_nd` |

In 2D each row is one n-D record (so the solvers balance the row fractions the
pipeline checks); repeated (e1, e2) pairs are then put in one split by majority vote.

Double-cold DataSAIL (`I2`, `C2`) drops interactions whose two entities land in
different splits; the pipeline rejects such incomplete splits unless
**Allow incomplete splits** is ticked (2D only).

## Demos

`demo_data/` holds six small public benchmarks (`demos.json` = titles, entity
columns, feature sets, colour column, default methods):

| id | domain | rows | colour |
|---|---|---|---|
| `esol` | molecules, regression | 1,128 | logS |
| `bbbp` | molecules, classification | 2,050 | permeable |
| `lp_pdbbind_proteins` | proteins (ESM2-650M) | 585 | protein class |
| `davis` | drug × kinase (2D) | 4,000 | pKd |
| `materials_project` | inorganic compositions | 1,500 | band gap |
| `rfam_rna` | non-coding RNA | 1,040 | Rfam family |

Rebuild the CSVs from PALM's bundled data with
`python PALM/webapp/demo_data/build_demos.py`. Pre-compute their features and maps
(protein demos need ESM2, so a GPU helps; on a shared cluster go through its GPU gate):

```bash
python PALM/webapp/demo_data/prewarm.py
```

## Caching & files

- Features: PALM's cache (`$PALM_CACHE_DIR`, default `~/.palm_cache`), shared with
  pipeline runs — computing a map warms the run and vice versa.
- Maps: `~/.palm_cache/webapp_maps/`, keyed by file content + featurization spec
  (same data → same layout before and after splitting). >5,000 entities are
  randomly subsampled for display (`WEBAPP_TSNE_MAX_POINTS`).
- Jobs: `$DATASAIL_JOBS_DIR` (default `/tmp/datasail_webapp`), deleted after
  `PALM_JOB_MAX_AGE_HOURS` (24).

Other settings: `PALM_MAX_UPLOAD_MB` (500), `PALM_RATE_LIMIT` (30 req/min/IP),
`PALM_CORS_ORIGINS` (`*`), `WEBAPP_SPLITTER_THREADS` (Mt-KaHyPar threads, 2),
`WEBAPP_MAX_CONCURRENT_RUNS` (2).

## Layout

```
PALM/webapp/
├── app.py           # FastAPI: upload, demos, maps, runs (+SSE), results, downloads
├── jobs.py          # persisted datasets + runs, worker queue, cancel, cleanup
├── worker.py        # runs/maps in child processes (python -m PALM.webapp.worker)
├── methods.py       # registers hypergraph + lowrank as pipeline techniques
├── embedding.py     # featurize like the pipeline → t-SNE map; split labels per point
├── reduction.py     # cached display reductions (t-SNE coordinates)
├── uploads.py       # bounded streaming upload / ZIP extraction
├── run.sh           # launcher (CPU default, --gpu via a gpu_reserve gate)
├── static/          # index.html + app.css + app.js (no build step; plotly served locally)
├── demo_data/       # demo CSVs, demos.json, build_demos.py, prewarm.py
├── deployment/      # Docker / Hugging Face Space bundle (export.py, Dockerfile, compose)
└── tests/           # python -m pytest PALM/webapp/tests
```

## Deployment and performance notes

See [deployment/README.md](deployment/README.md) for the portable CPU container,
persistent storage and access setup. Deployments use one shared budget for
run/map computation.
Scientific imports load in compute workers, display reductions are cached by
features/recipe/seed, and completed downloads are archived only once. Low-rank
splitting skips its duplicate internal dense diagnostics; the common pipeline
still computes the reported comparison metrics.

Uploads stream to disk and ZIP extraction is bounded. Browser-owned datasets,
runs, maps and downloads are checked consistently. Private deployments can set
`PALM_AUTH_USER` and `PALM_AUTH_PASSWORD`; credentials stay in server configuration.
`run.sh` binds to loopback by default; set `WEBAPP_BIND_HOST` deliberately when
using an authenticated reverse proxy or private container network.
