# Low-rank similarity splitting: publication revision

Technical manuscript revision, 21 October 2026. Author order, affiliations,
contributions, funding and competing interests remain pending author input.

- [Manuscript PDF](paper/manuscript.pdf) and [supplement PDF](paper/supplement.pdf).
- `package/`: MIT-licensed, namespaced `lowrank-split` distribution and tests.
- `paper/`: LaTeX sources, figure builders, retained assignments, numerical
  observations, input catalogue, environment versions and validation reports.
- `reproduction/`: frozen measurement sources and saved predictions needed for
  metric validation. These legacy source directories are not the public API.
- `inputs/`: exact benchmark snapshots. Most are relative links to the tracked
  repository data at the frozen base commit; clone this repository in full.

## Install and verify

```bash
python -m pip install ./package
cd package
python -m pytest -q
```

The wheel installs only `lowrank` and its distribution metadata. See the package
README for the precomputed-feature API. Molecular and experiment dependencies
are optional for the base API; the paper environment is recorded separately.

## Rebuild both PDFs from retained observations

Use Python 3.12, the versions in `paper/environment_versions.json`, and TeX Live
with pdflatex/bibtex. The build also needs Jinja2 and PyMuPDF for reporting and
PDF checks. It does not rerun expensive experiments.

```bash
MLST_PYTHON=python bash paper/build.sh
```

This rebuild validates the complete grid and derives tables and figures from
retained records. `paper/revision_20261021/completion_gate.json` records the
technical evidence gates. Manual scientific/visual reviews have their own
reports; passing these gates is not a publication acceptance claim.

## Rerun experiments

```bash
python reproduce.py revision_benchmark --output /absolute/fresh/run \
  --datasets moleculenet_freesolv --methods plain combined random --eps 0.01 --seeds 0
```

`reproduce.py` selects the frozen measurement sources, hides CUDA by default,
and caps native threads at two. Output is separate from the retained article
records. Use `--resume` only for an existing run with identical source/settings.
Run different dataset commands concurrently within your machine's CPU/memory
budget. For H100 scaling use `--gpu` and `revision_scaling_h100`; the full
ChEMBL 37 radius-2/2048-bit packed fingerprint cache must first be prepared with
`reproduction/comparisons/paper_lowrank/chembl_featurize.py`. The large cache is
not bundled; retained entity indices and checksums identify the measured input.

Study entry points include `revision_powered_fidelity`, `revision_robustness`,
`revision_lohi`, `revision_umap`, `revision_sensitivity_v2`,
`revision_calibration_v2`, `revision_external_control`,
`revision_datasail_matched_v2`, `revision_runtime`, and
`revision_metric_validation`. Each accepts `--help` through the launcher.
The runtime study is deliberately expensive (DataSAIL has a 7,200-second solver
limit, plus preprocessing). Lo-Hi uses version 1.1; UMAP 0.5.7;
Mt-KaHyPar 1.6.1; DataSAIL 1.3.0. The DataSAIL 1.4 solver equivalence check is
recorded in `paper/revision_20261021/datasail_version_check.json`.

## Comparison scope and provenance

The paper compares DataSAIL, Lo-Hi, the published UMAP/Ward recipe and an ordinary
weighted k-nearest-neighbour graph partitioned with Mt-KaHyPar's Graph API,
QUALITY preset and CUT objective. Random, scaffold and Butina are reference
controls. This publication folder contains no hypergraph benchmark results.
The surrounding repository contains separate historical research.

The version tag is `lowrank-v0.1.0-review-20261021-r1`. No archival DOI has been issued.

`SHA256SUMS.json` inventories this bundle. Individual records retain measurement
source hashes, assignments, native/returned counts, solver outcomes and scoring
scope. `reproduction/retained_input_transformations.json` documents filtering of
legacy synthetic observations to the permitted methods. Data and saved model
outputs retain their original source terms; the code MIT license does not
relicense third-party datasets. See the dataset catalogue and manuscript
references for MoleculeNet, SIMPD, QMOF, Materials Project and ChEMBL credits.
