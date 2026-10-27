# Low-rank similarity splitting: current publication revision

The current manuscript and package use v0.2.0.dev3. Numerical method source is
pinned to `f7e0866`; per-study manifests retain the actual producer hashes.
The v0.1 studies remain explicitly identified as archival supporting evidence.
Author order, affiliations, contributions, funding and interests await author input.

- [Manuscript PDF](paper/manuscript.pdf) and [supplement PDF](paper/supplement.pdf).
- [Seven-point revision and evidence plan](revision_20261026/PLAN.md).
- [Current study summary](revision_20261026/analysis/summary.json).
- [Technical completion gate](revision_20261026/verification/completion_gate.json).
- [Package API](package/README.md) and [joint audit policy](package/JOINT_BUDGET.md).

## Install and verify

```bash
python -m pip install ./package
python -m pytest -q package/tests
```

The distribution installs only the `lowrank` namespace. Domain and experiment
dependencies are optional extras. Actual producer versions are recorded in
`revision_20261026/verification/environment.json` and node preflight records.
Use the publication directory or a clean working directory for API imports:
the surrounding repository's legacy top-level research module can otherwise
shadow an installed distribution. The portable runner explicitly selects the
current package; its replay checks verify the loaded numerical implementation.

## Build both PDFs

Use Python 3.12, TeX Live (pdflatex and bibtex), and the pinned versions in
`paper/requirements-build.txt`. Building from saved observations does not rerun
expensive benchmarks or require the large feature caches.

```bash
MLST_PYTHON=python bash paper/build.sh
```

The default gate checks source/assignment identities, complete study outcomes,
generated figures, 18pt compiled figure text, TeX warnings and reviewed-render
signatures. `--allow-partial` is for an intermediate review build; it does not
declare submission readiness. Author-supplied metadata remains a separate flag.

## Reproduce a current numerical case

Prepare selected inputs from the included dataset snapshots, then choose a
fresh run directory and CPU cores allowed on your machine:

```bash
python revision_20261026/portable/run.py prepare --output /tmp/preparation \
  --cache-root /tmp/palm-inputs --data inputs \
  --datasets moleculenet_freesolv
python revision_20261026/portable/run.py plan --output /tmp/palm-run \
  --cache-root /tmp/palm-inputs --studies groups \
  --datasets moleculenet_freesolv --seeds 0 --cpus 0,1
python revision_20261026/portable/run.py launch --output /tmp/palm-run
```

The adapter keeps the measured numerical functions unchanged, verifies inputs,
uses disjoint two-core CPU workers, records native infeasibility without inventing
assignments, and retires claims safely on NFS. Omit filters for the full primary
grid. Runs resume only with identical source/settings; held claims require actual
owner verification. Case budgets and missing outputs remain visible.

## Scaling and comparison scope

The primary comparisons use DataSAIL and ordinary weighted k-NN graphs solved
through Mt-KaHyPar Graph/QUALITY/CUT. No hypergraph baseline is used.
Lo-Hi and other earlier comparisons are retained in the versioned archive.

Current scaling includes paired exact/1% two-core cases and three completed
two-million-entity cases on two disjoint 32-core CPU allocations on lambda13.
These are separate hardware protocols. The old 19.9-second H100 result belongs
to v0.1 and is not a current-code timing. All new GPU visibility is disabled.

The large ChEMBL 37 cache is not bundled. Obtain the versioned chemical
representations from the [official ChEMBL downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
and prepare radius-two/2048-bit packed fingerprints with
`reproduction/comparisons/paper_lowrank/chembl_featurize.py` (cap `--procs` to
available CPU resources). Input hashes and preparation recipes identify the
exact corpus. The staged large-CPU runner saves a completed split before its
independent audit so an interrupted audit does not repeat optimization.

## Archival material and attribution

`ARCHIVE_V01_README.md`, `reproduce.py`, `reproduction/` and the explicitly labelled
supplement section preserve the earlier release workflows and observations.
Use the current portable adapter for the new primary study. The running private
website remains separately pinned and is not changed by this revision.

`SHA256SUMS.json` inventories the delivered publication files. Code is MIT-licensed;
input data retain their source attribution and original terms. No archival DOI
is claimed. Publication and benchmark identities are documented in the manuscript.
