# Dense, component and CCPart controls

This revision implements the agreed computational-methods scope in PLAN.md.
It uses the unchanged publication package (`../package`, primary numerical
producer f7e0866). New experimental adapters have separate source manifests;
earlier observations keep their actual versions. No response labels enter the
new optimization controls.

The full added inventory is 540 fresh-process cases: 432 primary controls,
36 nested ChEMBL controls and 72 preempted equal-time controls. The primary
controls include 18 component cases with ten settings each and 54 CCPart cases
with native/count/group interventions. The 810 resulting saved assignments are
independently replayed using RDKit all-pairs Tanimoto. The existing 900 full-kernel
fidelity comparisons are reanalysed with true-cut and kernel-mass denominators.
These counts describe experimental settings, not independent datasets.

## Read the evidence

- `analysis/summary.json`: complete primary inventory and aggregate results.
- `analysis/all_observations.csv`: every primary setting, including original-kernel
  global and nearest-training scores, retained counts, cut errors and costs.
- `analysis/dense_factor_pairs.csv`: paired initializations, ranks, powers,
  capacity rules, actual true cuts, time, RSS and representation-array ratios.
- `analysis/scaling_observations.csv`: all 36 nested ChEMBL profiles.
- `analysis/preempted_clock_observations.csv`: equal-time results with eligible
  checkpoint times and actual subprocess termination times.
- `analysis/full_fidelity_cut_errors.csv`: all 900 reanalysed fidelity records.
- `verification/completion.json`: evidence gate, independent replays and tests.
- `run`, `scaling_run`, `strict_clock_run`: immutable manifests, observations,
  exact assignments and retired ownership records.

Reported cost excludes imports, feature loading, common warm-up and independent
quality scoring. Representation construction is included. Fresh-process RSS is
captured before scoring; it includes framework and input costs. Controlled
kernel/factor-array storage is separate. CCPart's native cost includes exhaustive
threshold similarity construction, component construction and packing. Its
matched interventions reuse groups and are explicitly not full standalone
group-construction pipelines. Input permutations measure CCPart order sensitivity.

The primary dense controls invoke the same current assignment/refinement
routines through `DenseOps` and `FactorOps`; they do not compare different
optimizers or compute a full-kernel square root. Dense solutions remain
heuristic. `test_controls.py` checks interface equivalence, direct cut/nearest
calculations, initial-label ownership and zero-cut handling. The earlier exact
12-entity enumeration remains the finite-instance optimality reference.

## Rebuild and verify saved evidence

Use Python 3.12 and the pinned scientific environment recorded in manifests.
Build dependencies are listed in `../paper/requirements-build.txt`. Verification
also needs RDKit, PyTorch and pytest. The vendored CCPart source is unchanged
Hestia-GOOD 1.2.2, commit recorded in `vendor/HESTIA_COMMIT`, under its included
MIT license. Polars 1.35.2 is the added runtime dependency. Its thread pool, and
all BLAS/OpenMP pools, are capped at two in the benchmark.

From this directory, with prepared input caches available:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q test_controls.py ../package/tests
CUDA_VISIBLE_DEVICES='' python validate.py --dataset moleculenet_freesolv
# Repeat for esol, bace, bbbp, lipophilicity, tox21.
CUDA_VISIBLE_DEVICES='' python verify_additions.py --study scaling
CUDA_VISIBLE_DEVICES='' python verify_additions.py --study clock
python report.py
MLST_PYTHON=python bash ../paper/build.sh
```

`report.py` requires every declared outcome, unchanged producer sources and
assignment hashes. `completion_check.py` additionally requires the independent
replays, 226 passing tests and the reviewed-PDF/evidence gate. Partial results
cannot silently become final manuscript tables.

## Rerun numerical studies

Prepared MoleculeNet caches belong in repository `data/lowrank_cache`, outside
the distributable evidence. Their identities must match the earlier primary
manifest. Recreate them with the pinned input-preparation recipe in
`../revision_20261026/portable/run.py` and the attributed snapshots in `../inputs`.
The large packed ChEMBL 37 cache is not redistributed. Its checksum and fixed
seed-zero nested sample recipe match the original scaling manifest.

Choose free, disjoint CPU pairs after checking existing affinities and load.
Use fresh output directories; preserve the supplied completed runs:

```bash
python suite.py plan --output /absolute/new-run --cpus 8,9 10,11 \
  --datasets moleculenet_freesolv --seeds 0
python suite.py launch --output /absolute/new-run
```

The original numerical adapters use relative package/input paths and recorded
host allocations. `dense_scaling.py` and `clock_budget.py` reproduce their own
separately pinned protocols. Their default CPU pairs and the ChEMBL path are
documented in source. Use `portable.py` to plan fresh filtered reruns and run
individual cases under assigned CPU pairs, without editing frozen numerical
producers. Primary/scaling inputs can be relocated; the clock subprocess uses
the repository-relative prepared-cache location. The new manifest records the
adapter identity, input locations and actual resource allocation. GPU visibility
stays disabled throughout.

Do not modify a producer while its manifest owns live work. A failed case,
deadline, incompatible capacity or missing artifact is an explicit outcome,
not an excuse to change its settings invisibly.
