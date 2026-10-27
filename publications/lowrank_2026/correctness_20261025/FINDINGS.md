# Low-rank correctness and pipeline review

The delivery fixes the reproduced audit and kernel failures, bounds default refinement effort, and removes dense n-D identity storage. It adds explicit evaluation-boundary weights and a supplied-date/order reference helper. Experimental penalties remain opt-in. Lower overlap does not by itself establish a better deployment test.

## Verification

- 213 package tests and 41 website compatibility tests pass. The built wheel is tested in an isolated target.
- All 189 final-source benchmark cases completed. Every assignment retains all featurized input rows at exact fold counts.
- Direct artifact checks preserve all 27 strict-group partitions.
- Independent NumPy dense scoring checks all 42 FreeSolv/ClinTox cases; maximum discrepancy is 3.44e-09.
- Ten end-to-end replays reproduce saved assignments. All 90 predictive checks use byte-identical features, targets and fold assignments to their corresponding final-source cases.
- Sixty fixed-classifier refits reproduce their saved Brier scores and add descriptive reliability bins and calibration-in-the-large.

The rare-mass regression no longer declares an empty factor accurate after missing all nonzero similarity. Zero observed leakage does not yield a false zero-width uncertainty interval. Dense/aligned kernels share normalization, extreme finite feature scales stay finite, count-valued Tanimoto is rejected, and undefined normalization/relative scores remain explicit.

## Primary comparison

Nine full datasets, three seeds and seven fixed configurations compare with the preceding development version. ClinTox, SIDER and QM8 extend the preceding six-dataset iteration. They are additional development tasks, not a permanently held-out or prospectively collected validation set. Defaults used unlabeled runtime/fidelity evidence, including QM8; model test outcomes did not select them. No hypergraph baseline is used. This study does not replace the paper’s external-method comparison.

Score columns average the nine dataset means. Call time is the median of 27 calls; the QM8 column separately exposes the large-dataset cost. Split calls include factorization, checks and graph work. Preparation, scoring, loading, warm-up, fresh-process time and process memory remain in each row.

| Configuration | Global ratio | Local ramp | MACCS nearest | Call seconds | QM8 call seconds |
|---|---:|---:|---:|---:|---:|
| previous_plain | 0.2133 | 0.0759 | 0.6551 | 0.317 | 91.36 |
| previous_combined | 0.2372 | 0.0040 | 0.6327 | 0.601 | 30.34 |
| new_default_plain | 0.2133 | 0.0779 | 0.6553 | 0.456 | 156.12 |
| new_default_combined | 0.2368 | 0.0034 | 0.6311 | 0.684 | 43.08 |
| new_default_strict | 0.2369 | 0.0032 | 0.6290 | 0.804 | 44.17 |
| new_default_cut | 0.2144 | 0.0885 | 0.6567 | 1.011 | 73.26 |
| new_default_weighted | 0.2170 | 0.0276 | 0.5766 | 0.384 | 6.28 |

The weighted setting expresses an example train/test priority; its objective differs from the ordinary cut. It is not presented as an unqualified winner. Refinement after candidate generation can improve the found objective while adding computation. The two-candidate default and explicit one/all-candidate choices expose that trade-off.

Cut auditing establishes its configured tolerance on 27/27 final cases. The exact returned-cut errors are independently checked against the full kernel totals in `verified_run/pipeline_review.json`. The tolerance is normalized by total absolute kernel mass, not by the cut value. Unestablished tolerances warn/raise or are explicitly collected with status; they are not silently passed.

## Structural performance and budget evidence

For the 100,000-by-16 cosine input, median factor construction is 1.340 s for the preceding Nyström path and 0.018 s for the exact feature-map backend (three seeds). These are factor-only timings, not whole-pipeline speedups.

The 100,000-record paired-identity factor uses 1,200,004 bytes of CSR arrays. Its dense equivalent is 20,000,000,000 bytes; no 20-GB dense array was allocated. Process RSS is reported separately, including framework/import overhead.

The native Tox21 calibration example costs 31.97 s initially and 1.05 s on the identical repeated request. The first call includes any rejected ANN/compilation cost; exact-backend reuse remains exact.

The rank-budget study holds refinement at one candidate and uses no target labels. It demonstrates why more audit samples can be cheaper than a larger factor. These explicit-budget cases remain separate from the final default settings.

| Dataset | Pair samples | Median seconds | Ranks | Independent audits passed |
|---|---:|---:|---|---:|
| moleculenet_bace | 2048 | 2.29 | [512, 512, 512] | 0/3 |
| moleculenet_bace | 8192 | 0.85 | [128, 128, 128] | 3/3 |
| moleculenet_bace | 32768 | 0.69 | [32, 32, 32] | 3/3 |
| moleculenet_freesolv | 2048 | 0.71 | [512, 512, 512] | 0/3 |
| moleculenet_freesolv | 8192 | 1.56 | [512, 512, 512] | 0/3 |
| moleculenet_freesolv | 32768 | 0.89 | [64, 64, 64] | 3/3 |
| moleculenet_qm8 | 2048 | 423.13 | [512, 512, 512] | 0/3 |
| moleculenet_qm8 | 8192 | 427.86 | [512, 512, 512] | 0/3 |
| moleculenet_qm8 | 32768 | 56.03 | [128, 128, 128] | 3/3 |

The earlier all-candidates study completed 242/243 cases. One QM8 plain case exceeded the 600-s protocol. Its missing observation and terminal log remain preserved, and it is not counted as a success. This cost motivated the bounded default before predictive outcomes were examined.

## Predictive and chronological scope

Two fixed models per task (linear and random forest) were fit on training labels only for five frozen policies, three seeds and three additional datasets: 90 fits. `summary.json` retains per-model AUROC, average precision and Brier where defined, or regression MAE/RMSE/R². Per-case records retain label availability and class composition. No test-set hyperparameter search or policy selection uses these scores. The calibration supplement uses fixed ten equal-width probability bins; its ECE and reliability summaries are sample-size/distribution dependent and descriptive.

Actual project dates/test order were not supplied. `chronological_reference` requires complete real ordering and explicit cutoffs, preserves ties and returns a zero-imbalance matched-count specification. Its synthetic policy tests are not real chronological experiments. Prospective model ranking and reliability still require actual dated outcomes.

## Provenance and reproduction

The final numerical grid is pinned to commit `356ae758af8f268028dd2534c4f7530054bd6b1c`. Subsequent Python changes align three docstrings only; `verification/doc_alignment.json` and the release audit verify equality of the complete non-doc executable AST. Original input, source, runner and assignment hashes remain in the manifests/rows. The isolated wheel check compares all shipped Python files with the delivery source.

Use `benchmark.py prepare/launch` for the final grid, `review.py` for artifact/source/numerical verification, `prediction_check.py` and `calibration_check.py` for fixed-model checks, and `scaling.py` for selected structural workloads. Frozen quality-profile runners preserve earlier protocols. Prepared public-feature caches and runtime logs are excluded from Git; input recipes and checksums are retained. The manuscript and running app stay on the reviewed release.
