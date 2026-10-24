# Low-rank development findings

The development package adds numerical safeguards, empirical adaptive rank, explicit group contracts, bounded sampled diagnostics and an experimental global/local objective. The evidence supports these controls and robustness fixes; it does not establish a universal speed or split-quality improvement.

All 216 cases completed across six full molecular datasets and three seeds. Every assignment retains all entities at exact 80/10/10 counts. Independent review checks the saved assignments, source/input hashes and all 18 strict group partitions. A separate NumPy dense reference reproduces the global/local scores on all 36 FreeSolv cases. No hypergraph method enters this comparison.

## What to use

Keep fixed rank for workloads with an already validated accuracy/runtime trade-off. Use adaptive rank when an explicit approximation check matters, and inspect both selection and independent-audit flags. The 5% entrywise tolerance often requires larger factors; it is more demanding than preserving a useful cut.

Use strict supplied groups when series/family retention is a real requirement. For constructed groups, strict preservation applies to those groups only. Keep count-first repair when exact counts take priority over group retention. Incompatible requirements raise a proven-infeasible or unresolved-feasibility error.

Keep the existing combined powered/grouped method for strong local separation on these datasets. The new local penalty offers a different global/local trade-off and remains experimental. Neither new weight should be selected by examining final model test performance. Power-only and group-only controls are included below rather than assuming the new objective adds something those settings cannot do.

Threshold 0.4 and k=32 match the existing grouping configuration. Weight 1 gives equal emphasis to the two mass-normalized components; weight 0.25 tests a smaller local contribution. These weights were fixed before the main benchmark, not selected from model test outcomes.

For these CPU fingerprint workloads, prefer explicit exact tiled neighbours. ANN needs an audited use case; auto calibration includes its rejected work and can cost more than a known explicit choice. The native audit caught two deficient Tox21 candidate rows; exact-query repair now produces finite, distinct, non-self neighbours and counts toward construction time.

## Complete comparison

Scores below average the six dataset means; split time is the median of 18 calls per configuration. Lower overlap is a measurement, not a guarantee of a better deployment test. The local score is the mean ramp above Morgan similarity 0.4 to the nearest training molecule. MACCS is used only for evaluation.

| Configuration | Global ratio | Local ramp | MACCS nearest | Split seconds | Prep + split + scoring seconds |
|---|---:|---:|---:|---:|---:|
| release_plain | 0.2128 | 0.0800 | 0.6682 | 0.422 | 8.103 |
| new_plain | 0.2128 | 0.0800 | 0.6682 | 0.470 | 8.139 |
| new_auto | 0.2128 | 0.0812 | 0.6640 | 1.491 | 8.768 |
| new_auto_sampled | 0.2128 | 0.0812 | 0.6640 | 1.547 | 8.796 |
| release_power3 | 0.2278 | 0.0498 | 0.6604 | 0.394 | 8.011 |
| release_groups | 0.2303 | 0.0223 | 0.6459 | 0.802 | 8.415 |
| release_combined | 0.2417 | 0.0053 | 0.6509 | 0.833 | 8.474 |
| new_combined | 0.2417 | 0.0053 | 0.6509 | 0.881 | 8.524 |
| new_auto_combined | 0.2391 | 0.0040 | 0.6480 | 2.102 | 9.740 |
| new_combined_strict | 0.2422 | 0.0097 | 0.6505 | 1.305 | 8.616 |
| new_local_025 | 0.2179 | 0.0376 | 0.6478 | 1.021 | 8.485 |
| new_local_1 | 0.2313 | 0.0205 | 0.6401 | 1.129 | 8.551 |

Shared preparation includes the Morgan construction and independent MACCS features. The table includes scoring cost separately from split-call cost; split calls include factorization, adaptive checks and neighbour construction. Imports, small warm-ups and cache-loading costs are excluded from these two columns and retained separately in each row. No trained models are timed.

## Fidelity, trade-offs and validation

- new_auto: selection tolerance met in 15/18; independent audit met in 15/18; landmark counts range 128–512.
- new_auto_combined: selection tolerance met in 3/18; independent audit met in 2/18; landmark counts range 512–512.
- new_auto_sampled: selection tolerance met in 15/18; independent audit met in 15/18; landmark counts range 128–512.
- new_local_025 has no larger mean global and local score than release_plain on 0/6 datasets (none).
- new_local_025 has no larger mean global and local score than release_power3 on 3/6 datasets (bbbp, freesolv, tox21).
- new_local_025 has no larger mean global and local score than release_groups on 1/6 datasets (freesolv).
- new_local_025 has no larger mean global and local score than release_combined on 1/6 datasets (freesolv).
- new_local_1 has no larger mean global and local score than release_plain on 0/6 datasets (none).
- new_local_1 has no larger mean global and local score than release_power3 on 2/6 datasets (esol, freesolv).
- new_local_1 has no larger mean global and local score than release_groups on 0/6 datasets (none).
- new_local_1 has no larger mean global and local score than release_combined on 0/6 datasets (none).

`summary.json` contains every dataset mean, target-distribution diagnostics and paired changes with exploratory dataset-bootstrap intervals. Strict preservation can trade overlap quality for the required contract. The 18 sampled leakage intervals all covered the exact ratios in this grid; this is descriptive coverage and does not establish a guarantee.

The 180-case main grid ran at source commit `95e43be`. The later ANN deficiency repair changes only a function that none of those settings calls. `review.py` checks that executable AST scope and verifies all measured neighbour paths are exact. The 36 additional power/group controls were specified in `DEVELOPMENT.md`; their separate pre-execution manifest preserves the main grid and fixed settings. The ANN repair is tested independently on the full BBBP/Tox21 fingerprints and by regression tests.

This is development evidence, not a new manuscript claim. No actual chronological project outcomes were supplied, and simulated dates are not substituted. Chronological generalization and broader representation validity remain external validation tasks. The current manuscript and running website continue to describe/use the reviewed release.

## Reproduction

`benchmark.py prepare` builds the public fingerprint caches; `benchmark.py launch` runs the main grid with three workers, two disjoint CPU slots and two native threads each. `controls.py` supplies the two planned controls, reusing identical prepared features. `neighbour_check.py` audits native backends; `review.py` checks saved artifacts; `report.py` creates this report and `summary.json`. Manifests, per-case records and assignments accompany the results. Source/input drift is rejected rather than silently mixing measurements. The benchmark commit and release source paths must be available when reproducing the frozen grid.
