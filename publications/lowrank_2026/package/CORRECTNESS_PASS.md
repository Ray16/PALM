# Correctness and clean-code pass

> Historical v0.2.0.dev2 ledger. See `../revision_20261026/PLAN.md` for the current v0.2.0.dev3 manuscript revision. The measurements below retain their original scope.

Base: `14904f256705e42c72256b4d93477dc5adfee163`.

The user approved the critical-review improvements. Completion requires fixes,
regressions, independent numerical checks, benchmark comparisons and reviewable
GitHub changes. Preserve the prior development evidence and production app.

1. Repair rare-mass audit failures and zero-width uncertainty. Use exact known
   structure where available, valid bounded-sample intervals otherwise, explicit
   insufficient-information states and visible tolerance failures.
2. Unify dense/aligned kernels; enforce binary Tanimoto; reject invalid feature
   values instead of silently turning them into legitimate observations.
3. Add cut-aligned adaptive checks and reusable landmark construction. Preserve
   independent auditing and distinguish empirical checks from certificates.
4. Select optimizer restarts after refinement, support a bounded shortlist and
   validate against exact small problems. Preserve count/group contracts.
5. Keep n-D identity factors sparse, exploit exact cosine factors when cheaper,
   and reuse measured neighbour choices without hiding calibration cost.
6. Add explicit fold-pair weights so objectives can reflect evaluation boundaries,
   with consistent optimization, scoring and diagnostics.
7. Validate on untouched public datasets and representations, keep experimental
   penalties opt-in, and prepare chronological validation when real dated input
   becomes available. Do not manufacture chronology or claim prospective accuracy.

## Delivery and evidence

All seven implementation areas are addressed. Their evidence is scoped below;
the exploratory model checks do not establish prospective generalization.

| Area | Implemented behavior and evidence |
|---|---|
| Audits | Known-support exact checks; bounded empirical-Bernstein ratio intervals; explicit insufficient mass; independent rank audit and visible failure policies. Rare-mass/zero-width counterexamples are regressions. |
| Kernels | Shared stable cosine normalization; overflow-safe Euclidean preparation; binary Tanimoto and finite-input contracts across factorization, scoring, neighbours and exact refinement. Undefined relative scores use null/status. |
| Adaptive rank | Entrywise or returned-cut criterion, reused landmarks/kernel blocks, 32,768 default cut samples, explicit limits and failures. All 27 final cut cases establish tolerance and pass an independent full-kernel returned-cut check. |
| Optimizer | Refine before selecting; two-candidate public default with one/all options; exact small-problem references and dense weighted/local objective checks. |
| Structure/scaling | CSR n-D identity factors, record-weighted deduplication, exact cosine feature maps and bounded calibration reuse. Current kernel/factor code has 24 native structural measurements; prior fixed-budget study is preserved. |
| Fold policy | Symmetric nonnegative fold-pair weights enter assignment, moves/swaps, grouped optimization and weighted scoring consistently. Integer label-dtype regressions cover byte/int32/int64 inputs. |
| Validation | Nine datasets, three seeds, seven policies: 189 final numerical-source cases; 90 fixed-model checks and 60 descriptive calibration cross-checks. A genuine supplied-order reference helper is implemented and synthetically tested. Real dated outcomes are still needed for prospective validation. |

The package suite passes 213 tests; website compatibility passes 41 tests.
All 189 saved final-source assignments retain every featurized row at exact fold
counts. All 27 strict-group artifacts preserve their constructed groups.
Independent NumPy scoring agrees on 42 full FreeSolv/ClinTox cases. Ten final
replays match assignments, and all predictive feature/target/assignment inputs
are byte-identical to final cases. The standalone wheel matches all 43 shipped
Python files and works with optional domain/ANN imports blocked.

The numerical grid is pinned to commit `356ae758af8f268028dd2534c4f7530054bd6b1c`.
Three subsequent docstring corrections are verified to leave the entire
non-doc executable AST unchanged. Original hashes remain preserved. The earlier
all-restarts study has 242/243 completed observations and one confirmed 600-s
QM8 timeout; that observation is not counted as a success. Its cost informed the
bounded default before any predictive outcomes were examined.

See [../correctness_20261025/FINDINGS.md](../correctness_20261025/FINDINGS.md),
`summary.json`, per-case records, `verified_run/pipeline_review.json` and the
verification directory. No hypergraph baseline is used. The running app and
published manuscript remain on their separately reviewed release.
