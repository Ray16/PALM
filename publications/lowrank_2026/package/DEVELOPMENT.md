# Next low-rank development pass

Base: published commit `9f7c34e1ded78b25b8f48cbb10dfb82f1d696cd3` (the exact base is also
recorded by Git; this development branch does not alter the frozen reproduction).

The user approved all five areas below. Completion requires implementation,
behavioral verification, matched benchmark evidence, documentation and a review
of the resulting pipeline. A faster import or a small unit test alone does not
establish a better splitting method.

1. **Adaptive rank and fidelity diagnostics.** Select rank using sampled kernel
   residuals under an explicit factor-array budget. Report independent validation,
   tolerance failures and the difference between sampled checks and guarantees.
2. **Global and local objective.** Evaluate a low-rank plus ordinary pairwise local
   penalty against plain, powered and grouped low rank. Preserve the archived
   negative evidence; no hypergraph baseline. Do not promote an unhelpful variant.
3. **Group contracts.** Distinguish count-first repair from strict group retention
   and explicitly allowed size corridors. Prove infeasibility when possible and
   distinguish solver/search exhaustion from proven infeasibility.
4. **Numerics and landmarks.** Prevent repeated landmarks, handle deficient
   landmark kernels without amplifying null directions, and check determinism,
   degenerate inputs and reconstruction against dense reference calculations.
5. **Complete-workflow scaling.** Add bounded sampled leakage diagnostics with
   uncertainty, tiled factor construction and measured neighbour backend selection
   with recall evidence. Include preprocessing and scoring costs in comparisons.

## Evidence and constraints

- Edit the namespaced `package/lowrank` implementation. The published manuscript,
  original dirty research checkout and running website remain preserved.
- Earlier ridge/leverage variants were neutral for leakage; sparse residuals were
  weaker than grouping (original `low_rank/docs/FINDINGS.md`, Steps 2 and 9).
- Split quality, approximation fidelity and deployment realism are separate.
  Exact counts, retained rows, group breaks, class composition, global/local
  overlap, runtime and memory must accompany comparisons.
- Fix benchmark settings before measuring. Use repeated seeds and matched
  resources; parallel independent cases with bounded threads and CPU affinity.
- Final model test outcomes must not select splitting parameters. Chronological
  validation requires actual chronological project data; do not invent it.

## Completed development pass

All five implementation areas are implemented and evaluated. This means the
controls and experiments exist and have been checked; it does not mean each
research candidate improved every benchmark.

| Area | Implementation and evidence |
|---|---|
| Adaptive rank | Geometric rank selection, budget planning, separate selection/audit flags and explicit limit stops. Dense/degenerate tests and 54 adaptive benchmark calls. Powered kernels often fail the 5% tolerance at the configured limit; this is reported. |
| Global/local objective | Positive normalized pairwise penalty integrated into guarded Lloyd and move/swap refinement. Dense-reference tests; comparisons against plain, power-only, group-only and combined release methods. Remains experimental because it does not consistently beat grouping. |
| Group contracts | Complete supplied-id mappings; count-first or strict preservation; explicit size corridor; single-threaded MILP capacity fallback; proven-infeasible vs unresolved search errors. Exhaustive small packing checks and direct review of all 18 strict benchmark assignments. |
| Numerical safeguards | Distinct landmarks, float64 landmark eigensystem, relative rank truncation and bounded row tiles. Full-kernel reconstruction, null/duplicate cases and the existing n-D regressions pass. |
| Complete workflow | Exact/sampled/auto diagnostic modes with Monte Carlo uncertainty; audited neighbour cost/recall selection; exact repair of deficient ANN rows. Native BBBP/Tox21 checks, package and website compatibility tests, isolated-wheel smoke and 216 benchmark cases with preparation/scoring/memory records. |

The final package suite passes 144 tests. The website compatibility suite passes
41 tests with the development source on its import path. The benchmark uses
three parallel CPU workers with disjoint two-core affinities on lambda2; no GPU
workload was launched. Every one of the 216 saved assignments retains all rows
and meets exact fold counts. All 18 strict-group artifacts preserve the
constructed groups. NumPy dense checks independently reproduce the global/local
scores on all 36 FreeSolv cases.

The later ANN repair affects no path executed by the main grid; `review.py`
verifies this by comparing executable AST scope and recorded settings. Its native
audit and regression test exercise missing/repeated/self candidates directly.

See [../development/FINDINGS.md](../development/FINDINGS.md) for measured results
and limitations, `summary.json` for all comparisons, and per-grid
`pipeline_review.json` files for artifact checks. Real chronological validation
needs real dated outcomes, which were not supplied. It is not simulated or
claimed by this development pass. The frozen manuscript and the running website
continue to use the reviewed release.
