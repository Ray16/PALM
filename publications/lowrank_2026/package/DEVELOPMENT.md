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

## Current status

Implementation starting. The full objective remains active; none of the five
areas is yet claimed complete.
