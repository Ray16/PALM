# Correctness and clean-code pass

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

Work is in progress. Missing dated data does not block the independent code fixes.
