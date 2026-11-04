# Controlled optimization revision

User scope: the attached assessment, read on 3 November 2026, and the preceding
agreement. This revision retains the existing molecular and ODAC25 evidence and
adds the missing algorithm controls. It does not expand the biological claims.

## Requirements and evidence

1. Dense versus factorized optimization: identical entities, capacities, initial
   assignments, proposal/refinement routines and iteration limits; additionally
   compare equal wall-clock budgets including representation construction.
   Independent original-kernel scoring, per-process memory, actual work and
   initialization identities must accompany every result.
2. Current-code component controls: feasible initialization, assignment,
   refinement, restart budget, landmark sampling, kernel power and grouping.
   Group-preserving comparisons use exactly the same retained groups. Removing
   assignment preserves a feasible initial partition rather than relaxing sizes.
3. CCPart: use the pinned upstream Hestia-GOOD implementation, declared similarity
   thresholds, native realized sizes and retention, complete similarity cost,
   and low-rank controls at the native counts, with and without native groups.
   Empty folds, infeasibility and budget outcomes remain explicit.
4. Report absolute cut error, error relative to true cut and to total kernel mass,
   separately from the true-kernel quality of optimized partitions. Zero cuts
   receive explicit undefined-relative-error status. Retain the existing exact
   enumerated small-instance approximation/optimization decomposition.
5. Reframe title and paper around formulation, accuracy/optimization, scalability
   and fair benchmarking. Describe global/local differences as observed behavior,
   not an impossibility theorem. Existing two-million evidence retains its actual
   version, hardware and timing scope; no new downstream prediction claim.
6. Reproducible source/input/assignment identities, independently replayed scores,
   meaningful numerical tests, complete outcome inventory, built and visually
   checked main/SI, GitHub main and Overleaf synchronization.

## Declared experimental grid (before outcomes)

Full datasets: FreeSolv, ESOL, BACE, BBBP, Lipophilicity, Tox21; seeds 0, 1, 2.
Use the existing ECFP-1024 input caches, verified against the earlier manifest.
Primary capacities are 80/10/10, exact or a 1% corridor where stated.

Dense controls use current `balanced_lloyd` and `fm_polish` through a direct
kernel oracle implementing the same FactorOps interface. No eigendecomposition
of the full kernel is used. The factorized arm uses the unchanged publication
package. Both arms start from the same seeded feasible labels, use at most 25
assignment iterations and n refinement operations. Ranks: 64, 256, 512; powers:
1 and 3; capacity tolerances: 0 and .01. Fresh-process timing/memory profiles
exclude imports/loading/warm-up/scoring, and include representation construction.

Equal-clock controls use 1 and 5 seconds on two CPU cores, power 1, 1% corridor,
rank 256. Cooperative checks can return a feasible incumbent, including the
initial assignment if representation construction exhausts the budget. Every
finished result includes measured overrun and work counts; operations are not
described as preemptible. The iteration-controlled arm remains the primary
algorithm comparison. Candidate restarts share the same ordered seeds.

Current-code ablations use rank 256 and 1% corridor. Group comparisons retain
the same capped groups strictly. Native CCPart uses thresholds .4/.6/.8, all
three folds, no response labels and no sample removal. CCPart is deterministic;
the three repetitions permute input order with declared seeds, invert the
permutation for scoring and retain the corresponding group array. This measures
order sensitivity, not three stochastic optimizer seeds.

CPU-only local workers use disjoint two-core allocations and capped native
thread pools. Sources are frozen before each production manifest. Pilot results
are kept separately and cannot silently enter the production tables. No
sub-agents or unrequested GPU workloads are used.

## Synchronization

Overleaf was fetched and merged before this edit pass; commit 8141834 was current
and its manuscript matched the local manuscript. A baseline patch and dirty-file
inventory preserve pre-existing publication work. Pull/merge and rebuild again
before the final push; preserve collaborator edits and unrelated work.
