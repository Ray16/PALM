# Critical review of lowrank-split v0.2.0.dev4

Reviewed 12 November 2026 against the current code in this directory. The main
manuscript's numerical results were produced by frozen v0.2.0.dev3; the dev4
change batches exact-count swap calculations without expanding candidate
coverage. Historical performance values below remain attributed to their
producer version. No new scientific benchmark was run for this review.

The method has a sound computational core: for a factor B, cross-fold similarity
is exactly the sum of inner products of fold sums. This makes a dense objective
affordable without storing its full matrix. The strongest evidence is the direct
dense control with the same initialization and search procedures, together with
matched-group controls. The contribution should remain a constrained
optimization framework with measured trade-offs. A universally better
anti-leakage split is not established by the objective or experiments.

## Improvements in priority order

### 1. Audit the quantity that matters to the decision

The implemented `cut_error` audit divides the signed cut residual by total
absolute kernel mass, then takes its absolute value
([diagnostics.py](lowrank/diagnostics.py:179)). It does not measure relative error
in the cut. A small cut can therefore pass a mass-normalized tolerance while
having a large cut-relative error. The existing frozen six-dataset comparisons
make this concrete: the largest rank-256 cut-relative errors across tested
partitions are 2.71% for ordinary Tanimoto and 64.47% for the cubed kernel. These
are finite observed errors, not optimizer gaps or worst-case certificates.

Add a separate cut-relative diagnostic and an explicit acceptance criterion,
with an absolute-error floor and an unresolved state for near-zero denominator
mass. Retain the existing normalization for comparison, but name both clearly.
For choosing between candidate splits, audit their *difference* on common
independent pairs: accurate absolute scores need not preserve the ordering of
near ties. Report uncertainty rather than forcing a ranking.

Success criterion: better calibration and fewer reversed candidate rankings at
a fixed audit budget, measured against exact kernels on small datasets and
fresh audit samples at larger sizes. Raising rank alone is not the objective.

### 2. Improve assignment proposals before adding more refinement

`balanced_lloyd` stops a restart when a simultaneous self-excluded proposal
worsens the quadratic objective ([optimize.py](lowrank/optimize.py:133)). This
protects monotonicity but is not a stationarity test. For three or more folds,
the linear assignment itself is a greedy peel, and fold order can matter.
Exact-count swap search considers eight first-endpoint candidates per ordered
fold pair; dev4 makes those checks faster but does not search additional swaps.

First record rejected proposals, stop reasons, restart dispersion, and search
work. Compare alternative fold orders and an exact transportation assignment
on small/moderate point-level cases. A promising experimental proposal for the
unweighted PSD factor objective is to maximize its full linearization rather
than subtracting self terms. If that linearized score is no worse than the
incumbent, convexity of the within-fold squared-norm objective ensures a
non-worsening quadratic step. This argument does not automatically extend to
arbitrary fold-pair weights or the added sparse penalty, and it is not a global
optimality claim. Benchmark it before adopting it.

Use fixed wall-time budgets and retain the best feasible incumbent. The frozen
component study already finds global scores 0.626 for assignment alone versus
0.625 for four starts/two refinements. Extra optimizer sophistication needs to
justify its cost. Exhaustive small cases remain useful because full-rank
representations still have nonzero heuristic gaps there.

### 3. Make powered-kernel accuracy a distinct development target

Cubing similarities changes their spectrum as well as their relative weights.
The current sampler still chooses landmarks using the unpowered
`(1 - maximum similarity)^2` rule
([nystrom.py](lowrank/nystrom.py:66)). A landmark strategy adequate for global
unpowered mass need not adequately represent rare strong relationships.

Compare uniform sampling, the existing spread heuristic, and an explicitly
power-aware or group-stratified strategy under matched memory and time. Evaluate
exact powered-cut quality *after reoptimization*, cut-relative fidelity, and
local contamination separately. A more accurate factor can still lead a
heuristic optimizer to a worse partition. The November development record
already rejects a single global residual direction; the fold-mean pilot is a
dense-oracle experiment, not a scalable validated solution. Do not present
those ideas as untried fixes or enable them by default.

### 4. Make runtime predictable in the public defaults

The current API defaults `leakage_diagnostic=None` and `score_exact_leakage=True`.
Consequently it performs exact quadratic scoring through 100,000 rows, although
explicit `leakage_diagnostic="auto"` switches to sampling above 2,000 rows
([splitter.py](lowrank/splitter.py:227)). I verified this routing with a mocked
scorer at 3,000 rows. A scalable factorization can therefore be followed by a
much more expensive default diagnostic.

Make `auto` the normal new-API behavior, with an explicit compatibility path for
legacy callers. Expose refinement move/time budgets rather than relying only on
the two-million-row cutoff. Separate factorization, assignment, refinement,
neighbour construction, and diagnostics in timings. At large n, accept arrays
or memory-mapped inputs directly and preserve packed binary fingerprints where
possible: the native PCQM4Mv2 run reports 55.4 GiB peak RSS, so factor storage
alone is not an adequate description of memory needs.

### 5. Make group promises and local separation easier to interpret

The newer version already supports supplied groups, strict preservation and
explicit infeasible/unresolved feasibility outcomes. These are substantial
improvements and should not be proposed as missing features. However, default
`group_policy="count_first"` can dissolve groups, and constructed groups use a
capped top-k graph. Preserving these groups is not equivalent to preserving all
pairs above the threshold.

For user-supplied scientific identities or series, prefer strict behavior or
require an explicit choice to relax it. Provide a compact split report covering
retained groups, actual broken original groups, capacity outcomes, and sampled
crossing strong-edge rates. Evaluate ANN recall for threshold-relevant edges,
not only returned top-k neighbours; a top-k reference itself omits strong edges
when degree exceeds k. Keep global overlap and nearest-training exposure as
separate objectives. The experimental local penalty already exists and has no
established consistent advantage over powered grouping.

### 6. Strengthen validation around a declared transfer task

The current molecular evidence is dominated by Morgan fingerprints and datasets
that informed development. Lower similarity in that representation can be a
representation-specific success. Lower predictive scores can also reflect
target imbalance or a task that differs from deployment. The synthetic
new-family experiment establishes a mechanism under its generator; its ridge
reference is given privileged stable features.

Lock a small number of settings on development data, then assess genuinely
untouched tasks using real dates or independently defined project boundaries.
Compare model selection and prediction error on independent outcomes, not just
which split produces the lowest test score. Use multiple feature views and
report target composition, sizes, group contracts and retention. Separate split
seed variation from learner initialization variation. Current code includes a
chronological helper; obtaining real dates/outcomes, rather than adding that
API again, is the missing work.

## What to retain

Keep the exact fold-sum identity, stable truncated eigensystem, exact cosine
feature-map option, sparse identity factors, independent final audits, strict
group feasibility checks, and explicit versioned evidence. Generic kernel
k-means theory is useful context but does not directly certify the unequal-size
cut objective ([Wang, Gittens and Mahoney, 2019](https://jmlr.org/papers/v20/17-517.html)).
The dense control and constraint-matched comparisons are more persuasive than
unqualified win counts against methods solving different problems.

I would prioritize predictable diagnostics and cut-relative auditing first,
then assignment quality under fixed budgets, then powered-kernel development.
A broad new parameter sweep or another increase in restart count would be a
lower-priority use of compute.

## Verification and changes made during this review

The current package was promoted into `low_rank/`; the old v0.1 runtime source
was replaced without a separate v0.1 checkout. Data and research outputs remain.
The ODAC25 working tree was moved using Git's worktree command to
`../maintenance/retired_workspaces/odac25_revision/`. All 263 current package tests
passed; the editable installation, bundled ESOL loading and an exact-count
public-API split were checked. A dataset-path adjustment supports the promoted
namespaced source layout. Numerical splitting algorithms were not changed.

The manuscript was refined separately: the author name was removed from both
papers, repetitive framing was shortened, and five main plus four supplementary
figures were redrawn from frozen observations. Those editorial changes do not
claim that the algorithmic recommendations above have been implemented.
