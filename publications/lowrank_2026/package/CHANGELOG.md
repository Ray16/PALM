# Changes

## Joint accuracy/cost policy: 0.2.0.dev3

- Optional joint rank/probe allocation distinguishes established approximation
  error from uncertainty, with finite-grid selection coverage and an independent
  final audit. The fixed-probe default is preserved.
- Explicit probe maxima, allocation history, actual pair-evaluation counts and
  visible budget failures support accuracy/cost comparisons. See JOINT_BUDGET.md.

## Correctness review: 0.2.0.dev2

- Support-aware exact checks and bounded empirical-Bernstein ratio intervals
  replace zero-width normal intervals for rare similarity/leakage events.
- Dense/aligned cosine kernels share stable normalization; large finite inputs
  avoid norm overflow. Euclidean preprocessing avoids variance overflow.
- Binary Tanimoto and finite feature contracts apply across factorization,
  scoring, neighbours and exact refinement. Undefined relative scores use null.
- Visible warn/raise/explicit-return policies for unestablished rank tolerance.
- Cut-aligned selection audits the returned partition, including fold weights.
  It reuses landmarks/kernel blocks and uses a larger default sampling count.
- Candidate refinement precedes restart selection, with a two-candidate public
  default and an explicit all-candidates option.
- Exact sparse n-D identity factors and record-weighted feature deduplication
  remove the dense records-by-identities allocation.
- Exact cosine feature maps avoid unnecessary Nyström work when dimensions fit.
- Symmetric fold-pair policies apply to optimization and weighted diagnostics.
- Reusable exact-backend calibration decisions retain complete cost reporting.
- A supplied-date/order helper preserves ties and returns exact matched counts;
  synthetic helper tests are not presented as real chronological validation.

## Development: 0.2.0.dev1

- Opt-in empirical adaptive rank with independent pair audit, explicit unmet
  tolerances, factor-array budget planning and tiled kernel construction.
- Distinct landmarks and float64, rank-revealing landmark eigensystems; avoid
  amplifying null directions in duplicate or degenerate inputs.
- Original-kernel sampled leakage estimates with Monte Carlo uncertainty;
  explicit exact/sampled/auto/disabled diagnostic modes.
- Supplied entity groups and strict preservation with a bounded, single-threaded
  capacity-feasibility fallback; distinguish proven infeasibility from timeout.
- Experimental normalized global plus positive pairwise local objective, checked
  against dense references; disabled by default, pending comparative evidence.
- Audited exact/ANN selection with sampled recall and complete construction costs.
- Repair missing/repeated/self ANN candidate rows using exact tiled queries;
  the full molecular native audit identified and verifies this case.
- Preserve frozen publication evidence and the running website. Development
  capabilities are not retroactively claimed by the v0.1.0 manuscript.

## Publication review, 21 October 2026

- Capacity repair considers every donor/recipient pair directly reducing a
  violation before breaking a group. Valid moves may exceed the original excess
  or deficit. Group preservation remains conditional under greedy repair.
- CPU FM refinement copies input labels, matching the nonmutating GPU behavior.
- Default low-rank and n-D splits preserve exact target counts. Explicit
  `allowed_size_imbalance` or `balance_slack` enables a corridor;
  `SplitSpec.epsilon` no longer silently relaxes the default. This changes calls
  relying on the older implicit relaxation.
- Ratios that round to an empty fold raise before factorization.
- NN leakage defaults to CPU with an explicit `device` option. The raw score is
  bounded by one; its random-normalized ratio is not.

Earlier implementation corrections and experiment provenance remain in the
manuscript package's source snapshots and reproduction manifests.
