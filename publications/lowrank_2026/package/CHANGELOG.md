# Changes

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
