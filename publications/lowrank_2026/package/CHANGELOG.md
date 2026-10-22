# Changes

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
