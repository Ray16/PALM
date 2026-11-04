# Native-split production protocol

Declared after metadata/identity preparation and before production split results.
All three metadata collections together used 90,293,593 metered archive/CSV bytes,
including ZIP catalog ranges; this excludes small source-discovery JSON/pages.
No coordinates or processed graph bundles were downloaded.

Datasets: PCQM4Mv2, MOSES, ogbg-molpcba. Use all rows successfully parsed by the
unchanged radius-2, 1,024-bit ECFP recipe. Preserve the original native fold names,
and report original and retained counts. Native source memberships must be
disjoint and complete before filtering. Canonical identities are audited but
not silently deduplicated or used as labels for optimization.

For each dataset evaluate:

* Native fixed assignments, with no imputed construction runtime.
* Uniform count-matched random assignments at seeds 0/1/2.
* Low rank with exact native counts, rank 256, four starts, 25 assignment
  iterations at most, refinement explicitly disabled. This is the bounded
  assignment-only control, labelled as such rather than as the full pipeline.
* The standard plain low-rank pipeline with the same native target proportions,
  rank 256, four starts, two refined candidates and a 1% corridor, seeds 0/1/2.
  The unchanged implementation skips point refinement above two million rows;
  this actual size rule remains visible. Do not reinterpret it as a fixed
  refinement budget across datasets.

Thus there are 30 fresh-process cases: ten per dataset, including native,
three random and six low-rank cases. No model outcomes or response labels
enter settings or optimization. The exact-count and corridor settings answer
different capacity/cost questions and are reported separately. These are
development benchmarks, not new untouched prospective holdouts.

Numerical workers use 16 disjoint assigned CPU slots on lambda2, CUDA hidden,
all native pools capped at 16, no overlapping pinned computations, and at least
120 GiB available RAM before a large case. Initial allocation is CPUs 24--39,
disjoint from metadata preparation on 8--16. Run one numerical case at a time;
verify ownership and source/input identities before claiming work. Timeout is
1,200 seconds end to end per case. Preserve a timed-out or failed attempt as an
explicit censored outcome; do not invent a successful split time.

The split clock covers only the public low-rank call after feature unpacking,
input-adapter construction and a common small warm-up. Peak process RSS is read
before independent scoring. Random generation time and native metadata loading
are distinct fields; the latter is never called native split construction time.
Fingerprint preparation, metadata loading and independent auditing are recorded
outside the split clock.

All methods share 262,144 independently sampled ordered distinct-entity pairs
per dataset. Evaluate exact original Tanimoto from packed bits. Global overlap
is normalized by the analytic random crossing fraction at each actual count
vector. Report approximate 95% Monte Carlo intervals from bootstrap resampling
64 independent pair blocks. The common pairs support paired comparisons.

For each held-out native role, sample 128 uniform queries without replacement
and compare every query with the entire returned training fold using exact
packed-bit Tanimoto. Report the mean, 95% query-bootstrap interval, fraction
above .6 and fraction at one; per-fold definitions are preserved. Aggregate
held-out means weight fold means by their actual populations. These intervals
describe sampling uncertainty, not generalization over future datasets. A
similarity of one is fingerprint identity, not proof of identical geometry.

Native scaffold/CID/generation tasks impose different boundaries from entity-
level cut minimization. This extension compares separation and cost, not the
isolated quality of optimizers with identical native grouping constraints.
Lower global overlap does not establish better scientific evaluation or make
native interpolation/generation tests erroneous.
