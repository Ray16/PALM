# lowrank-split

Construct capacity-controlled data splits by optimizing a compressed similarity
kernel. Fold sums replace a stored all-pairs matrix. Optional grouping controls
local analogue transfer; strict supplied groups express domain requirements.
Similarity separation is one property of an evaluation design: it does not by
itself establish realistic deployment performance.

This branch develops `0.2.0.dev2`. The published v0.1.0 reproduction and the
running website remain separately pinned.

```python
from lowrank import split, SplitSpec

spec = SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0)
result = split("lowrank", feature_data, spec,
               rank=256, leakage_diagnostic="auto")
```

`feature_data` maps entity IDs to finite vectors of equal nonzero dimension.
Impute missing values explicitly before splitting. Binary Tanimoto accepts only
0/1 fingerprints; count-valued inputs raise. Empty binary fingerprints have
similarity zero, including to another empty fingerprint. Dense and sampled
kernels share cosine normalization, including its zero-row convention. Signed
cosine values are permitted, but their normalized overlap is not a probability;
nonpositive normalization mass produces an undefined diagnostic rather than a
claim of zero leakage. Negative-mass n-D cosine axes raise explicitly.

Default fold counts are exact. `allowed_size_imbalance` opens an explicitly
requested corridor. Ratios that produce an empty fold raise. Four seeded Lloyd
starts are generated; by default the two best candidates are refined before the
winner is selected. `refine_restarts=1` limits effort, and `None` refines all
candidates. Diagnostics record the selected seed and refinement work. Refinement
checks the actual factor-space objective; it is a local heuristic, not a global
optimality certificate.

## Audited rank selection

```python
result = split("lowrank", feature_data, spec,
               rank="auto", rank_check="cut_error",
               rank_min=32, rank_max=512,
               rank_tolerance=0.05, on_rank_failure="raise",
               factor_memory_mb=1024,
               leakage_diagnostic="sampled")
audit = result.diagnostics["factorization"]
```

`cut_error` checks the powered-kernel cut of the partition actually returned,
including declared fold-pair weights. Error is divided by the kernel's total
absolute off-diagonal mass; it is **not** relative error in the cut value itself.
`pair_mae` checks entrywise mean absolute error with the same normalization.
Neither criterion certifies predictive realism or optimizer optimality.

The cut criterion samples 32,768 pairs by default; entrywise checks sample 2,048.
A larger sampling count can establish fidelity at a lower rank more cheaply
than increasing rank. `rank_probe_pairs` overrides the sampling count. Small
pair populations may instead be enumerated exactly, up to 65,536 evaluations.
Known zero rows are excluded only when their contributions are analytically
zero. Unsampled rare similarity mass is not declared absent.

Sampled intervals use bounded empirical Bernstein mean bounds and a union bound
for numerator/denominator uncertainty. Selection also accounts for its finite
rank grid. An independent pair sample audits the selected factor/partition.
Intervals describe sampling uncertainty conditional on the supplied data, not
uncertainty about future experiments. A weak denominator yields an explicit
insufficient-information state. The legacy `ci95` key can hold a more
conservative selection interval; its actual confidence is recorded.

`selection_tolerance_met`, `audit_tolerance_met` and `stop_reason` retain the
outcome. The default failure policy warns, including when the standalone
`nystrom_features` call omits `return_info`. `"raise"` raises `AccuracyError`;
`"return"` is an explicit choice to collect a failed/unestablished check for
research. A rank ceiling can prevent establishing the requested tolerance.

The Nyström path reuses one landmark sequence and previously computed landmark
kernel blocks as rank grows. Zero kernel rows do not consume landmarks. It uses
a float64 landmark eigensystem and relative numerical-rank truncation rather
than amplifying null directions. Point-to-landmark products are tiled.

`factor_backend="auto"` uses normalized input features as an exact cosine feature
map when power is one and their dimension fits the requested rank. This avoids
unnecessary landmark construction; other cases retain Nyström. Exact here
means the feature-map identity at working numerical precision.

`factor_memory_mb` limits a conservative model of factorization arrays. It is
not a process-RSS limit and excludes framework overhead and subsequent optimizer
work. Fixed ranks that cannot fit raise; automatic selection never silently
lowers the requested minimum to meet the budget.

## Explicit grouping and fold policy

```python
result = split("lowrank", feature_data, spec,
               groups=entity_to_series, group_policy="strict",
               allowed_size_imbalance=0.0,
               leakage_diagnostic="auto")
```

Supply a complete entity-ID-to-group mapping, or request constructed groups with
`must_link=threshold`. `group_policy="count_first"` permits capacity repair to
dissolve groups and reports `broken_groups`. `"strict"` preserves every supplied
or constructed group. A bounded single-threaded MILP fallback checks whole-group
capacity feasibility when greedy repair fails. `GroupInfeasibleError` means the
requirements are proven incompatible; `GroupFeasibilityUnresolved` means no
packing was established within the search limit. An independently checked
integer incumbent can be accepted on timeout.

An explicit size corridor can make whole-group packing feasible. It is never
widened automatically. Constructed groups depend on caps and the selected top-k
edges: preserving them does not guarantee that every pair above a similarity
threshold is kept together. Individual-label randomization with `hardness < 1`
is rejected when grouping is enabled.

```python
# Order follows spec.names. This example emphasizes train/test overlap.
weights = [[0, 1, 2], [1, 0, 0.25], [2, 0.25, 0]]
result = split("lowrank", feature_data, spec, fold_weights=weights)
```

Fold-pair weights must be finite, symmetric, nonnegative and zero-diagonal,
with at least one positive pair. They apply consistently to optimization and
weighted diagnostics. Ordinary overlap remains separately reported, so scores
from different policies are not presented as the same objective. Weights should
express the intended evaluation boundary before examining model test outcomes.

## Scoring and neighbour construction

`leakage_diagnostic="auto"` uses an exact original-kernel score through 2,000 rows
and support-aware diagnostics above that. Explicit modes are `"exact"`,
`"sampled"` and `"none"`. The historical `score_exact_leakage` flag still works
when the new option is omitted. Sample-based results appear in `leakage_sampled`
with actual sample count, interval, status and an `exact` flag; the known small
population path can be exact. Undefined mass is retained explicitly.

Use exact tiled neighbours for validated CPU fingerprint workloads.
`must_link_backend="auto"` / `local_backend="auto"` can perform a cost/recall
calibration. Sampled ANN recall is not a full-graph guarantee. Missing, repeated
or self-neighbour candidate rows are repaired with exact tiled queries. All
construction and rejected calibration costs count toward runtime.

An in-process bounded cache reuses exact-backend decisions for identical inputs,
parameters, dependency versions and resources for up to one hour. Raw features
are not stored in that cache. Approximate results continue to receive a recall
audit. A known explicit backend avoids repeated calibration work across separate
worker processes.

## Multi-component data

The `lowrank_nd` input remains `(records, axis_feature_maps)`. Identity factors
are CSR, so storage follows nonzeros rather than records times distinct
identities. Prepared feature rows are deduplicated before factorization while
preserving record-weighted standardization. The common optimizer handles sparse
fold sums and similarity columns without a dense pairwise matrix. n-D sparse
optimization currently runs on CPU. Per-axis storage and factorization metadata
are returned.

Record-level splitting jointly controls the declared axes. It does not provide
a hard cold-start exclusion contract for every component.

## Experimental local objective

`local_weight > 0` adds a normalized positive pairwise-neighbour penalty to the
global factor cut. It remains opt-in and currently rejects combination with
`groups`/`must_link`. The existing powered/grouped setting remains a strong local
separation control. The penalty does not consistently outperform it.

Development uses threshold 0.4 and k=32 to match existing grouping controls.
Weights 1 and 0.25 respectively test equal and smaller local emphasis after
normalization. They were fixed before measurement. An empty local graph is
reported explicitly; it does not prove absence of all analogues.

## Real chronological references

```python
from lowrank import chronological_reference

ordered_assignment, matched_spec = chronological_reference(
    ids, actual_experiment_dates, [train_cutoff, validation_cutoff])
matched_split = split("lowrank", feature_data, matched_spec)
```

The helper requires complete actual dates/test order and predeclared cutoffs.
Tied order values stay together. Fold sizes follow those cutoffs, and the
returned specification lets other methods match the resulting counts. It
rejects mixed numeric/date order, nonfinite order, mixed naive/aware timestamps
and empty folds. It does not infer dates from publication years or simulate
chronology. Real prospective validation still requires real dated outcomes.

## Installation and evidence

Install with `pip install .`; `.[molecules]` adds RDKit and `.[ann]` adds optional
NN-Descent. PyTorch uses visible CUDA for supported dense operations and CPU
otherwise. Benchmark/cluster jobs must configure resources explicitly. All
support packages are namespaced under `lowrank.*`; no generic top-level package
named `datasets`, `features`, `eval` or `splitters` is installed.

See `CORRECTNESS_PASS.md` and the sibling `correctness_20261025/` directory for
regressions, fixed benchmark protocols, measured trade-offs and limitations.
The older development and publication studies remain preserved separately.
