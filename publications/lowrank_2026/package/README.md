# lowrank-split

Low-rank similarity splitting with specified fold sizes. Dense similarity is
approximated by Nyström factors; integer powers and optional capped groups
provide controls over global and local overlap. Lower similarity is not a
guarantee of a realistic deployment test.

```python
from lowrank import split, SplitSpec

result = split("lowrank", feature_data,
               SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0),
               score_exact_leakage=False)
```

Default low-rank fold counts are exact; ratios rounding to an empty fold raise.
Set `allowed_size_imbalance` explicitly to allow a corridor. Group preservation
is conditional: caps, missing graph edges and capacity repair can split related
entities. Diagnostics retain `broken_groups` rather than claim hard exclusion.
Optional exact leakage diagnostics can dominate runtime and are disabled above.

Install with `pip install .`; `.[molecules]` adds RDKit and `.[ann]` adds the
approximate neighbour backend. PyTorch is required; it uses CUDA when visible
and CPU otherwise. The NN metric defaults to CPU and accepts an explicit device.

All support modules are under `lowrank.splitters`, `lowrank.datasets`,
`lowrank.features` and `lowrank.eval`. The wheel installs no top-level package
named `datasets`, `features`, `eval` or `splitters`.

Dataset loaders are optional research helpers. Public raw datasets and the
manuscript's frozen reproduction sources are distributed separately; this
package does not bundle input datasets or author publication identifiers.

## Development: adaptive and audited splitting

This branch develops the next version; the published v0.1.0 reproduction and
running website are preserved. New capabilities are opt-in. Fixed rank 256,
count-first grouping and the legacy exact-scoring flag remain available.

```python
result = split("lowrank", feature_data,
               SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0),
               rank="auto", rank_min=32, rank_max=512,
               rank_tolerance=0.05, factor_memory_mb=1024,
               leakage_diagnostic="auto")
audit = result.diagnostics["factorization"]
```

Rank increases geometrically until the upper end of an approximate 95% interval
for sampled entrywise error meets the tolerance, or the rank/array budget is
exhausted. Error is mean absolute residual divided by mean absolute powered
kernel similarity on uniformly sampled off-diagonal pairs. A separate pair
sample audits the selected factor. `selection_tolerance_met`,
`audit_tolerance_met` and `stop_reason` expose failures; a factor is still returned
when a rank limit prevents meeting the tolerance. An independent audit failure
is reported without tuning further on that audit sample.

These Monte Carlo checks are not a uniform bound on cut error or a proof of
optimal partitions. Entrywise fidelity can require more rank than useful split
quality. Automatic selection adds work and can be slower than fixed rank;
sharpened kernels often need a larger budget. Select tolerance and budgets before
examining final model test performance.

The factorizer uses distinct landmark indices, a float64 landmark eigensystem
and relative numerical-rank truncation. Duplicate/zero kernels may return fewer
columns than requested, including a zero-column factor for a zero kernel. It
evaluates point-to-landmark kernels in row tiles. `factor_memory_mb` constrains a
conservative model of factorization arrays, not total process RSS; it excludes
framework overhead, caller-owned inputs and subsequent optimization. Fixed ranks
raise `MemoryError` if this model cannot fit. Auto rank does not silently lower
`rank_min` to satisfy the budget.

`leakage_diagnostic="auto"` computes an exact original-kernel score through 2000
rows and a bounded sampled estimate above that. `"sampled"`, `"exact"` and
`"none"` select explicitly. Sampled results are stored in `leakage_sampled` with
Monte Carlo standard error and an approximate 95% interval; they never populate
the exact `leakage` field. The interval does not measure uncertainty about new
datasets. Negligible/nonpositive sampled total mass is explicitly undefined.
Without this option, `score_exact_leakage` retains its old behavior.

## Explicit group contracts

```python
# Every entity must occur once in this id -> group mapping.
result = split("lowrank", feature_data, spec,
               groups=entity_to_series, group_policy="strict",
               allowed_size_imbalance=0.0, leakage_diagnostic="sampled")
```

`group_policy="count_first"` retains the existing capacity repair, allowing group
dissolution and reporting `broken_groups`. `"strict"` forbids dissolution and
checks preservation of every supplied or constructed group. Whole-group packing
uses a bounded, single-threaded binary MILP fallback when greedy repair cannot
find a feasible assignment. This solves capacity feasibility; it does not certify
the globally optimal similarity cut. `GroupInfeasibleError` is a proof of
incompatible requirements; `GroupFeasibilityUnresolved` means no feasible packing
was found within the solver limit. A valid integer incumbent can be used on
timeout after independent checks. Increase `group_feasibility_seconds` explicitly
for a longer search.

For groups that cannot meet exact counts, an explicitly supplied
`allowed_size_imbalance` opens a corridor while keeping strict groups intact.
The algorithm never widens that corridor automatically. Capped top-k groups are
still only groups of selected edges: strict preservation of them does not certify
that **every** pair above a threshold remains together. Supplied groups and
`must_link` are alternatives, and randomizing individual labels with
`hardness < 1` is rejected when either form of grouping is active.

## Experimental global and local objective

```python
result = split("lowrank", feature_data, spec, rank=256,
               local_weight=0.25, local_threshold=0.4, local_k=32,
               local_backend="tiled", leakage_diagnostic="sampled")
```

This experimental variant minimizes the factor-space global cut divided by its
total off-diagonal mass, plus `local_weight` times a normalized cut on an ordinary
pairwise union k-NN graph. Positive local weights follow the ramp
`max(0, (similarity - threshold)/(1 - threshold))`. It has no hyperedges and no
signed residual correction. Guarded Lloyd updates and capacity-preserving
move/swap refinement include both terms. Graph construction is additional work;
no speed or quality improvement is assumed. This variant currently rejects
combination with `groups`/`must_link`; compare it against the established
powered/grouped setting rather than assuming it supersedes grouping.

The development comparison fixes threshold 0.4 and k=32 to match the existing
grouping configuration. After each component is normalized by its total mass,
weight 1 gives equal emphasis and weight 0.25 tests a smaller local contribution.
These two weights were fixed before benchmarking, not selected from model test
outcomes.

The local objective requires positive global factor mass. An empty local graph
is reported as `no_positive_edges_in_graph`, not as proof that analogues are
absent from the data. `local_objective` records both objective components,
normalization, graph construction and experimental status.

## Audited neighbour backend selection

For `must_link_backend="auto"` or `local_backend="auto"`, small datasets retain
exact tiled search. Larger binary-Tanimoto workloads use an exact-query timing
pilot; when projected exact cost exceeds one second, optional NN-Descent is
evaluated. Auto accepts it only if measured full construction is faster than that
projection and sampled top-k recall meets `neighbour_minimum_recall` (default
0.98). It separately measures recall for strong edges. Rejected ANN work remains
included in `total_seconds`. Timing projection is heuristic; sampled recall is
not an all-row guarantee. Explicit `nndescent` retains approximate output and
reports its audit even when recall is below the auto threshold. All returned ANN
edge weights are recomputed exactly. Deficient candidate rows are repaired by
exact tiled queries, with their cost included in construction. Other
representations retain exact search. Auto calibration can itself be expensive;
use a validated explicit backend for repeated work on a known workload.

See [DEVELOPMENT.md](DEVELOPMENT.md) and the sibling `development/` directory for
the fixed benchmark protocol, source identities, tests and measured findings.
The new configuration fields above apply to the 1-D splitter; existing n-D
splitting benefits from the common numerical factorization changes and retains
its existing public interface.
