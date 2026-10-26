# Joint rank and audit allocation (v0.2.0.dev3)

The fixed-probe policy remains the default. Joint allocation is explicit:

```python
result = split(
    "lowrank", features, spec, rank="auto", rank_check="cut_error",
    rank_probe_policy="adaptive", rank_probe_pairs=2048,
    rank_probe_max_pairs=32768, rank_min=32, rank_max=512,
    rank_tolerance=0.05, on_rank_failure="raise",
)
```

At each candidate rank, the partition is computed without looking at audit
pairs. If its interval establishes the tolerance, selection stops. If its lower
bound exceeds tolerance, rank grows. When uncertainty straddles tolerance,
sampling grows first. At the maximum pair count, a failed check grows rank if
possible; exhaustion of both budgets is an explicit failure, never a pass.
Known-small populations use exact enumeration and do not waste sampling stages.
Landmarks and landmark-kernel blocks are reused across rank increases. The pair
count reached at one rank carries forward to subsequent ranks.

Let R and Q be the numbers of declared doubling stages in rank and pair count.
Each selection interval uses failure probability .025/(R Q). A union bound
covers every possible selection check even though the path is adaptive and
pairs are reused/dependent across checks. The independent final audit uses .025,
so selection and final audit together have at least .95 coverage under the
bounded-kernel sampling assumptions. The historical `ci95` key contains the
interval at its accompanying `confidence`, which need not equal .95.

Diagnostics retain every allocation decision, requested and actual pair counts,
total selection pair evaluations, confidence, chosen rank, selected pair budget,
and the independent final audit. Pair-evaluation totals include repeated checks;
they do not imply cached or statistically independent observations.

The cut criterion measures absolute error for the returned partition divided by
total absolute off-diagonal powered-kernel mass. It is not relative cut error,
a uniform approximation bound over all partitions, or an optimizer certificate.
A successful final audit does not certify near-optimality. Tolerance selection
must reflect the intended evaluation policy and must not use model test outcomes.
