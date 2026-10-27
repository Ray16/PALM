# Publication revision: argument and evidence map

The primary manuscript will describe the current v0.2.0.dev3 method (numerical
source f7e0866), with a single coherent Methods description. Older v0.1 studies
remain explicitly identified as archival context, not relabelled current-code
measurements. New results replace older headline comparisons only after their
assignments, metrics, source identities and full study outcomes are checked.

## Central argument

Dense similarity-cut optimization is useful for specifying evaluation boundaries,
but an explicit all-pairs matrix is expensive. Shared low-rank coordinates make
the objective accessible through fold sums under declared capacities. Global
overlap and nearest-training-neighbour exposure describe different boundaries.
Their usefulness depends on the intended transfer task.

## Four questions in Results

1. **Does compression preserve the objective, and does optimization add value?**
   Current-code full-kernel ordinary/powered fidelity; identical-group feasible
   versus optimized assignments; exact small-instance approximation/optimization
   decomposition. Main: fidelity and matched-group effects. SI: exhaustive gaps,
   rank/probe choices and allocation history. An audit of the returned cut is
   distinguished from a uniform approximation or an optimality certificate.
2. **What quality/cost choices does the method offer?**
   Six full datasets and three split seeds for plain/combined and ordinary graph
   degrees 16/32/128; four datasets and a declared cluster/solver-budget grid for
   DataSAIL, with exact native-count controls and native-group controls. Retain
   native infeasibility, graph repair and objective differences. Main: representative
   quality and cost comparisons; SI: every configuration and solver outcome.
3. **How do scale and capacity tolerance change cost and fidelity?**
   Current-code exact and 1% corridor CPU cases with independent cut/NN audits,
   plus any verified reservation-backed GPU measurements. Report censored outcomes
   explicitly. The earlier 19.9-s H100 result belongs to v0.1 and must not become
   a current-code headline by implication. Space complexity is not a runtime claim.
4. **When does the boundary give a useful evaluation decision?**
   Twenty independently generated synthetic instances: interpolation versus unseen
   families, two fixed model recipes, held-out risk versus independent-cohort risk
   and model ranking. State the privileged invariant-feature assumption. Include
   the unfavorable regime. The archival external-assay failure case demonstrates
   target-composition sensitivity; it does not validate prospective prediction.

## Main-text choices

- Explain why low rank is useful before its numerical implementation. Cite the
  established Nyström/kernel-clustering relationship without claiming its novelty.
- Primary global score: dense cut divided by its analytic random expectation at
  actual counts. Primary local score: mean test-to-training maximum similarity,
  which is already retained by the full common scorer. Threshold ramps and
  alternate fingerprints remain supporting diagnostics with explicit definitions.
- Keep solver/API options, complete grids, archive comparisons, D-MPNN reuse,
  calibration recipes and backend details in SI/software docs.
- Keep equations for the objective and fold-sum identity in the main article;
  detailed move formulas and conditional bounds can live in SI.
- Use generated numbers and tables, 6.5-inch vector canvases with text >=18pt,
  geometric checks plus compiled-PDF inspection. Avoid overlapping panels and text.

## Checks required before publication

One comparability issue was identified while drafting: the original fixed-probe
policy uses separate 95% selection and final-audit coverage, whereas the new joint
policy budgets at least 95% for the whole procedure. The accuracy/cost comparison
must therefore include fixed-size probe grids run through the same joint-valid
policy (min=max pair budget), rather than attribute the cost of a stricter
confidence requirement to adaptation alone. No model outcomes select these settings.

Author order, affiliations, ORCIDs, funding and interests remain author-supplied
submission metadata. They were deferred by the user and must not be invented.
