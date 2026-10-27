# Systematic method and manuscript revision

Base: `aa3a54b871392f701ed319357e1754d7a48eb073`. The reviewed v0.1 manuscript,
v0.2.dev2 numerical evidence, earlier development worktrees and running website
remain preserved. No hypergraph benchmark is permitted. The intended canonical
revision is v0.2.0.dev3; old measurements must retain their producer version until
rerun, and must never be relabelled as measurements of new code.

## Requirements and completion evidence

| Requirement | Work and evidence required | Status |
|---|---|---|
| 1. Contribution and representation | Position against Nyström/kernel clustering; correct Figure 1 to show cluster coarsening with inter-cluster edges; explain capacity and evaluation policies. | Evidence complete; see SCIENTIFIC_REVIEW.md |
| 2. Isolate optimization and grouping | Same retained rows, groups, exact counts and preservation policy for feasible-assignment and factor-optimized controls; report exact small-instance approximation and optimization gaps separately. | Evidence complete; see SCIENTIFIC_REVIEW.md |
| 3. Joint accuracy/cost policy | Finite rank/probe grid with simultaneous selection coverage, independent final audit, visible budget failure, numerical and policy regressions; compare fixed and adaptive budgets without target-based tuning. | Evidence complete; see SCIENTIFIC_REVIEW.md |
| 4. Fair strong comparisons | Native DataSAIL and ordinary weighted k-NN Graph/CUT quality preset; declared cluster/solver/degree budgets and graph construction/repair costs; distinguish matched counts from matched admissible groups; report quality/time/memory and missing outcomes. | Evidence complete; see SCIENTIFIC_REVIEW.md |
| 5. Large-scale quality and capacities | Current-code plain/combined scaling over their measured ranges, exact and 1% corridor; independent returned-cut audits and sampled exact nearest-neighbour exposure; timings, memory, counts and hardware together. | Evidence complete; see SCIENTIFIC_REVIEW.md |
| 6. Evaluation usefulness | Independent-cohort worked example with declared interpolation/new-family regimes; compare estimated risk and model ranking against independent-cohort risk. Synthetic evidence must remain labelled synthetic. Real chronology is conditional on genuine dated data. | Evidence complete; see SCIENTIFIC_REVIEW.md |
| 7. Canonical paper and delivery | Align Methods/algorithms/results to code version, focus main text on four scientific questions, move exhaustive settings to SI/docs; compile and visually inspect main/SI PDFs with figure text >=18pt and no overlaps; push code, evidence and PDFs to GitHub. | Technical evidence and reviewed PDFs complete; delivered in lowrank-v0.2.0-review-20261026 |

## Experimental rules declared before new outcomes

- Use full public feature caches with saved input hashes. Additional analyses of
  previously examined datasets are development evidence, not prospective holdouts.
- Group mechanism study: FreeSolv, ESOL, BACE, BBBP, Lipophilicity and Tox21;
  three seeds; both ordinary and cubed factors, identical retained groups and
  exact 80/10/10 counts. Declare infeasibility rather than silently dissolving
  groups. Include a seeded feasible-assignment control.
- Adaptive-budget study: FreeSolv, BACE and QM8, three seeds; compare fixed
  2,048 and 32,768 probes against joint allocation from 2,048 to 32,768 probes,
  rank 32--512, cut tolerance .05. No model outcomes enter selection.
- Independent-cohort study: generated family data, independent training-pool and
  evaluation draws, interpolation and unseen-family regimes, fixed model recipes
  and seeds. Report both favorable and unfavorable split behavior.
- Exact references enumerate a prespecified small-instance suite and use the
  same capacity/group feasible set for exact and factor objectives.
- Quality/cost frontiers use the six group-study datasets, three seeds,
  ordinary graph degrees 16/32/128 and plain/combined low rank at a .01
  corridor. DataSAIL uses FreeSolv, ESOL, BACE and BBBP, cluster counts 9,
  max(9,n//50), max(9,n//20), and solver budgets 60/300 seconds. Native outcomes
  include distance construction and clustering; matched controls use exact
  native counts, with and without the groups observed at its solver boundary.
  The same group constraints do not imply identical coarsened/entity objectives.
  Native graph and repaired graph assignments/scores are both retained.
- Parallel workers must have disjoint CPU affinities and all native thread pools
  capped. GPU workers require the shared reservation gate, per-node dependency
  preflight, source/input manifests, atomic claims and verified ownership.
- Freeze producer source before launching jobs. Preserve failed/partial outcomes
  and old artifacts. Do not edit source underneath live runs.

The repository has a GitHub origin and no configured Overleaf remote or project
link found during initial inspection. If an Overleaf connection is established,
the standing pre-edit and post-edit pull/merge/build/push workflow applies.
