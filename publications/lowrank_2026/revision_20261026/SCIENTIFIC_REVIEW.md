# Final scientific and pipeline review

This is an internal revision review, not an external journal decision. The
canonical method is v0.2.0.dev3 with numerical source f7e0866. The manuscript
uses current-code primary evidence and identifies earlier-release studies as
archival context. Author-supplied submission metadata remain deferred.

| Accepted review point | Resolution and inspected evidence |
|---|---|
| Contribution and precedent | Introduction explains shared coordinates and fold sums before implementation. Established Nyström/kernel-clustering work is credited. Equal-count equivalence and unequal-count limits are explicit. Figure 1 represents coarsening with inter-group edges, selected graph pairs, and shared coordinates. |
| Optimization versus grouping | Eighteen same-group cases cover six datasets and three seeds. Entities, groups and exact counts are held fixed; factor optimization improves both recorded measures on all six datasets. Twenty exact-reference instances enumerate the same feasible sets and separate approximation from optimizer gaps. Nonzero full-landmark heuristic gaps remain in the paper. |
| Accuracy and computational budget | Optional joint rank/probe allocation uses a finite-grid correction and independent final audit. Fixed-size controls use the same joint-valid policy. All 27 equal-confidence policy cases remain visible, including failure to establish tolerance. The empirical-Bernstein constants and tail allocation were checked against Maurer and Pontil, Theorem 4. A returned-cut audit is never presented as an optimality certificate. |
| Strong and fair comparators | DataSAIL C1e/SCIP and ordinary Graph/QUALITY/CUT comparators address distinct dense/coarsened and sparse representations. Graph degrees 16/32/128 and DataSAIL cluster/time grids are retained. Native counts and observed groups are matched in additional controls. The average global advantage disappears under group matching and is reported. Eight infeasible native configurations are not scored. Native memory is captured before scoring/matched controls. No hypergraph baseline is used. |
| Largest-scale quality and exactness | Two-core scaling has ten completed cases and eight end-to-end budget failures. At 100k, exact/1% median calls are 567.2/40.1 s. Three current two-million-entity splits and independent audits complete on 32-core CPU allocations on lambda13: median 243.9 s, peak 58.8 GiB, returned-cut interval upper endpoints below .008 of kernel mass. Every sampled held-out query is compared with the entire training fold. Host/thread differences are explicit. |
| Evaluation decision | Twenty independently generated family instances compare two fixed model recipes with independent known/new-family cohorts. Both the favorable and unfavorable regimes are reported. The invariant reference's privileged generative features, conditional interpolation cohort and synthetic scope are explicit. No real dates or prospective outcomes are invented. The earlier assay sensitivity case is identified by release and is not treated as a causal intervention on labels alone. |
| Canonical manuscript and delivery | Primary Methods match truncated float64 landmark eigensystems and shortlist refinement before selection. Main text follows four questions; complete grids and archive studies are in SI. Current plots have fixed 6.5-inch canvases and >=18pt text, with text/panel geometry checks and a compiled-PDF font gate. Final reviewed-render signatures and publication checks are recorded separately. |

## Numerical and reproduction checks

The package suite has 223 passing tests. Independent full NumPy kernels check
246 saved global/local scores. The full-kernel fidelity grid has 900 cut
comparisons. Raw assignments, counts, strict groups, source identities and
native exclusions are checked by the publication gate. The standalone wheel
matches all 43 shipped Python sources and runs the joint cut policy in an
isolated target. A focused website-adapter smoke test explicitly loads the new namespace.
The unchanged website regression suite is recorded separately; it is not
claimed as a full new-package compatibility suite. The running service is unchanged.

The earlier CPU controller encountered an NFS claim-cleanup error after
retaining its outputs. Its results are preserved; no observation timeout was
treated as proof that a job was dead. Remaining claim directories are retired
only after producer PID/start identities and predictor absence are verified.
The portable adapter avoids unlink/rmdir races by retaining ownership metadata
in retired directories. Open-reader and changed-owner tests run on the actual
shared filesystem. A fresh case at a different cache path reproduces all saved
group and assignment arrays exactly.

ChEMBL 37 provenance follows the frozen preparation recipe and official release
metadata. A 128-molecule regeneration check reproduces the cached bits; it is
explicitly not a claim of regenerating the full corpus. Input and preparation
hashes identify the measured snapshot.

## Limits retained in the article

The datasets are development/convenience samples. Presentation uses raw mean
nearest similarity; original thresholded measures remain available. No claim
of preregistered confirmatory superiority is made. Powered-kernel fidelity is
substantially weaker than ordinary-kernel fidelity at the same rank. Feasible
sets and cluster objectives differ across methods. A harder split is not
automatically a more useful deployment test. Genuine chronological project
validation and a global optimizer certificate remain future work.

Final submission still requires the author order, affiliations, ORCIDs,
contributions, funding and interests that the user explicitly deferred. The
technical gate reports this separately from scientific/layout checks.
