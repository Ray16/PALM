# Master benchmark — insights

Generated from `PALM/benchmarks/results/master_benchmark.csv` (243 ok runs over 16 datasets, seeds [0, 1, 2]).

## 1. Leakage by method (vs the random baseline)

| method | mean L(pi) | vs random |
|---|--:|--:|
| random | 0.320 | +0% |
| datasail | 0.196 | -39% |
| graph | 0.236 | -26% |
| hypergraph | 0.228 | -29% |
| lowrank | 0.211 | -34% |
| hypergraph_nd | 0.309 | -4% |
| hypergraph_nd_knn | 0.256 | -20% |

## 2. Generalization gap by method

Larger gap / lower held-out metric = the split makes the task genuinely harder (more honest OOD evaluation).

| method | mean gap | mean held-out metric |
|---|--:|--:|
| random | +0.202 | 0.778 |
| scaffold | +0.358 | 0.625 |
| datasail | +0.454 | 0.529 |
| graph | +0.507 | 0.475 |
| hypergraph | +0.537 | 0.444 |
| lowrank | +0.508 | 0.474 |

## 3. Does lower leakage buy a larger (more honest) gap?

Across all (dataset, method) points, Pearson **r(leakage, gap) = -0.56** — see `figures/leakage_vs_gap.png`.
A negative r means: the less a split leaks, the larger the train->test gap it induces, i.e. leakage-minimized splits are harder / more realistic.

## 4. Per-dataset winner (lowest leakage)

| dataset | best method | L(pi) |
|---|---|--:|
| materials_project | lowrank | 0.229 |
| moleculenet_bace | lowrank | 0.232 |
| moleculenet_bbbp | lowrank | 0.215 |
| moleculenet_clintox | datasail | 0.203 |
| moleculenet_esol | lowrank | 0.160 |
| moleculenet_freesolv | lowrank | 0.127 |
| moleculenet_hiv | lowrank | 0.212 |
| moleculenet_lipophilicity | lowrank | 0.245 |
| moleculenet_muv | hypergraph | 0.235 |
| moleculenet_qm8 | hypergraph | 0.184 |
| moleculenet_sider | datasail | 0.182 |
| moleculenet_tox21 | lowrank | 0.174 |
| omol25 | graph | 0.217 |
| openpolymer26 | lowrank | 0.258 |
| qmof | lowrank | 0.239 |
| uspto_mcr | hypergraph_nd_knn | 0.256 |

## 5. CheMixHub mixture suite (split-quality only)

12 mixture tasks; mean mixture-level L(pi) per engine (generalization gap not available without the external clone):

| method | mean L(pi) |
|---|--:|
| random_group | 0.466 |
| butina | 0.339 |
| hypergraph | 0.299 |
| lowrank | 0.184 |
