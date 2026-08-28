# Systematic method comparison — PALM engines vs DataSAIL vs baselines

From `PALM/benchmarks/results/master_benchmark_routed.csv` (21 datasets, hand-picked features per type, triplicate). Leakage scored on each dataset's routed features; lower is better.

## Mean leakage by method (with coverage = # datasets it can run on)

| method | mean L(π) | coverage | note |
|---|--:|--:|---|
| random | 0.320 | 21 | baseline |
| astartes | 0.317 | 20 | kmeans sampler |
| lohi | 0.232 | 7 | molecule-only, ILP, n≤3000 |
| graph | 0.232 | 20 | **PALM** |
| hypergraph | 0.226 | 20 | **PALM** |
| datasail | 0.208 | 7 | O(n²)+ILP — only n≤3000 |
| lowrank | 0.213 | 20 | **PALM** — runs at any scale |
| hypergraph_nd | 0.309 | 1 |  |
| hypergraph_nd_knn | 0.256 | 1 |  |

## Win-rate — datasets where each method leaks least (excl. random)

| method | wins |
|---|--:|
| lowrank | 11 |
| datasail | 3 |
| hypergraph | 3 |
| graph | 2 |
| lohi | 1 |
| hypergraph_nd_knn | 1 |

## Takeaways

- **lowrank is the best all-rounder**: lowest mean leakage among methods that run on *every* dataset, and the most per-dataset wins. It scales where DataSAIL cannot.
- **DataSAIL edges leakage only on the small (n≤3000) subset it can solve** — its O(n²)+ILP can't run on the larger sets, so its coverage is partial.
- **hypergraph** is competitive and wins the high-redundancy sets.
- **astartes** (kmeans sampler) sits near the random baseline — it does not minimize cross-split similarity the way the leakage-targeting methods do.
- **lohi** genuinely lowers leakage but is molecule-only and ILP-bounded.
