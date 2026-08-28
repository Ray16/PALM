# PALM.hypergraph — the hypergraph / graph-cut leakage-minimizing splitters

Standalone package for developing the hypergraph data splitters (the original PALM
approach: build a k-NN similarity structure and cut it with Mt-KaHyPar so few
similar pairs land on opposite sides of the split). Kept separate from
`PALM.splitters.methods` — like [`PALM.lowrank`](../lowrank/README.md) — so the
algorithm can evolve on its own; it still registers its four methods into the
shared `PALM.splitters` registry on import.

## How it works
Construct a sparse k-NN similarity structure over the entities, then partition it
under a balanced Mt-KaHyPar objective (few crossing similarities = low leakage):

```
knn.py         k-NN hyperedges / weighted graph edges (GPU + sklearn fallback)  ← construction
partition.py   Mt-KaHyPar KM1 (hypergraph) / CUT (graph) balanced partition     ← cut objective
splitter.py    HypergraphSplitter (@register "hypergraph"), GraphSplitter ("graph")   — 1-D
nd_splitter.py HypergraphNDSplitter ("hypergraph_nd"), HypergraphNDKnnSplitter ("hypergraph_nd_knn"),
               NDInput / _as_nd                                                       — n-D
tests/         test_hypergraph.py
```

Shared low-level kernels (feature prep, the tanimoto/cosine/euclidean similarity,
the exact-leakage Fiduccia–Mattheyses polish, the scaled `L(π)` scorers) stay in
`PALM.splitters.common`.

## Methods & knobs
| name | arity | idea | key `Params` |
|---|---|---|---|
| `hypergraph` | 1-D | one mean-weighted k-NN **hyperedge** per node, KM1 cut | `k` (15), `metric`, `preset`, `use_gpu`, `threads` |
| `graph` | 1-D | weighted 2-uniform k-NN **edge-cut** (CUT) + exact-leakage FM polish | `k`, `threshold`, `max_deg`, `fm`, `fm_max_n`, `preset` |
| `hypergraph_nd` | n-D | per-axis identity/similarity-**cluster** hyperedges (multi-component records) | `sim_threshold` (1.0), `axis_weights`, `preset` |
| `hypergraph_nd_knn` | n-D | per-axis record-level **k-NN** hyperedges (high-cardinality axes) | `k` (25), `preset` |

`preset` maps to Mt-KaHyPar (`default`/`quality`/`highest_quality`/`deterministic`/`large_k`);
use `deterministic` for reproducible runs.

## Use
```python
from PALM.splitters import split, SplitSpec

# 1-D
r = split("hypergraph", feature_data, SplitSpec([8, 2], ["train", "test"], seed=0), k=15)
r.diagnostics["leakage"], r.diagnostics["km1"]

# n-D (multi-component / reaction records)
from PALM.hypergraph import NDInput
r = split("hypergraph_nd_knn", NDInput(records, axis_feature_maps),
          SplitSpec([8, 2], ["train", "test"], seed=0), k=25)
```
Tests: `python -m PALM.hypergraph.tests.test_hypergraph` (needs the `boltz-2` env;
one GPU job at a time — `CUDA_VISIBLE_DEVICES=<free gpu>`).
