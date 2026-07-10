# Low-rank factorized splitter — a graph-free leakage minimizer

An alternative splitting backend for PALM that minimizes train/test similarity
leakage **without building a graph or calling a partitioner**. It is a
fundamentally different model from the hypergraph/graph edge-cut backends, and
on the MoleculeNet 1D datasets it is competitive with them and with DataSAIL
while being essentially deterministic and O(n·r).

## The idea

The leakage PALM cares about is the full pairwise similarity across the split,
`L = Σ_{i∈train, j∈test} sim(i,j)` — an O(n²) object. Instead of approximating
it with a sparse k-NN graph (which truncates similarity in dense/congeneric
regions), we **factorize the similarity matrix**:

```
S ≈ B Bᵀ ,   B ∈ ℝ^{n×r}   (r ≪ n)      # Nyström; Tanimoto is a valid PD kernel
```

With per-block feature sums `p_c = Σ_{i∈block c} B_i`, the leakage decomposes
exactly in the factor space:

```
cross-leakage = Σ_{c<c'} p_c · p_{c'} = ½( ‖Σ_c p_c‖² − Σ_c ‖p_c‖² )
```

which is evaluated in **O(n·r)** and never materializes S. Minimizing it is a
**balanced k-means / max-diversity partition in B-space**, optimized with:

1. **Balanced-Lloyd** sweeps (batched, O(n·r·k)/iter, assigns to the exact
   target ratio), with several random restarts (selection is free — it uses the
   O(n·r) factor objective);
2. an optional **Fiduccia–Mattheyses** single-move polish that is provably
   monotone on the low-rank leakage.

No graph, no ILP, no k-NN truncation — the factorization captures the whole
similarity matrix.

### Why this is sound

`derisk_lowrank.py` verifies the two things the method relies on:
- Nyström `BBᵀ` recovers Tanimoto (pair-correlation 0.94–0.99 at r=256);
- the factor-space objective `p₀·p₁` tracks the exact ECFP `scaled_lpi` with
  **correlation 0.995–1.000** at rank 128–256.

So optimizing the cheap factor objective optimizes the real leakage metric.

## Files

| File | Purpose |
|---|---|
| `lowrank_split.py` | Core module: `nystrom_features`, `balanced_lloyd`, `fm_polish`, `lowrank_leakage`, and the entry point `run_lowrank_split`. |
| `test_lowrank_split.py` | Correctness tests: factorization recovery, objective exactness, balance, FM monotonicity, brute-force optimality on tiny n, end-to-end. |
| `derisk_lowrank.py` | Viability study (factorization recovery + objective↔metric correlation). |
| `benchmark_lowrank.py` | Head-to-head vs graph/hypergraph/DataSAIL, variance-controlled, **parallelized one-GPU-per-dataset**. |
| `results/` | `lowrank_benchmark.csv` and the run log. |

## Usage

```python
from PALM.lowrank_split.lowrank_split import run_lowrank_split
split = run_lowrank_split(feature_data, [8, 2], ["train", "test"], rank=256)
# -> {entity_id: "train" | "test"}
```

Run the tests and benchmark (in the `palm` env):

```bash
python  PALM/lowrank_split/test_lowrank_split.py                 # correctness
python -m PALM.lowrank_split.derisk_lowrank                      # viability
python -m PALM.lowrank_split.benchmark_lowrank --workers 8 --gpus 8   # full sweep
```

## Method comparison

| Backend | Model | Objective fidelity | Determinism | Scaling |
|---|---|---|---|---|
| `hypergraph` | per-node k-NN hyperedge, KM1 | coarse (mean, count-once) | noisy (multithread) | O(n·k) + partition |
| `graph` (Tier-1) | weighted k-NN edge-cut | faithful on retained edges | noisy (multithread) | O(n·k) + partition |
| **`lowrank`** | **factorized S≈BBᵀ, balanced k-means** | **full matrix, no truncation** | **~deterministic** | **O(n·r)** |

Results are written to `results/lowrank_benchmark.csv` (see the run log for the
formatted table).
