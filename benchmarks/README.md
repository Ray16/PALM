# PALM splitter benchmarks

Research + benchmark scripts for the PALM dataset splitters. Every split now goes
through the unified `PALM.splitters` API:

```python
from PALM.splitters import split, SplitSpec

r = split("hypergraph", feature_data,
          SplitSpec(splits=[8, 2], names=["train", "test"], seed=42),
          k=15, preset="quality")
r.assignment      # {id: "train" | "test"}
r.diagnostics     # {"km1", "imbalance", "leakage", "runtime_s", ...}
```

Registered methods: `hypergraph`, `graph`, `lowrank`, `hypergraph_nd`,
`hypergraph_nd_knn`, `datasail`, `scaffold`.

## Layout

```
benchmarks/
  common/       shared helpers (one copy each):
                  featurize.py     ecfp1024 / morgan_matrix
                  datasets.py      MoleculeNet loader (DATA, SMILES_COL, load_smiles)
                  datasail.py      DataSAIL C1e wrappers (fingerprint + distance-matrix)
                  gpu_pool.py      CUDA_VISIBLE_DEVICES + spawn round-robin pool
                  timing.py        CUDA-synced _sync / _time / warmup
                  random_split.py  random 80/20 label makers
  moleculenet/  1-D molecule benchmarks (hypergraph/graph/lowrank vs DataSAIL,
                astartes, DeepChem, Lo-Hi) + leakage.py (scaled_lpi wrapper)
  reactions/    n-D reaction / USPTO-MCR benchmarks + data-prep scripts
  omol25/       OMol25 + UMA-embedding studies (keeps its own _cache_uma/ & results/)
  charts/       figure generators (read the results/ CSVs, do not recompute)
  results/      committed CSV/JSON/PNG outputs for moleculenet + reactions + lowrank
  archive/      superseded scripts (not repointed — see archive/README.md)
```

Leakage scoring is centralized in `PALM.splitters.common.leakage_metrics`
(`scaled_lpi`, `scaled_lpi_smiles`, `macro_axis_lpi`); the MoleculeNet
`leakage.py` and the reaction `leakage_nd.py` are thin wrappers over it.

## Running

Run from the PALM **parent** directory (`/nfs/lambda_stor_01/homes/rzhu`) in an
env with torch + mtkahypar + rdkit (e.g. `boltz-2`):

```bash
# MoleculeNet 1-D: hypergraph vs DataSAIL vs random -> results/moleculenet1d_results.csv
python -m PALM.benchmarks.moleculenet.benchmark_moleculenet1d
python -m PALM.benchmarks.charts.make_chart            # plot it

# low-rank head-to-head (parallel across GPUs) -> results/lowrank_benchmark.csv
python -m PALM.benchmarks.moleculenet.benchmark_lowrank --workers 4 esol bace
python -m PALM.benchmarks.charts.make_benchmark_chart

# reactions (n-D) on the HTE sets -> results/hte_reactions_results.csv
python -m PALM.benchmarks.reactions.benchmark_reactions
# USPTO-MCR n-D split + scaling -> results/mcr_results.csv, mcr_scaling_results.csv
python -m PALM.benchmarks.reactions.benchmark_mcr
python -m PALM.benchmarks.charts.make_mcr_chart
```

The OMol25 / UMA studies live under `omol25/` and read their large caches from
`omol25/_cache_uma/`; see `omol25/README.md`.
