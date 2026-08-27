# PALM splitters — agent-ready leakage-minimizing dataset splitters

A self-contained library of train/test/val splitters that minimize **data
leakage** (cross-split similarity). Every method is a registered
`BaseSplitter` with a uniform interface, typed parameters, and a structured
result — so it can be selected, introspected, and driven by name from code, a
CLI, or an LLM agent / MCP tool.

## Methods

| name | arity | idea |
|---|---|---|
| `hypergraph` | 1-D | k-NN similarity **hyperedges**, Mt-KaHyPar KM1 cut |
| `graph` | 1-D | weighted k-NN **edge-cut** (CUT) + exact-leakage FM polish |
| `lowrank` | 1-D | Nyström factorization `S≈BBᵀ` + balanced-Lloyd + FM (graph-free, **O(n·r)**, scales to millions) |
| `hypergraph_nd` | n-D | per-axis identity/similarity-cluster hyperedges (multi-component records) |
| `hypergraph_nd_knn` | n-D | per-axis record-level k-NN hyperedges (high-cardinality axes) |
| `datasail` | 1-D | adapter: DataSAIL cluster cold-split (C1e) over a custom similarity |
| `scaffold` | 1-D | adapter: Bemis–Murcko generic-scaffold grouping |

## Install

See [`environment.yml`](environment.yml):

```bash
conda env create -f PALM/splitters/environment.yml
conda activate palm-splitters
```

**Core deps:** `torch`, `mtkahypar`, `rdkit`, `scikit-learn`, `numpy<2`, `scipy`.
GPU is optional (CPU sklearn fallback is automatic). Optional extras enable the
adapters and dataset loaders: `datasail` (the `datasail` method), `PyTDC`,
`mp-api` (Materials Project — needs `MP_API_KEY`), `ase` (OMol25 / QMOF).

> On this cluster the ready-made env is **`boltz-2`** (has torch + mtkahypar +
> rdkit + sklearn 1.6 + PyTDC). Run one GPU job at a time
> (`CUDA_VISIBLE_DEVICES=<free gpu>`); never share a GPU between jobs.

## Use it — three surfaces

**Library**
```python
from PALM.splitters import split, SplitSpec, list_splitters, describe_splitters

spec = SplitSpec(splits=[8, 2], names=["train", "test"], seed=0)
result = split("lowrank", feature_data, spec, rank=256)   # feature_data: {id: vector}
result.assignment     # {id: "train" | "test"}
result.diagnostics    # {"metric","leakage","imbalance","runtime_s","split_fractions",...}
```

**CLI**
```bash
python -m PALM.splitters list
python -m PALM.splitters describe --method lowrank
python -m PALM.splitters split --method hypergraph --features feats.npz \
    --splits 8 2 --names train test --seed 0 --out split.csv
```

**Agent / MCP tool** (`PALM.splitters.tool`)
```python
from PALM.splitters.tool import describe_splitters_tool, run_split_tool
describe_splitters_tool()                 # discovery: names + JSON param schemas
run_split_tool("lowrank", features={"a": [...], ...}, splits=[8,2], names=["train","test"])
```
`describe_splitters_tool()` returns each method's name, description, arity, and a
JSON schema of its parameters (derived from the method's `Params` dataclass — the
single source of truth), and `run_split_tool(...)` is fully JSON-in/JSON-out.

## Layout

```
splitters/
  base.py            SplitSpec, SplitResult, BaseSplitter, @register
  registry.py        get_splitter / list_splitters / describe_splitters
  dispatch.py        split(method, data, spec, **params)
  tool.py            JSON-in/out wrapper for agents/MCP
  cli.py             python -m PALM.splitters
  common/            shared kernels (one home for each concern):
    feature_preparation, pairwise_similarity, nearest_neighbors,
    balanced_assignment, split_naming, mtkahypar_partition,
    fiduccia_mattheyses, leakage_metrics
  methods/           hypergraph, lowrank, nD_hypergraph, adapters
  tests/             test_splitters.py (registry, every method, tool, low-rank correctness)
```

## Test

```bash
CUDA_VISIBLE_DEVICES=<free gpu> python PALM/splitters/tests/test_splitters.py
# or: pytest PALM/splitters/tests/test_splitters.py
```

## Add a new method

Drop a module in `methods/`, subclass `BaseSplitter`, declare `name` /
`description` / `arity` / a `Params` dataclass, implement
`split(self, data, spec) -> SplitResult` (use `self._result(...)` to attach the
standard diagnostics), decorate with `@register("your_name")`, and import it in
`methods/__init__.py`. It is then available everywhere — library, CLI, and tool —
with no other changes.
