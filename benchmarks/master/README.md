# Master benchmark — every dataset × every splitter

One unified sweep that scores **all** PALM splitters on **all** configured
datasets, and records the results in a single long-format table you can come
back to and mine for insights. It answers two questions per (dataset × splitter):

1. **Split quality** — how much does the split leak? (`leakage` = scaled
   cross-split similarity `L(π)`, plus `imbalance`, realized `test_fraction`,
   partition `runtime_s`).
2. **Generalization gap** — is the split genuinely *harder*? A fixed
   RandomForest is trained on the split's train bucket and scored on its test
   bucket (`test_metric` = ROC-AUC for classification, R² for regression);
   `gen_gap = train_metric − test_metric`. The model + features are held constant
   across splitters, so any change in the gap is attributable to the **split**,
   not the model.

The headline hypothesis: a leakage-minimizing split should push the test
molecules further from training, *lowering* the held-out metric and *raising* the
gap relative to a random split — i.e. it produces a more honest OOD evaluation.
`analyze.py` measures the Pearson correlation between leakage and gap directly.

## Coverage

| category | datasets |
|---|---|
| organic (SMILES→ECFP-1024) | MoleculeNet: bace, bbbp, esol, freesolv, lipophilicity, clintox, hiv, tox21, sider, muv, qm8 |
| inorganic (formula→MAGPIE) | qmof, omol25, materials_project* |
| reaction (n-D Morgan)      | uspto_mcr |
| polymer (formula→MAGPIE)   | openpolymer26 |
| mixture (mole-frac ECFP)   | CheMixHub: 12 tasks (folded in from `../chemixhub_splits`) |

\* `materials_project` needs `MP_API_KEY`; `tdc_*` need `python -m PALM.data.prepare_tdc`.
Datasets/methods that can't run are recorded as rows with a `status` + `reason`,
never dropped silently.

**Splitters:** 1-D — `random` (baseline), `hypergraph`, `graph`, `lowrank`,
`datasail` (capped at n≤3000), `scaffold` (SMILES only). n-D — `random`,
`hypergraph_nd`, `hypergraph_nd_knn`.

**Targets & the gap layer:** available for all MoleculeNet sets, qmof (PBE band
gap), openpolymer26 (DFT energy), materials_project (formation energy). Not
available for omol25 / uspto_mcr (no target wired) and CheMixHub (raw data is an
external clone that isn't vendored) — those contribute split-quality rows only,
with the reason recorded.

## Run

```bash
# whole suite on GPU 3 (registry sweep + CheMixHub fold-in + figures + INSIGHTS.md)
benchmarks/master/run_all.sh 3

# just the registry sweep, custom seeds/limit
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
/homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.run_benchmark \
    --datasets moleculenet_esol qmof --seeds 0 1 2 --limit 5000
```

Env: the dedicated `palm` conda env (torch + mtkahypar + rdkit + datasail +
sklearn). One GPU only.

## Outputs

```
results/master_benchmark.csv      registry: one row per (dataset, method, seed)
results/chemixhub_quality.csv     mixture suite: one row per (task, engine)
results/figures/
    leakage_by_method.png         mean L(π) per method (lower = better)
    gengap_by_method.png          mean train−test gap per method
    leakage_vs_gap.png            the headline correlation
    test_metric_by_method.png     mean held-out metric per method
master/INSIGHTS.md                the numbers, in words
```

### Schema (`master_benchmark.csv`)

`dataset, category, task_type, kind, n, method, seed, status, leakage,
imbalance, test_fraction, runtime_s, model, metric_name, train_metric,
test_metric, gen_gap, n_train_lab, n_test_lab, extra, reason`

Aggregate over `seed` (mean ± std) for headline numbers; `status != "ok"` rows
carry a `reason`.

## Default featurizers per entity type  ← the canonical table

Each entity type has one **hand-picked default featurizer**, chosen from the
feature sweep (`feature_sweep.py` → `derive_heuristics.py`) under a
**predictive-validity gate** (a representation qualifies only if a model on a
*random* split still learns; among those, pick lowest reference-space leakage that
doesn't hurt held-out prediction). Full evidence + tradeoff figures:
`../../insight.md` §5/§5b, `figures/feature_tradeoff_*.png`.

| entity type | **default featurizer** | candidates considered | validated per-dataset exception |
|---|---|---|---|
| **molecule** | **ECFP-1024** (Morgan r2, Tanimoto) | ecfp1024, maccs, rdkit_descriptors, chemberta | `moleculenet_bace` → chemberta |
| **material** | **MAGPIE** composition | magpie, mat2vec | — |
| **mof** | **MAGPIE** composition | magpie, mat2vec, linker_ecfp | `qmof` → mat2vec |
| **polymer** | **MAGPIE** composition | magpie, mat2vec | — |
| **protein** | **ESM2** (esm2_t30, 150M) | esm2, sequence_properties | — |
| **gene / RNA** | **canonical k-mer** frequencies | canonical_kmer, kmer, nucleotide_composition (, nt*) | — |

\* Nucleotide Transformer (`nt`) is a candidate but currently deferred (its ~2 GB
download needs local scratch the full local disk lacks).

The machine-readable version is **`data/feature_heuristics.json`**
(`per_entity_type` = the defaults above; `per_dataset` = the exceptions). Recurring
lesson: the leakage-cheaper embedding is repeatedly the *OOD-unsafe trap* (mat2vec
on polymer, seq-props on protein, linker_ecfp on MOF, chemberta off-bace), so the
robust default wins — a lower leakage number is not a better split unless the model
still generalizes.

### Using / regenerating the defaults

```python
from PALM.data.sources import load_dataset
b = load_dataset("moleculenet_bace", route=True)   # -> chemberta (its exception)
b.meta["feature_set"]                               # the featurizer actually used
```

The agent-ready router `PALM.data.routing` chooses features by
`override → per_dataset exception → per_entity_type default`; `describe_router()`
exposes the table to an MCP/agent, and LLM overrides are logged to
`data/routing_overrides.jsonl`. To regenerate from scratch:

```bash
CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
/homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.feature_sweep --seeds 0 1 2
/homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.derive_heuristics
/homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.demonstrate_features
```

## Extending

- **New dataset:** add a loader to `PALM/data/sources.py` returning a
  `DatasetBundle` (set `targets` + `task_type` to get the gap layer; set
  `identifiers` + `identifier_kind` to join the feature sweep + router) and
  register it — the sweep picks it up automatically.
- **New splitter:** register it in `PALM.splitters`; add its name to
  `METHODS_1D` / `METHODS_ND` in `run_benchmark.py`.
- **CheMixHub gap layer:** re-clone `chemcognition-lab/chemixhub`, then wire each
  task's mixture features + measured property into `run_benchmark` (see
  `chemixhub_ingest.py`'s note).
