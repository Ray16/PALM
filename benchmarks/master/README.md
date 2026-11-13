# Master benchmark — every dataset × every splitter

One unified sweep that scores **all** PALM splitters on **all** configured
datasets, and records the results in a single long-format table you can come
back to and mine for insights. It answers two questions per (dataset × splitter):

1. **Split quality** — how much does the split leak? (`leakage` = scaled
   cross-split similarity `L(π)`, plus `imbalance`, realized `test_fraction`,
   partition `runtime_s`).
2. **Generalization gap** — does the split change held-out performance? A fixed
   RandomForest is trained on the split's train bucket and scored on its test
   bucket (`test_metric` = ROC-AUC for classification, R² for regression;
   accuracy when a test bucket is single-class); `gen_gap = train_metric −
   test_metric`.

**What the gap does and does not establish.** Holding the model + features fixed
removes *model* choice as an explanation, but it does **not** make a gap change
attributable to "OOD-ness" alone. A split simultaneously changes (a) the train↔test
similarity (what we want to measure), (b) the test **label distribution** — test
positive rate, test label mean/variance (R² is normalized by test variance, so a
narrower test set lowers R² at identical error), (c) the **effective train size**
(the non-random splitters realize a 0.77–0.89 train fraction vs 0.80 for random), and
(d) class balance / test difficulty. A larger gap therefore shows that the split is
*harder for this model*, not that it is a more honest estimate of deployment error,
and a lower L(π) is not by itself evidence of a better benchmark. Pooling points
across heterogeneous datasets (different metrics, scales, and unequal method
coverage — DataSAIL only runs at n ≤ 3000, scaffold only on SMILES sets) compounds
this: cross-dataset correlations and method means mix dataset difficulty with
split effect. `analyze.py`'s pooled Pearson r(leakage, gap) is descriptive only.

`paired_analysis.py` re-reads the saved results (no retraining) and reports what
*is* supported — see [Paired, coverage-matched re-analysis](#paired-coverage-matched-re-analysis).

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
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
python -m PALM.benchmarks.master.run_benchmark \
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
results/paired_effects_per_dataset.csv   paired_analysis.py: method − random per dataset
results/paired_effects_summary.csv       paired_analysis.py: coverage-matched aggregates
results/label_shift.csv                  paired_analysis.py: train/test label stats per verified split
master/INSIGHTS.md                the numbers, in words
```

### Schema (`master_benchmark.csv`)

`dataset, category, task_type, kind, n, method, seed, status, leakage,
imbalance, test_fraction, runtime_s, model, metric_name, train_metric,
test_metric, gen_gap, n_train_lab, n_test_lab, extra, reason`

Aggregate over `seed` (mean ± std) for headline numbers; `status != "ok"` rows
carry a `reason`.

## Paired, coverage-matched re-analysis

`python -m PALM.benchmarks.master.paired_analysis [--reconstruct]` (CPU, no model
retraining) re-reads `master_benchmark.csv`. Numbers below are from the current
results (seeds 0–2, 1-D datasets with a target).

**Design.** Effects are *within dataset*: `method − random` on `test_metric` and
`gen_gap`, paired by seed, and a pair is dropped unless both runs used the same
metric (MUV silently falls back to accuracy on single-class test buckets, so it
drops out). Intervals are 95% t-intervals over the 3 paired seeds — note the
non-random splitters are deterministic, so their replicates are the *same*
partition with a re-seeded RandomForest; the interval is model + random-split
variance, not method variance. Aggregates are per metric family, only over
datasets where every compared method has a paired effect, with a
bootstrap-over-datasets 95% CI.

**Coverage-matched mean Δ held-out metric vs random (95% bootstrap CI; datasets below random)**

| method set / datasets | graph | hypergraph | lowrank | datasail | scaffold |
|---|--:|--:|--:|--:|--:|
| R², 7 sets (mp, esol, freesolv, lipo, qm8, openpolymer26, qmof) | −0.46 [−0.62, −0.33] 7/7 | −0.50 [−0.65, −0.36] 7/7 | −0.45 [−0.57, −0.34] 7/7 | n/a | n/a |
| ROC-AUC, 6 sets (bace, bbbp, clintox, hiv, sider, tox21) | −0.22 [−0.29, −0.15] 6/6 | −0.24 [−0.34, −0.16] 6/6 | −0.14 [−0.18, −0.11] 6/6 | n/a | n/a |
| R², all 6 methods (esol, freesolv) | −0.48 | −0.64 | −0.47 | −0.44 | −0.47 |
| ROC-AUC, all 6 methods (bace, bbbp, clintox, sider) | −0.24 [−0.33, −0.15] | −0.30 [−0.40, −0.23] | −0.16 [−0.20, −0.13] | −0.17 [−0.19, −0.12] | −0.05 [−0.09, −0.01] |

(Δgap mirrors Δtest because train metrics barely move.) Per-dataset cells with
seed CIs are in `results/paired_effects_per_dataset.csv`; the only per-dataset
CIs that include 0 are scaffold on bbbp/clintox/hiv/sider, hypergraph/lowrank
on hiv/tox21, and lowrank on openpolymer26.

**Leakage ↔ gap.** Pooled r = −0.55 (63 dataset×method points), within-dataset
(centered) r = −0.59, but **within-dataset among non-random methods only r = +0.25**:
the negative correlation is carried by the random-vs-everything contrast, not by
lower-L(π) methods producing larger gaps. The leakage ranking (lowrank lowest on
most sets) does not predict the gap ranking (hypergraph largest drop).

**Label shift and train size** (`results/label_shift.csv`). Only splits whose
assignment could be recovered *and* matched to the master row (identical labeled
train/test counts and, where recorded, identical L(π)) are used: saved seed-0
assignments for bace and esol (all methods), recomputed random seeds everywhere,
legacy scaffold on most SMILES sets, and hypergraph/graph on openpolymer26/qmof
(the other hypergraph/graph recomputations no longer reproduce the saved rows and
were dropped). Random splits keep test labels within ±0.14 SD of train (and
positive rate within ±0.05). Similarity-driven splits do not:

| dataset | method | test vs train label | Δ test metric vs random (seed 0) |
|---|---|---|---|
| bace | hypergraph | positive rate 0.28 vs 0.50 | AUC 0.47 vs 0.89 (below chance) |
| bace | lowrank | positive rate 0.27 vs 0.50 | AUC 0.77 vs 0.89 |
| bace | datasail | positive rate 0.57 vs 0.43 | AUC 0.71 vs 0.89 |
| esol | graph / lowrank | mean shift +0.60 / +0.47 SD | R² 0.29 / 0.09 vs 0.66 |
| esol | hypergraph | test std 1.65 vs 2.29 (random test) | R² −0.06 vs 0.66 |
| freesolv | scaffold (legacy) | mean shift −0.61 SD | R² 0.12 vs 0.59 |
| openpolymer26 | hypergraph | mean shift +0.62 SD, test std 0.43× random's | R² 0.55 vs 0.85 |
| qmof | graph | mean shift +0.75 SD | R² 0.33 vs 0.68 |

Train size runs the other way: non-random splits leave *more* training data
(train fraction 0.80–0.89; scaffold 0.77–0.82), which should make them easier, so
it does not explain the drop.

**What survives.** (1) Every leakage-minimizing splitter lowers the held-out
metric relative to random on every coverage-matched dataset, well outside seed
noise — the splits are reliably *harder for this model*. **What does not.**
(2) "Attributable to the split's OOD-ness": the split also moves the test label
distribution substantially (and R² is normalized by the test variance it shrinks),
so part of the drop is label shift, not similarity alone; a below-chance AUC (bace
hypergraph) is a label/structure artefact, not an honest OOD estimate.
(3) "Lower leakage ⇒ larger / more honest gap": not supported among non-random
methods (r = +0.25). (4) Cross-method rankings from `INSIGHTS.md` §2 pooled means
are not coverage-matched; on matched sets most method CIs overlap — the
separable differences are on ROC-AUC, where hypergraph's drop exceeds lowrank's
and datasail's and scaffold's is the smallest. None of this ranks methods by
benchmark *validity*.

**Missing to go further:** per-seed assignments are not saved by
`run_benchmark.py`, so label-shift diagnostics for lowrank/datasail on 14 of 16
datasets, and for hypergraph/graph on most, cannot be recovered without
re-partitioning at the exact code version used; a label-shift-controlled
comparison (e.g. stratifying the test label distribution, or reporting RMSE/MAE
alongside R²) needs new runs.

## Default featurizers per entity type  ← the canonical table

Each entity type has one **hand-picked default featurizer**, chosen from the
feature sweep (`feature_sweep.py` → `derive_heuristics.py`) under a
**predictive-validity gate** (a representation qualifies only if a model on a
*random* split still learns; among those, pick lowest reference-space leakage that
doesn't hurt held-out prediction). Full evidence + tradeoff figures:
`../../docs/insights.md` §5/§5b, `figures/feature_tradeoff_*.png`.

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
CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
python -m PALM.benchmarks.master.feature_sweep --seeds 0 1 2
python -m PALM.benchmarks.master.derive_heuristics
python -m PALM.benchmarks.master.demonstrate_features
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
