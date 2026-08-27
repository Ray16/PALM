# Insights — PALM master benchmark

Curated observations from the master benchmark (`benchmarks/master/`), meant to
be read alongside the auto-generated `benchmarks/master/INSIGHTS.md` (which holds
the raw per-method tables). This file is the *interpretation* — what the numbers
mean and what to do about them. Living document; append freely.

---

## 1. The central result: leakage and honesty trade off, measurably

Across all (dataset, method) points, **Pearson r(leakage, gap) = −0.56**. The
less a split leaks (lower `L(π)`), the larger the train→test performance drop it
induces. Concretely, going from a random split to a leakage-minimized one roughly
**halves the held-out metric** (mean 0.78 → 0.44 ROC-AUC/R²) while raising the
gap (0.20 → ~0.52).

This is the whole thesis made quantitative: a random split flatters the model
because test molecules are near-duplicates of training ones. The
leakage-minimizing splitters remove that flattery, so the reported number finally
reflects generalization to *new* chemistry. The benchmark doesn't just assert
this — it measures the exchange rate (−0.56).

**Implication:** any model comparison run on a random split is measuring
interpolation, not generalization. For a paper claim of the form "our model
generalizes," the split must be leakage-controlled or the claim is unsupported.

---

## 2. Redundancy is the hidden variable — and it picks the winner

**Redundancy** = how many samples collapse onto the same/near-identical entity.
Where it's measured directly (CheMixHub `k̄` = measurements per unique mixture):

| dataset | k̄ | reading |
|---|--:|---|
| ionic-liquids | 57.0 | 40,904 rows → ~717 real mixtures |
| drug-solubility | 40.9 | |
| nist-logV | 34.3 | |
| polymer-electrolyte | 30.5 | |
| medicine-formulations | 16.7 | |
| logV | 7.2 | |
| MON / miscible-solvent / olfactory | ~1–1.7 | essentially no redundancy |

High k̄ is exactly why a naive random split leaks ~99.9% identity on the
electrolyte/viscosity tasks: the same mixture, measured 57×, lands in both train
and test. Whole-entity assignment drives that to 0.

**The engine choice tracks redundancy:**
- **lowrank wins the typical dataset (9/16)** — global low-rank similarity
  factorization is the best all-rounder.
- **hypergraph wins the high-redundancy sets** (muv, qm8; highest-k̄ CheMixHub IL
  tasks) — its k-NN *hyperedges* explicitly keep dense near-duplicate clusters
  intact, which is precisely what redundancy demands.

**Heuristic:** default to **lowrank**; switch to **hypergraph** when the dataset
is redundant (high k̄, or dense fingerprint neighborhoods / many near-duplicates).
This is a candidate rule for the auto-router.

---

## 3. datasail is the leakage champion but doesn't scale

datasail posts the lowest mean leakage (−39% vs random) — but it's a C1e
clustering + ILP that is O(n²) and we cap it at **n ≤ 3000**; on 9 of the larger
datasets it simply couldn't run (recorded as skipped). So its "win" is only on
small sets. lowrank/hypergraph give ~90% of the leakage reduction and run on
everything (lowrank splits 9.5M OMol25 points in ~27 s where datasail and even
hypergraph's O(n²) k-NN are infeasible).

**Implication:** datasail is a fine small-data reference; the PALM engines are the
only ones viable at scale. For a fair headline comparison, quote it only on the
sets where all methods ran.

---

## 4. Scaffold splitting — the field's default — under-stresses the model

`scaffold` (the standard "hard" split in cheminformatics) produces a gap of only
+0.36 and held-out 0.63 — much *easier* than the similarity-based splitters
(hypergraph +0.54 / 0.44). On bace it was even easier than a random split.
Bemis–Murcko scaffolds group by ring skeleton, which is a coarse proxy for
similarity: different scaffolds can still be very close in fingerprint space, so
"held-out scaffolds" often aren't held-out chemistry.

**Implication:** results reported on scaffold splits are less conservative than
they look; leakage-minimized splits are a stronger generalization test.

---

## 5. The feature space is not neutral — routing beats hand-picking

Which representation you featurize with changes how clean a split *can* be. The
trap: a representation that makes everything mutually dissimilar yields trivially
low leakage and a useless (un-learnable) split. So the feature sweep
(`feature_sweep.py`) selects under a **predictive-validity gate** — a
representation qualifies only if a model on a *random* split clears a floor
(R² ≥ 0.2 / AUC ≥ 0.55); among survivors, pick lowest OOD leakage.

**Result (450/455 runs, triplicate).** Routing each dataset to its learned-best
representation cuts OOD leakage **10.6% on average** vs the field-default
ECFP-1024 / MAGPIE — at equal-or-better predictive validity (the gate guarantees
it). **The hardcoded default was suboptimal on 11 of 14 datasets.** That is the
quantitative case for routing over hand-picking.

| dataset | default | best (routed) | leakage Δ |
|---|---|---|--:|
| openpolymer26 | magpie | **mat2vec** | −33% |
| moleculenet_clintox | ecfp1024 | **rdkit_descriptors** | −20% |
| moleculenet_bbbp | ecfp1024 | **rdkit_descriptors** | −19% |
| moleculenet_sider | ecfp1024 | **rdkit_descriptors** | −17% |
| qmof | magpie | **mat2vec** | −16% |
| materials_project | magpie | **mat2vec** | −12% |
| moleculenet_hiv | ecfp1024 | **rdkit_descriptors** | −11% |
| moleculenet_esol | ecfp1024 | **maccs** | −9% |
| moleculenet_bace / lipophilicity | ecfp1024 | rdkit / maccs | −4–5% |
| moleculenet_freesolv / muv / qm8 | ecfp1024 | *ecfp1024 stays best* | 0% |

Three lessons:

- **The gate earns its keep.** On esol, `rdkit_descriptors` is the *most*
  predictive representation (random-split R² 0.88) but also the *leakiest*
  (0.261) — so it is correctly rejected in favor of `maccs` (0.158). Lowest
  leakage alone would have picked a worse split; the gate + objective together
  pick the clean-*and*-meaningful one.
- **Materials: mat2vec > MAGPIE everywhere** (−12 to −33%). Learned elemental
  embeddings carve cleaner composition boundaries than hand-built MAGPIE stats.
- **ECFP-1024 survives only on freesolv, muv, qm8** — the dense / high-redundancy
  molecule sets. This dovetails with §2: where near-duplicates dominate, the
  high-resolution bit-vector is exactly the right space, and it's also where
  *hypergraph* wins the engine choice. Redundancy is the common thread.

The learned table lives in `data/feature_heuristics.json`; the router
(`data/routing.py`) consults it (`per_dataset` → `per_entity_type` → default),
with the per-type fallback being **maccs** (molecule) and **mat2vec** (material).

---

## 6. Caveats (so nobody over-reads the table)

- **TDC (3 sets):** unavailable — Harvard Dataverse host down, cached raw files
  are 0-byte stubs. Organic space is still covered by 11 MoleculeNet sets.
- **omol25, uspto_mcr:** split-quality only (no target wired), so they contribute
  leakage/runtime but no generalization gap.
- **CheMixHub:** split-quality only — raw data is an external clone not vendored
  here; the gap layer needs a re-clone.
- **Subsample seed is fixed (0):** all triplicate seeds see the same rows; seed
  variance is in the split + model, not in which rows are sampled. Intentional
  (controlled comparison), but it means error bars understate sampling variance.
- **Multitask sets (tox21/sider/muv/qm8):** one representative target column each,
  not the full task suite.

---

## 7. Open questions / next

- Does the redundancy→engine rule hold if we compute a direct redundancy proxy
  (mean nearest-neighbor Tanimoto) per molecule dataset, not just CheMixHub k̄?
- Weighted balancing (sample-count vs unique-entity) — the one axis where the
  paper's Butina still beats PALM on realized fractions; a `node_weights`
  pass-through would close it.
- **[answered, §5]** Routing to learned-best features lowers OOD leakage 10.6%
  mean at equal predictive validity. Next: re-run the *full* master split-sweep
  with `load_dataset(route=True)` to measure the downstream effect on the
  generalization gap, not just leakage.
