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

## 5. The feature space is not neutral — so we hand-pick per type and demonstrate it

*(This section was rewritten twice: after hardening the pipeline (§5a), and again
after the sweep showed per-dataset routing isn't worth its complexity — so we
**hand-pick a safe featurizer per entity type and demonstrate it**, rather than
route.)*

**The decision:** don't route per dataset. Use one hand-picked featurizer per
entity type, chosen from the sweep evidence, with a short list of validated
exceptions:

| entity type | hand-pick | validated exception |
|---|---|---|
| molecule | **ECFP-1024** | bace → ChemBERTa |
| material | **MAGPIE** | qmof → mat2vec |

**The demonstration** (`demonstrate_features.py` → `figures/feature_tradeoff_*.png`),
mean over each type's datasets, reference-space leakage vs held-out OOD metric:

| type | feature | ref-leakage | mean OOD | worst OOD | verdict |
|---|---|--:|--:|--:|---|
| molecule | **ecfp1024** | **0.208** | 0.47 | 0.06 | hand-pick: lowest leakage, OOD-safe |
| molecule | maccs | 0.237 | 0.55 | 0.12 | higher leakage |
| molecule | rdkit_descriptors | 0.278 | 0.52 | 0.02 | leakiest |
| molecule | chemberta | 0.229 | 0.41 | **−0.91** | OOD-UNSAFE (good only on bace) |
| material | **magpie** | 0.24 | 0.20 | 0.12 | hand-pick: safe, best mean OOD |
| material | mat2vec | 0.22 | 0.10 | **−0.28** | OOD-UNSAFE (good only on qmof) |

So the hand-picks are justified directly: **ECFP-1024 has the lowest molecule
leakage and never goes OOD-harmful; MAGPIE is the safe material default.** The two
alternatives that *look* attractive on leakage (chemberta, mat2vec) each collapse
generalization on at least one dataset — which is exactly why they're per-dataset
exceptions, not defaults. A lower leakage number is not a better split unless the
model still generalizes.

*(The paragraphs below give the per-dataset detail and the post-mortem on why the
first, un-hardened pass over-claimed.)*

The feature sweep (`feature_sweep.py`) featurizes each dataset with every
candidate representation, then selects with three guards:

1. **Predictive-validity gate** — keep only spaces where a model on a *random*
   split clears a floor (R² ≥ 0.2 / AUC ≥ 0.55), so a space can't win by being
   uninformative.
2. **OOD-prediction guard** — among gated spaces, drop any whose *held-out* metric
   is materially worse than the best (tol 0.03), so we don't trade real signal for
   a cleaner-looking split.
3. **Reference-space leakage + significance margin** — score every split's leakage
   in one fixed space (ECFP / MAGPIE) so candidates are comparable, and only
   override the default when the win exceeds `max(5%, pooled-std)`.

**Result (450/455 runs, triplicate).** Under fair, noise-aware selection, routing
beats the ECFP/MAGPIE default on **only 2 of 14 datasets** — and both are real,
validated by *better* OOD prediction, not just lower leakage:

| dataset | default | routed | ref-leakage Δ | OOD metric (default → routed) |
|---|---|---|--:|---|
| moleculenet_bace | ecfp1024 | **chemberta** | −6.5% | 0.60 → **0.81** |
| qmof | magpie | **mat2vec** | −8.5% | 0.33 → **0.35** |

Everything else falls back to the default — ECFP is the right choice for **10/11**
molecule sets, MAGPIE for 2/3 material sets.

**The guard prevented two actively harmful picks** the naive version would have
made:

- **openpolymer26 → mat2vec** looked like the biggest win (−14% ref-leakage, −33%
  in the buggy first pass) — but its OOD R² *collapses to −0.28* (worse than
  predicting the mean). Cleaner split, un-generalizable model. Correctly rejected.
- **moleculenet_sider → chemberta** (−11% leakage) also lowers OOD performance →
  rejected; ECFP kept.

Lessons:

- **ECFP-1024 is a genuinely strong default for molecules.** Once leakage is
  scored fairly and noise is accounted for, the field default is hard to beat —
  it wins or ties 10 of 11 sets. The interesting exceptions are informative:
  **bace** rewards a semantic embedding (ChemBERTa), consistent with its being a
  single-target binding task where substructure bits alone under-describe.
- **Materials: mat2vec helps on qmof** (cleaner *and* better-predicting) but not
  universally — the "mat2vec beats MAGPIE everywhere" claim from the first pass
  was an artifact (see §5a).
- **A lower leakage number is not a better split** unless the model still
  generalizes. Two of the biggest raw leakage drops were the *worst* choices.

Table: `data/feature_heuristics.json`; router `data/routing.py`
(`per_dataset` → `per_entity_type` → default). Per-type fallback (minimax regret):
**ecfp1024** (molecule), **mat2vec** (material).

## 5a. Why the first-pass feature results were wrong (and what fixed them)

The initial sweep reported routing winning **11/14 datasets at −10.6% mean
leakage**. That was mostly artifact. Three bugs, three fixes:

- **Circular objective.** Leakage was measured *inside each candidate's own
  space*, so a 0.16 in maccs-space was compared to a 0.17 in ecfp-space — not the
  same ruler. *Fix:* score every split's leakage in one fixed **reference space**
  (`scaled_lpi(X_ref, labels)`). Under a common ruler, maccs's esol split went from
  looking cleaner (0.156) to actually leakier (0.207).
- **No significance test + shared rows.** All three "seeds" split the *same* rows,
  so deterministic methods had zero variance and any tiny difference read as a
  win. *Fix:* each seed draws a different subsample, and a win must exceed the
  pooled std.
- **Leakage-only objective.** It ignored the OOD performance it had already
  measured, so it happily picked un-generalizable spaces (openpolymer mat2vec).
  *Fix:* the OOD-prediction guard above.

The corrected pipeline is more conservative and more correct: it makes 2 confident
recommendations, blocks 2 harmful ones, and otherwise trusts the default. The
lesson for the router is that **feature routing helps in specific, verifiable
cases, not as a blanket win** — and the machinery now proves each case rather than
asserting it.

---

## 5b. Default featurizers for the other entity types (protein / RNA / MOF / polymer)

Same sweep + gate + demonstrate methodology, extended to four more types
(`figures/feature_tradeoff_{protein,gene,mof,polymer}.png`). Hand-picked default
per type:

| type | dataset | default | runner-up | reading |
|---|---|---|---|---|
| **protein** | lp_pdbbind | **esm2** (150M) | sequence_properties | ESM2 lower leakage (0.20 vs 0.27) and the only OOD-safe option; seq-props goes OOD-negative. *Target (binding affinity) is ligand-dependent, so absolute OOD is low for both — but ESM2 is clearly the better representation.* |
| **RNA** | rfam (13-class) | **canonical_kmer** | kmer | Canonical k-mers lowest leakage (0.244); OOD family-classification accuracy ~0.29 on the held-out split (chance = 0.077). NT deferred (needs the ~2 GB download the full local disk blocked). |
| **MOF** | qmof | **magpie** (type default); **qmof → mat2vec** (exception) | linker_ecfp | mat2vec is best on the one MOF we have (lower leakage + better OOD), so it's the validated exception; `linker_ecfp` is OOD-unsafe (−0.01). Type default stays MAGPIE for a novel MOF. |
| **polymer** | openpolymer26 | **magpie** | mat2vec | mat2vec is leakage-cheaper (0.233 vs 0.270) but **OOD-harmful (R² −0.28)** — the same trap as elsewhere. MAGPIE is the safe pick. |

Two honest caveats specific to these: each new type currently has **one dataset**,
so its per-type default is single-dataset evidence (the per-dataset exception
mechanism captures the actual best, e.g. qmof→mat2vec); and the protein target is
noisy (affinity averaged over ligands), so protein OOD magnitudes are weak even
though the featurizer ranking is clear. The recurring rule holds across all six
types: **the leakage-cheaper embedding is repeatedly the OOD-unsafe trap
(mat2vec on polymer, seq-props on protein, linker_ecfp on MOF), and the robust
default wins.**

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
- **[answered, §5]** Under fair, noise-aware selection, feature routing is a
  *specific* win, not a blanket one — 2/14 datasets (bace→chemberta, qmof→mat2vec),
  both validated by better OOD prediction; the default is right elsewhere. Next:
  re-run the *full* master split-sweep with `load_dataset(route=True)` to confirm
  the two routed picks also help the downstream generalization gap end-to-end.
- Does ChemBERTa help beyond bace if given a better pooling / a fine-tuned
  variant? It never wins elsewhere — is that the representation or the (frozen,
  mean-pooled) usage?

---

## 8. Systematic method comparison — PALM engines vs DataSAIL vs baselines

The head-to-head across all 20 datasets (6 entity types), each featurized with its
hand-picked default, triplicate — `benchmarks/master/COMPARISON.md`,
`figures/compare_*.png`. Methods: PALM (`hypergraph`, `lowrank`, `graph`),
`datasail`, external baselines (`astartes`, `lohi`, `scaffold`), `random`. DataSAIL
and lohi are ILP-based and only run at n≤3000; DataSAIL now uses mmseqs2 for
protein/RNA homology.

| method | mean L(π) | coverage (datasets) | note |
|---|--:|--:|---|
| datasail | **0.208** | 7 | lowest — but O(n²)+ILP, **n≤3000 only** |
| **lowrank** (PALM) | **0.213** | **20** | best among methods that scale; **11/20 wins** |
| hypergraph (PALM) | 0.226 | 20 | wins high-redundancy sets |
| graph (PALM) | 0.232 | 20 | |
| lohi | 0.232 | 7 | molecule-only, ILP |
| astartes | 0.317 | 20 | kmeans sampler ≈ random |
| random | 0.320 | 21 | baseline |

**Conclusion.** DataSAIL has marginally lower leakage *but only on the small
subset its ILP can solve* (7 of 20 datasets). **lowrank is the best all-round
splitter: it nearly matches DataSAIL's leakage while running on every dataset at
any scale, and wins the most datasets outright (11/20).** hypergraph takes the
high-redundancy sets. The external baselines are informative negatives: astartes'
kmeans sampler barely beats random (it doesn't target cross-split similarity), and
lohi — while a genuine leakage-minimizer — is molecule-only and ILP-bounded. The
PALM engines are the only leakage-minimizers here that combine low leakage with
scale and generality across all six entity types.

### Baseline installation note
mmseqs2 + cd-hit (`~/.local/bin`) and astartes + lohi (pip, `--no-deps` into the
`palm` env) were installed despite the chronically-full local root disk by routing
every conda/pip/HF cache and TMPDIR to NFS. The Nucleotide Transformer (RNA) is the
one baseline still deferred — its ~2 GB download needs local scratch that isn't
free.
