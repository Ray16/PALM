# Enriching the embedding layer for a multi-modal splitting agent

Notes toward making PALM's two engines (`hypergraph`, `lowrank`) split **any**
modality — molecules, mixtures, materials, polymers, proteins, genes — as well as
they currently split ECFP molecules. Grounded in the current code and in what the
CheMixHub chem-OOD exercise (`benchmarks/chemixhub_splits/`) exposed.

## Where we are today

- Both 1-D engines are **bring-your-own-vector**: `feature_matrix_from_dict`
  takes `{id: vector}` and that's the whole embedding story
  (`common/feature_preparation.py`).
- Metric is a **3-way heuristic** on the matrix shape — `choose_metric`:
  binary→tanimoto, sparse→cosine, dense→euclidean. It never sees the *modality*.
- Modality knowledge lives **outside** the engines: `scaffold` /`datasail`
  adapters, dataset loaders. There is no path from a raw SMILES / FASTA / CIF /
  gene-id to a vector + the right metric inside the package.

So "the agent takes molecules, materials, polymers, genes, proteins" is today an
aspiration carried by whatever the caller hand-builds (as I hand-built the
mole-fraction-weighted Morgan mixture vectors for CheMixHub). The enrichment is to
make that layer **first-class, introspectable, and modality-aware**, mirroring how
splitters are already registered + described + tool-exposed.

## 1. An embedder registry (the structural change)

Mirror `registry.py` / `describe_splitters_tool`: a registry of embedders keyed by
modality, each declaring how to embed, its natural metric, and its aggregation
rule.

```python
@register_embedder("smiles")
class ECFPEmbedder(BaseEmbedder):
    metric = "tanimoto"
    def embed(self, records) -> dict[id, np.ndarray]: ...
```

Payoff: the agent introspects `describe_embedders_tool()` (names + input schema +
metric), auto-selects by declared input type, records `embedder@version` in
`SplitResult.diagnostics` for reproducibility, and stops silently forcing every
modality through the same euclidean/cosine guess. `choose_metric` becomes the
*fallback*, not the only policy.

## 2. Per-modality embedders (what to actually put in the registry)

| modality | embedding | natural metric | notes |
|---|---|---|---|
| small molecule | ECFP4/count-Morgan; optional ChemBERTa/MolFormer | tanimoto / cosine | current default; add **count** fingerprints + learned option |
| **mixture / formulation** | mole-fraction-weighted pool of component FPs (salt/role weights) | tanimoto (binarized) or EMD | promote the CheMixHub code; see §3 |
| material | composition stats (Magpie) ⊕ structure (SOAP, M3GNet/ORB embeddings, CrystalNN graph) | cosine (learned) / EMD (composition) | wire to existing `pymatgen`/`mp-api` loaders |
| polymer | repeat-unit periodic Morgan on PSMILES/BigSMILES; polyBERT / Transpolymer | tanimoto / cosine | must handle `[*]` attachment points & stochastic SMILES |
| protein | ESM-2 / ESM-C mean-embedding; **or** MMseqs2 sequence-identity | cosine / precomputed identity | identity-threshold clustering is the leakage-relevant signal |
| gene | sequence, GO semantic similarity, or PPI/co-expression node2vec | precomputed similarity | usually a *graph*, not a euclidean vector — see §4 |

The cross-cutting lesson: several modalities (protein, gene, material) have a
**natural pairwise similarity, not a natural vector**. Forcing them into a
euclidean embedding to fit the current API throws away exactly the structure a
leakage split cares about. Hence §4.

## 3. Set / mixture embedding as a first-class primitive

Many records are **sets or multisets**, not single entities: mixtures, reactions,
protein complexes, alloys, drug combinations. The CheMixHub exercise hand-rolled
all of this; it should be a reusable `SetEmbedder` wrapping any per-component
embedder:

- **pooling**: weighted-mean (done), sum, max, attention/DeepSets;
- **weights**: composition/mole-fraction, with per-role overrides (the salt
  `w=0.5` trick), or learned;
- **identity collapse**: map a set to a canonical key (sorted component sets) so
  repeated measurements share a bucket — the mechanism that drove
  mixture-identity leakage to 0;
- **research upgrade — EMD**: a mixture is a *weighted point cloud* of component
  fingerprints; the **Earth-Mover / Wasserstein distance** between clouds is a
  strictly more faithful mixture metric than mean-pool (which collapses
  {A,B} and {C,D} to the same centroid when A+B ≈ C+D). Feed EMD in via the
  precomputed-similarity path (§4). This is a concrete win over both mean-pool
  and the paper's mean-pool.

## 4. Let both engines consume a precomputed similarity / kNN (highest leverage)

Today the engines start from a vector matrix and *build* similarity internally.
Add a **bring-your-own-similarity** entry point:

- `hypergraph`: it already builds a kNN graph — accept a caller-supplied kNN /
  sparse similarity (or a distance callable) and skip featurization entirely.
- `lowrank`: Nyström needs a PD kernel; accept a similarity **callable** or
  precomputed kernel block and factor that, instead of requiring `X`.

This single change unlocks proteins (MMseqs2 identity), genes (GO/PPI kernels),
materials (structure kernels), and mixtures (EMD) **without pretending they are
euclidean points**. It is the difference between "the agent accepts these
modalities" and "the agent accepts these modalities *well*."

## 5. Metric-layer upgrades (smaller, still valuable)

- Add **count-Tanimoto**, **RBF/Gaussian**, and normalized **learned-embedding
  cosine** to the metric set; today only tanimoto/cosine/euclidean exist.
- **Heterogeneous features**: real records mix blocks (fingerprint ⊕ temperature ⊕
  composition). Support **per-block metric + weight** instead of one global metric
  — e.g. chemistry(Tanimoto) ⊕ 0.2·thermo(euclidean). (For CheMixHub I dropped `T`
  to keep leakage chemical; a general agent should let the user *include* it with
  the right weighting rather than force an all-or-nothing choice.)
- Standardize / whiten dense learned embeddings before the euclidean path; expose
  optional PCA (the `lowrank` engine already Nyström-reduces, but pre-reduction
  stabilizes metric choice).

## 6. Robustness / reproducibility (things the exercise tripped on)

- **Component embedding cache** (unique component → vector), keyed + persisted —
  I did this by hand for every dataset; it belongs in `SetEmbedder`.
- **Sample-count node weights**: engines balance *vertex* count, so set-collapsed
  splits miss sample-fraction targets on high-redundancy data (nist-logV drifted
  to ~62/22/17). One `node_weights` pass-through in `mtkahypar_partition.py` +
  a weighted `target_sizes` in `balanced_lloyd` fixes it for every modality.
- **Graceful fallbacks**: unparseable SMILES / missing embedder → logged identity
  fallback, never a crash (hit this on a few compounds).
- **Versioned, deterministic embedders**: stamp `embedder`, params, and version
  into diagnostics so a split is reproducible from the record ids alone.

## Prioritized next steps

1. **Precomputed-similarity / BYO-kNN input path** for both engines (§4) — biggest
   unlock, turns "multi-modal" from claim into capability.
2. **`SetEmbedder`** (§3) + **sample-count node weights** (§6) — promote the
   CheMixHub code; makes mixtures/reactions/complexes first-class.
3. **Embedder registry + auto metric + tool introspection** (§1) — the structural
   home that makes 1 and 2 discoverable by the agent.
4. **EMD mixture distance** and **learned embedders** (proteins/materials/polymers)
   (§2–3) — research-grade quality upgrades once the plumbing exists.
