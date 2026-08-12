# OMol25 × low-rank splitting

Apply the low-rank leakage-minimizing splitter to **OMol25** (Open Molecules
2025, Meta FAIR — ~83M systems, 83 elements incl. transition metals, organics +
metal-organics + electrolytes + biomolecules, with per-system charge & spin).

## Why low-rank fits OMol25

- **Scale.** OMol25 is millions–tens-of-millions of structures. DataSAIL (O(n²)
  + ILP) and even the k-NN graph backend do not reach that scale; low-rank is
  O(n·r), GPU-native, and deterministic — the only one of the three with a
  realistic path to the full set.
- **Better than the native split.** OMol25 ships a **composition (formula)**
  split: all conformers of a formula stay together. That removes *conformer*
  leakage but not **cross-formula chemical similarity** (a molecule vs its
  homolog, or a complex differing by one ligand). Low-rank operates on a
  continuous similarity, so it suppresses that residual leakage — a concrete,
  defensible improvement over the released split.
- **No algorithm changes.** The splitter is modality-agnostic; only the
  featurizer differs. Dense features → `metric="cosine"`.

## Files

| File | Purpose |
|---|---|
| `omol25_features.py` | Dependency-free featurizer (ase + numpy): composition histogram \| elemental stats \| mini-RDF (3D) \| charge/spin. Runs today, no gated download. |
| `omol25_embed.py` | Quality featurizer: mean-pooled pre-readout embeddings from the pretrained UMA/eSEN model (needs `fairchem-core`). |
| `omol25_split.py` | Loader (`AseDBDataset` → ASE fallback), runs low-rank, compares leakage vs the formula split. |
| `test_omol25.py` | Plumbing test on synthetic ASE structures (no gated data needed). |

## Getting started

**1. Access the data (gated, CC-BY-4.0).** Accept terms at
`https://huggingface.co/facebook/OMol25`, then pull the **`4M`** subset (FAIR's
fast-iteration split) — full download is not needed to prototype.

**2. Smoke test the pipeline today** (no fairchem, no gated model):

```bash
python PALM/lowrank_split/omol25/test_omol25.py         # synthetic structures
```

**3. First real split — cheap features** (needs OMol25 `4M` on disk; `fairchem-core`
gives the native `*.aselmdb` reader, else point `--src` at any ASE-readable file):

```bash
pip install fairchem-core          # for AseDBDataset (native aselmdb reader)
python -m PALM.lowrank_split.omol25.omol25_split --src /path/to/omol25/train_4M --limit 100000
```

This featurizes with `omol25_features.py`, runs the low-rank split, and prints
cross-split leakage (cosine) + nearest-neighbor leakage for **low-rank vs the
formula split**.

**4. Quality run — learned embeddings.** Swap the featurizer for
`omol25_embed.embed_structures(...)` (pooled UMA/eSEN node features) and feed the
matrix to `run_lowrank_split(fd, [8,2], ["train","test"], metric="cosine")`.
Precompute embeddings once (one GPU forward pass per structure, batched) and
memory-map the matrix; then the split is O(n·r).

## Notes / decisions to make

- **Leakage definition for OMol25.** With dense embeddings the natural leakage
  is *embedding cosine similarity*, not formula identity. `omol25_split.py`
  reports both cross-split cosine leakage and NN leakage. (On z-scored features
  cosine is signed, so NN_mean is the cleaner interpretable number.)
- **Feature choice.** Start with `omol25_features.py` to validate plumbing and
  scale; move to `omol25_embed.py` for the quality result — learned embeddings
  are more informative per dimension (lower Nyström rank needed) and avoid
  SOAP's element-count blow-up over 83 elements.
- **Charge/spin matter.** OMol25 samples charge (−10..+10) and spin (1..11);
  the featurizer includes them so redox/spin variants of one formula are treated
  as distinct — important for a faithful leakage split.

Sources: OMol25 paper `arXiv:2505.08762`; `facebook/OMol25` (HF, gated);
fairchem `https://fair-chem.github.io/`; embedding-extraction (pre-readout,
mean-pool) `arXiv:2512.03750`.
