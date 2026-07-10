# OMol25 leakage study — results

Merged dataset = OMol25 `train_4M` + `val` + `test` = **9,553,821 structures**
(train 3,986,754 / val 2,762,021 / test 2,805,046), downloaded to
`PALM/data/DataSAIL_data/1D/omol25/` and featurized to
`_cache/features.npy` [9553821 × 115] + `_cache/meta.parquet`.

## Metric

DataSAIL's **L(π)** form — the fraction of total pairwise similarity that
crosses split boundaries, `L = 1 − Σ_c‖p_c‖²/‖s‖²` — computed with a **cosine**
similarity over a non-negative structural descriptor
(composition | elemental stats | 3D radial-distance histogram | charge± | spin).

*Why not ECFP/Tanimoto (DataSAIL's small-molecule default)?* OMol25 has no SMILES
and includes transition-metal complexes RDKit can't reliably perceive, so a
3D/composition structural similarity is the defensible choice. The similarity is
non-negative so L(π) ∈ [0, 1] like Tanimoto.

**Estimator validated:** factorized L(π) = **0.6573** vs exact O(n²) cosine =
**0.6573** (|diff| = 0.0000) on a 40k subsample — the low-rank estimate is exact
to ~1e-4, so it is trustworthy at full scale.

## 1. Leakage of the existing split vs a low-rank re-split (full 9.55M, 3-way native proportions)

| split | L(π) | sizes (train/val/test) |
|---|--:|---|
| **existing native** (train_4M/val/test) | **0.6571** | 3,986,754 / 2,762,021 / 2,805,046 |
| **low-rank re-split** (same proportions) | **0.6387** | 3,986,754 / 2,762,021 / 2,805,046 |
| random baseline (1 − Σ f_c²) | ≈0.657 | — |

**Reading:** under structural-cosine similarity the native **composition split is
essentially as leaky as random** (0.657), and low-rank lowers it modestly to
0.639 at identical block sizes. The reduction is small because this coarse
descriptor makes OMol25's similarity *diffuse* (most structures are somewhat
similar), leaving little separable structure. A learned UMA/eSEN embedding
(`omol25_embed.py`) would give a sharper similarity and larger achievable
reduction — the recommended next step.

## 2. Scaling: split time and L(π) vs dataset size (80/20, nested subsamples)

`omol25_scaling.csv`, plots `omol25_scaling_time.png`, `omol25_scaling_lpi.png`.

| n | low-rank time | low-rank L(π) | hypergraph time | hypergraph L(π) |
|--:|--:|--:|--:|--:|
| 10,000 | 0.07 s | 0.270 | 0.45 s | 0.246 |
| 100,000 | 0.30 s | 0.270 | 5.85 s | 0.247 |
| 300,000 | 0.67 s | 0.269 | **infeasible** | — |
| 1,000,000 | 2.57 s | 0.270 | infeasible | — |
| 9,553,821 | **28 s** | 0.270 | infeasible | — |

(random L(π) ≈ 0.32 throughout.)

**Reading:**
- **Scale (the headline):** low-rank's O(n·r) time grows gently — it splits the
  **full 9.55M set in ~30 s**. Hypergraph's O(n²) k-NN becomes infeasible past
  ~10⁵; it never reaches 10⁶–10⁷. DataSAIL would stop far earlier.
- **Quality:** low-rank's L(π) is rock-stable (~0.270) across four orders of
  magnitude; hypergraph is slightly *lower* (0.245–0.258) where it can run.
  Consistent with the MoleculeNet finding — the two are comparable on leakage,
  and low-rank's decisive advantage is **scale + determinism**, not a lower L(π).

## Saved artifacts (for manual inspection)

- `omol25_splits.parquet` — all 9.55M structures with `native_split`,
  `lowrank_split`, plus formula / charge / spin / natoms / data_id / db_id.
- `omol25_splits_sample.csv` — 20k-row human-readable sample.
- `omol25_lpi_summary.csv`, `omol25_scaling.csv`, and the two PNG plots.

## Honest caveats

- The **coarse structural descriptor** limits the meaningfulness of the absolute
  L(π): it shows the *method scales*, but the leakage-reduction *magnitude*
  depends on the similarity, which should be upgraded to learned embeddings for a
  real quality claim.
- The 3-way full-set experiment and the 2-way scaling experiment use different
  block counts/proportions, so their L(π) values are not directly comparable
  (each is compared to its own random baseline).
