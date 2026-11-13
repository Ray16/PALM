# archive/ — documented negative results

This directory holds method directions that were **evaluated and did not help**,
moved out of the live tree so the working code stays clean. Nothing here is imported
by the live package; each subfolder is self-contained and still runnable for
reproducibility. The synthesis of why each failed lives in
[`../docs/FINDINGS.md`](../docs/FINDINGS.md).

| direction | verdict | FINDINGS |
|---|---|---|
| `multilevel/` | Multilevel FM V-cycle gives **zero** leakage benefit over flat best-of-4 Lloyd + single-move FM (already at the global optimum). The real lever was the `fm_max_n` cap, fixed in the live `splitter.py`. | Step 4 |
| `nystrom_variants/` | Tight/adaptive Nyström sketches (leverage-score landmarks, `ridge`, adaptive `energy`-rank) give **negligible** leakage benefit over plain k-means++ — the split objective is robust to approximation error. | Step 2 |
| `sparse_residual/` | Low-rank + sparse kernel `BBᵀ + R` (exact k-NN residual) and a *soft* must-link boost on R. The residual is a mild positive but is dominated by live hard `must_link`; the soft boost is a clear negative (real nnl_rel 0.229 vs 0.108). `sparse_split.py` reproduces the archived path; `optimize_sparse.py` / `objective_sparse.py` are pre-strip snapshots. | Step 9 |

## Running the archived studies

Each script is standalone (adds the repo root and its own folder to `sys.path`):

```bash
# from the low_rank/ repo root, on CPU to avoid cluster GPU contention:
CUDA_VISIBLE_DEVICES="" python archive/multilevel/multilevel_fm.py --datasets moleculenet_esol --seeds 0
CUDA_VISIBLE_DEVICES="" python archive/nystrom_variants/nystrom_fidelity.py --datasets moleculenet_esol --seeds 0
CUDA_VISIBLE_DEVICES="" python archive/multilevel/test_multilevel.py
CUDA_VISIBLE_DEVICES="" python archive/nystrom_variants/test_nystrom_variants.py
CUDA_VISIBLE_DEVICES="" python -m pytest archive/sparse_residual/test_sparse_residual.py
```

`nystrom_variants/nystrom_full.py` is a frozen pre-strip snapshot of `lowrank/nystrom.py`
that still contains the removed variants, so the variant sweep and its tests reproduce
without the live module carrying dead options.

## Why keep negative results at all

They are evidence: they document *what was tried* so the same dead ends are not
re-explored, and they back the central finding that **the lever is the problem
definition (objective / balance / hardness), not solver fidelity or sketch quality**.
