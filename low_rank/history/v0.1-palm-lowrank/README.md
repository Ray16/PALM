# v0.1 — the former in-tree `PALM.lowrank` package

Until November 2026 PALM carried its own copy of the low-rank splitter in
`PALM/lowrank/` (v0.1 API: `nystrom_features`, `balanced_lloyd`, `fm_polish`,
`factor_leakage`, `LowRankSplitter`, `LowRankNDSplitter`, `target_gap`, `multilevel`).
It was replaced by this package (`lowrank-split`), which PALM now registers through
`PALM/splitters/methods/lowrank_adapter.py`.

This folder keeps that package's records:

| file | contents |
|---|---|
| `v0.1_README.md`, `FINDINGS.md`, `PROGRESS.md` | the v0.1 package's own documentation and research log |
| `experiments/*.py` | the v0.1 study scripts (ablation, balance/hardness trade-offs, k-way, multilevel FM, n-D, Nyström fidelity, target gap) |

Their outputs (CSV tables and figures) are already in [`../../results/`](../../results)
and [`../../figures/`](../../figures). The scripts import the removed `PALM.lowrank`
module and are kept for provenance, not maintained. To rerun them exactly, check out
PALM commit `1dbbbbf` (the last commit containing `PALM/lowrank/`).

## Behaviour change v0.1 → v0.2

Measured before the switch on five MoleculeNet sets (FreeSolv, ESOL, BACE, BBBP,
Lipophilicity; ECFP4-2048, 80/20, seeds 0–2, identical scorers):

- v0.1 used a 5% balance corridor and delivered ≈19% test sets; v0.2 hits the requested
  20% exactly by default.
- With exact counts, scaled L(π) is ≈4.7% higher (median), which is mostly the larger
  test set: with the same 5% corridor (`allowed_size_imbalance=0.05`) v0.2 reproduces
  v0.1's L(π) to within 0.002.
- Nearest-neighbour leakage (`nnl_rel`) is lower with v0.2 defaults in 4 of 5 datasets
  (median −0.014); using the corridor makes it worse (e.g. BACE 0.29 → 0.42).
