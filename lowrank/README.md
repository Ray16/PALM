# PALM.lowrank — the low-rank leakage-minimizing splitter

Standalone package for developing the low-rank data splitter (the best all-round
method in the master benchmark: near-DataSAIL leakage but runs at any scale). Kept
separate from `PALM.splitters.methods` so the algorithm can evolve on its own; it
still registers the `lowrank` method into the shared `PALM.splitters` registry on
import.

## How it works
`S ≈ B Bᵀ` (Nyström) → minimize cross-split leakage in the r-dim factor space with
balanced-Lloyd restarts + a monotone FM polish — O(n·r), no similarity matrix.

```
nystrom.py    Nyström factorization (landmarks, W, B)      ← Direction A
objective.py  factor_leakage, realized_imbalance           ← Directions B/C
optimize.py   balanced_lloyd, corridor_assign, fm_polish   ← Directions B/C
splitter.py   LowRankSplitter (@register "lowrank")
experiments/  balance_pareto.py — the leakage↔balance frontier
```

## Tunable knobs (`LowRankSplitter.Params`)
`rank` (256), `landmark` (kmeans++/uniform), `n_restarts` (4), `n_iter` (25),
`fm`/`fm_max_n`, `metric`, and **`balance_slack`** (0 = exact target sizes;
>0 opens a (1 ± slack) size corridor the optimizer exploits to lower leakage).

## Method-development roadmap
1. **Multi-objective core** *(done — `balance_slack`)*: the leakage↔balance
   tradeoff. `experiments/balance_pareto.py` shows ~30% slack cuts leakage 24–32%
   (bace/esol/qmof). **But** `experiments/balance_gap.py` shows this lower leakage
   does *not* raise the generalization gap (flat/falling) — `balance_slack` is a
   metric-tradeoff knob, **not** a difficulty controller (that's Step 3). See
   `FINDINGS.md`.
2. **Tight & adaptive approximation (A)** *(done — with an honest negative
   result)*: added `landmark="leverage"` (approx. ridge-leverage-score),
   `ridge` (regularized `W⁻¹ᐟ²`), and `energy` (adaptive rank). `experiments/
   nystrom_fidelity.py` finds **the approximation is not the bottleneck**: across
   bace/esol/qmof, reconstruction error keeps dropping with rank but **leakage
   plateaus by rank ≈ 32–64** (rank 256 is overkill), and leverage-score / ridge
   give negligible leakage benefit over k-means++. Net: the fidelity bound holds
   but is *loose* — the split objective is robust to approximation error. The
   practical win is **adaptive/lower rank as a speed optimization** (drop
   256→64 at ~no leakage cost); the real leverage stays in Steps 1 & 3.
3. **Controllable OOD-hardness (C)** *(done — the flagship, validated)*: a
   `hardness` dial ∈ [0,1] (1 = leakage-minimized/hardest, 0 = random/easiest)
   via balance-preserving interpolation toward random (`interpolate_to_random`).
   `experiments/hardness_control.py` validates it **controls the realized
   generalization gap**: Spearman(hardness, gap) = **+0.90 to +1.00** within a
   dataset (esol gap 0.35→0.84, bace 0.11→0.23, freesolv 0.26→0.71) — much cleaner
   than the noisy cross-dataset r(leakage,gap)=−0.56. Each dataset gets a linear
   calibration `gap ≈ b + a·α` you invert to request a target difficulty (slopes
   are dataset-specific, so calibrate per dataset — the honest scope).

## Use
```python
from PALM.splitters import split, SplitSpec
r = split("lowrank", feature_data, SplitSpec([8, 2], ["train", "test"], seed=0),
          balance_slack=0.10)     # 10% size corridor -> lower leakage
r.diagnostics["leakage"]
```
Tests: `python -m PALM.lowrank.tests.test_lowrank` (6 tests).
