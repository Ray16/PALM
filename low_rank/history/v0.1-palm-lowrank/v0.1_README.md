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

4. **Optimizer at scale (from the multilevel investigation)** *(done — negative
   result for the fancy solver, one real win)*: a coarsen→refine→uncoarsen
   multilevel FM V-cycle (`multilevel.py`) gives **0% leakage benefit** on real
   datasets — flat best-of-4 Lloyd + single-move FM is already at the global optimum
   (best-of-30 == best-of-4). The actual bottleneck was the conservative
   `fm_max_n=200_000` cap, which shipped *un-polished Lloyd* splits above 200k. Flat FM
   is O(n·r) and converges in ~12s at n=1M, so **raising the cap to 2M recovers ~4%
   leakage on all large splits** (`experiments/multilevel_fm.py`). `multilevel.py` is
   kept as documented negative evidence, not wired in.
5. **`target_gap` API (productizing Step 3)** *(done — validated)*: request a target
   generalization gap and get the split that delivers it. `calibrate_gap` fits the
   α→gap curve from cheap probe splits (the curve is **convex**, so a monotone
   piecewise-linear fit beats a naive linear one), and `split_for_gap` inverts it.
   Requested↔realized gap corr +0.90–0.997 (`experiments/target_gap.py`).

## Use
```python
from PALM.splitters import split, SplitSpec
r = split("lowrank", feature_data, SplitSpec([8, 2], ["train", "test"], seed=0),
          balance_slack=0.10)     # 10% size corridor -> lower leakage
r.diagnostics["leakage"]

# request a difficulty level instead of hand-picking hardness α (Step 3 / #5):
from PALM.lowrank import calibrate_gap, split_for_gap
cal = calibrate_gap(feature_data, X, y, task_type="regression")  # cheap probe splits
res, inv, _ = split_for_gap(feature_data, target_gap=0.5, calibrator=cal, seed=0)
print(inv.message)              # "-> alpha=0.68 (predicted gap 0.50)"; flags out-of-range
```
Tests: `python -m PALM.lowrank.tests.test_lowrank` (9) ·
`… .tests.test_multilevel` (4) · `… .tests.test_target_gap` (7).
