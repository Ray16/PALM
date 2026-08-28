# PALM.lowrank — progress log

Live status of low-rank splitter method development. Branch `benchmark/master-suite`.
See `README.md` for the method and `FINDINGS.md` for the synthesized results.

## Governing lesson (from FINDINGS.md)
The **generalization gap is ground truth**, not the leakage number. Lowering leakage
helps difficulty only when done *honestly* — increasing train/test separation at
**fixed balance** (Step 3), not by spending balance (Step 1). Every improvement is
therefore either (I) push the honest leakage lower, or (II) make the validated
difficulty capability more usable/general.

## Done
- **Step 1 — leakage↔balance tradeoff** (`balance_slack`): ~30% slack cuts leakage
  24–32%, but does NOT raise the gap → a metric knob, not a difficulty dial.
- **Step 2 — tight/adaptive Nyström**: honest negative result — approximation is not
  the bottleneck; leakage plateaus by rank ≈ 32–64. Win is speed, not leakage.
- **Step 3 — controllable OOD-hardness** (`hardness` α∈[0,1]): FLAGSHIP, validated.
  Within a dataset Spearman(α, gap) = +0.90…+1.00; calibration `gap ≈ b + a·α`.
- Ruled out (this session): "optimize `scaled_lpi` instead of `factor_leakage`" —
  they are equivalent up to the Nyström error (denom is a label-independent constant,
  `leakage_metrics.py:56`), so no lever there.

## Done this session
- **#1 Multilevel FM** — DONE. Honest NEGATIVE result for the fancy solver, one real
  win shipped. `multilevel.py` (coarsen→refine→uncoarsen V-cycle, provably never worse
  than flat) gives **+0.00%** leakage on bace/esol/freesolv/qmof (triplicate): flat
  best-of-4 Lloyd + single-move FM is already at the global optimum (best-of-30 ==
  best-of-4), so there are no local minima to escape. **The real bug was the
  `fm_max_n=200_000` cap** — above it the splitter shipped *un-polished Lloyd* splits.
  Scale sweep: at 300k the cap left 367M vs 352.7M with FM (**+3.99%**); flat FM runs in
  2.1s at 300k / ~12s at 1M, multilevel gets the same number ~48× slower.
  - **Applied**: `splitter.py` `fm_max_n` 200k → **2_000_000** (flat FM, not multilevel).
    `multilevel.py` kept as documented negative evidence, NOT wired in.
  - Files: `multilevel.py`, `experiments/multilevel_fm.py` (+ `multilevel_small.csv`,
    `multilevel_scale.csv`, `multilevel_scale.png`), `tests/test_multilevel.py` (4 pass;
    fixed a too-tight absolute tolerance → relative, GPU float-reduction noise).
  - **Implication for #2**: if flat FM already hits the global optimum here, "smarter
    Lloyd init + escape" (#2) is likely negative too — reconsider before spending a fork.
- **#4 target_gap API** — DONE & verified. Auto-calibrate the difficulty curve from
  probe splits, invert so the user requests a target *gap* instead of hand-picking α.
  Files: `target_gap.py`, `experiments/target_gap.py`, `tests/test_target_gap.py`
  (7 tests pass, no GPU), outputs `experiments/target_gap.csv` + `target_gap_{esol,bace,freesolv}.png`.
  - Reuses the Step-3 gap estimator (`benchmarks.master.model_eval.evaluate_gap`) so
    numbers are comparable to `hardness_control.py`.
  - **Requested↔realized gap tracks**: corr +0.90 (bace) / +0.997 (esol) / +0.90 (freesolv);
    MAE 0.014 / 0.090 / 0.050.
  - **Key finding**: the α→gap curve is **convex**, so a naive linear `gap≈b+a·α`
    over-predicts interior targets (esol MAE 0.122). A monotone **piecewise-linear**
    engine through the probe knots cut MAE −55/−26/−55%. esol residual (0.090) sits near
    the gap estimator's own noise floor (probe std ±0.08 at high α), not inversion error.
  - **Applied**: re-exported `calibrate_gap, split_for_gap, GapCalibrator` from
    `lowrank/__init__.py` + README "Use" snippet. Made target_gap's `PALM.splitters` /
    `benchmarks.master.model_eval` imports lazy (they'd otherwise be circular and pull
    the benchmark model stack into every registry import; verified registry import no
    longer touches `benchmarks.master`).
  - Follow-up idea (4b): report a convexity/nonlinearity diagnostic so users know when
    to add denser probes.

## Integration status
All applied to the working tree (branch benchmark/master-suite), NOT committed:
`splitter.py` (cap), `__init__.py` (re-exports), `README.md`, `target_gap.py` (lazy
imports), `tests/test_multilevel.py` (tolerance). Full suite green:
`test_lowrank` 9 · `test_multilevel` 4 · `test_target_gap` 7 = **20 pass**.

Both forks write only their own new files; the shared `splitter.py` wiring is applied
by the main session after review so the two cannot collide. Env `palm`; one GPU each.

## Done this session (cont.)
- **#3 Extend to k>2** — DONE & verified. Extended `corridor_assign` to k>2 (regret-based
  greedy peel; k==2 branch untouched/exact). Verified degenerate corridor
  (`floors==caps==sizes`) reproduces exact sizes, and feasibility clamps keep every
  block in-corridor.
  - **3-way leakage↔balance ([8,1,1], triplicate)**: opening `balance_slack` now lowers
    3-way leakage (it did nothing before — exact fallback). Spearman(slack,leakage)=−1.00:
    bace 0.270→0.200 (−26%), esol 0.172→0.111 (−35%), freesolv 0.129→0.078 (−39%),
    qmof 0.205→0.146 (−29%); sizes in-corridor.
  - **3-way hardness→gap ([8,1,1], triplicate)**: dial controls train→test gap at the
    endpoints (esol ~0.36→1.10) but noisier interior — Spearman +0.60/+0.60/+0.90 vs
    +0.90–1.00 at k=2, because the test block is half the size (10% vs 20%). Works;
    calibrate with more seeds on 3-way.
  - Files: edited `optimize.py`; new `experiments/kway_split.py` (+ `kway_balance.csv`,
    `kway_hardness.csv`, 7 PNGs), `tests/test_kway.py` (5 pass).
  - Suite: `test_lowrank` 9 · `test_multilevel` 4 · `test_target_gap` 7 · `test_kway` 5
    = **25 pass**, no k=2 regression.

## Done this session (cont.)
- **Ablation study** — DONE & verified (FINDINGS Step 7). Reports leakage AND gen-gap
  side by side, 4 configs, triplicate. **Corrected my earlier phrasing**: the anchor
  (lloyd+FM, exact balance) is near-MAX-gap but NOT global-min-leakage — `balance_slack`
  reaches lower leakage yet LOWER gap on all 4 (esol 0.81→0.51). So leakage & gap
  anti-correlate across mechanisms = the sharpened synthesis. Part B: rank & n_restarts
  neutral on BOTH metrics (n_restarts identical 1/4/16 → Lloyd+FM deterministic to the
  optimum), reinforcing #4 and the #2 skip. qmof had a usable target → gap column too.
  Files: `experiments/ablation.py` (+ `ablation_components.csv`, `ablation_grid.csv`,
  6 PNGs). Committed separately (see below).

## Skipped
- **#2 Smarter Lloyd init + local-minimum escape** — SKIPPED (decision). #1 showed flat
  FM already reaches the global optimum here (best-of-30 == best-of-4), so there are no
  local minima for a better init to escape → #2 would be a negative result. Revisit only
  if a future dataset shows best-of-N leakage NOT plateauing at best-of-4.

## Later (ranked list, not scheduled)
- Calibration robustness: validate `gap ≈ b + a·α` across multiple downstream models
  (pairs naturally with target_gap; also see 4b convexity diagnostic above).
- Directional/structured hardness (extrapolation vs. uniform randomization).
- Scale-regime leakage estimation (leakage is `None` for n>100k, `splitter.py:89`).
