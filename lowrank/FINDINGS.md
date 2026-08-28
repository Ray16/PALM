# Low-rank splitter — method-development findings

Findings from the multi-objective / controllable-hardness track built on
`PALM.lowrank`. Each step is a self-contained increment with an experiment under
`experiments/`; this file is the synthesis. All numbers are triplicate, hand-picked
features per type (`route=True`).

---

## Step 1 — Multi-objective core: the leakage ↔ balance tradeoff
*Code: `optimize.py` (`corridor_assign`, `balanced_lloyd(balance_slack=)`).
Experiment: `experiments/balance_pareto.py` → `balance_pareto.{csv,png}`.*

The default splitter pins **exact** train/test sizes. Relaxing that to a tunable
`balance_slack` corridor lets the optimizer cut cleaner and lowers leakage
monotonically:

| dataset | slack 0.0 | slack 0.30 | leakage ↓ |
|---|--:|--:|--:|
| bace (chemberta) | 0.257 | 0.195 | −24% |
| esol (ecfp) | 0.160 | 0.109 | −32% |
| qmof (mat2vec) | 0.194 | 0.141 | −27% |

**Finding:** exact balance costs ~25–30% of achievable leakage reduction. The
tradeoff is now an explicit knob, not a hard-coded constraint. **Mechanism:**
`balance_slack` moves leakage *below* the balanced optimum by spending balance.

> **Downstream caveat (see Synthesis):** this lower leakage does **not** translate
> into a harder evaluation — measuring the gap over the slack sweep shows it flat or
> *falling*. `balance_slack` is a knob on the leakage↔balance **metric**, not on
> real OOD difficulty; use `hardness` (Step 3) for the latter.

---

## Step 2 — Tight/adaptive Nyström: an honest negative result
*Code: `nystrom.py` (`landmark="leverage"`, `ridge`, `energy`).
Experiment: `experiments/nystrom_fidelity.py`.*

**Finding:** the Nyström approximation is **not** the bottleneck. Reconstruction
error `‖S−BBᵀ‖/‖S‖` keeps falling with rank, but **leakage plateaus by rank ≈ 32–64**
(rank 256 is overkill), and ridge-leverage-score landmarks / ridge `W⁻¹ᐟ²` give
negligible leakage benefit over plain k-means++. The fidelity bound holds but is
*loose* — the split objective is robust to approximation error. The practical win
is **lower/adaptive rank for speed**, not lower leakage.

---

## Step 3 — Controllable OOD-hardness (flagship, validated)
*Code: `optimize.py` (`interpolate_to_random`), `splitter.py` (`hardness`).
Experiment: `experiments/hardness_control.py` → `hardness_control.{csv,png}`.*

A `hardness` dial ∈ [0,1] (1 = leakage-minimized/hardest, 0 = random/easiest),
implemented as balance-preserving interpolation toward random. It **controls the
realized generalization gap**:

| dataset | leakage (α 0→1) | gap (α 0→1) | Spearman(α, gap) | calibration |
|---|--:|--:|--:|--|
| esol | 0.306 → 0.160 | 0.35 → 0.84 | **+1.00** | gap ≈ 0.22 + 0.50·α |
| bace | 0.307 → 0.257 | 0.11 → 0.23 | +0.90 | gap ≈ 0.10 + 0.10·α |
| freesolv | 0.316 → 0.127 | 0.26 → 0.71 | +0.90 | gap ≈ 0.26 + 0.40·α |

**Finding:** within a dataset, hardness controls the gap almost deterministically
(Spearman +0.90–1.00). Each dataset gets a linear calibration `gap ≈ b + a·α` to
invert for a target difficulty (slopes are dataset-specific). **Mechanism:**
`hardness` moves leakage *up* toward random by injecting randomness, holding balance
fixed — a near-pure leakage→gap effect.

---

## Synthesis — Steps 1 & 3 are NOT one capability (tested, hypothesis refuted)
*Experiment: `experiments/balance_gap.py` → `balance_vs_hardness_gap.png`,
`balance_gap.csv`.*

The tempting hypothesis was "both are leakage knobs, so both control difficulty."
Measuring the gap for the balance_slack sweep **refutes it.** The two knobs share
one anchor (slack 0 ≡ hardness 1 — the exact-balance, fully-optimized split) and
agree there exactly, but from that anchor they move the gap in **opposite
directions**:

| dataset | anchor L/gap | `balance_slack`→0.30 (leakage↓) | `hardness`→0 (leakage↑) |
|---|--:|--:|--:|
| esol | 0.160 / 0.81 | L 0.109, gap **0.51 ↓** | L 0.307, gap 0.30 ↓ |
| freesolv | 0.127 / 0.71 | L 0.079, gap **0.63 ↓** | L 0.303, gap 0.32 ↓ |
| bace | 0.257 / 0.23 | L 0.194, gap 0.22 (flat) | L 0.308, gap 0.11 ↓ |

**Key findings:**

1. **The anchor is the maximum-difficulty point.** The exact-balance,
   leakage-minimized split has the largest gap. Moving *either* way makes the eval
   easier — randomness (Step 3) via higher leakage, imbalance (Step 1) via the
   evaluation confound.

2. **Leakage is not a sufficient statistic for the gap — the *mechanism* matters.**
   Step 3 lowers leakage by increasing train/test *separation* at fixed balance →
   harder (bigger gap). Step 1 lowers leakage by *imbalance* → the gap flattens or
   **falls**. Same metric, opposite downstream effect. So the naive leakage→gap
   relation only holds when leakage is changed the *right* way.

3. **`balance_slack` is a metric-gaming trap for difficulty.** It improves the
   leakage *number* while making the benchmark no harder (often easier). Anyone
   reducing leakage via imbalance and expecting a harder OOD test would be fooled.
   `balance_slack` is a legitimate knob **for the leakage↔balance metric tradeoff**,
   but **not** a difficulty controller. **`hardness` (Step 3) is the only validated
   difficulty dial** — precisely because it isolates leakage at fixed balance.

4. **The lever is the problem definition, not solver fidelity** (why Step 2 failed).
   Steps 1 & 3 change *what* is optimized; Step 2 changed *how well*. The objective
   was already faithful enough.

5. **Leakage is dataset-relative, not a universal unit.** Within-dataset
   Spearman(hardness, gap) ≈ 1.0 but cross-dataset r(leakage, gap) = −0.56: leakage
   orders difficulty near-perfectly inside a fixed (dataset, features, model), but
   the leakage→gap *map* differs by dataset (esol slope 0.50 vs bace 0.10). Hence
   per-dataset calibration is inherent and the normalized dial α is the right
   abstraction.

**Takeaway:** controllable OOD difficulty is a real, calibratable capability — but
it is **`hardness` specifically** (separation at fixed balance), not "lower leakage"
in general. The balance_slack experiment is the control that proves the point:
lowering leakage the wrong way does not buy difficulty. The gap, not the leakage
number, is ground truth.
