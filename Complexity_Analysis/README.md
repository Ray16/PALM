# Complexity Analysis — step-2 neighbor-search backends

This directory analyzes the **runtime bottleneck** of PALM's hypergraph split and shows which
nearest-neighbor backend is optimal in which data regime. The short version: the neighbor search
(step 2) is the only expensive, scaling-sensitive part, and the right backend depends on the
feature dimension `d` and the number of records `n`.

## Contents

- `regime_study.ipynb` — self-contained notebook that sweeps `n` for a low-`d` and a
  very-high-`d` synthetic regime, times the three backends, and produces the plot + a
  confirmation table (empirical fastest vs. theory). No Mt-KaHyPar needed — it times step 2 only.
- `regime_scaling.png` — two-panel log-log plot of step-2 time vs. `n` for each regime.

## Background: where the time goes

The hypergraph split has three steps: (1) featurize, (2) build the k-NN similarity hyperedges,
(3) partition with Mt-KaHyPar. On real qm8 (n≈21.8k) the breakdown is roughly:

| step | time | scaling |
|---|---|---|
| 1 — featurize | ~2 s | one-time |
| 2 — **k-NN search** | **~5–150 s** (backend-dependent) | **the bottleneck** |
| 3 — Mt-KaHyPar | ~0.3 s | near-linear, negligible |

Step 3 is not the bottleneck and is already near-optimal, so **all runtime headroom is in step 2.**
Step 2 can be done three ways, all producing the same split quality (L(π) is invariant to how the
*exact* neighbors are found; the ANN backend is high-recall, so its L(π) matches too):

- **HG-kNN** — exact k-NN via a space-partitioning **tree** (KD-/ball-tree). ~O(n log n) in low
  `d`; collapses toward O(n²) in high `d` (curse of dimensionality — the tree stops pruning).
- **HG-Matmul** — exact k-NN via a **dense distance matmul** (BLAS/GPU). O(n²·d), but those FLOPs
  run at hardware peak. **This is what PALM currently uses.**
- **HG-ANN** — approximate k-NN via **HNSW** + exact rerank. ~O(n log n) with a worse constant.

## The regime map (validated empirically)

| regime | best backend | why |
|---|---|---|
| **low `d`** (≲20), any `n` | **HG-kNN (tree)** | trees prune well; exact and ~O(n log n) |
| **high `d`, small–moderate `n`** | **HG-Matmul** | tree collapses; n² is affordable and BLAS-fast |
| **high `d`, large `n`** | **HG-ANN** | tree ≈ n²; matmul n² prohibitive; ANN stays ~linear |

Measured fitted exponents (time ∝ nᵃ) confirm the mechanism:

- Low `d` (d=2): tree a≈0.5–1.0 (≈linear), matmul a≈1.9 (n²) → **tree wins** (≈270× faster than
  matmul at n=100k).
- High `d` (d=512): tree a≈1.4 (degrading), matmul a≈1.8–2.0 (n²), ANN a≈1.0 (linear) →
  **matmul at small `n`, ANN once `n` is past the matmul→ANN crossover.**

The matmul→ANN crossover is **hardware-dependent**: a GPU keeps the exact matmul fast to larger
`n`, pushing the crossover to the right (~10⁵ on a fast machine).

## What this means for PALM

Every MoleculeNet dataset in the paper's Table 2 is **high `d`** (2048-bit Morgan fingerprints,
Tanimoto) with **`n` from 642 to 93k** — i.e. the *high-`d`, small-to-moderate-`n`* quadrant. So:

- **Exact brute matmul is the correct, sufficient backend for the entire MoleculeNet benchmark** —
  which is exactly what `hypergraph.py` already does (GPU-accelerated when available). On a GPU it
  clears even MUV (93k) in ~10 s.
- The **tree is never** the right choice for fingerprint data — at d=2048 it is *slower than brute*
  (on real qm8: ball-tree 145 s vs. matmul 4.6 s), and KD-trees can't even use Tanimoto.
- **ANN is not worth it here** — at qm8's scale it is ~7× *slower* than matmul; its payoff is only
  at 10⁵–10⁶⁺ records (e.g. large multi-component/reaction corpora), or in the low-`d` per-axis
  reaction setting where a tree would win instead.

In short: the dispatch on `(d, n)` matters for datasets *outside* MoleculeNet; within it, PALM's
single hard-coded matmul backend is already in the right regime.

## Running the notebook

```bash
pip install numpy scikit-learn hnswlib   # optional: torch (GPU/CPU matmul path)
jupyter nbconvert --to notebook --execute regime_study.ipynb   # or just open and run
```

Config knobs at the top of the notebook: `D_LOW`, `D_HIGH`, `N_LOW`, `N_HIGH`, `K_NN`. Bump
`N_HIGH` past the matmul→ANN crossover for your hardware to see ANN overtake the matmul.
