"""Correctness tests for the low-rank factorized splitter (run in `palm` env).

Checks, in order:
  1. Nystrom factor recovers Tanimoto (BB^T ~= S).
  2. lowrank_leakage() equals the exact cross-block similarity in factor space.
  3. balanced_lloyd honours the 80/20 balance corridor.
  4. fm_polish is monotone on the low-rank objective and stays balanced.
  5. On tiny n, best-of-restarts + FM reaches the brute-force-optimal split.
  6. End-to-end run_lowrank_split beats a random split and is balanced.
"""

import itertools
import sys

import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.lowrank_split import lowrank_split as LR


# ── shared fixtures ────────────────────────────────────────────────────────

def planted_fps(n: int, dim: int = 256, n_clusters: int = 4, seed: int = 0) -> np.ndarray:
    """Binary fingerprints drawn from a few cluster centers with bit noise."""
    rng = np.random.default_rng(seed)
    centers = rng.integers(0, 2, size=(n_clusters, dim))
    X = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        c = centers[i % n_clusters].copy()
        flip = rng.random(dim) < 0.08
        c[flip] = 1 - c[flip]
        X[i] = c
    return X


def exact_cross_tanimoto(X: np.ndarray, labels) -> float:
    """Exact cross-block Tanimoto leakage (self pairs excluded), unordered."""
    X = X.astype(np.float64)
    card = X.sum(1)
    inter = X @ X.T
    union = card[:, None] + card[None, :] - inter
    sim = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(sim, 0.0)
    lab = np.asarray(labels)
    cross = (lab[:, None] != lab[None, :]).astype(float)
    return 0.5 * float((sim * cross).sum())


# ── tests ──────────────────────────────────────────────────────────────────

def test_factorization_recovers_similarity():
    X = planted_fps(400, seed=1)
    B, metric = LR.nystrom_features(X, rank=256, seed=0)
    assert metric == "tanimoto"
    # compare BB^T to exact Tanimoto on random pairs
    rng = np.random.default_rng(2)
    ii = rng.integers(0, 400, 3000)
    jj = rng.integers(0, 400, 3000)
    m = ii != jj
    ii, jj = ii[m], jj[m]
    card = X.sum(1)
    inter = (X[ii] * X[jj]).sum(1)
    true = inter / (card[ii] + card[jj] - inter + 1e-9)
    approx = (B[ii] * B[jj]).sum(1)
    corr = np.corrcoef(true, approx)[0, 1]
    rmse = np.sqrt(np.mean((true - approx) ** 2))
    print(f"[OK] factorization recovers Tanimoto: corr={corr:.3f} rmse={rmse:.3f}")
    assert corr > 0.9 and rmse < 0.05


def test_objective_matches_factor_space():
    """lowrank_leakage == exact cross similarity computed in the *same* factor space."""
    X = planted_fps(300, seed=3)
    B, _ = LR.nystrom_features(X, rank=200, seed=0)
    rng = np.random.default_rng(4)
    lab = np.array([0] * 240 + [1] * 60)
    rng.shuffle(lab)
    obj = LR.lowrank_leakage(B, lab, 2)
    # exact cross in factor space: sum over cross pairs of B_i . B_j
    G = B @ B.T
    np.fill_diagonal(G, 0.0)
    cross = (lab[:, None] != lab[None, :]).astype(float)
    exact = 0.5 * float((G * cross).sum())
    print(f"[OK] objective matches factor space: lowrank={obj:.2f} exact={exact:.2f}")
    assert abs(obj - exact) / max(abs(exact), 1.0) < 1e-4


def test_lloyd_balance():
    X = planted_fps(1000, seed=5)
    B, _ = LR.nystrom_features(X, rank=256, seed=0)
    lab = LR.balanced_lloyd(B, [8, 2], epsilon=0.05, seed=0)
    sizes = np.bincount(lab, minlength=2)
    print(f"[OK] Lloyd balance: sizes={sizes.tolist()} (want exactly [800,200])")
    # Lloyd targets the exact ratio, not the corridor bound
    assert sizes.tolist() == [800, 200], f"balance off: {sizes.tolist()}"


def test_fm_monotone_and_balanced():
    X = planted_fps(600, seed=6)
    B, _ = LR.nystrom_features(X, rank=256, seed=0)
    rng = np.random.default_rng(7)
    lab0 = np.array([0] * 480 + [1] * 120)
    rng.shuffle(lab0)
    L0 = LR.lowrank_leakage(B, lab0, 2)
    lab1, moves = LR.fm_polish(B, lab0.copy(), [8, 2], epsilon=0.05)
    L1 = LR.lowrank_leakage(B, lab1, 2)
    sizes = np.bincount(lab1, minlength=2)
    print(f"[OK] FM monotone: L {L0:.1f} -> {L1:.1f} in {moves} moves, sizes={sizes.tolist()}")
    assert L1 <= L0 + 1e-4, "FM increased the low-rank objective (not monotone)"
    assert 114 <= sizes[1] <= 126, f"FM broke balance: {sizes.tolist()}"


def test_vs_bruteforce_tiny():
    """best-of-restarts + FM should reach the exact optimum on n=12 (6/6)."""
    X = planted_fps(12, dim=64, n_clusters=2, seed=8)
    # brute force over all 6/6 splits with node 0 fixed to block 0
    best = np.inf
    for combo in itertools.combinations(range(1, 12), 6):
        lab = np.zeros(12, dtype=int)
        lab[list(combo)] = 1
        best = min(best, exact_cross_tanimoto(X, lab))
    B, _ = LR.nystrom_features(X, rank=12, seed=0)      # full rank -> exact
    fm_best = np.inf
    for s in range(20):
        # epsilon=0 pins the split to exactly 6/6 (same feasible set as brute force)
        lab = LR.balanced_lloyd(B, [1, 1], epsilon=0.0, seed=s)
        lab, _ = LR.fm_polish(B, lab, [1, 1], epsilon=0.0)
        assert np.bincount(lab, minlength=2).tolist() == [6, 6]
        fm_best = min(fm_best, exact_cross_tanimoto(X, lab))
    print(f"[OK] vs brute force (6/6): opt={best:.4f} lowrank-best={fm_best:.4f}")
    assert abs(fm_best - best) < 1e-6, f"low-rank {fm_best} != optimum {best}"


def test_end_to_end():
    X = planted_fps(800, seed=9)
    fd = {i: X[i] for i in range(800)}
    rng = np.random.default_rng(0)
    r = np.array([0] * 640 + [1] * 160)
    rng.shuffle(r)
    L_rand = exact_cross_tanimoto(X, r)
    split = LR.run_lowrank_split(fd, [8, 2], ["train", "test"], rank=256, n_restarts=4)
    lab = np.array([0 if split[i] == "train" else 1 for i in range(800)])
    L_lr = exact_cross_tanimoto(X, lab)
    n_test = int((lab == 1).sum())
    print(f"[OK] end-to-end: random L={L_rand:.1f} lowrank L={L_lr:.1f} test_size={n_test}")
    assert L_lr < L_rand, "low-rank split not better than random"
    assert 152 <= n_test <= 168, f"balance off: {n_test}"


if __name__ == "__main__":
    test_factorization_recovers_similarity()
    test_objective_matches_factor_space()
    test_lloyd_balance()
    test_fm_monotone_and_balanced()
    test_vs_bruteforce_tiny()
    test_end_to_end()
    print("\nALL TESTS PASSED")
