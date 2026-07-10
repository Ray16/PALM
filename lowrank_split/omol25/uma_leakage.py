"""Step 3 (palm env): does the UMA embedding give a discriminative similarity,
and does low-rank reduce leakage vs the native split?

Similarity for the L(pi) metric = RBF kernel over the UMA embedding
    k(i,j) = exp(-||z_i - z_j||^2 / (2 sigma^2)),  sigma = median pairwise distance,
which is non-negative + PSD (so L(pi) in [0,1]) and, unlike signed cosine, decays
to ~0 for dissimilar pairs so splitting actually matters. Cosine mean is reported
only as a "does it separate at all" sanity vs the old hand descriptor (~0.889).
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu/PALM")
from PALM.lowrank_split.lowrank_split import balanced_lloyd, lowrank_leakage

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache_uma")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mean_cosine(Z, k=4000, seed=0):
    idx = np.random.default_rng(seed).choice(len(Z), min(k, len(Z)), replace=False)
    M = torch.as_tensor(Z[idx], dtype=torch.float32, device=DEV)
    M = torch.nn.functional.normalize(M, dim=1)
    G = M @ M.T
    iu = torch.triu_indices(len(idx), len(idx), 1)
    v = G[iu[0], iu[1]]
    return v.mean().item(), v.std().item(), (v < 0).float().mean().item()


def median_sigma(Z, k=4000, seed=0):
    idx = np.random.default_rng(seed).choice(len(Z), min(k, len(Z)), replace=False)
    M = torch.as_tensor(Z[idx], dtype=torch.float32, device=DEV)
    d2 = torch.cdist(M, M) ** 2
    iu = torch.triu_indices(len(idx), len(idx), 1)
    return torch.sqrt(d2[iu[0], iu[1]].median()).item()


def rbf_factor(Z, rank=256, sigma=1.0, seed=0, chunk=200_000):
    """Nystrom factor B [n,rank] for the RBF kernel."""
    n = len(Z)
    rng = np.random.default_rng(seed)
    land = np.sort(rng.choice(n, min(rank, n), replace=False))
    L = torch.as_tensor(Z[land], dtype=torch.float32, device=DEV)
    g = 1.0 / (2 * sigma * sigma)
    W = torch.exp(-torch.cdist(L, L) ** 2 * g)
    ev, V = torch.linalg.eigh(W)
    ev = torch.clamp(ev, min=1e-6)
    Wih = (V * (1.0 / torch.sqrt(ev))[None]) @ V.T
    B = np.empty((n, len(land)), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xt = torch.as_tensor(Z[s:e], dtype=torch.float32, device=DEV)
        C = torch.exp(-torch.cdist(Xt, L) ** 2 * g)
        B[s:e] = (C @ Wih).cpu().numpy()
    return B


def lpi_from_factor(B, labels, k):
    r = B.shape[1]
    P = torch.zeros(k, r, device=DEV)
    lab = torch.as_tensor(labels, dtype=torch.long, device=DEV)
    Bt = torch.as_tensor(B, dtype=torch.float32, device=DEV)
    P.index_add_(0, lab, Bt)
    s = P.sum(0)
    tot = float(s @ s)
    within = float((P * P).sum())
    return max(0.0, 1.0 - within / tot) if tot > 0 else 0.0


def lpi_exact_rbf(Z, labels, sigma, block=2000):
    M = torch.as_tensor(Z, dtype=torch.float32, device=DEV)
    lab = torch.as_tensor(labels, dtype=torch.long, device=DEV)
    g = 1.0 / (2 * sigma * sigma)
    n = len(M); tot = torch.zeros((), device=DEV); cross = torch.zeros((), device=DEV)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = torch.exp(-torch.cdist(M[s:e], M) ** 2 * g)
        tot += sim.sum()
        cross += (sim * (lab[s:e][:, None] != lab[None, :]).float()).sum()
    return float((cross / tot).item())


def main():
    Z = np.load(os.path.join(CACHE, "uma_emb.npy"))
    native = np.load(os.path.join(CACHE, "native.npy"))
    n = len(Z)
    sizes = np.bincount(native, minlength=3)
    print(f"loaded UMA emb {Z.shape}; native sizes {sizes.tolist()}", flush=True)

    # standardize columns (whiten) so no channel dominates the distance
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    Z = Z.astype(np.float32)

    mc, sc, negfrac = mean_cosine(Z)
    print(f"\n[sanity] mean pairwise cosine = {mc:.3f} +/- {sc:.3f}  "
          f"(frac negative {negfrac:.2f})   [old hand-descriptor was 0.889]", flush=True)

    sigma = median_sigma(Z)
    print(f"[rbf] sigma (median dist) = {sigma:.3f}", flush=True)

    # validate factorized vs exact RBF L(pi) on a 6k subsample
    sub = np.random.default_rng(1).choice(n, 6000, replace=False)
    Bs = rbf_factor(Z[sub], rank=256, sigma=sigma, seed=0)
    lpi_fac = lpi_from_factor(Bs, native[sub], 3)
    lpi_ex = lpi_exact_rbf(Z[sub], native[sub], sigma)
    print(f"[validate] factorized L(pi)={lpi_fac:.4f} vs exact RBF={lpi_ex:.4f} "
          f"(|diff|={abs(lpi_fac-lpi_ex):.4f}) on 6k", flush=True)

    # full-subsample factor
    t0 = time.time()
    B = rbf_factor(Z, rank=256, sigma=sigma, seed=0)
    print(f"[rbf] factor B={B.shape} in {time.time()-t0:.1f}s", flush=True)

    lpi_native = lpi_from_factor(B, native, 3)

    # random baseline
    rnd = np.random.default_rng(2).permutation(native)
    lpi_random = lpi_from_factor(B, rnd, 3)

    # low-rank re-split at native proportions
    best_lab, best_obj = None, np.inf
    for r in range(3):
        lab = balanced_lloyd(B, sizes.tolist(), epsilon=0.0, n_iter=20, seed=r)
        obj = lowrank_leakage(B, lab, 3)
        if obj < best_obj:
            best_obj, best_lab = obj, lab
    lpi_lowrank = lpi_from_factor(B, best_lab, 3)
    lr_sizes = np.bincount(best_lab, minlength=3)

    print("\n=== L(pi) on UMA-RBF similarity (100k subsample, 3-way native proportions) ===")
    print(f"  {'random baseline':22s} {lpi_random:.4f}")
    print(f"  {'existing native split':22s} {lpi_native:.4f}")
    print(f"  {'low-rank re-split':22s} {lpi_lowrank:.4f}   sizes={lr_sizes.tolist()}")
    print(f"\n  native reduction vs random : {lpi_random - lpi_native:+.4f}")
    print(f"  low-rank reduction vs native: {lpi_native - lpi_lowrank:+.4f}")

    import json
    json.dump({"n": int(n), "mean_cosine": mc, "cos_neg_frac": negfrac,
               "sigma": sigma, "lpi_validate_fac": lpi_fac, "lpi_validate_exact": lpi_ex,
               "lpi_random": lpi_random, "lpi_native": lpi_native,
               "lpi_lowrank": lpi_lowrank, "lr_sizes": lr_sizes.tolist()},
              open(os.path.join(CACHE, "uma_leakage_result.json"), "w"), indent=2)
    print("\nsaved", os.path.join(CACHE, "uma_leakage_result.json"))


if __name__ == "__main__":
    main()
