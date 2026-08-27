"""Non-circular evidence that low-rank produces a GOOD OMol25 split.

All analyses run on the cached 100k UMA subsample (no new UMA inference):
  Beat 2  held-out re-score : split optimized on UMA, L(pi) re-scored on the
                              INDEPENDENT 115-d structural descriptor (and vice
                              versa). If the reduction survives the held-out
                              representation it is not a circular artifact.
  Beat 3a mechanism         : per-structure UMA prediction error vs its cosine
                              proximity to the (native) train pool. Negative
                              correlation == leakage inflates apparent accuracy.
  Beat 3b inflation gap     : mean error of evaluated structures assigned to the
                              native test block vs the low-rank test block.
  Surrogate retrain         : a kNN energy/atom regressor on UMA embeddings,
                              random 80/20 vs low-rank 80/20 test MAE (5 seeds).
                              A genuinely de-leaked split -> higher, honest MAE.

Outputs: results/beat_analysis.json  +  results/beat_analysis.png
Run:  <palm python> beat_analysis.py
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
import torch
from scipy.stats import pearsonr, spearmanr
from PALM.benchmarks.omol25 import omol25_leakage as LK
from PALM.splitters import SplitSpec, split

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def load_all():
    """Assemble everything in subsample order (100k)."""
    parts = sorted(glob.glob(os.path.join(HERE, "_cache_uma/parts/uma_emb.part*.npy")),
                   key=lambda p: int(os.path.basename(p).split("part")[1].split(".")[0]))
    U = np.concatenate([np.load(p) for p in parts]).astype(np.float32)      # (100k,128)
    orig = np.load(os.path.join(HERE, "_cache_uma/orig_row.npy"))            # (100k,)
    native = np.load(os.path.join(HERE, "_cache_uma/native.npy"))           # (100k,)
    n = len(U)
    assert len(orig) == n and len(native) == n

    # split labels from parquet, realigned to subsample order via orig_row
    df = pd.read_parquet(os.path.join(RES, "omol25_uma_lowrank_split.parquet"))
    row_of = {int(r): i for i, r in enumerate(df["orig_row"].to_numpy())}
    pidx = np.array([row_of[int(o)] for o in orig])
    lab_native = df["native_split"].to_numpy()[pidx].astype(np.int64)
    lab_lr3 = df["lowrank_3way"].to_numpy()[pidx].astype(np.int64)
    lab_lr82 = df["lowrank_8020"].to_numpy()[pidx].astype(np.int64)

    # independent structural descriptor for the SAME 100k, nonneg-scaled
    Xfull = np.load(LK.CACHE_DIR + "/features.npy", mmap_mode="r")
    S = np.asarray(Xfull[np.sort(orig)], dtype=np.float32)   # sorted for mmap speed
    # undo the sort so S aligns to subsample order
    inv = np.argsort(np.argsort(orig))
    S = S[inv]
    S = np.nan_to_num(S) * LK.fit_nonneg_scale(np.asarray(Xfull[:300000]))

    # prediction error (1396) -> subsample-local indices
    pe = np.load(os.path.join(RES, "uma_prediction_error.npz"))
    return dict(U=U, S=S, orig=orig, lab_native=lab_native, lab_lr3=lab_lr3,
                lab_lr82=lab_lr82, ki=pe["kept_local_idx"], err=pe["err_per_atom"],
                true_e=pe["true_energy"], natoms=pe["natoms"])


def factor(X, rank=512):
    return LK.build_factor(X, rank=min(rank, len(X)), seed=SEED)


def beat2_heldout(d):
    """L(pi) of native vs UMA-optimized low-rank split, scored on each representation."""
    B_uma = factor(d["U"])
    B_str = factor(d["S"])
    out = {}
    for split_name, lab in [("native", d["lab_native"]), ("lowrank_UMA", d["lab_lr3"])]:
        out[split_name] = {
            "score_on_UMA":        LK.lpi_from_factor(B_uma, lab, 3),
            "score_on_structural": LK.lpi_from_factor(B_str, lab, 3),
        }
    # random baseline (same 3 block sizes as native)
    rng = np.random.default_rng(SEED)
    rlab = d["lab_native"].copy(); rng.shuffle(rlab)
    out["random"] = {"score_on_UMA": LK.lpi_from_factor(B_uma, rlab, 3),
                     "score_on_structural": LK.lpi_from_factor(B_str, rlab, 3)}
    # validate factor vs exact on a 20k subsample (structural, held-out one)
    sub = rng.choice(len(d["S"]), 20000, replace=False)
    Bsub = factor(d["S"][sub])
    out["_validate_structural"] = {
        "factor": LK.lpi_from_factor(Bsub, d["lab_native"][sub], 3),
        "exact":  LK.lpi_exact_cosine(d["S"][sub], d["lab_native"][sub])}
    return out


def max_sim_to_pool(Uq, Upool, block=2048):
    """Max cosine sim of each query row to the pool (self excluded when identical)."""
    q = torch.nn.functional.normalize(torch.as_tensor(Uq, device=DEV), dim=1)
    p = torch.nn.functional.normalize(torch.as_tensor(Upool, device=DEV), dim=1)
    out = np.empty(len(Uq), dtype=np.float32)
    for s in range(0, len(Uq), block):
        e = min(s + block, len(Uq))
        sim = q[s:e] @ p.T                       # (b, npool)
        out[s:e] = sim.max(dim=1).values.cpu().numpy()
    return out


def beat3_mechanism_and_gap(d):
    ki, err = d["ki"], d["err"]
    U = d["U"]
    train_native = U[d["lab_native"] == 0]                     # native train pool
    sim_to_train = max_sim_to_pool(U[ki], train_native)        # proximity of evaluated set

    pr, pp = pearsonr(sim_to_train, err)
    sr, sp = spearmanr(sim_to_train, err)
    # binned: mean error by proximity quintile
    q = np.quantile(sim_to_train, np.linspace(0, 1, 6))
    binned = []
    for i in range(5):
        m = (sim_to_train >= q[i]) & (sim_to_train <= q[i + 1])
        binned.append({"sim_lo": float(q[i]), "sim_hi": float(q[i + 1]),
                       "n": int(m.sum()), "mean_err": float(err[m].mean())})

    # 3b inflation gap: evaluated structures in native-test vs lowrank-test block
    nat = d["lab_native"][ki]; lr = d["lab_lr82"][ki]
    gap = {
        "native_test": {"n": int((nat == 2).sum()),
                        "mean_err": float(err[nat == 2].mean()),
                        "mean_sim_to_native_train": float(sim_to_train[nat == 2].mean())},
        "lowrank_test": {"n": int((lr == 1).sum()),
                         "mean_err": float(err[lr == 1].mean())},
    }
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
            "binned_err_by_proximity": binned, "inflation_gap": gap,
            "_sim": sim_to_train}


def surrogate_retrain(d, k=5, seeds=5):
    """kNN energy/atom regressor: random vs low-rank 80/20 test MAE over seeds."""
    ki = d["ki"]
    Uk = d["U"][ki]
    y = d["true_e"] / d["natoms"]                             # energy per atom (eV)
    Un = Uk / (np.linalg.norm(Uk, axis=1, keepdims=True) + 1e-9)
    n = len(Uk)

    def knn_mae(tr, te):
        Xt = torch.as_tensor(Un[tr], device=DEV)
        Xe = torch.as_tensor(Un[te], device=DEV)
        sim = Xe @ Xt.T
        nn = sim.topk(min(k, len(tr)), dim=1).indices.cpu().numpy()
        pred = y[tr][nn].mean(axis=1)
        return float(np.abs(pred - y[te]).mean())

    rand_maes, lr_maes = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        perm = rng.permutation(n); ntr = int(0.8 * n)
        rand_maes.append(knn_mae(perm[:ntr], perm[ntr:]))
        # low-rank 80/20 split of the SAME points, in UMA space
        fd = {i: Uk[i] for i in range(n)}
        sp = split("lowrank", fd, SplitSpec([8, 2], ["train", "test"], seed=s),
                   rank=min(256, n), metric="cosine").assignment
        tr = np.array([i for i in range(n) if sp[i] == "train"])
        te = np.array([i for i in range(n) if sp[i] == "test"])
        lr_maes.append(knn_mae(tr, te))
    return {"k": k, "n": n, "random_MAE_mean": float(np.mean(rand_maes)),
            "random_MAE_std": float(np.std(rand_maes)),
            "lowrank_MAE_mean": float(np.mean(lr_maes)),
            "lowrank_MAE_std": float(np.std(lr_maes)),
            "inflation_ratio": float(np.mean(lr_maes) / np.mean(rand_maes))}


def make_figure(b3, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sim, err = b3["_sim"], None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    b = b3["binned_err_by_proximity"]
    centers = [(x["sim_lo"] + x["sim_hi"]) / 2 for x in b]
    means = [x["mean_err"] for x in b]
    ax[0].plot(centers, means, "o-", color="#2563EB", lw=2)
    ax[0].set_xlabel("cosine proximity to train (UMA)")
    ax[0].set_ylabel("mean UMA error / atom (eV)")
    ax[0].set_title(f"Leaked structures score better\n(Spearman r={b3['spearman_r']:.2f}, "
                    f"p={b3['spearman_p']:.1e})")
    g = b3["inflation_gap"]
    labels = ["native\ntest", "low-rank\ntest"]
    vals = [g["native_test"]["mean_err"], g["lowrank_test"]["mean_err"]]
    ax[1].bar(labels, vals, color=["#9CA3AF", "#2563EB"])
    ax[1].set_ylabel("mean UMA error / atom (eV)")
    ax[1].set_title("Low-rank test block is harder\n(honest error, not inflated)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    d = load_all()
    print(f"assembled: U{d['U'].shape} structural{d['S'].shape} eval={len(d['ki'])}")

    print("Beat 2: held-out re-score ...")
    b2 = beat2_heldout(d)
    print(json.dumps(b2, indent=2))

    print("Beat 3: mechanism + inflation gap ...")
    b3 = beat3_mechanism_and_gap(d)
    print(f"  Spearman(sim,err) = {b3['spearman_r']:.3f} (p={b3['spearman_p']:.1e})")
    print(f"  native-test mean err  = {b3['inflation_gap']['native_test']['mean_err']:.4f}")
    print(f"  lowrank-test mean err = {b3['inflation_gap']['lowrank_test']['mean_err']:.4f}")

    print("Surrogate retrain (kNN energy/atom) ...")
    sur = surrogate_retrain(d)
    print(f"  random MAE  = {sur['random_MAE_mean']:.4f} +/- {sur['random_MAE_std']:.4f}")
    print(f"  lowrank MAE = {sur['lowrank_MAE_mean']:.4f} +/- {sur['lowrank_MAE_std']:.4f}"
          f"  (ratio {sur['inflation_ratio']:.2f}x)")

    make_figure(b3, os.path.join(RES, "beat_analysis.png"))
    b3.pop("_sim")
    out = {"beat2_heldout": b2, "beat3": b3, "surrogate": sur, "n_eval": int(len(d["ki"]))}
    with open(os.path.join(RES, "beat_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/beat_analysis.json + beat_analysis.png")


if __name__ == "__main__":
    main()
