"""Hypergraph (n-D) splitting on the 3D+ reaction datasets + leakage metrics.

The reaction datasets in ``data/DataSAIL_data/3D+`` have MORE THAN TWO component
axes per record (Buchwald-Hartwig: ligand/additive/base/aryl-halide; Suzuki-
Miyaura: reactant1/reactant2/ligand/base/solvent). DataSAIL's splitting engine is
2-D and cannot express a joint split over >2 axes, so here we report the
hypergraph n-D split (``hypergraph.run_hypergraph_split_nd``) against a random
baseline, scored with the n-D generalization of the molecular scaled L(pi).

Metrics (lower = less leakage), reported per axis and as an unweighted macro
average across axes:
  - scaled L(pi): fraction of total pairwise component similarity that crosses
    the train/test boundary. Similarity on an axis is the feature similarity
    (Tanimoto for fingerprints, cosine for real-valued descriptors) when both
    records have a feature vector for that component, and identity (1 iff the
    component value is the same) otherwise. This is the direct n-D analogue of
    DataSAIL's L(pi).
  - identity leakage: fraction of test records whose component value also
    appears in the training split (the "was this exact component seen in
    training?" cold-split metric).

Run (palm env, from the PALM parent dir):
    python -m PALM.benchmark.benchmark_reactions               # sim_threshold=1.0
    python -m PALM.benchmark.benchmark_reactions 0.8           # similarity-aware
Writes benchmark/reactions_results.csv and prints a per-axis breakdown.

Dependencies: numpy, pandas, openpyxl, rdkit, scikit-learn/scipy, mtkahypar
(no torch needed — the n-D path does not use GPU k-NN). The Suzuki base axis
additionally uses PALM's MAGPIE material featurizer.
"""

import csv
import os
import random
import sys
import time

import numpy as np

from .. import reactions as R
from ..hypergraph import run_hypergraph_split_nd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "reactions_results.csv")


# ── n-D leakage metric ─────────────────────────────────────────────────────

def _feature_matrix(vals, feat_map):
    """(F[n,d] float32, has[n] bool); missing/all-zero features -> zero row, has=False."""
    dim = 0
    for v in vals:
        f = feat_map.get(v)
        if f is not None and np.any(f):
            dim = len(np.asarray(f).ravel())
            break
    n = len(vals)
    F = np.zeros((n, dim), dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i, v in enumerate(vals):
        f = feat_map.get(v)
        if dim and f is not None and np.any(f):
            F[i] = np.asarray(f, dtype=np.float32).ravel()
            has[i] = True
    return F, has


def _is_binary(F):
    return F.shape[1] >= 8 and np.all((F == 0) | (F == 1))


def axis_scaled_lpi(vals, feat_map, labels, block=1024):
    """Scaled L(pi) for one axis over records. See module docstring for the
    feature-vs-identity similarity definition."""
    n = len(vals)
    _, codes = np.unique(np.asarray(vals), return_inverse=True)   # identity codes
    labels = np.asarray(labels)
    F, has = _feature_matrix(vals, feat_map)
    d = F.shape[1]
    binary = _is_binary(F) if d else False
    if d and not binary:
        Fs = (F - F.mean(0)) / (F.std(0) + 1e-8)
        nrm = np.linalg.norm(Fs, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Fn = Fs / nrm                       # cosine via normalized dot product

    total = 0.0
    leak = 0.0
    for s in range(0, n, block):
        e = min(s + block, n)
        if d and binary:
            inter = F[s:e] @ F.T
            card = F.sum(1)
            union = card[s:e][:, None] + card[None, :] - inter
            fsim = np.where(union > 0, inter / union, 0.0)
        elif d:
            fsim = np.clip(Fn[s:e] @ Fn.T, 0.0, None)
        else:
            fsim = np.zeros((e - s, n), dtype=np.float32)
        idsim = (codes[s:e][:, None] == codes[None, :]).astype(np.float32)
        both = has[s:e][:, None] & has[None, :]
        sim = np.where(both, fsim, idsim)
        cross = (labels[s:e][:, None] != labels[None, :])
        total += float(sim.sum())
        leak += float((sim * cross).sum())
    total -= n                              # remove diagonal (self-similarity == 1)
    return leak / total if total > 0 else 0.0


def identity_leakage(vals, labels):
    """Fraction of test records whose component value also appears in train."""
    vals = np.asarray(vals)
    labels = np.asarray(labels)
    train_vals = set(vals[labels == "train"].tolist())
    test = vals[labels == "test"]
    if test.size == 0:
        return 0.0
    return float(np.mean([v in train_vals for v in test]))


def evaluate(records, axis_feature_maps, labels):
    axes = list(axis_feature_maps.keys())
    per_axis = {}
    for a in axes:
        vals = [str(r[a]) for r in records]
        per_axis[a] = {
            "lpi": round(axis_scaled_lpi(vals, axis_feature_maps[a], labels), 4),
            "id_leak": round(identity_leakage(vals, labels), 4),
            "n_unique": len(set(vals)),
        }
    macro_lpi = round(float(np.mean([per_axis[a]["lpi"] for a in axes])), 4)
    macro_id = round(float(np.mean([per_axis[a]["id_leak"] for a in axes])), 4)
    return macro_lpi, macro_id, per_axis


def random_split(n, seed=42):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(0.8 * n)
    lab = ["train"] * n
    for i in idx[cut:]:
        lab[i] = "test"
    return lab


# ── runner ─────────────────────────────────────────────────────────────────

DATASETS = {
    "buchwald_hartwig": R.load_buchwald_hartwig,
    "suzuki_miyaura": R.load_suzuki_miyaura,
}

COLS = ["dataset", "n", "n_axes", "axes",
        "hg_macro_lpi", "random_macro_lpi",
        "hg_macro_id_leak", "random_macro_id_leak",
        "hg_time_s", "train_frac", "km1", "imbalance"]


def run(name, loader, sim_threshold=1.0):
    records, afm, _target = loader()
    n = len(records)
    t0 = time.time()
    assignment, info = run_hypergraph_split_nd(
        records, afm, [8, 2], ["train", "test"], sim_threshold=sim_threshold)
    hg_t = round(time.time() - t0, 2)

    hg_lpi, hg_id, hg_axis = evaluate(records, afm, assignment)
    r_lpi, r_id, _ = evaluate(records, afm, random_split(n))
    n_train = sum(1 for a in assignment if a == "train")
    return {
        "dataset": name, "n": n, "n_axes": len(afm), "axes": ";".join(afm.keys()),
        "hg_macro_lpi": hg_lpi, "random_macro_lpi": r_lpi,
        "hg_macro_id_leak": hg_id, "random_macro_id_leak": r_id,
        "hg_time_s": hg_t, "train_frac": round(n_train / n, 3),
        "km1": info["km1"], "imbalance": info["imbalance"],
        "per_axis": hg_axis,
    }


def main():
    sim_threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"sim_threshold = {sim_threshold} "
          f"({'identity grouping' if sim_threshold >= 1.0 else 'similarity-aware grouping'})\n",
          flush=True)
    rows = []
    for name, loader in DATASETS.items():
        print(f"[{name}] ...", flush=True)
        try:
            row = run(name, loader, sim_threshold=sim_threshold)
        except Exception as e:
            import traceback
            traceback.print_exc()
            row = {"dataset": name, "hg_macro_lpi": f"ERROR ({type(e).__name__}: {str(e)[:60]})"}
        rows.append(row)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    for r in rows:
        print(f"\n=== {r['dataset']} (n={r.get('n', '?')}, axes={r.get('n_axes', '?')}) ===")
        if "per_axis" not in r:
            print(f"  {r['hg_macro_lpi']}")
            continue
        print(f"  macro L(pi):    hypergraph={r['hg_macro_lpi']:.4f}   random={r['random_macro_lpi']:.4f}")
        print(f"  macro id-leak:  hypergraph={r['hg_macro_id_leak']:.4f}   random={r['random_macro_id_leak']:.4f}")
        print(f"  split: train={r['train_frac']:.0%}  KM1={r['km1']}  imbalance={r['imbalance']}  time={r['hg_time_s']}s")
        for a, m in r["per_axis"].items():
            print(f"    {a:12s} L(pi)={m['lpi']:.4f}  id_leak={m['id_leak']:.4f}  n_unique={m['n_unique']}")
    print("\nDONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
