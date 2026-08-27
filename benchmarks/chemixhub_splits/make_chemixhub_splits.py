#!/usr/bin/env python
"""Chem-OOD splits for the CheMixHub mixture-property suite via PALM's engines.

Reproduces the MixUni chem-OOD split *recipe* (Rajaonson et al. CheMixHub +
MixUni, ICML 2026 GFM workshop, Appendix A.4 / Table 4) but replaces the
Butina + LPT-bin-packing partitioner with PALM's two leakage-minimizing engines:

  * ``hypergraph``  — k-NN similarity hyperedges, Mt-KaHyPar KM1 cut
  * ``lowrank``     — Nyström factorization + balanced-Lloyd + FM polish

Featurization matches the paper exactly:
  * per-component Morgan fingerprint, radius 2, 1024 bits, from canonical SMILES
  * per-mixture fingerprint = mole-fraction-weighted mean; salt components get a
    fixed pseudo-weight w_salt = 0.5 (Eq. 13); binarized at 0.5
  * samples collapsed to unique mixture identity (sorted solvent set + sorted
    salt set); every (T, c) sample of a mixture inherits its bucket
  * split fractions 0.70 / 0.20 / 0.10 (train / val / test), seed 42

The whole-mixture assignment guarantees zero mixture-identity leakage (the paper's
Table-16 failure mode); L(pi) then measures the residual *chemical-similarity*
leakage across buckets, which is exactly what the two engines minimize.

Run (GPU recommended):
    CUDA_VISIBLE_DEVICES=<free gpu> \
    /homes/rzhu/miniforge3/envs/palm/bin/python make_chemixhub_splits.py \
        --data-root <clone>/datasets --out <this dir>
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# PALM lives one dir above the repo root on this box.
sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.splitters import split, SplitSpec  # noqa: E402
from PALM.splitters.common.leakage_metrics import scaled_lpi  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BUTINA_TAU = 0.4          # distance cutoff (Table 4); merge if Tanimoto >= 1-tau = 0.6

FP_RADIUS = 2
FP_BITS = 1024
W_SALT = 0.5
SEED = 42
SPLITS = [70, 20, 10]
NAMES = ["train", "val", "test"]

# dataset dir -> (processed csv basename, list of cmp_ids columns)
DATASETS = {
    "ionic-liquids":         ("processed_IlThermoData",          ["cmp_ids"]),
    "miscible-solvent":      ("processed_MiscibleSolventData",   ["cmp_ids"]),
    "drug-solubility":       ("processed_DrugSolubilityData",    ["cmp_ids_solvent", "cmp_ids_drug"]),
    "polymer-electrolyte":   ("processed_PolymerElectrolyteData",["cmp_ids"]),
    "olfactory-similarity":  ("processed_OlfactorySimilarity",   ["cmp_ids_1", "cmp_ids_2"]),
    "logV":                  ("processed_logV",                  ["cmp_ids"]),
    "nist-logV":             ("processed_NISTlogV",              ["cmp_ids"]),
    "MON":                   ("processed_MON",                   ["cmp_ids"]),
    "medicine-formulations": ("processed_MedicineFormulations",  ["cmp_ids"]),
}


def slug(s: str) -> str:
    return s.lower().replace(" ", "_")


def morgan(smiles: str) -> np.ndarray | None:
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    a = np.zeros(FP_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_BITS), a)
    return a


def parse_list(cell):
    """'[0.0, 625.0]' -> [0, 625]; robust to NaN / already-list."""
    if isinstance(cell, (list, tuple)):
        return list(cell)
    if pd.isna(cell):
        return []
    return ast.literal_eval(str(cell))


def load_compounds(path: str):
    """Return (id -> fingerprint-or-None, id -> is_salt bool)."""
    df = pd.read_csv(path)
    fp, is_salt = {}, {}
    has_salt = "salt" in df.columns
    for _, row in df.iterrows():
        cid = int(row["compound_id"])
        fp[cid] = morgan(row["smiles"])
        is_salt[cid] = bool(int(row["salt"])) if has_salt and not pd.isna(row.get("salt")) else False
    return fp, is_salt


def mixture_fingerprint_and_key(row, id_cols, fp_map, is_salt):
    """Weighted-mean binarized Morgan fp + mixture identity key for one sample.

    Solvent components weighted by mole fraction; salts by fixed W_SALT (Eq. 13).
    Columns with no matching mole-fraction column use equal weights.
    Returns (fp float32[FP_BITS] or None, key) where key = (solvent set, salt set).
    """
    num = np.zeros(FP_BITS, dtype=np.float64)
    den = 0.0
    solv_ids, salt_ids = [], []
    for col in id_cols:
        ids = [int(round(float(x))) for x in parse_list(row[col])]
        frac_col = col.replace("cmp_ids", "cmp_mole_fractions")
        if frac_col in row.index and not (isinstance(row[frac_col], float) and pd.isna(row[frac_col])):
            fracs = [float(x) for x in parse_list(row[frac_col])]
        else:
            fracs = []
        if len(fracs) != len(ids):
            fracs = [1.0 / max(len(ids), 1)] * len(ids)   # equal weights fallback
        for cid, fr in zip(ids, fracs):
            f = fp_map.get(cid)
            salt = is_salt.get(cid, False)
            (salt_ids if salt else solv_ids).append(cid)
            if f is None:
                continue
            w = W_SALT if salt else fr
            num += w * f
            den += w
    if den <= 0:
        return None, None
    mean = num / den
    binv = (mean >= 0.5).astype(np.float32)          # binarize at 0.5 (Table 4)
    key = (tuple(sorted(solv_ids)), tuple(sorted(salt_ids)))
    return binv, key


def build_unique_mixtures(sub, id_cols, fp_map, is_salt):
    """Collapse a property-filtered dataframe to unique mixtures.

    Returns ``(key_of_sample, samples_of_key, feature_data)`` where
    ``feature_data`` maps ``str(key) -> binarized weighted-mean fingerprint``.
    """
    n = len(sub)
    key_of_sample = [None] * n
    samples_of_key = defaultdict(list)
    fp_of_key = {}
    for i in range(n):
        binv, key = mixture_fingerprint_and_key(sub.iloc[i], id_cols, fp_map, is_salt)
        if key is None:
            key = (("__unpar?%d" % i,), ())          # keep row; isolated identity
            binv = np.zeros(FP_BITS, dtype=np.float32)
        key_of_sample[i] = key
        samples_of_key[key].append(i)
        fp_of_key.setdefault(key, binv)
    feature_data = {str(k): fp_of_key[k] for k in samples_of_key}
    return key_of_sample, samples_of_key, feature_data


def butina_clusters(X, tau=BUTINA_TAU, block=2048):
    """Exact Butina sphere-exclusion clustering on binarized-fingerprint Tanimoto.

    Two mixtures are neighbors iff Tanimoto similarity >= 1 - tau. Points are
    processed in descending neighbor count; each unclustered point seeds a cluster
    that claims all its still-unclustered neighbors (Butina, 1999). Returns a
    cluster label per row (row order = the order of ``X``).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    n = Xt.shape[0]
    card = Xt.sum(1)
    sim_cut = 1.0 - tau

    rows, cols = [], []
    for s in range(0, n, block):
        e = min(s + block, n)
        inter = Xt[s:e] @ Xt.T
        union = card[s:e, None] + card[None, :] - inter
        sim = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        r_idx, c_idx = torch.nonzero(sim >= sim_cut, as_tuple=True)
        rows.append((r_idx + s).cpu().numpy())
        cols.append(c_idx.cpu().numpy())
    rows = np.concatenate(rows) if rows else np.empty(0, np.int64)
    cols = np.concatenate(cols) if cols else np.empty(0, np.int64)

    counts = np.bincount(rows, minlength=n)
    srt = np.argsort(rows, kind="stable")            # CSR neighbor lists
    cols_s = cols[srt]
    offs = np.concatenate([[0], np.cumsum(counts)])

    order = np.argsort(-counts, kind="stable")
    assigned = np.full(n, -1, dtype=np.int64)
    cid = 0
    for i in order:
        if assigned[i] != -1:
            continue
        nbrs = cols_s[offs[i]:offs[i + 1]]
        take = nbrs[assigned[nbrs] == -1]
        assigned[take] = cid
        assigned[i] = cid
        cid += 1
    return assigned


def lpt_bin_pack(cluster_of_key, sample_count_of_key, sorted_keys, fractions=(0.70, 0.20, 0.10)):
    """Cluster-aware LPT bin packing to train/val/test by SAMPLE count (paper A.4).

    Whole clusters, processed in descending sample count, each placed in the
    bucket with the largest remaining sample-count deficit. Returns {str_key: bucket}.
    """
    clus_samples = defaultdict(int)
    clus_keys = defaultdict(list)
    for k in sorted_keys:
        c = int(cluster_of_key[k])
        clus_samples[c] += sample_count_of_key[k]
        clus_keys[c].append(k)
    total = sum(clus_samples.values())
    targets = [f * total for f in fractions]
    current = [0, 0, 0]
    order = sorted(clus_samples, key=lambda c: (-clus_samples[c], c))   # size-desc, deterministic
    bucket_of = {}
    names = ["train", "val", "test"]
    for c in order:
        b = int(np.argmax([targets[j] - current[j] for j in range(3)]))
        current[b] += clus_samples[c]
        for k in clus_keys[c]:
            bucket_of[k] = names[b]
    return bucket_of


def sample_random_identity_leakage(key_of_sample, n, seed=SEED, fractions=(0.70, 0.20, 0.10)):
    """Paper Table-16 metric: sample-level random 70/20/10, fraction of test
    samples whose mixture identity also appears in train."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    train_keys = {key_of_sample[i] for i in tr}
    if len(te) == 0:
        return 0.0
    return sum(1 for i in te if key_of_sample[i] in train_keys) / len(te)


def mixture_identity_leakage(idx_by_bucket, key_of_sample):
    """Paper Table-16 metric: fraction of test samples whose mixture key is in train."""
    train_keys = {key_of_sample[i] for i in idx_by_bucket["train"]}
    test = idx_by_bucket["test"]
    if not test:
        return 0.0
    return sum(1 for i in test if key_of_sample[i] in train_keys) / len(test)


def random_group_split(keys, sample_count, rng):
    """Assign whole unique mixtures to train/val/test ~70/20/10 by mixture count."""
    order = list(keys)
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.20 * n))
    lab = {}
    for i, k in enumerate(order):
        lab[k] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return lab


def process(dataset_dir, csv_name, id_cols, out_root, records):
    ddir = os.path.join(dataset_dir, "processed_data")
    df = pd.read_csv(os.path.join(ddir, f"{csv_name}.csv"))
    fp_map, is_salt = load_compounds(os.path.join(ddir, "compounds.csv"))

    props = df["property"].unique() if "property" in df.columns else ["value"]
    for prop in props:
        sub = df[df["property"] == prop].reset_index(drop=True) if "property" in df.columns else df
        n = len(sub)

        key_of_sample, samples_of_key, feature_data = build_unique_mixtures(
            sub, id_cols, fp_map, is_salt)
        keys = list(samples_of_key.keys())
        m = len(keys)
        kbar = n / m
        sorted_keys = sorted(feature_data.keys())
        X = np.vstack([feature_data[k] for k in sorted_keys])

        name = f"{os.path.basename(dataset_dir)} / {prop}"
        print(f"\n=== {name}: {n} samples, {m} unique mixtures (k̄={kbar:.1f}) ===",
              flush=True)

        row_common = dict(dataset=os.path.basename(dataset_dir), property=prop,
                          n_samples=n, n_unique_mixtures=m, k_bar=round(kbar, 3))
        out_dir = os.path.join(out_root, os.path.basename(dataset_dir), f"{slug(prop)}_splits")
        os.makedirs(out_dir, exist_ok=True)
        sample_count_of_key = {str(k): len(samples_of_key[k]) for k in keys}

        def _save_and_score(engine, bucket_of_strkey):
            """Expand a {str_key: bucket} assignment to samples, save, score, record."""
            idx_by_bucket = {nm: [] for nm in NAMES}
            for k in keys:
                idx_by_bucket[bucket_of_strkey[str(k)]].extend(samples_of_key[k])
            for nm in NAMES:
                idx_by_bucket[nm].sort()
            save_file(
                {f"{nm}_indices": np.asarray(idx_by_bucket[nm], dtype=np.int64) for nm in NAMES},
                os.path.join(out_dir, f"{engine}_chemood_split.safetensors"),
            )
            labels = np.array([{"train": 0, "val": 1, "test": 2}[bucket_of_strkey[k]]
                               for k in sorted_keys])
            lpi = round(float(scaled_lpi(X, labels, metric="tanimoto")), 6)
            samp_total = sum(len(idx_by_bucket[nm]) for nm in NAMES)
            samp_frac = {nm: round(len(idx_by_bucket[nm]) / samp_total, 4) for nm in NAMES}
            id_leak = mixture_identity_leakage(idx_by_bucket, key_of_sample)
            return lpi, id_leak, samp_frac

        # ---- paper's Table-16 metric: sample-level random-split identity leakage ----
        rand_idleak = sample_random_identity_leakage(key_of_sample, n)
        records.append({**row_common, "engine": "random_sample",
                        "mixture_identity_leakage": round(rand_idleak, 6)})
        print(f"  random_sample     id-leak={rand_idleak:.4f}  (paper Table 16)")

        # ---- random-group baseline (whole mixtures, mixture-count balanced) ----
        rng = np.random.default_rng(SEED)
        rg_lab = random_group_split(sorted_keys, samples_of_key, rng)   # keyed by str(key)
        rg_lpi, rg_idleak, rg_frac = _save_and_score("random_group", rg_lab)
        records.append({**row_common, "engine": "random_group",
                        "Lpi_unique_mixture": rg_lpi, "mixture_identity_leakage": round(rg_idleak, 6),
                        "sample_fractions": rg_frac})
        print(f"  random_group      L(pi)={rg_lpi:.4f}")

        # ---- Butina chem-OOD (the paper's method): cluster tau=0.4 + LPT bin-pack ----
        t0 = time.time()
        clus = butina_clusters(X)
        cluster_of_key = {sorted_keys[i]: int(clus[i]) for i in range(m)}
        but_lab = lpt_bin_pack(cluster_of_key, sample_count_of_key, sorted_keys)
        but_lpi, but_idleak, but_frac = _save_and_score("butina", but_lab)
        n_clusters = int(clus.max()) + 1
        records.append({**row_common, "engine": "butina", "Lpi_unique_mixture": but_lpi,
                        "mixture_identity_leakage": round(but_idleak, 6),
                        "n_clusters": n_clusters, "sample_fractions": but_frac,
                        "runtime_s": round(time.time() - t0, 3)})
        print(f"  butina (paper)    L(pi)={but_lpi}  id-leak={but_idleak:.4f}  "
              f"{n_clusters} clusters  samples {but_frac}")

        for engine, kwargs in (("hypergraph", dict(metric="tanimoto", preset="quality")),
                               ("lowrank", dict(metric="tanimoto"))):
            spec = SplitSpec(splits=SPLITS, names=NAMES, seed=SEED, epsilon=0.10)
            t0 = time.time()
            res = split(engine, feature_data, spec, **kwargs)
            dt = time.time() - t0
            bucket_of = {str(k): res.assignment[str(k)] for k in keys}
            lpi, id_leak, samp_frac = _save_and_score(engine, bucket_of)
            records.append({**row_common, "engine": engine,
                            "Lpi_unique_mixture": lpi,
                            "mixture_identity_leakage": round(id_leak, 6),
                            "mixture_count_fractions": res.diagnostics["split_fractions"],
                            "sample_fractions": samp_frac,
                            "imbalance": res.diagnostics.get("imbalance"),
                            "runtime_s": round(dt, 3)})
            print(f"  {engine:16s} L(pi)={lpi}  id-leak={id_leak:.4f}  "
                  f"samples {samp_frac}  ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="path to <clone>/datasets")
    ap.add_argument("--out", required=True, help="output dir for splits + report")
    ap.add_argument("--only", nargs="*", default=None, help="subset of dataset dir names")
    args = ap.parse_args()

    records = []
    for dset, (csv_name, id_cols) in DATASETS.items():
        if args.only and dset not in args.only:
            continue
        ddir = os.path.join(args.data_root, dset)
        if not os.path.isdir(ddir):
            print(f"!! missing {ddir}, skip", flush=True)
            continue
        try:
            process(ddir, csv_name, id_cols, args.out, records)
        except Exception as e:
            import traceback
            print(f"!! {dset} FAILED: {e}", flush=True)
            traceback.print_exc()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "leakage_report.json"), "w") as f:
        json.dump(records, f, indent=2)
    pd.DataFrame(records).to_csv(os.path.join(args.out, "leakage_report.csv"), index=False)
    print(f"\nWrote report to {args.out}/leakage_report.{{json,csv}}", flush=True)


if __name__ == "__main__":
    main()
