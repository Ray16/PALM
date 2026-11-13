"""Within-dataset, coverage-matched re-analysis of the master benchmark.

``analyze.py`` pools (dataset, method) points across heterogeneous datasets
(ROC-AUC, R^2 and accuracy on one axis; unequal method coverage — DataSAIL only
on n<=3000, scaffold only on SMILES sets). This script asks the narrower,
defensible questions from the **saved** results only (no model is retrained):

1. **Paired effects vs random.** For each dataset and seed, ``method - random``
   on ``test_metric`` and ``gen_gap`` (pairs are dropped unless both runs used
   the *same* metric — MUV falls back to accuracy for some splits). Per-dataset
   mean with a t-interval over the paired seeds.

   Caveat on replicates: non-random splitters are deterministic, so their three
   "seeds" are the *same* partition with a re-seeded RandomForest (model
   variance only); random re-draws the partition per seed. The interval is
   therefore model + random-split variance, not the variance of the method.

2. **Coverage-matched aggregates.** Methods are compared only on datasets where
   *every* compared method has a paired effect, per metric family; the across-
   dataset mean paired effect gets a bootstrap-over-datasets 95% CI and a
   sign count.

3. **Within- vs pooled leakage/gap correlation.** Pearson r pooled (as in
   ``analyze.py``) vs after centering leakage and gap within each dataset.

4. **Split-induced confounds** (label shift, train size). Train/test label
   mean/std (regression) or positive rate (classification) and train size per
   split, for splits whose assignment can be recovered *and verified* against
   the master row (identical n_train_lab / n_test_lab / leakage): first the
   assignments ``run_benchmark.py`` saves alongside every row
   (``benchmarks/results/assignments``, all seeds), then
   saved assignments in ``PALM/data/split_results/assignments`` (from
   ``data/run_splits.py``, seed 0), and — with ``--reconstruct`` — cheap
   deterministic re-partitioning (random seeds, legacy scaffold, hypergraph,
   graph). Re-partitioning is CPU-only and trains nothing.

Outputs (``benchmarks/results/``): ``paired_effects_per_dataset.csv``,
``paired_effects_summary.csv``, ``label_shift.csv``; a markdown digest on stdout.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONPATH=<dir containing PALM/> \
    python \
        -m PALM.benchmarks.master.paired_analysis [--reconstruct]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
MASTER = os.path.join(RESULTS, "master_benchmark.csv")
SAVED_ASG = os.path.abspath(os.path.join(HERE, "..", "..", "data", "split_results", "assignments"))
# per-(dataset, method[, seed]) assignments written by run_benchmark.py itself
BENCH_ASG = os.path.join(RESULTS, "assignments")

RANDOM = "random"
FAMILY = {"roc_auc": "classification (ROC-AUC)", "r2": "regression (R^2)",
          "accuracy": "accuracy (degenerate test)"}
# method sets compared under coverage matching
COMPARE_SETS = {
    "core":  ["graph", "hypergraph", "lowrank"],
    "all":   ["scaffold", "datasail", "graph", "hypergraph", "lowrank"],
}
N_BOOT = 10_000


def _t_ci(x, level=0.95):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def _boot_ci(x, rng, level=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    bs = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return tuple(np.quantile(bs, [0.5 - level / 2, 0.5 + level / 2]))


def load_master(path=MASTER):
    m = pd.read_csv(path)
    ok = m[(m["status"] == "ok") & (m["kind"] == "1d")].copy()
    for c in ("leakage", "gen_gap", "test_metric", "train_metric", "test_fraction",
              "n_train_lab", "n_test_lab", "seed", "n"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    return ok


# ── 1. paired effects ────────────────────────────────────────────────────────

def paired_effects(ok):
    g = ok.dropna(subset=["gen_gap", "test_metric"])
    base = g[g["method"] == RANDOM][["dataset", "seed", "metric_name", "test_metric",
                                     "gen_gap", "n_train_lab", "n_test_lab"]]
    other = g[g["method"] != RANDOM]
    pr = other.merge(base, on=["dataset", "seed"], suffixes=("", "_rand"))
    pr["metric_match"] = pr["metric_name"] == pr["metric_name_rand"]
    pr = pr[pr["metric_match"]].copy()
    pr["d_test"] = pr["test_metric"] - pr["test_metric_rand"]
    pr["d_gap"] = pr["gen_gap"] - pr["gen_gap_rand"]
    pr["d_train_n"] = pr["n_train_lab"] - pr["n_train_lab_rand"]
    rows = []
    for (ds, meth), s in pr.groupby(["dataset", "method"]):
        lo_t, hi_t = _t_ci(s["d_test"])
        lo_g, hi_g = _t_ci(s["d_gap"])
        rows.append(dict(dataset=ds, method=meth, metric=s["metric_name"].iloc[0],
                         n_pairs=len(s), d_test=s["d_test"].mean(), d_test_lo=lo_t, d_test_hi=hi_t,
                         d_gap=s["d_gap"].mean(), d_gap_lo=lo_g, d_gap_hi=hi_g,
                         d_train_n=s["d_train_n"].mean(),
                         train_frac=(s["n_train_lab"] / (s["n_train_lab"] + s["n_test_lab"])).mean(),
                         leakage=s["leakage"].mean()))
    return pd.DataFrame(rows), pr


def coverage_matched(per, rng):
    out = []
    for set_name, methods in COMPARE_SETS.items():
        for metric, sub in per.groupby("metric"):
            if metric == "accuracy":
                continue
            piv = sub.pivot(index="dataset", columns="method", values="d_test")
            have = [mm for mm in methods if mm in piv.columns]
            if len(have) < len(methods):
                continue
            piv = piv[methods].dropna()
            gp = sub.pivot(index="dataset", columns="method", values="d_gap").loc[piv.index, methods]
            for meth in methods:
                x, y = piv[meth].values, gp[meth].values
                lo, hi = _boot_ci(x, rng)
                glo, ghi = _boot_ci(y, rng)
                out.append(dict(compare_set=set_name, metric=metric, method=meth,
                                n_datasets=len(x), datasets=";".join(piv.index),
                                mean_d_test=x.mean() if len(x) else np.nan,
                                d_test_lo=lo, d_test_hi=hi,
                                n_test_below_random=int((x < 0).sum()),
                                mean_d_gap=y.mean() if len(y) else np.nan,
                                d_gap_lo=glo, d_gap_hi=ghi,
                                n_gap_above_random=int((y > 0).sum())))
    return pd.DataFrame(out)


def leak_gap_correlations(ok):
    per = (ok.dropna(subset=["leakage", "gen_gap"])
             .groupby(["dataset", "method", "metric_name"])[["leakage", "gen_gap"]]
             .mean().reset_index())
    res = {"pooled_all": np.corrcoef(per["leakage"], per["gen_gap"])[0, 1], "n": len(per)}
    c = per.copy()
    c[["leakage", "gen_gap"]] -= c.groupby("dataset")[["leakage", "gen_gap"]].transform("mean")
    res["within_dataset"] = np.corrcoef(c["leakage"], c["gen_gap"])[0, 1]
    nonr = c[c["method"] != RANDOM]
    nonr = nonr.copy()
    nonr[["leakage", "gen_gap"]] -= nonr.groupby("dataset")[["leakage", "gen_gap"]].transform("mean")
    res["within_dataset_excl_random"] = np.corrcoef(nonr["leakage"], nonr["gen_gap"])[0, 1]
    for mt, s in per.groupby("metric_name"):
        if len(s) > 3:
            res[f"pooled_{mt}"] = np.corrcoef(s["leakage"], s["gen_gap"])[0, 1]
    return res


# ── 4. label shift / train size from recoverable assignments ────────────────

def _legacy_scaffold(smiles, spec):
    """The pre-fix scaffold split (shared '' / '_no_scaffold' buckets), to reproduce saved runs."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    groups = {}
    for eid, s in smiles.items():
        mol, key = Chem.MolFromSmiles(str(s)), "_no_scaffold"
        if mol is not None:
            try:
                key = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)))
            except Exception:
                key = "_no_scaffold"
        groups.setdefault(key, []).append(eid)
    groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    total = sum(len(e) for _, e in groups)
    tgt = {n: total * s / sum(spec.splits) for n, s in zip(spec.names, spec.splits)}
    cur = {n: 0 for n in spec.names}
    asg = {}
    for _k, eids in groups:
        best = max(spec.names, key=lambda n: tgt[n] - cur[n])
        for e in eids:
            asg[e] = best
        cur[best] += len(eids)
    return asg


def _label_stats(y, asg, ids, task):
    tr = np.array([y[i] for i in ids if asg.get(i) == "train"], float)
    te = np.array([y[i] for i in ids if asg.get(i) == "test"], float)
    tr, te = tr[np.isfinite(tr)], te[np.isfinite(te)]
    d = dict(n_train_lab=len(tr), n_test_lab=len(te))
    if task == "classification":
        d.update(train_pos_rate=tr.mean() if len(tr) else np.nan,
                 test_pos_rate=te.mean() if len(te) else np.nan)
    else:
        d.update(train_mean=tr.mean() if len(tr) else np.nan, test_mean=te.mean() if len(te) else np.nan,
                 train_std=tr.std() if len(tr) else np.nan, test_std=te.std() if len(te) else np.nan)
        pooled = np.concatenate([tr, te]).std() if len(tr) + len(te) else np.nan
        d["test_mean_shift_sd"] = (d["test_mean"] - d["train_mean"]) / pooled if pooled else np.nan
    return d


def _saved_summary():
    p = os.path.join(os.path.dirname(SAVED_ASG), "summary.csv")
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p)
    s["leakage"] = pd.to_numeric(s["leakage"], errors="coerce")
    return {(r.dataset, r.method): r.leakage for r in s.itertuples()}


def label_shift(ok, reconstruct=False, limit=10_000):
    from PALM.data.sources import load_dataset
    from PALM.splitters import SplitSpec, split
    rows = []
    for ds in sorted(ok["dataset"].unique()):
        mrows = ok[(ok["dataset"] == ds) & ok["n_train_lab"].notna()]
        if mrows.empty:
            continue
        b = load_dataset(ds, limit=limit)
        if not b.available or not b.targets:
            continue
        ids = list(b.feature_data.keys())
        y, task = b.targets, b.task_type
        cands = []                          # (method, seed, source, assignment, its leakage)
        saved_summary = _saved_summary()
        key = type(ids[0])
        for (meth, seed), mr in mrows.groupby(["method", "seed"]):
            seed = int(seed)
            for p in (os.path.join(BENCH_ASG, f"{ds}__{meth}__s{seed}.csv"),
                      os.path.join(BENCH_ASG, f"{ds}__{meth}.csv")):
                if os.path.exists(p):
                    a = pd.read_csv(p)
                    # written by the same run as the master row -> its own leakage is
                    # the row's; counts are still re-checked below
                    cands.append((meth, seed, "benchmark", {key(i): s for i, s in
                                                           zip(a["id"], a["split"])},
                                  float(mr["leakage"].iloc[0])))
                    break
        for meth in mrows["method"].unique():
            p = os.path.join(SAVED_ASG, f"{ds}__{meth}.csv")
            if os.path.exists(p):
                a = pd.read_csv(p)
                cands.append((meth, 0, "saved", {key(i): s for i, s in zip(a["id"], a["split"])},
                              saved_summary.get((ds, meth), np.nan)))
        if reconstruct:
            spec0 = SplitSpec([8, 2], ["train", "test"], seed=0)
            for seed in sorted(mrows["seed"].dropna().unique().astype(int)):
                r = split(RANDOM, b.feature_data, SplitSpec([8, 2], ["train", "test"], seed=seed))
                cands.append((RANDOM, seed, "recomputed", r.assignment, r.diagnostics.get("leakage")))
            if b.smiles and "scaffold" in set(mrows["method"]):
                cands.append(("scaffold", 0, "recomputed_legacy",
                              _legacy_scaffold(b.smiles, spec0), np.nan))
            for meth in ("hypergraph", "graph"):
                if meth in set(mrows["method"]):
                    try:
                        r = split(meth, b.feature_data, spec0, preset="deterministic")
                        cands.append((meth, 0, "recomputed", r.assignment, r.diagnostics.get("leakage")))
                    except Exception as exc:          # noqa: BLE001
                        print(f"  [{ds}] {meth} reconstruct failed: {exc}")
        seen = set()
        for meth, seed, src, asg, leak in cands:
            if (meth, seed) in seen:
                continue
            mr = mrows[(mrows["method"] == meth) & (mrows["seed"] == seed)]
            if mr.empty:
                continue
            st = _label_stats(y, asg, ids, task)
            mr = mr.iloc[0]
            counts_ok = (st["n_train_lab"] == mr["n_train_lab"] and st["n_test_lab"] == mr["n_test_lab"]
                         and (set(asg) == set(ids) or (meth == "scaffold" and set(asg) <= set(ids))))
            leak_known = leak is not None and np.isfinite(leak) and np.isfinite(mr["leakage"])
            leak_ok = leak_known and abs(float(leak) - mr["leakage"]) < 1e-5
            # counts alone are weak (any same-size partition matches); require the
            # leakage fingerprint whenever the master row has one
            verified = counts_ok and (leak_ok or not np.isfinite(mr["leakage"]))
            if not verified:
                print(f"  [{ds}] {meth} s{seed} ({src}) does NOT match master row "
                      f"(train {st['n_train_lab']} vs {mr['n_train_lab']}, "
                      f"test {st['n_test_lab']} vs {mr['n_test_lab']}) — dropped")
                continue
            seen.add((meth, seed))
            rows.append(dict(dataset=ds, task_type=task, method=meth, seed=seed, source=src,
                             verified_by=("saved by run_benchmark + counts" if src == "benchmark" else "counts+leakage" if leak_ok else "counts only"),
                             test_metric=mr["test_metric"], gen_gap=mr["gen_gap"], **st))
    return pd.DataFrame(rows)


# ── report ───────────────────────────────────────────────────────────────────

def _fmt(v, p=3):
    return "-" if not np.isfinite(v) else f"{v:+.{p}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconstruct", action="store_true",
                    help="re-partition random/scaffold/hypergraph/graph (CPU, no training) for label-shift")
    ap.add_argument("--skip-label-shift", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    ok = load_master()

    per, _pairs = paired_effects(ok)
    per.to_csv(os.path.join(RESULTS, "paired_effects_per_dataset.csv"), index=False)
    summ = coverage_matched(per, rng)
    summ.to_csv(os.path.join(RESULTS, "paired_effects_summary.csv"), index=False)
    corr = leak_gap_correlations(ok)

    print("## Per-dataset paired effect vs random (test_metric: method - random, 95% t-CI over seeds)\n")
    piv = per.assign(cell=lambda d: d.apply(
        lambda r: f"{r.d_test:+.3f} [{r.d_test_lo:+.3f},{r.d_test_hi:+.3f}]", axis=1))
    print(piv.pivot(index=["dataset", "metric"], columns="method", values="cell")
             .fillna("n/a").to_markdown())
    print("\n## Coverage-matched aggregates (bootstrap-over-datasets 95% CI)\n")
    cols = ["compare_set", "metric", "method", "n_datasets", "mean_d_test", "d_test_lo",
            "d_test_hi", "n_test_below_random", "mean_d_gap", "d_gap_lo", "d_gap_hi",
            "n_gap_above_random"]
    print(summ[cols].to_markdown(index=False, floatfmt="+.3f"))
    print("\n## Leakage-gap correlation\n")
    for k, v in corr.items():
        print(f"- {k}: {v:.3f}" if k != "n" else f"- n points: {v}")

    if not args.skip_label_shift:
        ls = label_shift(ok, reconstruct=args.reconstruct)
        ls.to_csv(os.path.join(RESULTS, "label_shift.csv"), index=False)
        if not ls.empty:
            print("\n## Label shift / train size (verified assignments only)\n")
            print(ls.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
