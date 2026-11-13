"""Wire every splitting method to every configured dataset.

For each available dataset in ``PALM.data.sources.REGISTRY`` this runs the
applicable splitters via the ``PALM.splitters`` registry, writes each assignment
to ``data/split_results/assignments/<dataset>__<method>.csv`` (git-ignored) and
appends a diagnostics row (leakage, imbalance, runtime, test fraction) to
``data/split_results/summary.csv`` (committed — the results matrix).

Datasets that are unavailable (missing data / credentials) are recorded with the
reason, not skipped silently. ``--methods`` runs a subset and MERGES into the
existing ``summary.csv`` (rows for the rerun (dataset, method) pairs replaced,
all others kept — e.g. slow DataSAIL rows or lowrank rows produced by the
sibling ``low_rank`` repo). One GPU only (see the one-job-per-GPU rule):

    CUDA_VISIBLE_DEVICES=<free gpu> python -m PALM.data.run_splits --limit 10000

Env: the dedicated ``palm`` conda env (torch + mtkahypar + rdkit + datasail).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback

from PALM.splitters import SplitSpec, split
from PALM.data.sources import REGISTRY, load_dataset

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "split_results")
ASG = os.path.join(OUT, "assignments")

# 1-D methods and the fixed params that make them reproducible / bounded.
# "random" is the baseline every method should beat.
METHODS_1D = [
    ("random", dict()),
    ("hypergraph", dict(preset="deterministic")),
    ("graph", dict(preset="deterministic")),
    ("lowrank", dict()),
    ("datasail", dict(max_sec=120)),
    ("scaffold", dict()),                       # only when SMILES are available
]
METHODS_ND = [
    ("random", dict()),
    ("hypergraph_nd", dict(preset="deterministic", sim_threshold=0.6)),
    ("hypergraph_nd_knn", dict(preset="deterministic", k=25)),
]

# DataSAIL's C1e solve is O(n^2) clustering + an ILP; cap it by size.
DATASAIL_MAX_N = 3000


def _write_assignment(dataset, method, assignment):
    os.makedirs(ASG, exist_ok=True)
    path = os.path.join(ASG, f"{dataset}__{method}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split"])
        for k, v in assignment.items():
            w.writerow([k, v])
    return path


_EXTRA_KEYS = ("km1", "cut", "rank", "fm_moves", "n_hyperedges", "technique", "n_scaffolds",
               "within_corridor", "balance_repair_moves", "balance_ok", "giant_group",
               "max_fraction_deviation", "n_acyclic", "n_invalid", "invalid_policy")


def _split(method, data_m, spec, params):
    """``split`` with the scaffold invalid-SMILES fallback (recorded, never silent)."""
    try:
        return split(method, data_m, spec, **params)
    except ValueError as exc:
        if method != "scaffold" or "could not be parsed" not in str(exc):
            raise
        res = split(method, data_m, spec, **{**params, "invalid": "singleton"})
        res.diagnostics["invalid_policy"] = "singleton"
        return res


def _row(dataset, category, method, n, status, diag=None, reason=""):
    diag = diag or {}
    return {
        "dataset": dataset, "category": category, "method": method, "n": n,
        "status": status,
        "leakage": diag.get("leakage", ""),
        "imbalance": diag.get("imbalance", ""),
        "test_fraction": (diag.get("split_fractions", {}) or {}).get("test", ""),
        "runtime_s": diag.get("runtime_s", ""),
        "extra": ";".join(f"{k}={diag[k]}" for k in _EXTRA_KEYS if k in diag),
        "reason": reason,
    }


def run(datasets, limit, spec, methods_filter=None):
    rows = []
    for name in datasets:
        bundle = load_dataset(name, limit=limit)
        cat = bundle.category
        if not bundle.available:
            rows.append(_row(name, cat, "-", 0, "dataset_unavailable", reason=bundle.reason))
            print(f"[{name}] UNAVAILABLE: {bundle.reason[:80]}")
            continue

        if bundle.kind == "nd":
            n = len(bundle.records)
            data = (bundle.records, bundle.axis_feature_maps)
            methods = METHODS_ND
        else:
            n = len(bundle.feature_data)
            data = bundle.feature_data
            methods = METHODS_1D
        if methods_filter is not None:
            methods = [(m, p) for m, p in methods if m in methods_filter]
        print(f"[{name}] {cat} {bundle.kind} n={n} — running {len(methods)} methods")

        for method, params in methods:
            if method == "scaffold":
                if not bundle.smiles:
                    continue
                data_m = bundle.smiles
            elif method == "datasail" and n > DATASAIL_MAX_N:
                rows.append(_row(name, cat, method, n, "skipped",
                                 reason=f"n={n} > DATASAIL_MAX_N={DATASAIL_MAX_N} (too slow)"))
                print(f"    {method}: skipped (n>{DATASAIL_MAX_N})")
                continue
            else:
                data_m = data
            try:
                t0 = time.time()
                res = _split(method, data_m, spec, params)
                _write_assignment(name, method, res.assignment)
                why = (f"scaffold: unparseable SMILES given singleton groups "
                       f"(n_invalid={res.diagnostics.get('n_invalid')})"
                       if res.diagnostics.get("invalid_policy") == "singleton" else "")
                rows.append(_row(name, cat, method, n, "ok", res.diagnostics, reason=why))
                d = res.diagnostics
                print(f"    {method:16s} leakage={d.get('leakage')} imbalance={d.get('imbalance')} "
                      f"test={d.get('split_fractions',{}).get('test')} {d.get('runtime_s')}s")
            except Exception as exc:
                rows.append(_row(name, cat, method, n, "error", reason=f"{type(exc).__name__}: {exc}"))
                print(f"    {method}: ERROR {type(exc).__name__}: {exc}")
                traceback.print_exc()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY))
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    ap.add_argument("--names", nargs="+", default=["train", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="run only these splitters; rows MERGE into summary.csv")
    ap.add_argument("--no-merge", action="store_true", help="overwrite summary.csv")
    args = ap.parse_args(argv)

    spec = SplitSpec(splits=args.splits, names=args.names, seed=args.seed)
    os.makedirs(OUT, exist_ok=True)
    mf = set(args.methods) if args.methods else None
    rows = run(args.datasets, args.limit, spec, methods_filter=mf)

    fields = ["dataset", "category", "method", "n", "status", "leakage",
              "imbalance", "test_fraction", "runtime_s", "extra", "reason"]
    summary = os.path.join(OUT, "summary.csv")
    out_rows = rows
    if not args.no_merge and os.path.exists(summary):
        with open(summary, newline="") as fh:
            old = list(csv.DictReader(fh))
        ds_run = set(args.datasets)
        new_keys = {(r["dataset"], r["method"]) for r in rows}
        kept = [r for r in old
                if (r["dataset"], r["method"]) not in new_keys
                and not (r["dataset"] in ds_run and (r["method"] == "-" or mf is None
                                                     or r["method"] in mf))]
        order = {}
        for r in old + rows:
            order.setdefault(r["dataset"], len(order))
        mord = {}
        for r in old + rows:
            mord.setdefault((r["dataset"], r["method"]), len(mord))
        out_rows = sorted(kept + rows, key=lambda r: (order[r["dataset"]],
                                                      mord[(r["dataset"], r["method"])]))
        print(f"merge: kept {len(kept)}/{len(old)} existing rows, added {len(rows)}")
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in out_rows])
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n== {ok}/{len(rows)} runs ok -> {summary}")


if __name__ == "__main__":
    main()
