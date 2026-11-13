"""Master benchmark driver: every dataset x every splitter x every seed.

For each dataset in ``PALM.data.sources.REGISTRY`` this runs the applicable
splitters (1-D or n-D) over N seeds, and for each run records both the split's
*quality* (leakage, imbalance, realized test fraction, runtime) and — when the
dataset carries a target — the *generalization gap* from a fixed RandomForest
trained on the split (see ``model_eval``). Every row lands in one long-format
table, ``benchmarks/results/master_benchmark.csv``.

Unavailable datasets / skipped methods / errors are recorded as rows with a
``reason`` (never dropped silently). Every split's assignment (``id,split``) is
saved under ``benchmarks/results/assignments/`` — ``<dataset>__<method>.csv``
for deterministic splitters (one partition reused across seeds) and
``<dataset>__<method>__s<seed>.csv`` for stochastic ones — so label-shift and
other post-hoc analyses never need re-partitioning.

``--methods`` runs a subset and MERGES into the existing ``--out`` table: rows
for the rerun (dataset, method) pairs are replaced, all others kept (e.g. the
slow DataSAIL rows, or rows from splitters run elsewhere such as lowrank).
``--no-merge`` overwrites instead. One GPU only (one-job-per-GPU rule):

    CUDA_VISIBLE_DEVICES=<free gpu> LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
    python -m PALM.benchmarks.master.run_benchmark \
        --seeds 0 1 2 3 4 --limit 10000

Env: the dedicated ``palm`` conda env (torch + mtkahypar + rdkit + datasail + sklearn).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback

import numpy as np

from PALM.splitters import SplitSpec, split
from PALM.data.sources import REGISTRY, load_dataset
from .model_eval import evaluate_gap

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
MASTER = os.path.join(RESULTS, "master_benchmark.csv")
ASSIGNMENTS = os.path.join(RESULTS, "assignments")

# 1-D methods + fixed, reproducible params. "random" is the baseline everything
# else should beat on leakage (and match on gap, being in-distribution).
METHODS_1D = [
    ("random",     dict()),
    ("hypergraph", dict(preset="deterministic")),
    ("graph",      dict(preset="deterministic")),
    ("lowrank",    dict()),
    ("datasail",   dict(max_sec=120)),
    ("scaffold",   dict()),                     # SMILES only
    ("astartes",   dict()),                     # external baseline (feature-based)
    ("lohi",       dict()),                     # external baseline (SMILES only, ILP)
]
SMILES_METHODS = {"scaffold", "lohi"}           # need bundle.smiles
ILP_METHODS = {"datasail", "lohi"}              # O(n^2) / ILP -> cap by size
METHODS_ND = [
    ("random",            dict()),
    ("hypergraph_nd",     dict(preset="deterministic", sim_threshold=0.6)),
    ("hypergraph_nd_knn", dict(preset="deterministic", k=25)),
    ("lowrank_nd",        dict()),
]

DATASAIL_MAX_N = 3000                            # C1e is O(n^2) clustering + ILP

# Only these two vary with the seed; every other splitter is deterministic, so
# its split is computed once and reused across seeds (the RandomForest still
# re-runs per seed, giving model-variance error bars without re-partitioning).
STOCHASTIC = {"random", "lowrank"}

FIELDS = [
    "dataset", "category", "task_type", "kind", "n", "method", "seed", "status",
    "leakage", "imbalance", "test_fraction", "runtime_s",
    "model", "metric_name", "train_metric", "test_metric", "gen_gap",
    "n_train_lab", "n_test_lab", "extra", "reason",
]

_EXTRA_KEYS = ("km1", "cut", "rank", "fm_moves", "n_hyperedges", "technique", "n_scaffolds",
               # balance diagnostics (hypergraph/graph corridor repair; scaffold tolerance check)
               "within_corridor", "balance_repair_moves",
               "balance_ok", "giant_group", "max_fraction_deviation", "largest_group_frac",
               "n_acyclic", "n_invalid", "invalid_policy", "feat_leakage")


def _row(**kw):
    r = {k: "" for k in FIELDS}
    r.update(kw)
    return r


def _extra(diag):
    return ";".join(f"{k}={diag[k]}" for k in _EXTRA_KEYS if k in diag)


def assignment_path(dataset, method, seed=None, root=ASSIGNMENTS):
    """Where ``run`` saves a split: per-seed file only for stochastic methods."""
    tag = f"__s{seed}" if method in STOCHASTIC and seed is not None else ""
    return os.path.join(root, f"{dataset}__{method}{tag}.csv")


def _save_assignment(path, assignment):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split"])
        for k, v in assignment.items():
            w.writerow([k, v])


def _run_split(method, data_m, spec, params):
    """``split`` with the scaffold invalid-SMILES fallback.

    The scaffold splitter raises on unparseable SMILES by default; the benchmark
    retries with ``invalid="singleton"`` (each invalid entry its own group) and
    records the policy in the diagnostics (-> ``extra``) so it is never silent.
    """
    try:
        return split(method, data_m, spec, **params)
    except ValueError as exc:
        if method != "scaffold" or "could not be parsed" not in str(exc):
            raise
        res = split(method, data_m, spec, **{**params, "invalid": "singleton"})
        res.diagnostics["invalid_policy"] = "singleton"
        return res


def _feat_leakage(X, ids, asg):
    """L(pi) on the dataset features for splitters that don't report one (scaffold).

    Computed over the assigned entities only (scaffold covers just the ids with
    SMILES, e.g. a subset of qmof).
    """
    from PALM.splitters.common.feature_preparation import choose_metric
    from PALM.splitters.common.leakage_metrics import scaled_lpi
    rows = [p for p, i in enumerate(ids) if i in asg]
    Xs = X[rows]
    labels = [asg[ids[p]] for p in rows]
    return round(scaled_lpi(Xs, labels, metric=choose_metric(Xs)), 6)


def _prep_1d(bundle):
    """Ordered ids + aligned X and y (y is NaN where a label is missing)."""
    ids = list(bundle.feature_data.keys())
    X = np.stack([np.asarray(bundle.feature_data[i], dtype=np.float32) for i in ids])
    y = None
    if bundle.targets:
        y = np.array([bundle.targets.get(i, np.nan) for i in ids], dtype=np.float64)
    pos_of_id = {i: p for p, i in enumerate(ids)}
    return ids, X, y, pos_of_id


def run(datasets, seeds, limit, split_geom, names, route=False, methods_filter=None,
        save_dir=ASSIGNMENTS):
    rows = []
    for name in datasets:
        try:
            bundle = load_dataset(name, limit=limit, route=route)
        except Exception as exc:                                    # loader blew up
            rows.append(_row(dataset=name, method="-", status="error",
                             reason=f"load {type(exc).__name__}: {exc}"))
            print(f"[{name}] LOAD ERROR: {exc}")
            continue

        cat = bundle.category
        if not bundle.available:
            rows.append(_row(dataset=name, category=cat, method="-", status="dataset_unavailable",
                             reason=bundle.reason))
            print(f"[{name}] UNAVAILABLE: {bundle.reason[:80]}")
            continue

        if bundle.kind == "nd":
            n = len(bundle.records)
            data = (bundle.records, bundle.axis_feature_maps)
            methods, ids, X, y, pos_of_id = METHODS_ND, None, None, None, None
        else:
            ids, X, y, pos_of_id = _prep_1d(bundle)
            n = len(ids)
            data = bundle.feature_data
            methods = METHODS_1D
        if methods_filter is not None:
            methods = [(m, p) for m, p in methods if m in methods_filter]
        tt = bundle.task_type
        print(f"[{name}] {cat} {bundle.kind} n={n} task={tt} "
              f"seeds={list(seeds)} methods={len(methods)}")

        # method-outer so a deterministic split is computed once and reused; the
        # generalization-gap model still re-runs per seed.
        for method, params in methods:
            if method in SMILES_METHODS:
                if not bundle.smiles:
                    continue
                data_m = bundle.smiles
            else:
                data_m = data
            if method in ILP_METHODS and n > DATASAIL_MAX_N:
                rows.append(_row(dataset=name, category=cat, task_type=tt, kind=bundle.kind,
                                 n=n, method=method, seed=seeds[0], status="skipped",
                                 reason=f"n={n} > DATASAIL_MAX_N={DATASAIL_MAX_N} (ILP too slow)"))
                continue

            cached = None                       # (assignment, diagnostics) for deterministic reuse
            for seed in seeds:
                spec = SplitSpec(splits=split_geom, names=names, seed=seed)
                try:
                    if method in STOCHASTIC or cached is None:
                        res = _run_split(method, data_m, spec, params)
                        d = dict(res.diagnostics)
                        if (d.get("leakage") in (None, "") and bundle.kind == "1d"
                                and method == "scaffold"):
                            d["feat_leakage"] = _feat_leakage(X, ids, res.assignment)
                        if save_dir:
                            _save_assignment(assignment_path(name, method, seed, save_dir),
                                             res.assignment)
                        cached = (res.assignment, d)
                    asg, d = cached
                    row = _row(
                        dataset=name, category=cat, task_type=tt, kind=bundle.kind, n=n,
                        method=method, seed=seed, status="ok",
                        leakage=d.get("leakage", ""), imbalance=d.get("imbalance", ""),
                        test_fraction=(d.get("split_fractions", {}) or {}).get("test", ""),
                        runtime_s=d.get("runtime_s", ""), extra=_extra(d),
                        reason=("scaffold: unparseable SMILES given singleton groups "
                                f"(n_invalid={d.get('n_invalid')})"
                                if d.get("invalid_policy") == "singleton" else ""),
                    )
                    # generalization gap (1-D with a target only)
                    if bundle.kind == "1d" and y is not None:
                        tr = [pos_of_id[i] for i in ids if asg.get(i) == "train"]
                        te = [pos_of_id[i] for i in ids if asg.get(i) == "test"]
                        g = evaluate_gap(X, y, tt, tr, te, seed=seed)
                        row.update({k: g[k] for k in
                                    ("model", "metric_name", "train_metric", "test_metric",
                                     "gen_gap", "n_train_lab", "n_test_lab")})
                        if g.get("gap_reason"):
                            row["reason"] = "; ".join(x for x in (row["reason"], g["gap_reason"]) if x)
                    rows.append(row)
                    print(f"  s{seed} {method:16s} leak={row['leakage']} "
                          f"test_metric={row['test_metric']} gap={row['gen_gap']}")
                except Exception as exc:                            # noqa: BLE001
                    rows.append(_row(dataset=name, category=cat, task_type=tt, kind=bundle.kind,
                                     n=n, method=method, seed=seed, status="error",
                                     reason=f"{type(exc).__name__}: {exc}"))
                    print(f"  s{seed} {method}: ERROR {type(exc).__name__}: {exc}")
                    traceback.print_exc()
    return rows


def merge_rows(path, new_rows, datasets, methods_filter):
    """Existing table minus the rerun (dataset, method) pairs, plus ``new_rows``.

    A pair counts as rerun if it appears in ``new_rows`` or if its dataset was in
    this run and its method was requested (so a method that is now skipped for a
    dataset doesn't leave a stale row). Dataset-level ``-`` rows (load errors /
    unavailable) of a rerun dataset are replaced too. Row order: existing dataset
    order, then new datasets; within a dataset, the existing method order.
    """
    with open(path, newline="") as fh:
        old = list(csv.DictReader(fh))
    ds_run = set(datasets)
    rerun = {(r["dataset"], r["method"]) for r in new_rows}
    rerun |= {(d, "-") for d in ds_run}
    if methods_filter is None:          # full run of these datasets: replace them wholesale
        kept = [r for r in old if r["dataset"] not in ds_run]
    else:
        kept = [r for r in old if (r["dataset"], r["method"]) not in rerun
                and not (r["dataset"] in ds_run and r["method"] in methods_filter)]
    ds_order, meth_order = {}, {}
    for r in old + new_rows:
        ds_order.setdefault(r["dataset"], len(ds_order))
        meth_order.setdefault((r["dataset"], r["method"]), len(meth_order))
    merged = [{k: r.get(k, "") for k in FIELDS} for r in kept + new_rows]
    merged.sort(key=lambda r: (ds_order[r["dataset"]], meth_order[(r["dataset"], r["method"])],
                               str(r["seed"])))
    print(f"merge: kept {len(kept)}/{len(old)} existing rows, added {len(new_rows)}")
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])  # triplicate
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    ap.add_argument("--names", nargs="+", default=["train", "test"])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="run only these splitters; results MERGE into --out (rerun "
                         "(dataset, method) pairs replaced, all other rows kept)")
    ap.add_argument("--no-merge", action="store_true",
                    help="overwrite --out instead of merging into it")
    ap.add_argument("--assignments-dir", default=ASSIGNMENTS,
                    help="where per-split id,split CSVs are saved ('' to disable)")
    ap.add_argument("--out", default=MASTER)
    ap.add_argument("--route", action="store_true",
                    help="featurize each dataset with its hand-picked default (PALM.data.routing)")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mf = set(args.methods) if args.methods else None
    rows = run(args.datasets, args.seeds, args.limit, args.splits, args.names, route=args.route,
               methods_filter=mf, save_dir=args.assignments_dir)

    out_rows = rows
    if not args.no_merge and os.path.exists(args.out):
        out_rows = merge_rows(args.out, rows, args.datasets, mf)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    gaps = sum(1 for r in rows if r["gen_gap"] != "")
    print(f"\n== {ok}/{len(rows)} runs ok; {gaps} with a generalization gap -> {args.out}")


if __name__ == "__main__":
    main()
