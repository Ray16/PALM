"""Master benchmark driver: every dataset x every splitter x every seed.

For each dataset in ``PALM.data.sources.REGISTRY`` this runs the applicable
splitters (1-D or n-D) over N seeds, and for each run records both the split's
*quality* (leakage, imbalance, realized test fraction, runtime) and — when the
dataset carries a target — the *generalization gap* from a fixed RandomForest
trained on the split (see ``model_eval``). Every row lands in one long-format
table, ``benchmarks/results/master_benchmark.csv``.

Unavailable datasets / skipped methods / errors are recorded as rows with a
``reason`` (never dropped silently). One GPU only (one-job-per-GPU rule):

    CUDA_VISIBLE_DEVICES=<free gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.run_benchmark \
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

# 1-D methods + fixed, reproducible params. "random" is the baseline everything
# else should beat on leakage (and match on gap, being in-distribution).
METHODS_1D = [
    ("random",     dict()),
    ("hypergraph", dict(preset="deterministic")),
    ("graph",      dict(preset="deterministic")),
    ("lowrank",    dict()),
    ("datasail",   dict(max_sec=120)),
    ("scaffold",   dict()),                     # only when SMILES are available
]
METHODS_ND = [
    ("random",            dict()),
    ("hypergraph_nd",     dict(preset="deterministic", sim_threshold=0.6)),
    ("hypergraph_nd_knn", dict(preset="deterministic", k=25)),
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

_EXTRA_KEYS = ("km1", "cut", "rank", "fm_moves", "n_hyperedges", "technique", "n_scaffolds")


def _row(**kw):
    r = {k: "" for k in FIELDS}
    r.update(kw)
    return r


def _extra(diag):
    return ";".join(f"{k}={diag[k]}" for k in _EXTRA_KEYS if k in diag)


def _prep_1d(bundle):
    """Ordered ids + aligned X and y (y is NaN where a label is missing)."""
    ids = list(bundle.feature_data.keys())
    X = np.stack([np.asarray(bundle.feature_data[i], dtype=np.float32) for i in ids])
    y = None
    if bundle.targets:
        y = np.array([bundle.targets.get(i, np.nan) for i in ids], dtype=np.float64)
    pos_of_id = {i: p for p, i in enumerate(ids)}
    return ids, X, y, pos_of_id


def run(datasets, seeds, limit, split_geom, names, route=False):
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
        tt = bundle.task_type
        print(f"[{name}] {cat} {bundle.kind} n={n} task={tt} "
              f"seeds={list(seeds)} methods={len(methods)}")

        # method-outer so a deterministic split is computed once and reused; the
        # generalization-gap model still re-runs per seed.
        for method, params in methods:
            if method == "scaffold":
                if not bundle.smiles:
                    continue
                data_m = bundle.smiles
            elif method == "datasail" and n > DATASAIL_MAX_N:
                rows.append(_row(dataset=name, category=cat, task_type=tt, kind=bundle.kind,
                                 n=n, method=method, seed=seeds[0], status="skipped",
                                 reason=f"n={n} > DATASAIL_MAX_N={DATASAIL_MAX_N}"))
                continue
            else:
                data_m = data

            cached = None                       # (assignment, diagnostics) for deterministic reuse
            for seed in seeds:
                spec = SplitSpec(splits=split_geom, names=names, seed=seed)
                try:
                    if method in STOCHASTIC or cached is None:
                        res = split(method, data_m, spec, **params)
                        cached = (res.assignment, res.diagnostics)
                    asg, d = cached
                    row = _row(
                        dataset=name, category=cat, task_type=tt, kind=bundle.kind, n=n,
                        method=method, seed=seed, status="ok",
                        leakage=d.get("leakage", ""), imbalance=d.get("imbalance", ""),
                        test_fraction=(d.get("split_fractions", {}) or {}).get("test", ""),
                        runtime_s=d.get("runtime_s", ""), extra=_extra(d),
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
                            row["reason"] = g["gap_reason"]
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])  # triplicate
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    ap.add_argument("--names", nargs="+", default=["train", "test"])
    ap.add_argument("--out", default=MASTER)
    ap.add_argument("--route", action="store_true",
                    help="featurize each dataset with its hand-picked default (PALM.data.routing)")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = run(args.datasets, args.seeds, args.limit, args.splits, args.names, route=args.route)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    gaps = sum(1 for r in rows if r["gen_gap"] != "")
    print(f"\n== {ok}/{len(rows)} runs ok; {gaps} with a generalization gap -> {args.out}")


if __name__ == "__main__":
    main()
