"""Feature-space sweep: which representation gives the cleanest *meaningful* split?

For every dataset that has raw identifiers + a target, this featurizes it with
each candidate representation for its entity type (molecules: ecfp1024 / maccs /
rdkit_descriptors / chemberta; materials: magpie / mat2vec / matminer), then runs
a representative splitter set — ``random`` (the predictive-validity gate),
``hypergraph`` and ``lowrank`` (the OOD partitioners) — across triplicate seeds,
recording leakage and the generalization gap for each.

Its output, ``results/feature_sweep.csv``, is reduced by
``derive_heuristics.py`` into ``data/feature_heuristics.json``: per dataset /
entity type, the feature space with the lowest OOD leakage *among those where a
model on a random split still learns* (test metric above a floor). That guards
against the trap where a representation looks "clean" only because it makes
everything mutually dissimilar and destroys the signal.

One GPU only, and NOT the one the split-method sweep is using:

    CUDA_VISIBLE_DEVICES=<free gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.feature_sweep \
        --seeds 0 1 2 --limit 3000
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
from PALM.data.routing import FEATURE_CANDIDATES, apply_featurizer
from .model_eval import evaluate_gap

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RESULTS, "feature_sweep.csv")

# random = the gate (its test metric says whether the space is learnable at all);
# hypergraph + lowrank = the OOD partitioners whose leakage we want to minimize.
METHODS = [("random", dict()), ("hypergraph", dict(preset="deterministic")),
           ("lowrank", dict())]
STOCHASTIC = {"random", "lowrank"}

_KIND_TO_TYPE = {"smiles": "molecule", "formula": "material"}

FIELDS = ["dataset", "category", "entity_type", "feature_set", "dim", "n",
          "task_type", "method", "seed", "status", "leakage", "imbalance",
          "test_metric", "train_metric", "gen_gap", "runtime_s", "reason"]


def _row(**kw):
    r = {k: "" for k in FIELDS}
    r.update(kw)
    return r


def run(datasets, seeds, limit, split_geom, names):
    rows = []
    for name in datasets:
        try:
            b = load_dataset(name, limit=limit)
        except Exception as exc:
            rows.append(_row(dataset=name, method="-", status="error",
                             reason=f"load {type(exc).__name__}: {exc}"))
            continue
        if not b.available or not b.identifiers or b.targets is None:
            reason = (b.reason if not b.available
                      else "no raw identifiers" if not b.identifiers else "no target")
            rows.append(_row(dataset=name, category=getattr(b, "category", ""),
                             method="-", status="skipped", reason=reason))
            print(f"[{name}] skip: {reason[:70]}")
            continue

        etype = _KIND_TO_TYPE.get(b.identifier_kind)
        if etype not in FEATURE_CANDIDATES:
            rows.append(_row(dataset=name, category=b.category, method="-",
                             status="skipped", reason=f"no candidates for kind={b.identifier_kind}"))
            continue

        print(f"[{name}] {b.category} type={etype} n={len(b.identifiers)} "
              f"candidates={FEATURE_CANDIDATES[etype]}")
        for fs in FEATURE_CANDIDATES[etype]:
            # featurize once per representation
            try:
                t0 = time.time()
                Xmap = apply_featurizer(etype, fs, b.identifiers)
                feat_s = time.time() - t0
            except Exception as exc:
                rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                                 feature_set=fs, method="-", status="error",
                                 reason=f"featurize {type(exc).__name__}: {exc}"))
                print(f"    {fs}: FEATURIZE ERROR {type(exc).__name__}: {exc}")
                continue
            ids = [i for i in Xmap if i in b.targets]
            if len(ids) < 30:
                rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                                 feature_set=fs, method="-", status="skipped",
                                 reason=f"too few featurized+labeled ({len(ids)})"))
                continue
            X = np.stack([Xmap[i] for i in ids])
            y = np.array([b.targets.get(i, np.nan) for i in ids], dtype=np.float64)
            data = {i: Xmap[i] for i in ids}
            pos = {i: p for p, i in enumerate(ids)}
            dim = int(X.shape[1])
            print(f"    {fs:16s} dim={dim} featurize={feat_s:.1f}s")

            for method, params in METHODS:
                cached = None
                for seed in seeds:
                    spec = SplitSpec(splits=split_geom, names=names, seed=seed)
                    try:
                        if method in STOCHASTIC or cached is None:
                            res = split(method, data, spec, **params)
                            cached = (res.assignment, res.diagnostics)
                        asg, d = cached
                        tr = [pos[i] for i in ids if asg.get(i) == "train"]
                        te = [pos[i] for i in ids if asg.get(i) == "test"]
                        g = evaluate_gap(X, y, b.task_type, tr, te, seed=seed)
                        rows.append(_row(
                            dataset=name, category=b.category, entity_type=etype,
                            feature_set=fs, dim=dim, n=len(ids), task_type=b.task_type,
                            method=method, seed=seed, status="ok",
                            leakage=d.get("leakage", ""), imbalance=d.get("imbalance", ""),
                            test_metric=g["test_metric"], train_metric=g["train_metric"],
                            gen_gap=g["gen_gap"], runtime_s=d.get("runtime_s", ""),
                            reason=g.get("gap_reason", "")))
                    except Exception as exc:
                        rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                                         feature_set=fs, dim=dim, method=method, seed=seed,
                                         status="error", reason=f"{type(exc).__name__}: {exc}"))
                        traceback.print_exc()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    ap.add_argument("--names", nargs="+", default=["train", "test"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = run(args.datasets, args.seeds, args.limit, args.splits, args.names)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n== {ok}/{len(rows)} ok -> {args.out}")


if __name__ == "__main__":
    main()
