"""Feature-space sweep: which representation gives the cleanest *meaningful* split?

For every dataset that has raw identifiers + a target, this featurizes it with
each candidate representation for its entity type (molecules: ecfp1024 / maccs /
rdkit_descriptors / chemberta; materials: magpie / mat2vec), then runs a
representative splitter set — ``random`` (the predictive-validity gate),
``hypergraph`` and ``lowrank`` (the OOD partitioners) — across triplicate seeds.

Two design points that make the result trustworthy (vs. the naive first pass):

1. **Reference-space leakage.** Each candidate's *split* is scored not only in its
   own feature space (which would be circular — you can't compare 0.16 in
   maccs-space to 0.17 in ecfp-space) but also in a **fixed reference space**
   (ECFP for molecules, MAGPIE for materials, using the pipeline's own
   ``choose_metric``). ``ref_leakage`` is comparable across candidates and is the
   quantity ``derive_heuristics`` selects on.

2. **Independent reps.** Each seed draws a *different* row subsample from a shared
   pool, so even the deterministic splitters produce genuine seed-to-seed
   variance — which the significance margin in ``derive_heuristics`` needs to tell
   a real win from noise.

One GPU only, and NOT the one any other sweep is using:

    CUDA_VISIBLE_DEVICES=<free gpu> LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib \
    /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.benchmarks.master.feature_sweep \
        --seeds 0 1 2 --pool-limit 4000 --limit 2500
"""

from __future__ import annotations

import argparse
import csv
import os
import traceback

import numpy as np

from PALM.splitters import SplitSpec, split
from PALM.splitters.common.feature_preparation import choose_metric
from PALM.splitters.common.leakage_metrics import scaled_lpi
from PALM.data.sources import REGISTRY, load_dataset
from PALM.data.routing import FEATURE_CANDIDATES, FEATURE_DEFAULTS, apply_featurizer
from .model_eval import evaluate_gap

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RESULTS, "feature_sweep.csv")

# random = the gate (its test metric says whether the space is learnable at all);
# hypergraph + lowrank = the OOD partitioners whose leakage we want to minimize.
METHODS = [("random", dict()), ("hypergraph", dict(preset="deterministic")),
           ("lowrank", dict())]

_KIND_TO_TYPE = {"smiles": "molecule", "formula": "material",
                 "protein": "protein", "nucleotide": "gene"}


def _id_source(bundle, etype, feature_set):
    """Which identifier map to featurize a candidate from (MOF linker uses SMILES)."""
    if etype == "mof" and feature_set == "linker_ecfp":
        return bundle.meta.get("linker_smiles") or {}
    return bundle.identifiers

FIELDS = ["dataset", "category", "entity_type", "feature_set", "dim", "n",
          "task_type", "method", "seed", "status", "leakage", "ref_leakage",
          "ref_metric", "imbalance", "test_metric", "train_metric", "gen_gap",
          "runtime_s", "reason"]


def _row(**kw):
    r = {k: "" for k in FIELDS}
    r.update(kw)
    return r


def run(datasets, seeds, pool_limit, rep_limit, split_geom, names):
    rows = []
    for name in datasets:
        try:
            b = load_dataset(name, limit=pool_limit)
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
        etype = b.entity_type or _KIND_TO_TYPE.get(b.identifier_kind)
        if etype not in FEATURE_CANDIDATES:
            rows.append(_row(dataset=name, category=b.category, method="-",
                             status="skipped", reason=f"no candidates for entity_type={etype}"))
            continue

        cands = FEATURE_CANDIDATES[etype]
        ref_fs = FEATURE_DEFAULTS[etype]
        print(f"[{name}] {b.category} type={etype} pool={len(b.identifiers)} "
              f"candidates={cands} ref={ref_fs}")

        # featurize every candidate + the reference space ONCE on the pool
        feat_maps, errored = {}, {}
        for fs in cands:
            try:
                feat_maps[fs] = apply_featurizer(etype, fs, _id_source(b, etype, fs))
            except Exception as exc:
                errored[fs] = f"{type(exc).__name__}: {exc}"
                print(f"    {fs}: FEATURIZE ERROR {errored[fs]}")
        try:
            ref_map = feat_maps.get(ref_fs) or apply_featurizer(etype, ref_fs, _id_source(b, etype, ref_fs))
        except Exception as exc:
            rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                             method="-", status="error",
                             reason=f"reference featurize {type(exc).__name__}: {exc}"))
            continue
        for fs, msg in errored.items():
            rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                             feature_set=fs, method="-", status="error", reason=msg))
        if not feat_maps:
            continue

        # common pool: rows featurized by ALL candidates, in reference, and labeled
        common = set(ref_map) & set(b.targets)
        for fs in feat_maps:
            common &= set(feat_maps[fs])
        common = sorted(common, key=str)
        if len(common) < 60:
            rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                             method="-", status="skipped",
                             reason=f"common featurized+labeled pool too small ({len(common)})"))
            continue
        ref_metric = choose_metric(np.stack([ref_map[i] for i in common]))
        print(f"    common pool={len(common)} ref_metric={ref_metric} dims="
              f"{{{', '.join(f'{fs}:{len(next(iter(feat_maps[fs].values())))}' for fs in feat_maps)}}}")

        for seed in seeds:
            rng = np.random.default_rng(seed)
            m = min(rep_limit, len(common))
            sel = sorted(rng.choice(len(common), m, replace=False))
            ids = [common[k] for k in sel]                       # SAME rows for every candidate this seed
            pos = {i: p for p, i in enumerate(ids)}
            y = np.array([b.targets[i] for i in ids], dtype=np.float64)
            Xref = np.stack([ref_map[i] for i in ids])
            spec = SplitSpec(splits=split_geom, names=names, seed=seed)

            for fs in feat_maps:
                X = np.stack([feat_maps[fs][i] for i in ids])
                data = {i: feat_maps[fs][i] for i in ids}
                dim = int(X.shape[1])
                for method, params in METHODS:
                    try:
                        res = split(method, data, spec, **params)
                        asg, d = res.assignment, res.diagnostics
                        labels = [0 if asg.get(i) == "train" else 1 for i in ids]
                        ref_leak = round(scaled_lpi(Xref, labels, metric=ref_metric), 6)
                        tr = [pos[i] for i in ids if asg.get(i) == "train"]
                        te = [pos[i] for i in ids if asg.get(i) == "test"]
                        g = evaluate_gap(X, y, b.task_type, tr, te, seed=seed)
                        rows.append(_row(
                            dataset=name, category=b.category, entity_type=etype,
                            feature_set=fs, dim=dim, n=len(ids), task_type=b.task_type,
                            method=method, seed=seed, status="ok",
                            leakage=d.get("leakage", ""), ref_leakage=ref_leak,
                            ref_metric=ref_metric, imbalance=d.get("imbalance", ""),
                            test_metric=g["test_metric"], train_metric=g["train_metric"],
                            gen_gap=g["gen_gap"], runtime_s=d.get("runtime_s", ""),
                            reason=g.get("gap_reason", "")))
                    except Exception as exc:
                        rows.append(_row(dataset=name, category=b.category, entity_type=etype,
                                         feature_set=fs, dim=dim, method=method, seed=seed,
                                         status="error", reason=f"{type(exc).__name__}: {exc}"))
                        traceback.print_exc()
            print(f"    seed {seed}: n={m} done")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--pool-limit", type=int, default=4000, help="rows to featurize per dataset")
    ap.add_argument("--limit", type=int, default=2500, help="rows per seed (subsampled from pool)")
    ap.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    ap.add_argument("--names", nargs="+", default=["train", "test"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = run(args.datasets, args.seeds, args.pool_limit, args.limit, args.splits, args.names)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n== {ok}/{len(rows)} ok -> {args.out}")


if __name__ == "__main__":
    main()
