"""Reduce the feature sweep to ``data/feature_heuristics.json`` (the router's table).

The predictive-validity-gated rule, per dataset:

1. **Gate** — keep only feature spaces where a model on a *random* split still
   learns: mean random-split test metric >= a floor (R^2 >= 0.2 for regression,
   ROC-AUC >= 0.55 for classification). This removes representations that look
   "clean" only because they make everything mutually dissimilar.
2. **Select** — among the gated feature spaces, pick the one with the lowest mean
   **OOD leakage** (averaged over the hypergraph + lowrank partitioners). Tie-break
   on the higher random-split test metric (more informative representation).

Per-entity-type recommendations are the per-type winner by average OOD-leakage
rank across that type's gated datasets. The router (``PALM.data.routing``) then
uses ``per_dataset`` first, ``per_entity_type`` as the fallback.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
SWEEP = os.path.join(RESULTS, "feature_sweep.csv")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "feature_heuristics.json"))

FLOORS = {"regression": 0.20, "classification": 0.55}   # random-split "still learns" floor
OOD_METHODS = ("hypergraph", "lowrank")


def derive(sweep=SWEEP, out=OUT, floors=FLOORS):
    df = pd.read_csv(sweep)
    ok = df[df["status"] == "ok"].copy()
    for c in ("leakage", "test_metric", "gen_gap"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    # mean over seeds per (dataset, entity_type, task_type, feature_set, method)
    agg = (ok.groupby(["dataset", "entity_type", "task_type", "feature_set", "method"])
             .agg(leakage=("leakage", "mean"), test_metric=("test_metric", "mean"))
             .reset_index())

    per_dataset, rows_report = {}, []
    for (ds, etype, task), g in agg.groupby(["dataset", "entity_type", "task_type"]):
        floor = floors.get(task, -np.inf)
        cand = {}
        for fs, gf in g.groupby("feature_set"):
            rnd = gf[gf["method"] == "random"]["test_metric"]
            ood = gf[gf["method"].isin(OOD_METHODS)]["leakage"]
            gate_val = float(rnd.mean()) if len(rnd) else np.nan
            ood_leak = float(ood.mean()) if len(ood) else np.nan
            passed = np.isfinite(gate_val) and gate_val >= floor
            cand[fs] = dict(gate=gate_val, ood_leakage=ood_leak, passed=passed)
            rows_report.append(dict(dataset=ds, entity_type=etype, task=task, feature_set=fs,
                                    random_test=gate_val, ood_leakage=ood_leak, gated_in=passed))
        gated = {fs: v for fs, v in cand.items() if v["passed"] and np.isfinite(v["ood_leakage"])}
        pool = gated or {fs: v for fs, v in cand.items() if np.isfinite(v["ood_leakage"])}
        if not pool:
            continue
        # lowest OOD leakage; tie-break higher gate (predictive) metric
        best = min(pool, key=lambda fs: (pool[fs]["ood_leakage"], -(pool[fs]["gate"] or 0)))
        per_dataset[ds] = best

    # per entity type: winner by average OOD-leakage rank across its gated datasets
    rep = pd.DataFrame(rows_report)
    per_entity_type = {}
    if not rep.empty:
        for etype, ge in rep[rep["gated_in"]].groupby("entity_type"):
            ge = ge.dropna(subset=["ood_leakage"])
            if ge.empty:
                continue
            ge = ge.copy()
            ge["rank"] = ge.groupby("dataset")["ood_leakage"].rank()
            order = ge.groupby("feature_set")["rank"].mean().sort_values()
            per_entity_type[etype] = order.index[0]

    payload = {
        "per_dataset": per_dataset,
        "per_entity_type": per_entity_type,
        "gate_floors": floors,
        "provenance": {"sweep": os.path.relpath(sweep),
                       "rule": "min OOD leakage among feature spaces passing the "
                               "random-split predictive gate; tie-break higher gate metric",
                       "ood_methods": list(OOD_METHODS)},
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"== heuristics -> {out}")
    print(f"   per_entity_type: {per_entity_type}")
    for ds, fs in per_dataset.items():
        print(f"   {ds:28s} -> {fs}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--out", default=OUT)
    ap.parse_args()
    derive()
