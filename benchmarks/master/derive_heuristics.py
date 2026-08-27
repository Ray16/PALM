"""Reduce the feature sweep to ``data/feature_heuristics.json`` (the router's table).

Trustworthy selection rule, per dataset:

1. **Gate (predictive validity).** Keep only feature spaces where a model on a
   *random* split still learns: mean random-split test metric >= a floor
   (R^2 >= 0.2 regression, ROC-AUC >= 0.55 classification). Removes spaces that
   look "clean" only because they make everything mutually dissimilar.
2. **Don't tank prediction.** Among gated spaces, keep those whose *OOD* test
   metric is within ``OOD_TOL`` of the best gated space — so we never trade real
   predictive signal for a marginally cleaner split.
3. **Select on REFERENCE-space leakage.** Pick the lowest ``ref_leakage`` — leakage
   of the split scored in one fixed space (ECFP / MAGPIE), comparable across
   candidates (unlike each space's self-measured leakage).
4. **Significance margin.** Only override the canonical default when the winner
   beats it by more than ``max(MIN_REL * default, pooled_std)`` — otherwise keep
   the default, so noise-level differences don't masquerade as wins.

Per-entity-type recommendations use **minimax regret**: the representation whose
worst-case gap to the per-dataset best (across that type's datasets) is smallest —
the safest fallback for a novel dataset, not the modal winner.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from PALM.data.routing import FEATURE_DEFAULTS

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
SWEEP = os.path.join(RESULTS, "feature_sweep.csv")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "feature_heuristics.json"))

FLOORS = {"regression": 0.20, "classification": 0.55}   # random-split "still learns" floor
OOD_TOL = 0.03            # allowed OOD test-metric shortfall vs best gated space
MIN_REL = 0.05            # a win must beat the default by >=5% (and >= pooled std)
OOD_METHODS = ("hypergraph", "lowrank")


def _stats(sweep):
    df = pd.read_csv(sweep)
    ok = df[df["status"] == "ok"].copy()
    for c in ("ref_leakage", "test_metric"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    # per-seed OOD reference leakage + OOD test metric, then mean/std across seeds
    ood = ok[ok["method"].isin(OOD_METHODS)]
    per_seed = (ood.groupby(["dataset", "entity_type", "task_type", "feature_set", "seed"])
                   .agg(ref=("ref_leakage", "mean"), test=("test_metric", "mean")).reset_index())
    stats = (per_seed.groupby(["dataset", "entity_type", "task_type", "feature_set"])
                     .agg(ref_mean=("ref", "mean"), ref_std=("ref", "std"),
                          ood_test=("test", "mean")).reset_index())
    stats["ref_std"] = stats["ref_std"].fillna(0.0)
    gate = (ok[ok["method"] == "random"]
            .groupby(["dataset", "feature_set"]).agg(gate=("test_metric", "mean")).reset_index())
    return stats.merge(gate, on=["dataset", "feature_set"], how="left")


def derive(sweep=SWEEP, out=OUT, floors=FLOORS):
    stats = _stats(sweep)
    per_dataset, regret_rows = {}, []

    for (ds, etype, task), g in stats.groupby(["dataset", "entity_type", "task_type"]):
        floor = floors.get(task, -np.inf)
        g = g.dropna(subset=["ref_mean"])
        gated = g[(g["gate"].notna()) & (g["gate"] >= floor)]
        pool = gated if not gated.empty else g            # fall back to all if nothing gated
        if pool.empty:
            continue
        default_fs = FEATURE_DEFAULTS.get(etype)

        # (2) don't tank prediction: keep spaces within OOD_TOL of the best OOD test
        best_ood_test = pool["ood_test"].max()
        keep = pool[pool["ood_test"] >= best_ood_test - OOD_TOL] if pd.notna(best_ood_test) else pool
        keep = keep if not keep.empty else pool

        # (3) lowest reference-space leakage; tiebreak higher gate
        keep = keep.sort_values(["ref_mean", "gate"], ascending=[True, False])
        best = keep.iloc[0]

        # (4) significance margin vs the canonical default
        drow = g[g["feature_set"] == default_fs]
        chosen = best["feature_set"]
        if not drow.empty and chosen != default_fs:
            d_leak, d_std = float(drow["ref_mean"].iloc[0]), float(drow["ref_std"].iloc[0])
            margin = max(MIN_REL * d_leak, d_std, float(best["ref_std"]))
            if not (best["ref_mean"] < d_leak - margin):
                chosen = default_fs                       # improvement not significant -> keep default
        per_dataset[ds] = chosen

        # regret bookkeeping for the per-type fallback (over gated spaces only)
        gmin = float(gated["ref_mean"].min()) if not gated.empty else float(pool["ref_mean"].min())
        for _, r in (gated if not gated.empty else pool).iterrows():
            regret_rows.append(dict(dataset=ds, entity_type=etype,
                                    feature_set=r["feature_set"],
                                    regret=float(r["ref_mean"]) - gmin))

    # per entity type: minimax regret (smallest worst-case gap to per-dataset best)
    per_entity_type = {}
    rr = pd.DataFrame(regret_rows)
    if not rr.empty:
        for etype, ge in rr.groupby("entity_type"):
            n_ds = ge["dataset"].nunique()
            # only consider features present (gated) on a majority of the type's datasets
            cov = ge.groupby("feature_set")["dataset"].nunique()
            eligible = cov[cov >= max(1, n_ds // 2)].index
            ge = ge[ge["feature_set"].isin(eligible)]
            if ge.empty:
                per_entity_type[etype] = FEATURE_DEFAULTS.get(etype)
                continue
            worst = ge.groupby("feature_set")["regret"].max()          # worst-case regret per feature
            per_entity_type[etype] = worst.idxmin()                    # minimize the worst case

    payload = {
        "per_dataset": per_dataset,
        "per_entity_type": per_entity_type,
        "gate_floors": floors,
        "selection": {"objective": "min reference-space leakage among gated spaces",
                      "gate": "random-split test >= floor",
                      "ood_tolerance": OOD_TOL, "significance_margin_rel": MIN_REL,
                      "per_type_rule": "minimax regret", "ood_methods": list(OOD_METHODS)},
        "provenance": {"sweep": os.path.relpath(sweep)},
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"== heuristics -> {out}")
    print(f"   per_entity_type (minimax regret): {per_entity_type}")
    for ds, fs in sorted(per_dataset.items()):
        tag = "" if fs != FEATURE_DEFAULTS.get(stats[stats.dataset == ds]["entity_type"].iloc[0]) else "  (default; no significant win)"
        print(f"   {ds:28s} -> {fs}{tag}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    derive(a.sweep, a.out)
