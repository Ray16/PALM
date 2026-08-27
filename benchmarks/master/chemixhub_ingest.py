"""Fold the CheMixHub mixture suite into the master schema (split-quality only).

The CheMixHub raw datasets are an external clone (`chemcognition-lab/chemixhub`)
that is not vendored here, so the generalization-gap layer cannot be recomputed
for these 12 mixture tasks without re-cloning. What *is* committed is the
per-task, per-engine split-quality report produced by
``benchmarks/chemixhub_splits/make_chemixhub_splits.py``
(``splits/leakage_report.csv``). This adapter reshapes that report into the same
long-format columns as the registry benchmark so the two live in one table; the
generalization-gap fields are left blank with a reason.

To add the gap dimension for CheMixHub, re-clone the repo and extend
``run_benchmark`` to load each task's mixture features + measured property.
"""

from __future__ import annotations

import ast
import csv
import os

import pandas as pd

from .run_benchmark import FIELDS

HERE = os.path.dirname(__file__)
REPORT = os.path.abspath(os.path.join(
    HERE, "..", "chemixhub_splits", "splits", "leakage_report.csv"))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RESULTS, "chemixhub_quality.csv")


def _test_frac(cell):
    try:
        d = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else {}
        return round(float(d.get("test", "")), 4) if d else ""
    except Exception:
        return ""


def ingest(report=REPORT, out=OUT):
    if not os.path.exists(report):
        raise FileNotFoundError(f"CheMixHub report not found: {report} "
                                "(run make_chemixhub_splits.py first)")
    df = pd.read_csv(report)
    rows = []
    for _, r in df.iterrows():
        row = {k: "" for k in FIELDS}
        row.update(
            dataset=f"chemixhub:{r['dataset']}/{r['property']}",
            category="mixture",
            task_type="",                        # no target without the external clone
            kind="mixture",
            n=int(r["n_samples"]),
            method=r["engine"],
            seed=42,                             # the recipe's fixed seed
            status="ok",
            leakage=r.get("Lpi_unique_mixture", ""),
            imbalance=r.get("imbalance", ""),
            test_fraction=_test_frac(r.get("sample_fractions", "")),
            runtime_s=r.get("runtime_s", ""),
            extra=(f"n_unique={r.get('n_unique_mixtures','')};"
                   f"k_bar={r.get('k_bar','')};"
                   f"id_leak={r.get('mixture_identity_leakage','')};"
                   f"n_clusters={r.get('n_clusters','')}"),
            reason="chemixhub split-quality only (no target without external clone)",
        )
        rows.append(row)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"== {len(rows)} CheMixHub rows -> {out}")
    return out


if __name__ == "__main__":
    ingest()
