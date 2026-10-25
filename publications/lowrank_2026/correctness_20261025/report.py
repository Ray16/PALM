"""Generate the measured findings without selecting methods by model test scores."""

from __future__ import annotations
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
run = HERE / "verified_run"
manifest = json.loads((run / "manifest.json").read_text())
audit = json.loads((run / "pipeline_review.json").read_text())
rows = [
    json.loads((run / "rows" / (j["name"] + ".json")).read_text())
    for j in manifest["jobs"]
]
assert len(rows) == 189 and all(r["status"] == "ok" for r in rows)
datasets = sorted({r["dataset"] for r in rows})
settings = list(dict.fromkeys(r["setting"] for r in rows))
metrics = [
    "global_ratio",
    "local_ramp_04",
    "maccs_mean_nearest_training",
    "weighted_global_ratio",
    "target_wasserstein",
]
summary = {}
for setting in settings:
    selected = [r for r in rows if r["setting"] == setting]
    by_dataset = {}
    for d in datasets:
        group = [r for r in selected if r["dataset"] == d]
        by_dataset[d] = {
            m: statistics.mean(r["scores"][m] for r in group) for m in metrics
        }
        by_dataset[d]["median_split_seconds"] = statistics.median(
            r["split_seconds"] for r in group
        )
    summary[setting] = dict(
        by_dataset=by_dataset,
        mean_scores={
            m: statistics.mean(v[m] for v in by_dataset.values()) for m in metrics
        },
        median_split_seconds=statistics.median(r["split_seconds"] for r in selected),
        median_process_wall_seconds=statistics.median(
            r["fresh_process_wall_seconds"] for r in selected
        ),
        median_process_peak_mb=statistics.median(
            r["process_peak_rss_mb"] for r in selected
        ),
    )
predictions = [
    json.loads(p.read_text()) for p in (HERE / "predictions/rows").glob("*.json")
]
assert len(predictions) == 90
predictive = {}
for dataset in ["moleculenet_clintox", "moleculenet_sider", "moleculenet_qm8"]:
    predictive[dataset] = {}
    for setting in [
        "previous_plain",
        "previous_combined",
        "new_default_plain",
        "new_default_cut",
        "new_default_weighted",
    ]:
        predictive[dataset][setting] = {}
        for model in ["linear", "forest"]:
            group = [
                r
                for r in predictions
                if r["dataset"] == dataset
                and r["setting"] == setting
                and r["model"] == model
            ]
            keys = set().union(
                *(r["metrics"] for r in group if r["metrics"] is not None)
            )
            predictive[dataset][setting][model] = {
                k: dict(
                    mean=statistics.mean(
                        r["metrics"][k]
                        for r in group
                        if r["metrics"] and r["metrics"][k] is not None
                    )
                    if any(r["metrics"] and r["metrics"][k] is not None for r in group)
                    else None,
                    defined=sum(
                        r["metrics"] is not None and r["metrics"][k] is not None
                        for r in group
                    ),
                    total=len(group),
                )
                for k in keys
            }
calibration = [
    json.loads(p.read_text()) for p in (HERE / "calibration/rows").glob("*.json")
]
assert len(calibration) == 60 and all(
    sum(b["n"] for b in r["bins"]) == r["n_test"] for r in calibration
)
cosine = [
    json.loads(p.read_text())
    for p in (HERE / "scaling_final/rows").glob("cosine_100000_*.json")
]
cos_times = {
    impl: statistics.median(r["seconds"] for r in cosine if r["implementation"] == impl)
    for impl in ["previous", "new"]
}
identity = json.loads(
    (HERE / "scaling_final/rows/identity_100000_new.json").read_text()
)
cache = json.loads((HERE / "scaling_final/rows/calibration_reuse.json").read_text())[
    "records"
]
budgets = [
    json.loads(p.read_text()) for p in (HERE / "scaling/rows").glob("cut_budget_*.json")
]
budget_summary = {}
for dataset in sorted({r["dataset"] for r in budgets}):
    budget_summary[dataset] = {}
    for count in [2048, 8192, 32768]:
        group = [r for r in budgets if r["dataset"] == dataset and r["pairs"] == count]
        budget_summary[dataset][str(count)] = dict(
            median_seconds=statistics.median(r["seconds"] for r in group),
            ranks=[r["diagnostics"]["rank"] for r in group],
            audits_passed=sum(
                r["diagnostics"]["factorization"]["audit_tolerance_met"] for r in group
            ),
        )
record = dict(
    primary_cases=len(rows),
    datasets=datasets,
    seeds=[0, 1, 2],
    settings=summary,
    numerical_commit="356ae758af8f268028dd2534c4f7530054bd6b1c",
    predictive=predictive,
    calibration_cases=len(calibration),
    rank_budget_study=budget_summary,
    cosine_factor_median_seconds=cos_times,
    identity_storage=identity,
    interpretation="Exploratory development evidence, not independent prospective validation. "
    "Defaults used unlabeled runtime/fidelity observations before predictive checks.",
)
(HERE / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
cut_rows = [r for r in rows if r["setting"] == "new_default_cut"]
passes = sum(r["diagnostics"]["factorization"]["audit_tolerance_met"] for r in cut_rows)
quality = json.loads((HERE / "quality_profile/complete.json").read_text())
lines = [
    "# Low-rank correctness and pipeline review",
    "",
    "The delivery fixes the reproduced audit and kernel failures, bounds default refinement effort, "
    "and removes dense n-D identity storage. It adds explicit evaluation-boundary weights and a "
    "supplied-date/order reference helper. Experimental penalties remain opt-in. Lower overlap "
    "does not by itself establish a better deployment test.",
    "",
    "## Verification",
    "",
    "- 213 package tests and 41 website compatibility tests pass. The built wheel is tested in an isolated target.",
    "- All 189 final-source benchmark cases completed. Every assignment retains all featurized input rows at exact fold counts.",
    "- Direct artifact checks preserve all 27 strict-group partitions.",
    f"- Independent NumPy dense scoring checks all 42 FreeSolv/ClinTox cases; maximum discrepancy is {audit['maximum_dense_score_error']:.3g}.",
    "- Ten end-to-end replays reproduce saved assignments. All 90 predictive checks use byte-identical features, targets and fold assignments to their corresponding final-source cases.",
    "- Sixty fixed-classifier refits reproduce their saved Brier scores and add descriptive reliability bins and calibration-in-the-large.",
    "",
    "The rare-mass regression no longer declares an empty factor accurate after missing all nonzero similarity. "
    "Zero observed leakage does not yield a false zero-width uncertainty interval. Dense/aligned kernels "
    "share normalization, extreme finite feature scales stay finite, count-valued Tanimoto is rejected, "
    "and undefined normalization/relative scores remain explicit.",
    "",
    "## Primary comparison",
    "",
    "Nine full datasets, three seeds and seven fixed configurations compare with the preceding development "
    "version. ClinTox, SIDER and QM8 extend the preceding six-dataset iteration. They are additional "
    "development tasks, not a permanently held-out or prospectively collected validation set. Defaults "
    "used unlabeled runtime/fidelity evidence, including QM8; model test outcomes did not select them. "
    "No hypergraph baseline is used. This study does not replace the paper’s external-method comparison.",
    "",
    "Score columns average the nine dataset means. Call time is the median of 27 calls; the QM8 column "
    "separately exposes the large-dataset cost. Split calls include factorization, checks and graph work. "
    "Preparation, scoring, loading, warm-up, fresh-process time and process memory remain in each row.",
    "",
    "| Configuration | Global ratio | Local ramp | MACCS nearest | Call seconds | QM8 call seconds |",
    "|---|---:|---:|---:|---:|---:|",
]
for setting in settings:
    s = summary[setting]
    v = s["mean_scores"]
    lines.append(
        f"| {setting} | {v['global_ratio']:.4f} | {v['local_ramp_04']:.4f} | {v['maccs_mean_nearest_training']:.4f} | {s['median_split_seconds']:.3f} | {s['by_dataset']['moleculenet_qm8']['median_split_seconds']:.2f} |"
    )
lines += [
    "",
    "The weighted setting expresses an example train/test priority; its objective differs from the "
    "ordinary cut. It is not presented as an unqualified winner. Refinement after candidate generation "
    "can improve the found objective while adding computation. The two-candidate default and explicit "
    "one/all-candidate choices expose that trade-off.",
    "",
    f"Cut auditing establishes its configured tolerance on {passes}/27 final cases. The exact returned-cut "
    "errors are independently checked against the full kernel totals in `verified_run/pipeline_review.json`. "
    "The tolerance is normalized by total absolute kernel mass, not by the cut value. Unestablished "
    "tolerances warn/raise or are explicitly collected with status; they are not silently passed.",
    "",
    "## Structural performance and budget evidence",
    "",
    f"For the 100,000-by-16 cosine input, median factor construction is {cos_times['previous']:.3f} s for the preceding Nyström path and {cos_times['new']:.3f} s for the exact feature-map backend (three seeds). These are factor-only timings, not whole-pipeline speedups.",
    "",
    f"The 100,000-record paired-identity factor uses {identity['storage_bytes']:,} bytes of CSR arrays. Its dense equivalent is {identity['dense_equivalent_bytes']:,} bytes; no 20-GB dense array was allocated. Process RSS is reported separately, including framework/import overhead.",
    "",
    f"The native Tox21 calibration example costs {cache[0]['seconds']:.2f} s initially and {cache[1]['seconds']:.2f} s on the identical repeated request. The first call includes any rejected ANN/compilation cost; exact-backend reuse remains exact.",
    "",
    "The rank-budget study holds refinement at one candidate and uses no target labels. It demonstrates "
    "why more audit samples can be cheaper than a larger factor. These explicit-budget cases remain "
    "separate from the final default settings.",
    "",
    "| Dataset | Pair samples | Median seconds | Ranks | Independent audits passed |",
    "|---|---:|---:|---|---:|",
]
for dataset, counts in budget_summary.items():
    for count, s in counts.items():
        lines.append(
            f"| {dataset} | {count} | {s['median_seconds']:.2f} | {s['ranks']} | {s['audits_passed']}/3 |"
        )
lines += [
    "",
    f"The earlier all-candidates study completed {quality['completed']}/243 cases. One QM8 plain case exceeded the 600-s protocol. Its missing observation and terminal log remain preserved, and it is not counted as a success. This cost motivated the bounded default before predictive outcomes were examined.",
    "",
    "## Predictive and chronological scope",
    "",
    "Two fixed models per task (linear and random forest) were fit on training labels only for five "
    "frozen policies, three seeds and three additional datasets: 90 fits. `summary.json` retains per-model "
    "AUROC, average precision and Brier where defined, or regression MAE/RMSE/R². Per-case records retain "
    "label availability and class composition. No test-set hyperparameter search or policy selection "
    "uses these scores. The calibration supplement uses fixed ten equal-width probability bins; its ECE "
    "and reliability summaries are sample-size/distribution dependent and descriptive.",
    "",
    "Actual project dates/test order were not supplied. `chronological_reference` requires complete real "
    "ordering and explicit cutoffs, preserves ties and returns a zero-imbalance matched-count specification. "
    "Its synthetic policy tests are not real chronological experiments. Prospective model ranking and "
    "reliability still require actual dated outcomes.",
    "",
    "## Provenance and reproduction",
    "",
    "The final numerical grid is pinned to commit `356ae758af8f268028dd2534c4f7530054bd6b1c`. "
    "Subsequent Python changes align three docstrings only; `verification/doc_alignment.json` and the "
    "release audit verify equality of the complete non-doc executable AST. Original input, source, runner "
    "and assignment hashes remain in the manifests/rows. The isolated wheel check compares all shipped "
    "Python files with the delivery source.",
    "",
    "Use `benchmark.py prepare/launch` for the final grid, `review.py` for artifact/source/numerical "
    "verification, `prediction_check.py` and `calibration_check.py` for fixed-model checks, and "
    "`scaling.py` for selected structural workloads. Frozen quality-profile runners preserve earlier "
    "protocols. Prepared public-feature caches and runtime logs are excluded from Git; input recipes and "
    "checksums are retained. The manuscript and running app stay on the reviewed release.",
    "",
]
(HERE / "FINDINGS.md").write_text("\n".join(lines))
print(
    "Report generated:",
    len(rows),
    "final cases,",
    len(predictions),
    "predictive checks,",
    len(calibration),
    "calibration checks",
)
