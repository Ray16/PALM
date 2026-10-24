"""Summarize all predeclared configurations, including the missing planned controls."""
import json
from pathlib import Path
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
ROOTS = [HERE / "run_20261023_v1", HERE / "controls_20261023_v1"]
rows = []
for root in ROOTS:
    manifest = json.loads((root / "manifest.json").read_text())
    review = json.loads((root / "pipeline_review.json").read_text())
    if review["completed"] != len(manifest["jobs"]): raise RuntimeError("review incomplete")
    for job in manifest["jobs"]:
        row = json.loads((root / "rows" / (job["name"] + '.json')).read_text())
        if row["status"] != "ok": raise RuntimeError("non-success case")
        rows.append(row)
datasets = sorted({r["dataset"] for r in rows})
settings = sorted({r["setting"] for r in rows})
metrics = ["global_ratio", "local_ramp_04", "maccs_mean_nearest_training",
           "test_fraction_above_06", "target_wasserstein"]
summaries, paired = {}, {}
for setting in settings:
    selected = [r for r in rows if r["setting"] == setting]
    if len(selected) != 18: raise RuntimeError("setting is not complete")
    by_dataset = {}
    for dataset in datasets:
        subset = [r for r in selected if r["dataset"] == dataset]
        by_dataset[dataset] = {m: float(np.mean([r["scores"][m] for r in subset])) for m in metrics}
        by_dataset[dataset]["median_split_seconds"] = float(np.median([r["split_seconds"] for r in subset]))
    summaries[setting] = dict(by_dataset=by_dataset,
        mean_scores={m: float(np.mean([v[m] for v in by_dataset.values()])) for m in metrics},
        median_split_seconds=float(np.median([r["split_seconds"] for r in selected])),
        median_process_peak_mb=float(np.median([r["process_peak_rss_mb"] for r in selected])),
        median_preparation_split_scoring_seconds=float(np.median([r["preparation_plus_split_plus_scoring_seconds"] for r in selected])))

rng = np.random.default_rng(0)
for candidate in ["new_auto", "new_auto_combined", "new_local_025", "new_local_1", "new_combined_strict"]:
    for baseline in ["release_plain", "release_power3", "release_groups", "release_combined"]:
        a, b = summaries[candidate]["by_dataset"], summaries[baseline]["by_dataset"]
        result = {}
        for m in metrics:
            differences = np.array([a[d][m] - b[d][m] for d in datasets])
            boot = differences[rng.integers(len(datasets), size=(20000, len(datasets)))].mean(1)
            result[m] = dict(mean_paired_change=float(differences.mean()),
                             dataset_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist())
        result["dominates_on_global_and_local"] = [d for d in datasets if
            a[d]["global_ratio"] <= b[d]["global_ratio"] and a[d]["local_ramp_04"] <= b[d]["local_ramp_04"]]
        paired[candidate + "__vs__" + baseline] = result

adaptive = {}
for setting in ["new_auto", "new_auto_combined", "new_auto_sampled"]:
    factors = [r["diagnostics"]["factorization"] for r in rows if r["setting"] == setting]
    adaptive[setting] = dict(selection_pass=sum(x["selection_tolerance_met"] for x in factors),
        independent_audit_pass=sum(x["audit_tolerance_met"] for x in factors), total=len(factors),
        landmark_counts=[x["landmark_count"] for x in factors],
        ranks=[x["effective_rank"] for x in factors])
record = dict(completed_cases=len(rows), datasets=datasets, seeds=[0, 1, 2],
              settings=summaries, paired=paired, adaptive=adaptive,
              interpretation="Six development datasets, descriptive exploratory comparisons. "
                             "Bootstrap resamples datasets, not seeds as independent datasets. "
                             "No chronological or predictive-realism claim.")
(HERE / "summary.json").write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

lines = ["# Low-rank development findings", "",
    "The development package adds numerical safeguards, empirical adaptive rank, explicit group contracts, "
    "bounded sampled diagnostics and an experimental global/local objective. The evidence supports these "
    "controls and robustness fixes; it does not establish a universal speed or split-quality improvement.", "",
    "All 216 cases completed across six full molecular datasets and three seeds. Every assignment retains "
    "all entities at exact 80/10/10 counts. Independent review checks the saved assignments, source/input "
    "hashes and all 18 strict group partitions. A separate NumPy dense reference reproduces the global/local "
    "scores on all 36 FreeSolv cases. No hypergraph method enters this comparison.", "",
    "## What to use", "",
    "Keep fixed rank for workloads with an already validated accuracy/runtime trade-off. Use adaptive rank "
    "when an explicit approximation check matters, and inspect both selection and independent-audit flags. "
    "The 5% entrywise tolerance often requires larger factors; it is more demanding than preserving a useful cut.", "",
    "Use strict supplied groups when series/family retention is a real requirement. For constructed groups, "
    "strict preservation applies to those groups only. Keep count-first repair when exact counts take priority "
    "over group retention. Incompatible requirements raise a proven-infeasible or unresolved-feasibility error.", "",
    "Keep the existing combined powered/grouped method for strong local separation on these datasets. "
    "The new local penalty offers a different global/local trade-off and remains experimental. Neither new "
    "weight should be selected by examining final model test performance. Power-only and group-only controls "
    "are included below rather than assuming the new objective adds something those settings cannot do.", "",
    "Threshold 0.4 and k=32 match the existing grouping configuration. Weight 1 gives equal emphasis to "
    "the two mass-normalized components; weight 0.25 tests a smaller local contribution. These weights "
    "were fixed before the main benchmark, not selected from model test outcomes.", "",
    "For these CPU fingerprint workloads, prefer explicit exact tiled neighbours. ANN needs an audited use "
    "case; auto calibration includes its rejected work and can cost more than a known explicit choice. "
    "The native audit caught two deficient Tox21 candidate rows; exact-query repair now produces finite, "
    "distinct, non-self neighbours and counts toward construction time.", "",
    "## Complete comparison", "",
    "Scores below average the six dataset means; split time is the median of 18 calls per configuration. "
    "Lower overlap is a measurement, not a guarantee of a better deployment test. The local score is the "
    "mean ramp above Morgan similarity 0.4 to the nearest training molecule. MACCS is used only for evaluation.", "",
    "| Configuration | Global ratio | Local ramp | MACCS nearest | Split seconds | Prep + split + scoring seconds |",
    "|---|---:|---:|---:|---:|---:|"]
for setting in ["release_plain", "new_plain", "new_auto", "new_auto_sampled", "release_power3",
                "release_groups", "release_combined", "new_combined", "new_auto_combined",
                "new_combined_strict", "new_local_025", "new_local_1"]:
    s = summaries[setting]; v = s["mean_scores"]
    lines.append(f"| {setting} | {v['global_ratio']:.4f} | {v['local_ramp_04']:.4f} | "
                 f"{v['maccs_mean_nearest_training']:.4f} | {s['median_split_seconds']:.3f} | "
                 f"{s['median_preparation_split_scoring_seconds']:.3f} |")
lines += ["", "Shared preparation includes the Morgan construction and independent MACCS features. "
    "The table includes scoring cost separately from split-call cost; split calls include factorization, "
    "adaptive checks and neighbour construction. Imports, small warm-ups and cache-loading costs are "
    "excluded from these two columns and retained separately in each row. No trained models are timed.", "",
    "## Fidelity, trade-offs and validation", ""]
for name, values in adaptive.items():
    lines.append(f"- {name}: selection tolerance met in {values['selection_pass']}/18; independent audit met "
                 f"in {values['independent_audit_pass']}/18; landmark counts range "
                 f"{min(values['landmark_counts'])}–{max(values['landmark_counts'])}.")
for name in ["new_local_025", "new_local_1"]:
    for baseline in ["release_plain", "release_power3", "release_groups", "release_combined"]:
        ds = paired[name + '__vs__' + baseline]["dominates_on_global_and_local"]
        lines.append(f"- {name} has no larger mean global and local score than {baseline} on "
                     f"{len(ds)}/6 datasets ({', '.join(d.removeprefix('moleculenet_') for d in ds) or 'none'}).")
lines += ["", "`summary.json` contains every dataset mean, target-distribution diagnostics and paired changes "
    "with exploratory dataset-bootstrap intervals. Strict preservation can trade overlap quality for the "
    "required contract. The 18 sampled leakage intervals all covered the exact ratios in this grid; this "
    "is descriptive coverage and does not establish a guarantee.", "",
    "The 180-case main grid ran at source commit `95e43be`. The later ANN deficiency repair changes only "
    "a function that none of those settings calls. `review.py` checks that executable AST scope and verifies "
    "all measured neighbour paths are exact. The 36 additional power/group controls were specified in "
    "`DEVELOPMENT.md`; their separate pre-execution manifest preserves the main grid and fixed settings. "
    "The ANN repair is tested independently on the full BBBP/Tox21 fingerprints and by regression tests.", "",
    "This is development evidence, not a new manuscript claim. No actual chronological project outcomes "
    "were supplied, and simulated dates are not substituted. Chronological generalization and broader "
    "representation validity remain external validation tasks. The current manuscript and running website "
    "continue to describe/use the reviewed release.", "",
    "## Reproduction", "",
    "`benchmark.py prepare` builds the public fingerprint caches; `benchmark.py launch` runs the main "
    "grid with three workers, two disjoint CPU slots and two native threads each. `controls.py` supplies "
    "the two planned controls, reusing identical prepared features. `neighbour_check.py` audits native "
    "backends; `review.py` checks saved artifacts; `report.py` creates this report and `summary.json`. "
    "Manifests, per-case records and assignments accompany the results. Source/input drift is rejected "
    "rather than silently mixing measurements. The benchmark commit and release source paths must be "
    "available when reproducing the frozen grid.", ""]
(HERE / "FINDINGS.md").write_text('\n'.join(lines))
inputs = HERE / "inputs"; inputs.mkdir(exist_ok=True)
for source in (ROOTS[0] / "cache").glob('*.json'): shutil.copy(source, inputs / source.name)
print(f"reported {len(rows)} cases, {len(datasets)} datasets, {len(settings)} settings")
