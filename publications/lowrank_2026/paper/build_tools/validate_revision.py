"""Strict evidence gates for the revised article; partial drafts opt out explicitly.

Passing these mechanical gates is necessary, not sufficient for publication.
The requirement ledger and final scientific/visual review remain separate gates.
"""
from pathlib import Path
import json
import sys

PAPER = Path(__file__).resolve().parents[1]
D = PAPER / "revision_20261021"


def main():
    issues = []
    external = list((D / "external").glob("*/result.json"))
    if len(external) != 10:
        issues.append(f"external assays: {len(external)}/10")
    for path in external:
        r = json.loads(path.read_text())
        if len(r["rows"]) != 20 or len(r["models"]) != 80:
            issues.append(f"external record coverage: {path.parent.name}")
    fidelity = list((D / "powered_fidelity").glob("*/result.json"))
    if len(fidelity) != 6:
        issues.append(f"powered fidelity: {len(fidelity)}/6 datasets")
    datasets = ["moleculenet_" + name for name in ("freesolv", "esol", "bace", "bbbp", "sider",
        "clintox", "lipophilicity", "tox21", "qm8", "hiv", "muv")] + ["qmof", "materials_project"]
    methods = ("random", "plain", "combined", "combined_tau06", "knn_graph", "scaffold", "butina", "datasail")
    for dataset in datasets:
        missing = 0
        for method in methods:
            for epsilon in (.01, .05):
                for seed in ([0] if method == "datasail" else [0, 1, 2]):
                    p = D / "benchmark" / dataset / f"{method}__eps{epsilon}__s{seed}.json"
                    if not p.exists():
                        missing += 1
                    elif json.loads(p.read_text()).get("status") not in ("ok", "unavailable"):
                        issues.append(f"unresolved benchmark: {p.name}")
        if missing:
            issues.append(f"{dataset}: {missing} benchmark records missing")
    for name in ("benchmark_revision.tex",):
        if not (PAPER / name).exists():
            issues.append(f"missing integrated article evidence: {name}")
    for gate in ("lohi_validation.json", "umap_validation.json", "scaling_validation.json",
                 "benchmark_validation.json", "datasail_matched_validation.json", "runtime_validation.json",
                 "robustness_validation.json", "sensitivity_validation.json", "calibration_validation.json",
                 "legacy_calibration_reconciliation.json", "namespaced_release_validation.json",
                 "metric_validation.json",
                 "pipeline_review.json", "final_visual_review.json"):
        path = D / gate
        if not path.exists() or json.loads(path.read_text()).get("passed") is not True:
            issues.append(f"open revision gate: {gate}")
    result = dict(passed=not issues, issues=issues,
        note="author metadata deferred; this validator does not certify scientific conclusions")
    (D / "completion_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    for issue in issues:
        print(issue)
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
