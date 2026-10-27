"""Evidence, source, layout and reviewed-render gates for the current paper."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

from pdf_review import signature, review_issues

PAPER = Path(__file__).resolve().parents[1]
REVISION = PAPER.parent / "revision_20261026"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    issues = []
    evidence = json.loads((REVISION / "analysis/build_evidence.json").read_text())
    if evidence["preliminary"] and not args.allow_incomplete:
        issues.append("unfinished evidence: " + ", ".join(evidence["unmet"]))
    spec = importlib.util.spec_from_file_location(
        "publication_report", REVISION / "analysis/report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    manifest, rows, outcomes = report.verified_rows()
    if len(outcomes) != 199 or len(rows) != 191 or len(outcomes) - len(rows) != 8:
        issues.append("primary outcome inventory mismatch")
    if report.summarize(rows, outcomes) != json.loads(
        (REVISION / "analysis/summary.json").read_text()
    ):
        issues.append("generated summary disagrees with raw observations")
    source_checks = {}
    for folder, expected in [
        ("run", 199),
        ("fidelity/run", 18),
        ("publication_checks/run", 72),
        ("scaling/cpu_run", 18),
        ("scaling/large_cpu_run", 3),
    ]:
        root = REVISION / folder
        m = json.loads((root / "manifest.json").read_text())
        if len(m["jobs"]) != expected:
            issues.append("manifest size mismatch: " + folder)
        current = {name: sha(PAPER.parent / name) for name in m["sources"]}
        same = current == m["sources"]
        source_checks[folder] = same
        if not same:
            issues.append("producer source drift: " + folder)
        if folder == "publication_checks/run":
            for job in m["jobs"]:
                p = root / "rows" / f"{job['name']}.json"
                if not p.exists():
                    issues.append("missing profile/control: " + job["name"])
                    continue
                r = json.loads(p.read_text())
                if r["manifest_sha256"] != sha(root / "manifest.json"):
                    issues.append("profile provenance: " + job["name"])
                if r["status"] == "native_infeasible":
                    if not any(
                        "SCIP status: infeasible" in s for s in r["solver_evidence"]
                    ):
                        issues.append("unproved exclusion")
                elif r["status"] == "ok":
                    if not r["artifacts"] or any(
                        sha(root / name) != digest
                        for name, digest in r["artifacts"].items()
                    ):
                        issues.append("profile/control artifact drift")
                else:
                    issues.append("unexpected native/control status")
        if folder.startswith("scaling/"):
            for job in m["jobs"]:
                p = root / job["name"] / "result.json"
                fail = root / "failures" / f"{job['name']}.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    if r["manifest_sha256"] != sha(root / "manifest.json"):
                        issues.append("scaling provenance mismatch")
                    if any(
                        sha(p.parent / name) != digest
                        for name, digest in r["artifacts"].items()
                    ):
                        issues.append("scaling artifact drift")
                    if sum(r["counts"]) != job["n"]:
                        issues.append("scaling retention mismatch")
                    if job["n"] >= 100000 and r["method"] == "plain":
                        interval = r["fidelity"]["relative_pair_mae"]["ci95"]
                        if (
                            interval is None
                            or not math.isfinite(interval[1])
                            or interval[1] >= 0.008
                        ):
                            issues.append("headline cut-error claim unsupported")
                elif folder == "scaling/cpu_run" and fail.exists():
                    if json.loads(fail.read_text())["exit_code"] != -999:
                        issues.append("unexpected scaling failure")
                else:
                    issues.append("unaccounted scaling case: " + job["name"])
    independent = json.loads(
        (REVISION / "analysis/independent_validation.json").read_text()
    )
    errors = [
        independent["max_global_discrepancy"],
        independent["max_nearest_discrepancy"],
    ]
    if independent["comparisons"] != 246 or any(
        not math.isfinite(v) or v > 1e-6 for v in errors
    ):
        issues.append("independent numerical validation incomplete")
    provenance = json.loads((REVISION / "analysis/provenance.json").read_text())
    if provenance.get("report_sha256") != sha(REVISION / "analysis/report.py"):
        issues.append("numerical report source changed since verification")
    if provenance.get("summary_sha256") != sha(REVISION / "analysis/summary.json"):
        issues.append("summary differs from verified report")
    if provenance.get("independent_validation_sha256") != sha(
        REVISION / "analysis/independent_validation.json"
    ):
        issues.append("independent checks differ from verified report")
    figures = json.loads((REVISION / "analysis/figure_audit.json").read_text())
    for f in figures:
        if (
            f["minimum_source_font_pt"] < 18
            or f["detected_text_overlaps"]
            or f.get("detected_panel_overlaps")
            or f["clipped_text"]
        ):
            issues.append("figure geometry/font failure: " + f["figure"])
    rendered = {
        name: signature(PAPER / name) for name in ["manuscript.pdf", "supplement.pdf"]
    }
    for name, stats in rendered.items():
        if not stats["all_figure_fonts_at_least_18"] or stats["clipped_figure_text"]:
            issues.append("compiled figure font/clip failure: " + name)
    pattern = (
        r"Overfull \\[hv]box|Float too large|undefined references|Citation .*undefined"
    )
    for name in ["manuscript_pass3.log", "supplement_pass3.log"]:
        p = PAPER / "logs" / name
        if not p.exists():
            issues.append("missing final TeX log: " + name)
            continue
        if re.search(pattern, p.read_text(errors="replace")):
            issues.append("TeX layout/reference warning: " + name)
    review = REVISION / "verification/final_visual_review.json"
    if not args.allow_incomplete:
        if not review.exists():
            issues.append("final manual visual review missing")
        else:
            issues += review_issues(PAPER, json.loads(review.read_text()))
    pending = (
        "Author information pending" in (PAPER / "manuscript.tex").read_text()
        or "Pending" in (PAPER / "submission_metadata.tex").read_text()
    )
    result = dict(
        technical_checks_passed=not issues,
        issues=issues,
        source_checks=source_checks,
        rendered=rendered,
        author_submission_metadata_pending=pending,
        submission_ready=not issues and not pending,
        scope="Technical/reproducibility checks; author metadata was explicitly deferred and is never invented.",
    )
    (REVISION / "verification/completion_gate.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--allow-incomplete", action="store_true")
    main(p.parse_args())
