"""Verify publication-study outputs and derive manuscript summaries.

Infeasible native outcomes are classified from retained solver evidence, never
silently filled with assignments or treated as successful numerical cases.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run"
CACHE = Path(
    "/nfs/lambda_stor_01/homes/rzhu/data_splitting_lowrank_reviewed/publications/lowrank_2026/correctness_20261025/verified_run/cache"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def expected_cross(counts):
    n = int(sum(counts))
    return float(1 - sum(c * (c - 1) for c in counts) / (n * (n - 1)))


def group_check(labels, groups):
    lo, hi = np.full(groups.max() + 1, 3), np.full(groups.max() + 1, -1)
    np.minimum.at(lo, groups, labels)
    np.maximum.at(hi, groups, labels)
    if not np.array_equal(lo, hi):
        raise AssertionError("saved partition breaks a required group")


def verified_rows():
    manifest = json.loads((RUN / "manifest.json").read_text())
    manifest_sha = sha(RUN / "manifest.json")
    outcomes, rows = [], []
    for job in manifest["jobs"]:
        path = RUN / "rows" / f"{job['name']}.json"
        if not path.exists():
            failures = sorted((RUN / "failures").glob(job["name"] + "__attempt*.json"))
            if not failures:
                raise RuntimeError("nonterminal missing observation: " + job["name"])
            failure = json.loads(failures[-1].read_text())
            log_path = Path(failure["log"])
            archived = ROOT / "analysis/native_logs" / (log_path.stem + ".txt")
            if archived.exists():
                log_path = archived
            lines = log_path.read_text().splitlines()
            evidence = [
                line
                for line in lines
                if "SCIP status:" in line or "No assignments could be made" in line
            ]
            if job.get("method") != "datasail" or not any(
                "SCIP status: infeasible" in line for line in evidence
            ):
                raise RuntimeError("unexplained failure: " + job["name"])
            outcomes.append(
                dict(
                    job,
                    status="native_solver_reported_infeasible",
                    evidence=evidence,
                    log_sha256=sha(log_path),
                    adapter_note="v1 adapter subsequently raised TypeError when reading absent native assignments; no score or runtime is imputed",
                )
            )
            continue
        row = json.loads(path.read_text())
        if row["status"] != "ok" or row["manifest_sha256"] != manifest_sha:
            raise AssertionError("row provenance/status mismatch")
        for name, digest in row["artifacts"].items():
            artifact = RUN / name
            if sha(artifact) != digest:
                raise AssertionError("assignment/artifact hash mismatch")
            a = np.load(artifact)
            if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
                raise AssertionError("nonintegral/invalid assignment artifact")
        if row["study"] == "groups":
            directory = RUN / "assignments" / row["name"]
            gid = np.load(directory / "groups.npy")
            for method in ["feasible", "optimized_p1", "optimized_p3"]:
                labels = np.load(directory / f"{method}.npy")
                if not np.array_equal(np.bincount(labels, minlength=3), row["counts"]):
                    raise AssertionError("group-study exact count violation")
                group_check(labels, gid)
        if row["study"] == "frontier" and row.get("method") == "datasail":
            directory = RUN / "assignments" / row["name"]
            gid = np.load(directory / "native_groups.npy")
            native = np.load(directory / "datasail.npy")
            counts = np.bincount(native, minlength=3)
            for method in ["datasail", "matched_counts", "matched_native_groups"]:
                labels = np.load(directory / f"{method}.npy")
                if not np.array_equal(np.bincount(labels, minlength=3), counts):
                    raise AssertionError("DataSAIL matched-count violation")
                if method != "matched_counts":
                    group_check(labels, gid)
        outcomes.append(dict(name=row["name"], status="ok"))
        rows.append(row)
    return manifest, rows, outcomes


def summarize(rows, outcomes):
    groups = [r for r in rows if r["study"] == "groups"]
    group_summary = {}
    for method in ["feasible_groups", "optimized_p1", "optimized_p3"]:
        values = []
        for case in groups:
            row = next(r for r in case["rows"] if r["method"] == method)
            values.append(
                dict(
                    dataset=case["dataset"],
                    seed=case["seed"],
                    global_relative=row["scores"]["global_ratio"]
                    / expected_cross(case["counts"]),
                    mean_nearest=row["scores"]["mean_nearest_training"],
                    raw_global=row["scores"]["global_ratio"],
                )
            )
        group_summary[method] = dict(
            cases=len(values),
            global_relative_mean=float(np.mean([v["global_relative"] for v in values])),
            mean_nearest=float(np.mean([v["mean_nearest"] for v in values])),
            values=values,
        )
    exact = [x for r in rows if r["study"] == "exact" for x in r["rows"]]
    exact_summary = []
    for grouped in [False, True]:
        for power in [1, 3]:
            for rank in [2, 4, 8, 12]:
                v = [
                    r
                    for r in exact
                    if r["grouped"] == grouped
                    and r["power"] == power
                    and r["rank"] == rank
                ]
                exact_summary.append(
                    dict(
                        grouped=grouped,
                        power=power,
                        rank=rank,
                        cases=len(v),
                        max_optimization_gap=float(
                            max(r["normalized_optimization_gap"] for r in v)
                        ),
                        max_true_gap=float(max(r["normalized_true_gap"] for r in v)),
                        max_approximation_error=float(
                            max(r["approximation_uniform_error"] / r["mass"] for r in v)
                        ),
                    )
                )
    deployment = []
    for method in ["random", "plain", "combined", "oracle_family_holdout"]:
        values = []
        for case in [r for r in rows if r["study"] == "deployment"]:
            r = {x["model"]: x for x in case["rows"] if x["method"] == method}
            a, b = r["invariant_ridge"], r["family_knn"]
            values.append(
                dict(
                    seed=case["seed"],
                    estimated_difference=a["estimated_rmse"] - b["estimated_rmse"],
                    interpolation_difference=a["interpolation_rmse"]
                    - b["interpolation_rmse"],
                    new_family_difference=a["new_family_rmse"] - b["new_family_rmse"],
                    knn_interpolation_bias=b["interpolation_bias"],
                    knn_new_family_bias=b["new_family_bias"],
                    test_seen_family_fraction=b["test_seen_family_fraction"],
                )
            )
        deployment.append(
            dict(
                method=method,
                instances=len(values),
                interpolation_ranking_agreement=sum(
                    v["estimated_difference"] * v["interpolation_difference"] > 0
                    for v in values
                ),
                new_family_ranking_agreement=sum(
                    v["estimated_difference"] * v["new_family_difference"] > 0
                    for v in values
                ),
                knn_interpolation_bias=float(
                    np.mean([v["knn_interpolation_bias"] for v in values])
                ),
                knn_new_family_bias=float(
                    np.mean([v["knn_new_family_bias"] for v in values])
                ),
                values=values,
            )
        )
    fidelity = []
    for path in (ROOT / "fidelity/run/rows").glob("*.json"):
        row = json.loads(path.read_text())
        if (
            row["manifest_sha256"] != sha(ROOT / "fidelity/run/manifest.json")
            or len(row["rows"]) != 50
        ):
            raise AssertionError("fidelity row provenance mismatch")
        fidelity.extend(
            dict(dataset=row["dataset"], seed=row["seed"], **r) for r in row["rows"]
        )
    if len(fidelity) != 900:
        raise RuntimeError("incomplete current-version fidelity study")
    curves = [
        dict(
            power=p,
            rank=r,
            comparisons=sum(x["power"] == p and x["rank"] == r for x in fidelity),
            max_error=max(
                x["abs_cut_error_over_mass"]
                for x in fidelity
                if x["power"] == p and x["rank"] == r
            ),
        )
        for p in [1, 3]
        for r in [32, 64, 128, 256, 512]
    ]
    frontier = []
    for case in [r for r in rows if r["study"] == "frontier"]:
        for r in case["rows"]:
            if r["method"] == "native_graph":
                continue
            frontier.append(
                dict(
                    dataset=case["dataset"],
                    seed=case["seed"],
                    method=r["method"],
                    degree=case.get("degree"),
                    cluster_policy=case.get("cluster_policy"),
                    solver_seconds=case.get("solver_seconds"),
                    seconds=r.get("seconds", case["seconds"]),
                    global_relative=r["scores"]["global_ratio"]
                    / expected_cross(r["counts"]),
                    mean_nearest=r["scores"]["mean_nearest_training"],
                    counts=r["counts"],
                    parent_case=case["name"],
                )
            )
    return dict(
        expected=199,
        successful=len(rows),
        native_infeasible=sum(x["status"] != "ok" for x in outcomes),
        completed_by_study=dict(Counter(r["study"] for r in rows)),
        grouping=group_summary,
        exact=exact_summary,
        deployment=deployment,
        fidelity=curves,
        frontier=frontier,
        outcomes=outcomes,
        units=dict(
            global_relative="cross-fold kernel mass / total off-diagonal mass / analytic random fraction at actual counts",
            local="mean maximum test-to-training Tanimoto similarity, unnormalized",
        ),
    )


def independent_scores(rows):
    comparisons, worst_global, worst_nearest = 0, 0.0, 0.0
    datasets = sorted(
        {r["dataset"] for r in rows if r["study"] in ["groups", "frontier"]}
    )
    for dataset in datasets:
        X = np.load(CACHE / f"{dataset}.npz")["X"]
        if not np.isfinite(X).all() or not np.all((X == 0) | (X == 1)):
            raise AssertionError("reference requires finite binary input")
        dot = X @ X.T
        sums = X.sum(1)
        den = sums[:, None] + sums[None, :] - dot
        S = np.divide(dot, den, out=np.zeros_like(den, dtype=np.float64), where=den > 0)
        del dot, den
        np.fill_diagonal(S, 0)
        if not np.isfinite(S).all():
            raise AssertionError("nonfinite independent reference")
        total = float(S.sum(dtype=np.float64))
        for case in rows:
            if case.get("dataset") != dataset or case["study"] not in [
                "groups",
                "frontier",
            ]:
                continue
            directory = RUN / "assignments" / case["name"]
            for r in case["rows"]:
                method = r["method"]
                name = {
                    "feasible_groups": "feasible",
                    "native_graph": "native_graph",
                    "repaired_graph": "repaired_graph",
                }.get(method, method)
                lab = np.load(directory / f"{name}.npy")
                if len(lab) != len(X):
                    raise AssertionError("row retention violation")
                within = sum(
                    S[np.ix_(lab == c, lab == c)].sum(dtype=np.float64)
                    for c in range(3)
                )
                value = (total - float(within)) / total
                nearest = float(S[np.ix_(lab == 2, lab == 0)].max(1).mean())
                dg = abs(value - r["scores"]["global_ratio"])
                dn = abs(nearest - r["scores"]["mean_nearest_training"])
                if not np.isfinite([value, nearest, dg, dn]).all():
                    raise AssertionError("nonfinite reference/score comparison")
                worst_global = max(worst_global, dg)
                worst_nearest = max(worst_nearest, dn)
                comparisons += 1
                if dg > 1e-6 or dn > 1e-6:
                    raise AssertionError("independent NumPy score disagreement")
        print("independently scored", dataset, flush=True)
    return dict(
        comparisons=comparisons,
        max_global_discrepancy=worst_global,
        max_nearest_discrepancy=worst_nearest,
        reference="Independent full NumPy binary Tanimoto; float64 division/sums; self excluded; all test queries against every training row.",
    )


if __name__ == "__main__":
    manifest, rows, outcomes = verified_rows()
    summary = summarize(rows, outcomes)
    write(ROOT / "analysis/summary.json", summary)
    validation = independent_scores(rows)
    write(ROOT / "analysis/independent_validation.json", validation)
    write(
        ROOT / "analysis/provenance.json",
        dict(
            manifest_sha256=sha(RUN / "manifest.json"),
            report_sha256=sha(__file__),
            producer_commit="f7e0866",
            summary_sha256=sha(ROOT / "analysis/summary.json"),
            independent_validation_sha256=sha(ROOT / "analysis/independent_validation.json"),
            observations=len(outcomes),
            successes=len(rows),
            native_infeasible=len(outcomes) - len(rows),
        ),
    )
    print(json.dumps(validation), flush=True)
