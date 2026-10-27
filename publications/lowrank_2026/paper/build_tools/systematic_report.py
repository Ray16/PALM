"""Generate current-version manuscript numbers/figures from verified observations.

Primary metrics are count-normalized global overlap and raw mean nearest-training
similarity. Archival v0.1 thresholds/normalizations stay in the archive sections.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
REVISION = PAPER.parent / "revision_20261026"
FIG = PAPER / "figures/systematic"
NAMES = {
    "moleculenet_" + s: n
    for s, n in [
        ("freesolv", "FreeSolv"),
        ("esol", "ESOL"),
        ("bace", "BACE"),
        ("bbbp", "BBBP"),
        ("lipophilicity", "Lipophilicity"),
        ("tox21", "Tox21"),
    ]
}


def balanced_mean(values, field):
    by = defaultdict(list)
    for row in values:
        by[row["dataset"]].append(row[field])
    return float(np.mean([np.mean(v) for v in by.values()]))


def longtable(name, headers, rows, caption):
    lines = [
        r"\begin{longtable}{l" + "r" * (len(headers) - 1) + "}",
        r"\caption{" + caption + r"}\\",
        r"\toprule",
        " & ".join(headers) + r"\\\midrule",
        r"\endfirsthead",
        r"\multicolumn{"
        + str(len(headers))
        + r"}{l}{\tablename~\thetable\ (continued)}\\",
        r"\toprule",
        " & ".join(headers) + r"\\\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(len(headers)) + r"}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    lines += [" & ".join(map(str, row)) + r"\\" for row in rows]
    lines += [r"\end{longtable}"]
    (PAPER / name).write_text("\n".join(lines) + "\n")


def supporting_tables(summary, scale_rows, scale_outcomes, large_rows):
    native = defaultdict(dict)
    for row in summary["frontier"]:
        if row["method"] in ["datasail", "matched_counts", "matched_native_groups"]:
            native[row["parent_case"]][row["method"]] = row
    for field, suffix, title in [
        (
            "global_relative",
            "global",
            "Global overlap relative to count-matched random",
        ),
        ("mean_nearest", "local", "Mean maximum test-to-training similarity"),
    ]:
        values = []
        for name, methods in sorted(native.items()):
            source = json.loads((REVISION / "run/rows" / (name + ".json")).read_text())
            n = methods["datasail"]
            values.append(
                [
                    NAMES[n["dataset"]],
                    source["cluster_count"],
                    n["solver_seconds"],
                    f"{methods['datasail'][field]:.3f}",
                    f"{methods['matched_counts'][field]:.3f}",
                    f"{methods['matched_native_groups'][field]:.3f}",
                ]
            )
        longtable(
            "systematic_native_" + suffix + ".tex",
            ["Dataset", "$K$", "Budget (s)", "DataSAIL", "Counts", "Groups"],
            values,
            title
            + ". Both low-rank controls use exact native counts. The Groups control also preserves every group passed to DataSAIL's solver. Equal groups do not imply identical coarsened and entity-level objectives. One native solve is recorded per configuration; these configurations are not independent datasets.",
        )
    values = [
        [
            "Pairs" if r["grouped"] else "None",
            r["power"],
            r["rank"],
            f"{r['max_approximation_error']:.4f}",
            f"{r['max_optimization_gap']:.4f}",
            f"{r['max_true_gap']:.4f}",
        ]
        for r in summary["exact"]
    ]
    longtable(
        "systematic_exact_table.tex",
        ["Groups", "Power", "Landmarks", r"$\delta_{\rm enum}$", r"$\eta$", "True gap"],
        values,
        r"Exact small-instance references. Each value is the maximum over twenty 12-entity instances, divided by the true kernel's off-diagonal mass. Here $\delta_{\rm enum}$ is the maximum cut-value error over the enumerated feasible set, $\eta$ is heuristic suboptimality for the factor objective, and True gap is suboptimality for the original kernel. Counts are 8/2/2; paired groups have two entities each.",
    )
    values = []
    by_name = {r["name"]: r for r in scale_rows}
    for outcome in scale_outcomes:
        row = by_name.get(outcome["name"])
        if row:
            interval = row["fidelity"]["relative_pair_mae"]["ci95"]
            ci = (
                f"[{interval[0]:.4f}, {interval[1]:.4f}]"
                if interval
                else "Unestablished"
            )
            values.append(
                [
                    "2/Plain" if row["method"] == "plain" else "2/Combined",
                    f"{row['n']:,}",
                    row["epsilon"],
                    row["seed"],
                    f"{row['seconds']:.1f}",
                    f"{row['peak_split_rss_gib']:.2f}",
                    ci,
                ]
            )
        else:
            values.append(
                [
                    "2/Plain" if outcome["method"] == "plain" else "2/Combined",
                    f"{outcome['n']:,}",
                    outcome["epsilon"],
                    outcome["seed"],
                    "--",
                    "--",
                    "Case budget",
                ]
            )
    for row in large_rows:
        interval = row["fidelity"]["relative_pair_mae"]["ci95"]
        values.append(
            [
                "32/Plain",
                f"{row['n']:,}",
                row["epsilon"],
                row["seed"],
                f"{row['seconds']:.1f}",
                f"{row['peak_split_rss_gib']:.2f}",
                f"[{interval[0]:.4f}, {interval[1]:.4f}]"
                if interval
                else "Unestablished",
            ]
        )
    longtable(
        "systematic_scaling_table.tex",
        [
            "Cores/setting",
            "Entities",
            "Tolerance",
            "Seed",
            "Call (s)",
            "RSS (GiB)",
            "Cut-error interval",
        ],
        values,
        "Current CPU scaling. Two-core cases run on lambda2; 32-core cases run on lambda13 and are a separate hardware protocol. Call time excludes feature loading, warm-up and independent scoring. RSS is the fresh-process peak immediately after splitting, including loaded input/features and framework overhead. Cut-error intervals concern the returned partition and use total powered-kernel mass. A case-budget outcome supplies no imputed split-call time or memory.",
    )
    checks = [
        json.loads(p.read_text())
        for p in (REVISION / "publication_checks/run/rows").glob("*.json")
    ]
    controls = [
        dict(r, policy_name="Fixed " + ("2k" if r["pairs"] == 2048 else "32k"))
        for r in checks
        if r["study"] == "equal_confidence"
    ]
    for p in (REVISION / "run/rows").glob("budget*.json"):
        r = json.loads(p.read_text())
        if r["policy"] == "adaptive":
            controls.append(dict(r, policy_name="Joint 2--32k"))
    values = []
    for dataset in ["moleculenet_freesolv", "moleculenet_bace", "moleculenet_qm8"]:
        for policy in ["Fixed 2k", "Fixed 32k", "Joint 2--32k"]:
            rows = [
                r
                for r in controls
                if r["dataset"] == dataset and r["policy_name"] == policy
            ]
            if len(rows) != 3:
                continue
            ranks = [r["diagnostics"]["factorization"]["effective_rank"] for r in rows]
            passed = sum(
                r["diagnostics"]["factorization"]["selection_tolerance_met"]
                and r["diagnostics"]["factorization"]["audit_tolerance_met"]
                for r in rows
            )
            values.append(
                [
                    NAMES.get(dataset, "QM8"),
                    policy,
                    f"{np.median([r['seconds'] for r in rows]):.2f}",
                    f"{min(ranks)}--{max(ranks)}",
                    f"{passed}/3",
                ]
            )
    longtable(
        "systematic_accuracy_table.tex",
        ["Dataset", "Probe policy", "Median call (s)", "Ranks", "Established"],
        values,
        "Equal-confidence accuracy/cost controls. Every policy uses the same 95 percent joint selection/final-audit coverage, cut tolerance 0.05, rank range 32--512, two refined candidates and exact counts. A fixed policy is a single-size probe grid, not the older policy with separate unadjusted 95 percent checks. Established requires both selection and independent final audit to establish tolerance. These datasets are development evidence; no model outcome selects a policy.",
    )
    values = []
    for row in sorted(
        [r for r in checks if r["study"] == "memory"],
        key=lambda r: (
            r["dataset"],
            r["method"],
            r.get("degree", 0),
            r.get("cluster_policy", ""),
            r.get("solver_seconds", 0),
        ),
    ):
        method = {
            "plain": "Plain",
            "combined": "Combined",
            "graph": "Graph",
            "datasail": "DataSAIL",
        }[row["method"]]
        configuration = str(row.get("degree", row.get("cluster_count", "--")))
        if row["method"] == "datasail":
            configuration = row["cluster_policy"] + "/" + str(row["solver_seconds"])
        if row["status"] == "native_infeasible":
            values.append(
                [NAMES[row["dataset"]], method, configuration, "--", "Infeasible"]
            )
        else:
            values.append(
                [
                    NAMES[row["dataset"]],
                    method,
                    configuration,
                    f"{row['seconds']:.2f}",
                    f"{row['native_peak_rss_before_scoring_mb'] / 1024:.3f}",
                ]
            )
    longtable(
        "systematic_memory_table.tex",
        ["Dataset", "Method", "Configuration", "Call (s)", "RSS (GiB)"],
        values,
        "Fresh-process native memory profiles at seed zero. Memory is captured before the first scoring call or any matched controls and includes framework/input costs and the common small warm-up. Graph configuration is degree; DataSAIL configuration is cluster policy/solver seconds. Native profiling repeats can differ from a time-limited earlier solve; their own assignments and quality scores are retained. Solver-reported infeasibility has no imputed timing or memory.",
    )


def figures(summary, helper):
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    for power, color in [(1, "#0072B2"), (3, "#D55E00")]:
        rows = [r for r in summary["fidelity"] if r["power"] == power]
        ax.plot(
            [r["rank"] for r in rows],
            [r["max_error"] for r in rows],
            "o-",
            color=color,
            label=f"Kernel power {power}",
            ms=8,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
    ax.set(xlabel="Landmark count", ylabel="Cut error / kernel mass", ylim=(0, 0.25))
    ax.set_yticks([0, 0.1, 0.2])
    ax.grid(alpha=0.2)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    helper.save(fig, "fidelity")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.3), layout="constrained")
    methods = [
        ("plain", None, "Plain", "#0072B2", "o"),
        ("combined", None, "Combined", "#D55E00", "D"),
        ("repaired_graph", 16, "Graph 16", "#66A61E", "^"),
        ("repaired_graph", 32, "Graph 32", "#009E73", "s"),
        ("repaired_graph", 128, "Graph 128", "#006D2C", "v"),
    ]
    datasets = list(NAMES)
    for ax, field, title in zip(
        axes,
        ["global_relative", "mean_nearest"],
        ["a  Dense global overlap", "b  Nearest-training similarity"],
    ):
        for index, (method, degree, label, color, marker) in enumerate(methods):
            for y, dataset in enumerate(datasets):
                values = [
                    r[field]
                    for r in summary["frontier"]
                    if r["dataset"] == dataset
                    and r["method"] == method
                    and r["degree"] == degree
                ]
                if len(values) != 3:
                    raise ValueError("incomplete three-seed primary comparison")
                ax.errorbar(
                    np.mean(values),
                    y + (index - 2) * 0.11,
                    xerr=np.std(values, ddof=1),
                    fmt=marker,
                    color=color,
                    ms=6,
                    capsize=3,
                    label=label if y == 0 else None,
                )
        ax.set_yticks(range(len(datasets)), [NAMES[d] for d in datasets])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Overlap / count-matched random")
    axes[0].set_xlim(0.2, 1.05)
    axes[0].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].set_xlabel("Mean maximum similarity")
    axes[1].set_xlim(0.05, 0.85)
    axes[1].set_xticks([0.2, 0.4, 0.6, 0.8])
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    helper.save(fig, "quality")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.1), layout="constrained")
    methods = ["random", "plain", "combined", "oracle_family_holdout"]
    labels = ["Random", "Plain", "Combined", "Family\nreference"]
    for field, color, offset, label in [
        ("interpolation", "#0072B2", -0.08, "Known families"),
        ("new_family", "#D55E00", 0.08, "New families"),
    ]:
        for x, method in enumerate(methods):
            row = next(r for r in summary["deployment"] if r["method"] == method)
            fraction = row[field + "_ranking_agreement"] / row["instances"]
            axes[0].plot(
                x + offset,
                fraction,
                "o",
                color=color,
                ms=8,
                label=label if x == 0 else None,
            )
            values = np.array([r["knn_" + field + "_bias"] for r in row["values"]])
            boot = (
                np.random.default_rng(31415)
                .choice(values, (10000, len(values)))
                .mean(1)
            )
            lo, hi = np.quantile(boot, [0.025, 0.975])
            mean = values.mean()
            axes[1].errorbar(
                x + offset,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=color,
                ms=8,
                capsize=4,
            )
    for ax in axes:
        ax.set_xticks(range(4), labels)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(
        title="a  Model-ranking agreement",
        ylabel="Agreement",
        ylim=(-0.05, 1.05),
    )
    axes[0].set_yticks([0, 0.5, 1.0])
    axes[1].set(title="b  Family-model error estimate", ylabel="RMSE bias")
    axes[1].axhline(0, color="gray", ls=":", lw=1)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    helper.save(fig, "deployment")


def main(args):
    summary = json.loads((REVISION / "analysis/summary.json").read_text())
    verified = REVISION / "analysis/independent_validation.json"
    if not verified.exists() and not args.allow_incomplete:
        raise ValueError("independent full-kernel validation not complete")
    unmet = []
    checks = REVISION / "publication_checks/run"
    if (
        not (checks / "progress.json").exists()
        or json.loads((checks / "progress.json").read_text()).get("completed") != 72
    ):
        unmet.append("equal-confidence controls and native memory profiles")
    scaling = REVISION / "scaling/cpu_run"
    manifest = json.loads((scaling / "manifest.json").read_text())
    scale_rows = []
    scale_outcomes = []
    for job in manifest["jobs"]:
        p = scaling / job["name"] / "result.json"
        failure = scaling / "failures" / f"{job['name']}.json"
        if p.exists():
            scale_rows.append(json.loads(p.read_text()))
            scale_outcomes.append(dict(job, status="ok"))
        elif failure.exists():
            scale_outcomes.append(
                dict(
                    job,
                    status="end_to_end_budget_failure",
                    failure=json.loads(failure.read_text()),
                )
            )
        else:
            unmet.append("unfinished scaling case " + job["name"])
    large = REVISION / "scaling/large_cpu_run"
    large_rows = sorted(
        [json.loads(p.read_text()) for p in large.glob("*/result.json")],
        key=lambda row: (row["method"], row["n"], row["epsilon"], row["seed"]),
    )
    if (
        len(large_rows) != 3
        or not (large / "progress.json").exists()
        or json.loads((large / "progress.json").read_text()).get("completed") != 3
    ):
        unmet.append("largest-scale independent audits")
    if unmet and not args.allow_incomplete:
        raise ValueError("Unfinished publication evidence: " + ", ".join(unmet))
    FIG.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "revision_figures", PAPER / "build_tools/revision_figures.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.FIG = FIG
    helper.concept()
    figures(summary, helper)
    fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
    for epsilon, color, label in [
        (0.0, "#0072B2", "2 cores, exact"),
        (0.01, "#D55E00", "2 cores, 1%"),
    ]:
        values = [
            r for r in scale_rows if r["method"] == "plain" and r["epsilon"] == epsilon
        ]
        ns = sorted({r["n"] for r in values})
        ys = [np.median([r["seconds"] for r in values if r["n"] == n]) for n in ns]
        ax.plot(ns, ys, "o-", color=color, label=label, ms=8)
    if large_rows:
        ax.plot(
            [2000000],
            [np.median([r["seconds"] for r in large_rows])],
            "D",
            color="#CC79A7",
            label="32 cores, 1%",
            ms=9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 2000)
    ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1k"])
    ax.set(xlabel="Molecules", ylabel="Split time (s)")
    ns = sorted(
        {r["n"] for r in scale_rows if r["method"] == "plain"}
        | {r["n"] for r in large_rows}
    )
    ax.set_xticks(
        ns, [f"{n // 1000}k" if n < 1000000 else f"{n // 1000000}M" for n in ns]
    )
    ax.grid(alpha=0.2)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    helper.save(fig, "cpu_scaling")
    macros = {
        "CurrentCases": str(summary["successful"]),
        "CurrentInfeasible": str(summary["native_infeasible"]),
    }
    if large_rows:
        macros["CurrentLargeCpuSeconds"] = (
            f"{np.median([r['seconds'] for r in large_rows]):.1f}"
        )
        macros["CurrentLargeCpuMinutes"] = (
            f"{np.median([r['seconds'] for r in large_rows]) / 60:.1f}"
        )
        macros["CurrentLargeCpuMemory"] = (
            f"{max(r['peak_split_rss_gib'] for r in large_rows):.1f}"
        )
        macros["CurrentLargeNearest"] = (
            f"{np.mean([r['local']['nearest_mean'] for r in large_rows]):.3f}"
        )
    for power, prefix in [(1, "Plain"), (3, "Power")]:
        for rank, suffix in [(256, "Fidelity"), (512, "HighRankFidelity")]:
            macros["Current" + prefix + suffix] = (
                f"{next(r['max_error'] for r in summary['fidelity'] if r['power'] == power and r['rank'] == rank):.4f}"
            )
    for method, prefix in [
        ("plain", "Plain"),
        ("combined", "Combined"),
        ("repaired_graph", "Graph"),
    ]:
        values = [
            r
            for r in summary["frontier"]
            if r["method"] == method and r["degree"] in [None, 32]
        ]
        macros["Current" + prefix + "Global"] = (
            f"{balanced_mean(values, 'global_relative'):.3f}"
        )
        macros["Current" + prefix + "Nearest"] = (
            f"{balanced_mean(values, 'mean_nearest'):.3f}"
        )
    for method, prefix in [
        ("datasail", "Native"),
        ("matched_counts", "Matched"),
        ("matched_native_groups", "MatchedGroups"),
    ]:
        values = [r for r in summary["frontier"] if r["method"] == method]
        macros["Current" + prefix + "Global"] = (
            f"{balanced_mean(values, 'global_relative'):.3f}"
        )
        macros["Current" + prefix + "Nearest"] = (
            f"{balanced_mean(values, 'mean_nearest'):.3f}"
        )
    for epsilon, prefix in [(0.0, "Exact"), (0.01, "Corridor")]:
        values = [
            r["seconds"]
            for r in scale_rows
            if r["method"] == "plain" and r["n"] == 100000 and r["epsilon"] == epsilon
        ]
        if len(values) != 3:
            raise ValueError("100k capacity comparison incomplete")
        macros["Current" + prefix + "CpuSeconds"] = f"{np.median(values):.1f}"
    (PAPER / "systematic_numbers.tex").write_text(
        "% Generated from verified current-version observations.\n"
        + "".join("\\newcommand{\\" + k + "}{" + v + "}\n" for k, v in macros.items())
    )
    table = [
        "\\begin{table}[htbp]\\centering",
        "\\begin{tabular}{lrr}\\toprule",
        "Assignment & Global overlap / random & Mean nearest similarity\\\\\\midrule",
    ]
    for method, name in [
        ("feasible_groups", "Feasible groups"),
        ("optimized_p1", "Groups + ordinary factor"),
        ("optimized_p3", "Groups + cubed factor"),
    ]:
        r = summary["grouping"][method]
        table.append(
            f"{name} & {r['global_relative_mean']:.3f} & {r['mean_nearest']:.3f}\\\\"
        )
    table += [
        "\\bottomrule\\end{tabular}",
        r"\caption{Optimization beyond grouping. Every comparison retains the same entities, constructed groups and exact 80/10/10 counts. Values average six dataset means after averaging three seeds. The factor optimizer starts from the feasible control and preserves every group. Both optimized settings improve both measures on all six datasets. Lower values indicate less overlap.}",
        r"\label{tab:matched-groups}\end{table}",
    ]
    (PAPER / "systematic_group_table.tex").write_text("\n".join(table) + "\n")
    supporting_tables(summary, scale_rows, scale_outcomes, large_rows)
    (REVISION / "analysis/figure_audit.json").write_text(
        json.dumps(helper.AUDIT, indent=2) + "\n"
    )
    (REVISION / "analysis/build_evidence.json").write_text(
        json.dumps(
            dict(
                preliminary=bool(unmet),
                unmet=unmet,
                primary_successes=summary["successful"],
                native_infeasible=summary["native_infeasible"],
                scaling_outcomes=scale_outcomes,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "generated current-version figures and numbers; preliminary=",
        bool(unmet),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--allow-incomplete", action="store_true")
    main(p.parse_args())
