"""Validate the prespecified robustness grid and derive paired dataset summaries."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PAPER = Path(__file__).resolve().parents[1]
D = PAPER / "revision_20261021"


def main():
    records = [json.loads(p.read_text()) for p in sorted((D / "robustness").glob("*/result.json"))]
    issues = []
    if len(records) != 6:
        issues.append("require six completed full datasets")
    for record in records:
        if len(record["rows"]) != 51:
            issues.append(record["dataset"] + ": expected 17 settings by three seeds")
        for row in record["rows"]:
            path = D / "robustness" / record["dataset"] / f"{row['setting']}__s{row['seed']}.npy"
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != row["assignment_sha256"]:
                issues.append(str(path) + ": missing/changed assignment")
                continue
            labels = np.load(path)
            if len(labels) != record["n"] or set(labels) != {"train", "val", "test"}:
                issues.append(str(path) + ": invalid row/fold coverage")
    if issues:
        (D / "robustness_validation.json").write_text(json.dumps(dict(passed=False, issues=issues), indent=2)+'\n')
        raise ValueError(issues)
    data = pd.DataFrame([row for r in records for row in r["rows"]])
    plain = data[data.setting == "plain"].groupby("dataset").mean(numeric_only=True)
    combined = data[(data.power == 3) & np.isclose(data.tau, .4)].groupby("dataset").mean(numeric_only=True)
    definitions = [("morgan_nnl_rel", "Morgan ramp 0.4--0.8"),
        ("morgan_nnl_03_07_rel", "Morgan ramp 0.3--0.7"),
        ("morgan_nnl_05_09_rel", "Morgan ramp 0.5--0.9"),
        ("morgan_top5_ramp_rel", "Morgan top-five ramp"),
        ("morgan_frac_gt06", "Morgan fraction above 0.6"),
        ("morgan_frac_gt08", "Morgan fraction above 0.8"),
        ("maccs_nn_mean", "MACCS nearest similarity"),
        ("maccs_nnl_rel", "MACCS ramp 0.4--0.8")]
    rows = []
    for key, label in definitions:
        delta = (combined[key] - plain[key]).to_numpy()
        bootstrap = np.random.default_rng(31415).choice(delta, size=(10000, len(delta))).mean(1)
        lo, hi = np.quantile(bootstrap, [.025, .975])
        wins, losses = int(np.sum(delta < -1e-12)), int(np.sum(delta > 1e-12))
        n = wins + losses
        rows.append(dict(metric=key, label=label, mean_difference=float(delta.mean()),
            ci95_low=float(lo), ci95_high=float(hi), wins=wins, losses=losses,
            ties=len(delta)-n, n_datasets=len(delta),
            sign_test_p=float(binomtest(wins, n, .5).pvalue) if n else 1.))
    summary = pd.DataFrame(rows)
    summary.to_csv(D / "robustness_summary.csv", index=False)
    combined.join(plain, lsuffix="_combined", rsuffix="_plain").to_csv(D / "robustness_paired_datasets.csv")
    data[data.tau.notna() & data.power.isin([2, 3, 4])].groupby(["power", "tau", "dataset"])[["morgan_global_ratio", "morgan_nnl_rel"]].mean().groupby(["power", "tau"]).mean().to_csv(D / "robustness_frontier.csv")
    table = pd.DataFrame({"Metric": summary.label,
        "Mean change": summary.mean_difference,
        "95% interval": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(summary.ci95_low, summary.ci95_high)],
        "Wins/ties": [f"{w}/{t}" for w, t in zip(summary.wins, summary.ties)], "Sign p": summary.sign_test_p})
    body = table.to_latex(index=False, escape=True, float_format="%.3f")
    (PAPER / "robustness_table.tex").write_text("\\begin{table}[htbp]\\centering\\small\n" + body +
        "\\caption{Combined minus plain on six full datasets after averaging three split seeds. "
        "Intervals are descriptive dataset bootstraps. Sign-test p-values are two-sided, exclude ties, "
        "and are unadjusted across correlated exploratory diagnostics. MACCS cutoffs are a sensitivity "
        "check, not a universal scale of chemical relatedness.}\\end{table}\n")
    values = summary.set_index("metric")
    commands = {"RobustMorganChange": f"{values.loc['morgan_nnl_rel','mean_difference']:+.3f}",
        "RobustMaccsChange": f"{values.loc['maccs_nn_mean','mean_difference']:+.3f}",
        "RobustMaccsLow": f"{values.loc['maccs_nn_mean','ci95_low']:+.3f}",
        "RobustMaccsHigh": f"{values.loc['maccs_nn_mean','ci95_high']:+.3f}",
        "RobustMaccsWins": str(int(values.loc['maccs_nn_mean','wins']))}
    (PAPER / "robustness_numbers.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in commands.items())+'\n')
    (D / "robustness_validation.json").write_text(json.dumps(dict(passed=True, issues=[], datasets=6,
        rows=len(data), settings_per_dataset=17, seeds=3,
        validation_scope="prespecified completed grid, assignment hashes and row/fold coverage; paired diagnostics derived without supervised parameter selection"), indent=2)+'\n')


if __name__ == "__main__":
    main()
