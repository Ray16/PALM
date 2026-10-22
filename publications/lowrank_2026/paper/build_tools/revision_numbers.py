"""Generate numeric inputs and supplement tables; manuscript prose is handwritten."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
D = PAPER / "revision_20261021"


def main():
    external = pd.read_csv(D / "external_summary.csv").set_index(["method", "condition", "model"])
    if not (external.assays == 10).all():
        raise ValueError("require all ten external assays")
    f = pd.read_csv(D / "powered_fidelity/rows.csv")
    s = pd.read_csv(D / "metric_validation/synthetic_correlations.csv").set_index(["regime", "proxy"])
    rows = pd.read_csv(D / "external_rows.csv")
    commands = {}
    for method, name in [("combined", "Combined"), ("plain", "Plain"), ("random", "Random"), ("published_simpd", "Simpd")]:
        for condition, cn in [("original", ""), ("label_balanced", "Balanced")]:
            for model in ("rf", "xgb"):
                commands["Ext" + name + cn + model.upper()] = f"{external.loc[(method, condition, model), 'auc']:.3f}"
    commands["PlainFidelityError"] = f"{f[(f.power == 1) & (f['rank'] == 256)].abs_cut_error_over_mass.max():.4f}"
    commands["PowerFidelityError"] = f"{f[(f.power == 3) & (f['rank'] == 256)].abs_cut_error_over_mass.max():.3f}"
    commands["PowerHighRankError"] = f"{f[(f.power == 3) & (f['rank'] == 512)].abs_cut_error_over_mass.max():.3f}"
    commands["PowerOptimizedError"] = f"{f[(f.power == 3) & (f['rank'] == 256) & (f.alpha == 1)].abs_cut_error_over_mass.max():.3f}"
    quality = pd.read_csv(D / "powered_fidelity/quality.csv")
    for rank, name in [(32, "Small"), (256, "Default"), (512, "Large")]:
        commands["BaceRank" + name] = f"{quality[(quality.dataset == 'moleculenet_bace') & (quality.power == 3) & (quality['rank'] == rank)].nnl_rel.mean():.3f}"
    per_molecule = pd.read_csv(D / "metric_validation/per_molecule_correlations.csv")
    for dataset, name in [("esol", "Esol"), ("freesolv", "Freesolv"), ("lipophilicity", "Lipo"), ("bbbp", "Bbbp")]:
        cell = per_molecule[(per_molecule.dataset == dataset) & (per_molecule.reference == "train") &
                            (per_molecule.split == "pooled") & (per_molecule.proxy == "nn_similarity")]
        commands["Local" + name + "Rho"] = f"{cell.rho.iloc[0]:+.3f}"
    cell = per_molecule[(per_molecule.dataset == "lipophilicity") & (per_molecule.reference == "train") &
                        (per_molecule.split == "pooled") & (per_molecule.proxy == "per_point_global")]
    commands["GlobalLipoRho"] = f"{cell.rho.iloc[0]:+.3f}"
    commands["SyntheticLocalRho"] = f"{s.loc[('pooled','nnl_rel'),'rho']:.3f}"
    commands["SyntheticGlobalRho"] = f"{s.loc[('pooled','leak_ratio'),'rho']:.3f}"
    commands["SyntheticCount"] = str(int(s.loc[("pooled", "nnl_rel"), "n"]))
    commands["ExternalMoved"] = f"{100 * rows[(rows.condition == 'label_balanced') & (rows.method == 'combined')].moved_fraction.median():.1f}"
    for condition, name in [("original", "Original"), ("label_balanced", "Balanced")]:
        commands["ExternalShift" + name] = f"{external.loc[('combined', condition, 'rf'), 'y_shift']:.3f}"
    (PAPER / "revision_numbers.tex").write_text("% Generated from completed revision evidence.\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + value + "}" for name, value in commands.items()) + "\n")
    # Normalized spectral estimates remain clearly distinguished from cut errors.
    summary = f.groupby(["power", "rank"]).agg(
        max_cut_error=("abs_cut_error_over_mass", "max"),
        median_delta=("numerical_delta_over_mass", "median"),
        max_delta=("numerical_delta_over_mass", "max")).reset_index()
    summary.to_csv(D / "powered_fidelity_summary.csv", index=False)
    tables = [r"\begin{table}[htbp]\centering\small",
        summary.rename(columns={"power": "Power", "rank": "Rank", "max_cut_error": "Max error / mass",
            "median_delta": "Median bound / mass", "max_delta": "Max bound / mass"}).to_latex(escape=True,index=False, float_format="%.4f"),
        r"\caption{Six full development datasets, three seeds and five partition perturbations per factor. Numerical spectral estimates use a symmetric eigensolver and are not certified upper bounds. Values above one give a vacuous bound on a cut ratio.}\end{table}"]
    per = pd.read_csv(D / "external_per_assay.csv")
    for model in ("rf", "xgb"):
        pivot = per[per.model == model].pivot(index="dataset", columns=["method", "condition"], values="auc")
        t = pd.DataFrame({"Assay": pivot.index,
            "Random": pivot["random", "original"].values, "SIMPD": pivot["published_simpd", "original"].values,
            "Plain": pivot["plain", "original"].values, "Combined": pivot["combined", "original"].values,
            "Balanced": pivot["combined", "label_balanced"].values})
        t["Assay"] = t.Assay.str.replace("CHEMBL", "", regex=False)
        tables += [r"\begin{table}[htbp]\centering\small", t.to_latex(escape=True,index=False, float_format="%.3f"),
            "\\caption{" + model.upper() + " mean test ROC-AUC by assay. Balanced refers to the combined partition after the target-aware intervention; it is not the label-blind method.}\\end{table}"]
    rates = rows[rows.condition == "original"].groupby(["dataset", "method"])[["train_positive_rate", "test_positive_rate"]].mean()
    rates.to_csv(D / "external_positive_rates.csv")
    for method in ("random", "published_simpd", "plain", "combined"):
        t = rates.xs(method, level="method").reset_index()
        t["dataset"] = t.dataset.str.replace("CHEMBL", "", regex=False)
        tables += [r"\begin{table}[htbp]\centering\small", t.rename(columns={"dataset": "Assay",
            "train_positive_rate": "Train positive rate", "test_positive_rate": "Test positive rate"}).to_latex(escape=True,index=False, float_format="%.3f"),
            "\\caption{Class proportions for " + method.replace("_", " ") + "; averages over split seeds.}\\end{table}"]
    (PAPER / "revision_tables.tex").write_text("\n".join(tables) + "\n")
    (D / "number_sources.json").write_text(json.dumps(dict(commands=commands,
        sources=["external_summary.csv", "external_rows.csv", "external_per_assay.csv",
                 "powered_fidelity/rows.csv", "powered_fidelity/quality.csv",
                 "metric_validation/synthetic_correlations.csv", "metric_validation/per_molecule_correlations.csv"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
