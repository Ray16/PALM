"""Comparison bar charts for the CheMixHub chem-OOD splits.

Two figures, numbers straight from splits/leakage_report.json:
  1. L(pi) per task: random-group vs Butina (paper) vs hypergraph vs low-rank
     -> figures/chemixhub_lpi_comparison.png
  2. mixture-identity leakage: standard random split (paper Table 16) vs every
     whole-mixture chem-OOD split (== 0 by construction)
     -> figures/chemixhub_identity_leakage.png

    /homes/rzhu/miniforge3/envs/palm/bin/python make_comparison_chart.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# palette consistent with benchmarks/charts/make_benchmark_chart.py
C_RANDOM = "#9aa0a6"   # grey  — random baseline
C_BUTINA = "#f4a36c"   # orange — the paper's method
C_HYPER = "#3b6fb6"    # blue  — hypergraph
C_LOWRANK = "#5aa469"  # green — low-rank

SHORT = {
    ("ionic-liquids", "Electrical conductivity"): "IL cond",
    ("ionic-liquids", "Viscosity"): "IL visc",
    ("miscible-solvent", "Density"): "MS density",
    ("miscible-solvent", "Heat of vaporization"): "MS Hvap",
    ("miscible-solvent", "Enthalpy of mixing"): "MS Hmix",
    ("drug-solubility", "Log solubility"): "drug solub.",
    ("polymer-electrolyte", "log Conductivity"): "PE cond",
    ("olfactory-similarity", "Mixture similarity"): "olfactory",
    ("logV", "Log viscosity"): "logV",
    ("nist-logV", "Log viscosity"): "NIST logV",
    ("MON", "Motor octane number"): "MON",
    ("medicine-formulations", "solubility"): "medicine",
}


def load():
    recs = json.load(open(os.path.join(HERE, "splits", "leakage_report.json")))
    tasks, data = [], {}
    for r in recs:
        key = (r["dataset"], r["property"])
        if key not in data:
            data[key] = {}
            tasks.append(key)
        data[key][r["engine"]] = r
    return tasks, data


def plot_lpi(tasks, data):
    labels = [f"{SHORT[k]}\n(n={data[k]['lowrank']['n_samples']:,})" for k in tasks]
    x = np.arange(len(tasks))
    w = 0.20
    fig, ax = plt.subplots(figsize=(15, 6))

    def col(engine):
        return np.array([data[k].get(engine, {}).get("Lpi_unique_mixture", np.nan) for k in tasks],
                        dtype=float)

    ax.bar(x - 1.5 * w, col("random_group"), w, label="Random (group)", color=C_RANDOM)
    ax.bar(x - 0.5 * w, col("butina"), w, label="Butina — paper chem-OOD", color=C_BUTINA)
    ax.bar(x + 0.5 * w, col("hypergraph"), w, label="Hypergraph (PALM)", color=C_HYPER)
    ax.bar(x + 1.5 * w, col("lowrank"), w, label="Low-rank (PALM)", color=C_LOWRANK)

    # flag the degenerate Butina cluster (single cluster -> empty val/test)
    for i, k in enumerate(tasks):
        b = data[k].get("butina", {})
        if b.get("n_clusters", 99) <= 1:
            ax.text(x[i] - 0.5 * w, 0.01, "degenerate\n(1 cluster)", rotation=90, ha="center",
                    va="bottom", fontsize=6.5, color="#b5651d", fontweight="bold")

    ax.set_ylabel(r"$L(\pi)$  — residual chemical leakage (lower is better)", fontsize=12)
    ax.set_title("CheMixHub chem-OOD splits: PALM engines vs the paper's Butina split\n"
                 "(all four have 0% mixture-identity leakage; bars show remaining "
                 "cross-split chemical similarity)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, framealpha=0.95, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.08)
    fig.tight_layout()
    out = os.path.join(FIGS, "chemixhub_lpi_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def plot_identity(tasks, data):
    labels = [SHORT[k] for k in tasks]
    x = np.arange(len(tasks))
    w = 0.5
    fig, ax = plt.subplots(figsize=(15, 5))
    rand = np.array([data[k]["random_sample"]["mixture_identity_leakage"] * 100 for k in tasks])
    ax.bar(x, rand, w, color="#c0392b", label="Standard random split (paper Table 16)")
    ax.axhline(0, color="#333", lw=0.8)
    for i, v in enumerate(rand):
        ax.text(x[i], v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#c0392b")
    # the OOD splits all sit at 0
    ax.plot(x, np.zeros_like(x, dtype=float), "o", color=C_LOWRANK, ms=7,
            label="All chem-OOD splits (Butina / hypergraph / low-rank) = 0%")

    ax.set_ylabel("mixture-identity leakage\n(% test samples whose mixture is in train)", fontsize=11)
    ax.set_title("Why the split matters: a standard random split memorises mixture identity; "
                 "whole-mixture chem-OOD splits eliminate it", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-5, 112)
    ax.legend(fontsize=10, loc="center", bbox_to_anchor=(0.62, 0.72), framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGS, "chemixhub_identity_leakage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    tasks, data = load()
    plot_lpi(tasks, data)
    plot_identity(tasks, data)
