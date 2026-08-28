"""Demonstrate that the hand-picked featurizers are good choices.

We do NOT route per dataset (the sweep showed that isn't worth it). Instead we
hand-pick a safe, strong featurizer per entity type and *demonstrate* it from the
sweep data:

    molecule -> ECFP-1024      material -> MAGPIE
    exceptions (validated on leakage AND OOD): bace -> ChemBERTa, qmof -> mat2vec

For each entity type this makes one figure — a leakage-vs-OOD tradeoff scatter,
one point per (dataset, candidate feature) — and prints a summary table. The
hand-pick should sit in the good corner (low reference-space leakage, high held-out
metric); features that only *look* clean (low leakage, low/negative OOD) expose
themselves as bad picks.

    python -m PALM.benchmarks.master.demonstrate_features
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.join(RESULTS, "figures")
SWEEP = os.path.join(RESULTS, "feature_sweep.csv")

HAND_PICK = {"molecule": "ecfp1024", "material": "magpie"}
EXCEPTIONS = {"moleculenet_bace": "chemberta", "qmof": "mat2vec"}
COLORS = {"ecfp1024": "#0072B2", "maccs": "#E69F00", "rdkit_descriptors": "#009E73",
          "chemberta": "#CC79A7", "magpie": "#0072B2", "mat2vec": "#D55E00"}
OOD = ("hypergraph", "lowrank")


def _agg(sweep):
    d = pd.read_csv(sweep)
    d = d[d["status"] == "ok"].copy()
    for c in ("ref_leakage", "test_metric"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ood = d[d["method"].isin(OOD)]
    # per (dataset, entity_type, feature): mean reference-leakage + mean OOD held-out metric
    g = (ood.groupby(["dataset", "entity_type", "feature_set"])
            .agg(ref_leak=("ref_leakage", "mean"), ood_test=("test_metric", "mean")).reset_index())
    return g


def _scatter(g, etype, path):
    sub = g[g["entity_type"] == etype]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for fs in sorted(sub["feature_set"].unique()):
        s = sub[sub["feature_set"] == fs]
        ax.scatter(s["ref_leak"], s["ood_test"], s=55, alpha=0.85,
                   color=COLORS.get(fs, "#666"), label=fs, edgecolor="white", linewidth=0.6)
    # star the hand-picked feature's points; mark validated exceptions
    hp = HAND_PICK[etype]
    hpx = sub[sub["feature_set"] == hp]
    ax.scatter(hpx["ref_leak"], hpx["ood_test"], s=210, marker="*",
               facecolor="none", edgecolor=COLORS.get(hp, "#000"), linewidth=1.4,
               label=f"hand-pick ({hp})")
    for ds, fs in EXCEPTIONS.items():
        e = sub[(sub["dataset"] == ds) & (sub["feature_set"] == fs)]
        if not e.empty:
            ax.annotate(f"{ds.replace('moleculenet_','')}→{fs}",
                        (e["ref_leak"].iloc[0], e["ood_test"].iloc[0]),
                        fontsize=7, xytext=(6, 6), textcoords="offset points")
    ax.axhline(0, color="#b00", lw=0.8, ls=":")           # OOD R^2 < 0 = worse than the mean
    ax.set_xlabel("reference-space leakage $L(\\pi)$  (lower = cleaner split)", fontsize=12)
    ax.set_ylabel("held-out OOD metric (ROC-AUC / $R^2$; higher = generalizes)", fontsize=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def demonstrate(sweep=SWEEP):
    os.makedirs(FIGS, exist_ok=True)
    g = _agg(sweep)
    for etype in sorted(g["entity_type"].unique()):
        _scatter(g, etype, os.path.join(FIGS, f"feature_tradeoff_{etype}.png"))

    # per-type summary: mean reference-leakage + mean OOD metric per feature
    print("== hand-pick demonstration ==")
    for etype in sorted(g["entity_type"].unique()):
        sub = g[g["entity_type"] == etype]
        tab = (sub.groupby("feature_set")
                  .agg(ref_leak=("ref_leak", "mean"), ood_test=("ood_test", "mean"),
                       worst_ood=("ood_test", "min")).round(3).sort_values("ref_leak"))
        print(f"\n{etype}  (hand-pick: {HAND_PICK[etype]})")
        print(tab.to_string())
        # flag features that go OOD-harmful on some dataset
        bad = tab[tab["worst_ood"] < 0]
        if not bad.empty:
            print(f"  OOD-UNSAFE (negative held-out on some dataset): {list(bad.index)}")
    print(f"\n== figures -> {FIGS}/feature_tradeoff_*.png")
    return g


if __name__ == "__main__":
    demonstrate()
