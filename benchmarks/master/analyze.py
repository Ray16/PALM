"""Turn the master benchmark table into figures + a written INSIGHTS.md.

Reads ``results/master_benchmark.csv`` (registry: split quality + generalization
gap) and, if present, ``results/chemixhub_quality.csv`` (mixture split quality),
aggregates over seeds, and writes:

- ``results/figures/leakage_by_method.png`` — mean leakage per method (lower = better)
- ``results/figures/gengap_by_method.png``  — mean generalization gap per method
- ``results/figures/leakage_vs_gap.png``    — the headline: does lower leakage buy
  a larger (more honest) train->test gap?
- ``results/figures/test_metric_by_method.png`` — mean held-out metric per method
- ``INSIGHTS.md`` — the numbers, in words.

Figures follow the repo conventions: single-panel, PNG only, no titles, an
Okabe-Ito colorblind palette; captions/interpretation live in INSIGHTS.md.
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
MASTER = os.path.join(RESULTS, "master_benchmark.csv")
CHEMIX = os.path.join(RESULTS, "chemixhub_quality.csv")
INSIGHTS = os.path.abspath(os.path.join(HERE, "..", "..", "benchmarks", "master", "INSIGHTS.md"))

# Okabe-Ito; random baselines in grey.
COLORS = {
    "random": "#999999", "random_group": "#999999", "random_sample": "#bbbbbb",
    "hypergraph": "#0072B2", "graph": "#56B4E9", "lowrank": "#009E73",
    "datasail": "#E69F00", "scaffold": "#CC79A7", "butina": "#D55E00",
    "hypergraph_nd": "#0072B2", "hypergraph_nd_knn": "#56B4E9",
}
# order methods sensibly on the x-axis
ORDER = ["random", "random_group", "random_sample", "scaffold", "butina",
         "datasail", "graph", "hypergraph", "lowrank",
         "hypergraph_nd", "hypergraph_nd_knn"]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _ordered(methods):
    seen = [m for m in ORDER if m in methods]
    return seen + [m for m in methods if m not in seen]


def _bar(agg_mean, agg_std, ylabel, path, zero_line=False):
    methods = _ordered(list(agg_mean.index))
    x = np.arange(len(methods))
    means = [agg_mean[m] for m in methods]
    stds = [agg_std.get(m, 0) for m in methods]
    colors = [COLORS.get(m, "#333333") for m in methods]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="white", linewidth=0.6)
    if zero_line:
        ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _scatter_leak_gap(df, path):
    """(leakage, gen_gap) per (dataset, method), colored by method + Pearson r."""
    d = df.dropna(subset=["leakage", "gen_gap"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for m in _ordered(d["method"].unique()):
        sub = d[d["method"] == m]
        ax.scatter(sub["leakage"], sub["gen_gap"], s=42, alpha=0.8,
                   color=COLORS.get(m, "#333333"), label=m, edgecolor="white", linewidth=0.5)
    r = np.corrcoef(d["leakage"], d["gen_gap"])[0, 1]
    # least-squares guide line
    b, a = np.polyfit(d["leakage"], d["gen_gap"], 1)
    xs = np.linspace(d["leakage"].min(), d["leakage"].max(), 50)
    ax.plot(xs, b * xs + a, color="#000", lw=1.2, ls="--",
            label=f"fit (Pearson r = {r:.2f})")
    ax.set_xlabel(r"leakage $L(\pi)$  (lower = less similarity leak)", fontsize=12)
    ax.set_ylabel("generalization gap (train - test)", fontsize=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return r


def analyze(master=MASTER, chemix=CHEMIX):
    os.makedirs(FIGS, exist_ok=True)
    m = pd.read_csv(master)
    ok = m[m["status"] == "ok"].copy()
    for c in ("leakage", "gen_gap", "test_metric", "runtime_s", "imbalance"):
        ok[c] = _num(ok, c)

    lines = ["# Master benchmark — insights", ""]
    n_ds = ok["dataset"].nunique()
    seeds = sorted({int(s) for s in ok["seed"].dropna()})
    lines += [f"Generated from `{os.path.relpath(master)}` "
              f"({len(ok)} ok runs over {n_ds} datasets, seeds {seeds}).", ""]

    # ---- per (dataset, method): mean over seeds --------------------------------
    per = (ok.groupby(["dataset", "category", "task_type", "method"], dropna=False)
             .agg(leakage=("leakage", "mean"), gen_gap=("gen_gap", "mean"),
                  test_metric=("test_metric", "mean"), runtime_s=("runtime_s", "mean"))
             .reset_index())

    # ---- figure 1: leakage by method (registry 1-D) ---------------------------
    leak = per.dropna(subset=["leakage"])
    lmean, lstd = leak.groupby("method")["leakage"].mean(), leak.groupby("method")["leakage"].std().fillna(0)
    _bar(lmean, lstd, r"mean leakage $L(\pi)$  (lower is better)",
         os.path.join(FIGS, "leakage_by_method.png"))

    # ---- figure 2: generalization gap by method -------------------------------
    gap = per.dropna(subset=["gen_gap"])
    gmean, gstd = gap.groupby("method")["gen_gap"].mean(), gap.groupby("method")["gen_gap"].std().fillna(0)
    _bar(gmean, gstd, "mean generalization gap (train - test)",
         os.path.join(FIGS, "gengap_by_method.png"), zero_line=True)

    # ---- figure 3: leakage vs gap (headline) ----------------------------------
    r = _scatter_leak_gap(per, os.path.join(FIGS, "leakage_vs_gap.png"))

    # ---- figure 4: test metric by method --------------------------------------
    tm = per.dropna(subset=["test_metric"])
    tmean, tstd = tm.groupby("method")["test_metric"].mean(), tm.groupby("method")["test_metric"].std().fillna(0)
    _bar(tmean, tstd, "mean held-out metric (ROC-AUC / R^2; higher = easier split)",
         os.path.join(FIGS, "test_metric_by_method.png"), zero_line=True)

    # ---- insight text ---------------------------------------------------------
    lines += ["## 1. Leakage by method (vs the random baseline)", ""]
    base = lmean.get("random", np.nan)
    lines += ["| method | mean L(pi) | vs random |", "|---|--:|--:|"]
    for meth in _ordered(list(lmean.index)):
        v = lmean[meth]
        rel = f"{(v-base)/base*100:+.0f}%" if np.isfinite(base) and base else "-"
        lines.append(f"| {meth} | {v:.3f} | {rel} |")
    lines.append("")

    lines += ["## 2. Generalization gap by method", "",
              "Larger gap / lower held-out metric = the split makes the task genuinely "
              "harder (more honest OOD evaluation).", "",
              "| method | mean gap | mean held-out metric |", "|---|--:|--:|"]
    for meth in _ordered(list(gmean.index)):
        lines.append(f"| {meth} | {gmean[meth]:+.3f} | {tmean.get(meth, float('nan')):.3f} |")
    lines.append("")

    lines += ["## 3. Does lower leakage buy a larger (more honest) gap?", ""]
    if r is not None:
        lines += [f"Across all (dataset, method) points, Pearson **r(leakage, gap) = {r:.2f}** "
                  "— see `figures/leakage_vs_gap.png`.",
                  "A negative r means: the less a split leaks, the larger the train->test "
                  "gap it induces, i.e. leakage-minimized splits are harder / more realistic.", ""]

    # win-rate: lowest-leakage method per dataset
    lines += ["## 4. Per-dataset winner (lowest leakage)", "", "| dataset | best method | L(pi) |", "|---|---|--:|"]
    for ds, g in leak.groupby("dataset"):
        nonrand = g[~g["method"].isin(("random", "random_group", "random_sample"))]
        if nonrand.empty:
            continue
        best = nonrand.loc[nonrand["leakage"].idxmin()]
        lines.append(f"| {ds} | {best['method']} | {best['leakage']:.3f} |")
    lines.append("")

    # chemixhub mixture suite (leakage only)
    if os.path.exists(chemix):
        cx = pd.read_csv(chemix)
        cx["leakage"] = _num(cx, "leakage")
        cxm = cx.dropna(subset=["leakage"]).groupby("method")["leakage"].mean()
        lines += ["## 5. CheMixHub mixture suite (split-quality only)", "",
                  f"12 mixture tasks; mean mixture-level L(pi) per engine "
                  f"(generalization gap not available without the external clone):", "",
                  "| method | mean L(pi) |", "|---|--:|"]
        for meth in _ordered(list(cxm.index)):
            lines.append(f"| {meth} | {cxm[meth]:.3f} |")
        lines.append("")

    with open(INSIGHTS, "w") as fh:
        fh.write("\n".join(lines))
    print(f"== figures -> {FIGS}\n== insights -> {INSIGHTS}")
    if r is not None:
        print(f"   headline: Pearson r(leakage, gap) = {r:.2f}")


if __name__ == "__main__":
    analyze()
