"""Poster figures for low-rank vs state-of-the-art splitters.

All methods run at an *exact* 80/20 block (epsilon=0) and scored with the same
scaled_lpi, so no method gains from a smaller test set.  Lo-Hi is the exception
by construction -- it discards molecules -- so it is shown separately, scored on
its own retained subset against low-rank run on that same subset at the same
realized ratio.

    python poster/make_baseline_figures.py
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "lowrank_split", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.benchmarks.moleculenet.benchmark_lowrank import REF_DATASAIL

LOWRANK = "#2563EB"
DATASAIL = "#C0392B"
INK = "#14243B"
GOOD = "#1B7A3D"

# method -> (label, colour)
STYLE = {
    "random":         ("Random",        "#C7CDD6"),
    "scaffold":       ("Scaffold",      "#E8A33D"),
    "butina":         ("Butina",        "#8E7CC3"),
    "astartes":       ("astartes",      "#17A2B8"),
    "datasail":       ("DataSAIL",      DATASAIL),
    "lohi":           ("Lo-Hi",         "#7B1FA2"),
    "lowrank":        ("Low-rank",      LOWRANK),
}
ORDER = ["random", "scaffold", "butina", "astartes", "datasail", "lohi", "lowrank"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 20,
    "axes.titlesize": 24, "axes.titleweight": "bold",
    "axes.labelsize": 22, "xtick.labelsize": 16, "ytick.labelsize": 18,
    "legend.fontsize": 17, "axes.linewidth": 1.5,
    "lines.linewidth": 3.5, "lines.markersize": 12,
    "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
})


# fig_moleculenet_leakage_vs_baselines and fig_moleculenet_time_vs_baselines are
# each placed at less than full column width on the poster, so the base rcParams
# above print smaller than their nominal point size. These per-figure multipliers
# bump just those two calls (via rc_context, not the shared rcParams) so their
# smallest on-poster text is >=18pt; tuned against poster/lowrank_poster.tex's
# actual placed width (measured empirically, since bbox_inches="tight" changes
# the native figure size as fonts grow).
_BASE_RC = {k: plt.rcParams[k] for k in
            ("font.size", "axes.titlesize", "axes.labelsize",
             "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
LEAKAGE_FONT_SCALE = 1.79


def _scaled_rc(factor):
    return {k: v * factor for k, v in _BASE_RC.items()}


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("saved", p)


def _load():
    df = pd.read_csv(os.path.join(RES, "baseline_benchmark.csv"))
    lpi = df.pivot(index=["dataset", "n"], columns="method",
                   values="lpi_mean").reset_index().sort_values("n")
    tim = df.pivot(index=["dataset", "n"], columns="method",
                   values="time_s").reset_index().sort_values("n")
    # DataSAIL leakage comes from a separate re-run (REF_DATASAIL in
    # benchmark_lowrank.py), scored with the same ECFP scaled_lpi -- muv has no
    # value (genuine DataSAIL timeout), datasail_extra.json can override/add.
    ds_ref = dict(REF_DATASAIL)
    extra_path = os.path.join(RES, "datasail_extra.json")
    if os.path.exists(extra_path):
        for name, v in json.load(open(extra_path)).items():
            ds_ref[name] = v["datasail_lpi"]
    lpi["datasail"] = lpi["dataset"].map(ds_ref)
    # Lo-Hi leakage: scored on its own retained subset (it discards 0.1-2% of
    # molecules to hit its ratio, so n/split aren't exactly matched to the
    # other methods) -- hiv/muv/qm8 have no value. run_lohi_all.sh wraps each
    # run in `timeout 1500`: rc=124 is that cap actually firing (genuine
    # timeout); any other nonzero rc is the script itself erroring out before
    # the cap, which is a DIFFERENT failure mode and must not be mislabeled
    # "timed out" (qm8/muv are rc=124; hiv is rc=1 -- an unconfirmed error).
    lohi_ref, lohi_annot = {}, {}
    for f in sorted(glob.glob(os.path.join(RES, "lohi_*.json"))):
        j = json.load(open(f))
        if not j.get("status"):
            lohi_ref[j["dataset"]] = j["hi_lpi"]
        else:
            lohi_annot[j["dataset"]] = "timed out" if j.get("rc") == 124 else "failed"
    lpi["lohi"] = lpi["dataset"].map(lohi_ref)

    # astartes (Burns et al., JOSS 2023): the field-standard splitting toolbox.
    # Represent it by its BEST (lowest-leakage) sampler per dataset over the four
    # it uniquely adds -- kennard_stone / spxy / sphere_exclusion / optisim --
    # with that sampler's split time.  best-of-4 is a per-dataset oracle, i.e.
    # deliberately generous, so no "you picked a weak sampler" objection stands.
    # astartes has no leakage objective and cannot hit an exact 20% test block
    # (its realized fraction drifts down, which if anything *lowers* L(pi)); it
    # still loses to low-rank.  Its O(n^2) interpolative samplers are capped, so
    # on muv (n=93k) no sampler runs -> "not run", like Butina.
    astartes_lpi, astartes_time = {}, {}
    ast_path = os.path.join(RES, "astartes_benchmark.csv")
    if os.path.exists(ast_path):
        adf = pd.read_csv(ast_path)
        samplers = ["kennard_stone", "spxy", "sphere_exclusion", "optisim"]
        a = adf[adf["method"].isin(samplers)].dropna(subset=["lpi_mean"])
        for ds, g in a.groupby("dataset"):
            i = g["lpi_mean"].idxmin()
            astartes_lpi[ds] = float(g.loc[i, "lpi_mean"])
            astartes_time[ds] = float(g.loc[i, "time_s"])
    lpi["astartes"] = lpi["dataset"].map(astartes_lpi)

    # split-time counterparts for the same two methods
    ds_time = pd.read_csv(os.path.join(RES, "lowrank_timing.csv"))
    tim["datasail"] = tim["dataset"].map(
        dict(zip(ds_time["dataset"], pd.to_numeric(ds_time["datasail_s"], errors="coerce"))))
    lohi_time = {}
    for f in sorted(glob.glob(os.path.join(RES, "lohi_*.json"))):
        j = json.load(open(f))
        if not j.get("status"):
            lohi_time[j["dataset"]] = j["hi_time_s"]
    tim["lohi"] = tim["dataset"].map(lohi_time)
    tim["astartes"] = tim["dataset"].map(astartes_time)
    return lpi, tim, lohi_annot


# ============ 1. leakage across all baselines + DataSAIL/Lo-Hi, exact 80/20 =
def leakage_vs_baselines():
    lpi, _, lohi_annot = _load()
    x = np.arange(len(lpi))
    w = 0.114
    with plt.rc_context(_scaled_rc(LEAKAGE_FONT_SCALE)):
        fig, ax = plt.subplots(figsize=(24.5, 7.6))
        for j, m in enumerate(ORDER):
            lab, col = STYLE[m]
            vals = lpi[m].to_numpy(float)
            off = (j - (len(ORDER) - 1) / 2) * w
            ax.bar(x + off, np.nan_to_num(vals), w, label=lab, color=col,
                   edgecolor=INK, linewidth=0.9,
                   zorder=4 if m == "lowrank" else 3)
            if m == "astartes":
                for i, v in enumerate(vals):
                    if np.isnan(v):        # all samplers past their O(n^2) cap
                        ax.annotate("not run", (x[i] + off, 0.006),
                                    rotation=90, ha="center", va="bottom",
                                    fontsize=13 * LEAKAGE_FONT_SCALE,
                                    color="#7c4a12", zorder=5)
            if m == "datasail":
                for i, v in enumerate(vals):
                    if np.isnan(v):        # genuine DataSAIL max_sec timeout
                        ax.annotate("timed out", (x[i] + off, 0.006),
                                    rotation=90, ha="center", va="bottom",
                                    fontsize=13 * LEAKAGE_FONT_SCALE,
                                    color="#7c4a12", zorder=5)
            if m == "lohi":
                for i, ds in enumerate(lpi["dataset"]):
                    if ds in lohi_annot:
                        ax.annotate(lohi_annot[ds], (x[i] + off, 0.006),
                                    rotation=90, ha="center", va="bottom",
                                    fontsize=13 * LEAKAGE_FONT_SCALE,
                                    color="#7c4a12", zorder=5)
        ax.set_ylabel(r"$L(\pi)$")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.dataset}\n$n$="
                            + (f"{r.n/1000:.1f}k" if r.n >= 1000 else f"{int(r.n)}")
                            for r in lpi.itertuples()],
                            fontsize=15 * LEAKAGE_FONT_SCALE)
        ax.set_ylim(0, 0.40)
        ax.legend(loc="upper center", ncol=7, frameon=True, framealpha=0.96,
                  columnspacing=1.0, handlelength=1.3, borderpad=0.4)
        _despine(ax)
        fig.tight_layout()
        save(fig, "fig_moleculenet_leakage_vs_baselines.png")


# ============ 2. split time across all baselines + DataSAIL/Lo-Hi ==========
def time_vs_baselines():
    """Grouped bars, log-scale y, same layout as leakage_vs_baselines().

    DataSAIL/Lo-Hi split time is driven by each dataset's specific ILP/MILP
    problem structure, not n -- e.g. DataSAIL takes 100x longer on bace than
    on sider despite nearly identical n (real solver behavior, see
    lowrank_split/results/lowrank_timing.csv). A log-log scaling *line* falsely
    implies a smooth n-to-time relationship that doesn't exist; discrete bars
    per dataset make no such claim. Random is excluded (~0s, invisible/
    undefined on a log axis).
    """
    _, tim, lohi_annot = _load()
    methods = [m for m in ORDER if m != "random"]
    x = np.arange(len(tim))
    w = 0.133
    with plt.rc_context(_scaled_rc(LEAKAGE_FONT_SCALE)):
        fig, ax = plt.subplots(figsize=(22.5, 7.6))
        ax.set_yscale("log")
        for j, m in enumerate(methods):
            lab, col = STYLE[m]
            vals = tim[m].to_numpy(float)
            off = (j - (len(methods) - 1) / 2) * w
            ok = ~np.isnan(vals)
            ax.bar(x[ok] + off, vals[ok], w, label=lab, color=col,
                   edgecolor=INK, linewidth=0.9,
                   zorder=4 if m == "lowrank" else 3)
            if m in ("butina", "astartes"):
                for i in np.where(~ok)[0]:      # skipped above its O(n^2) cutoff
                    ax.annotate("not run", (x[i] + off, 0.05),
                                rotation=90, ha="center", va="bottom",
                                fontsize=13 * LEAKAGE_FONT_SCALE,
                                color="#7c4a12", zorder=5)
            if m == "datasail":
                for i in np.where(~ok)[0]:      # genuine DataSAIL max_sec timeout
                    ax.annotate("timed out", (x[i] + off, 0.05),
                                rotation=90, ha="center", va="bottom",
                                fontsize=13 * LEAKAGE_FONT_SCALE,
                                color="#7c4a12", zorder=5)
            if m == "lohi":
                for i, ds in enumerate(tim["dataset"]):
                    if ds in lohi_annot:
                        ax.annotate(lohi_annot[ds], (x[i] + off, 0.05),
                                    rotation=90, ha="center", va="bottom",
                                    fontsize=13 * LEAKAGE_FONT_SCALE,
                                    color="#7c4a12", zorder=5)
        ax.set_ylabel("split time (s, log scale)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.dataset}\n$n$="
                            + (f"{r.n/1000:.1f}k" if r.n >= 1000 else f"{int(r.n)}")
                            for r in tim.itertuples()],
                            fontsize=15 * LEAKAGE_FONT_SCALE)
        ax.set_ylim(0.04, 4000)
        # In-axes legend would collide with the tall bars (bace/lipophilicity/
        # tox21/hiv all exceed 700s, close to the axis top on this log scale) --
        # unlike the leakage chart, nothing here leaves headroom for it.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6,
                  frameon=True, framealpha=0.96, columnspacing=1.0,
                  handlelength=1.3, borderpad=0.4)
        _despine(ax)
        # Unlike the log-log scatter version of this chart, this one has dense
        # two-line x-tick labels that need tight_layout()'s sizing pass to
        # avoid colliding with each other -- keep it despite the external
        # legend (re-check for reintroduced dead-margin whitespace).
        fig.tight_layout()
        save(fig, "fig_moleculenet_time_vs_baselines.png")


if __name__ == "__main__":
    leakage_vs_baselines(); time_vs_baselines()
    print("\nBaseline figures ->", OUT)
