"""Per-dataset grouped bars: low-rank vs the full DataSAIL-benchmark splitter set.

Shows L(pi) on every dataset (no aggregation -- L(pi) reducibility varies by
dataset, so a mean would be misleading). Set = Random + the 5 DeepChem splitters
from the DataSAIL paper (Fingerprint, MaxMin, Weight, Scaffold, Butina) +
DataSAIL + Low-rank, over the 9 MoleculeNet datasets <=30k (MaxMin is O(n^2), so
hiv/muv are out). Invalid SMILES (0.02-0.54%) are zero-filled: DeepChem
recomputes features from SMILES and crashes on unparseable ones, so those few
are swapped for a placeholder on input and scored as zero vectors -- exactly how
low-rank handles them. Every method is at an exact 20% test block.

    python poster/make_standard_splitter_figure.py   (palm env)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.benchmarks.moleculenet.benchmark_lowrank import REF_DATASAIL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "lowrank_split", "results")
OUT = os.path.join(HERE, "figures")
INK = "#14243B"

# ordered by n. Up to qm8 (<=30k) every DeepChem splitter runs; at hiv/muv the
# O(n^2) ones (MaxMin, Fingerprint, Butina) can't -- shown as gaps + an O(n^2) tag.
DATASETS = ["freesolv", "esol", "sider", "clintox", "bace", "bbbp",
            "lipophilicity", "tox21", "qm8", "hiv", "muv"]
# method -> (label, colour); Low-rank last, DataSAIL/Random distinct, DC set warm
STYLE = [
    ("random",         "Random",         "#C7CDD6"),
    ("dc_fingerprint", "DC-Fingerprint", "#E8A33D"),
    ("dc_maxmin",      "DC-MaxMin",      "#E86A9A"),
    ("dc_weight",      "DC-Weight",      "#A9A44C"),
    ("dc_scaffold",    "DC-Scaffold",    "#5FA85B"),
    ("dc_butina",      "DC-Butina",      "#8E7CC3"),
    ("datasail",       "DataSAIL",       "#C0392B"),
    ("lowrank",        "Low-rank",       "#2563EB"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 34,
    "axes.labelsize": 40, "xtick.labelsize": 25, "ytick.labelsize": 34,
    "legend.fontsize": 31, "axes.linewidth": 1.6,
    "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
})


def load():
    dc = pd.read_csv(os.path.join(RES, "deepchem_splitters.csv"))
    dc = dc[dc.status == "ok"]
    dc_p = dc.pivot(index="dataset", columns="method", values="lpi")
    base = pd.read_csv(os.path.join(RES, "baseline_benchmark.csv"))
    lr = base[base.method == "lowrank"].set_index("dataset")["lpi_mean"]
    rnd = base[base.method == "random"].set_index("dataset")["lpi_mean"]
    ns = base.drop_duplicates("dataset").set_index("dataset")["n"]
    vals = {}
    for m, _, _ in STYLE:
        if m == "lowrank":
            vals[m] = lr.reindex(DATASETS).values
        elif m == "random":
            vals[m] = rnd.reindex(DATASETS).values
        elif m == "datasail":
            vals[m] = np.array([REF_DATASAIL[d] if REF_DATASAIL[d] is not None
                                else np.nan for d in DATASETS])   # muv: timeout
        else:
            vals[m] = dc_p[m].reindex(DATASETS).values
    return vals, ns.reindex(DATASETS).values


def main():
    vals, ns = load()
    x = np.arange(len(DATASETS)) * 1.18   # extra group spacing so labels don't collide
    w = 0.115
    fig, ax = plt.subplots(figsize=(29, 8.4))
    on2 = {"dc_fingerprint", "dc_maxmin", "dc_butina"}   # O(n^2) DeepChem splitters
    for j, (m, lab, col) in enumerate(STYLE):
        off = (j - (len(STYLE) - 1) / 2) * w
        v = vals[m]
        ax.bar(x + off, np.nan_to_num(v), w, label=lab, color=col,
               edgecolor=INK, linewidth=0.8,
               zorder=4 if m == "lowrank" else 3)
        # tag the gaps at hiv/muv: O(n^2) DeepChem can't scale; DataSAIL times out
        for i in np.where(np.isnan(v))[0]:
            txt = "timed out" if m == "datasail" else r"$O(n^2)$"
            ax.annotate(txt, (x[i] + off, 0.008), rotation=90, ha="center",
                        va="bottom", fontsize=17, color="#7c4a12", zorder=5)
    ax.set_ylabel(r"$L(\pi)$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n$n$="
                        + (f"{n/1000:.1f}k" if n >= 1000 else f"{int(n)}")
                        for d, n in zip(DATASETS, ns)])
    ax.set_ylim(0, 0.40)
    ax.legend(loc="upper center", ncol=8, frameon=True, framealpha=0.96,
              columnspacing=1.0, handlelength=1.2, handletextpad=0.4,
              borderpad=0.4, bbox_to_anchor=(0.5, 1.10))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_moleculenet_standard_splitters.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    print("saved", p)


if __name__ == "__main__":
    main()
