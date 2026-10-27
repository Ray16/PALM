"""Fixed-width vector figures: all text >=18 pt at 6.5-inch manuscript width.

No tight bounding-box crop: changing the PDF canvas width would silently change
the final font size when LaTeX fits it to the text column. Geometry and font
checks supplement, rather than replace, rendered visual inspection.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "revision_20261021"
FIG = PAPER / "figures/revision"
COLORS = {"random": "#777777", "published_simpd": "#56B4E9", "plain": "#0072B2", "combined": "#D55E00", "knn_graph": "#009E73", "datasail": "#CC79A7"}
AUDIT = []
plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    "figure.titlesize": 18, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "figure.constrained_layout.h_pad": .14,
    "figure.constrained_layout.w_pad": .14})


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    # Some tick Text objects belong to invisible axes; discard empty extents.
    texts = [t for t in texts if t.get_window_extent(renderer).width > 0 and t.get_window_extent(renderer).height > 0]
    collisions = []
    for i, a in enumerate(texts):
        aa = a.get_window_extent(renderer)
        for b in texts[i + 1:]:
            bb = b.get_window_extent(renderer)
            intersection = aa.intersection(aa, bb)
            if intersection is not None and intersection.width > 1 and intersection.height > 1:
                collisions.append([a.get_text(), b.get_text()])
    minimum = min(t.get_fontsize() for t in texts)
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    panel_collisions = []
    for i, a in enumerate(panels):
        aa = a.get_window_extent(renderer)
        for b in panels[i + 1:]:
            bb = b.get_window_extent(renderer)
            intersection = aa.intersection(aa, bb)
            if intersection is not None and intersection.width > 1 and intersection.height > 1:
                panel_collisions.append(["panel", i, panels.index(b)])
    for t in texts:
        box = t.get_window_extent(renderer)
        for i, panel in enumerate(panels):
            if t.axes is panel:
                continue
            intersection = box.intersection(box, panel.get_window_extent(renderer))
            if intersection is not None and intersection.width > 1 and intersection.height > 1:
                panel_collisions.append([t.get_text(), i])
    clipped = []
    for t in texts:
        box = t.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            clipped.append(t.get_text())
    if minimum < 18 or collisions or clipped or panel_collisions:
        raise ValueError(f"{name}: font {minimum}, text overlaps {collisions}, panel overlaps {panel_collisions}, clipped {clipped}")
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=160)
    AUDIT.append(dict(figure=name, width_inches=float(fig.get_figwidth()),
        height_inches=float(fig.get_figheight()), minimum_source_font_pt=minimum,
        detected_text_overlaps=collisions, detected_panel_overlaps=panel_collisions,
        clipped_text=clipped, final_width_inches=6.5,
        visual_inspection="pending", final_manuscript_font_check="pending"))
    plt.close(fig)


def external():
    d = pd.read_csv(DATA / "external_per_assay.csv")
    for model, title in [("rf", "Random forest"), ("xgb", "Boosted trees")]:
        fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
        methods = ["random", "published_simpd", "plain", "combined"]
        for condition, offset, marker, label in [("original", -.10, "o", "Original split"),
                ("label_balanced", .10, "^", "Label balanced")]:
            for j, method in enumerate(methods):
                values = d[(d.model == model) & (d.method == method) & (d.condition == condition)].auc.to_numpy()
                if len(values) != 10:
                    raise ValueError("external figure requires all ten assays")
                boot = np.random.default_rng(31415).choice(values, (10000, len(values))).mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                mean = values.mean()
                ax.errorbar(j + offset, mean, yerr=[[mean - lo], [hi - mean]], fmt=marker,
                    ms=8, capsize=4, color=COLORS[method],
                    markerfacecolor=COLORS[method] if condition == "original" else "white",
                    label=label if j == 0 else None)
        ax.axhline(.5, color="gray", ls=":", lw=1)
        ax.set_xticks(range(4), ["Random", "SIMPD", "Plain", "Combined"])
        ax.set_ylabel("Mean test ROC-AUC")
        ax.set_ylim(.25, 1.02)
        ax.set_yticks([.3, .5, .7, .9])
        ax.set_title(title + ": ten external assays")
        ax.grid(axis="y", alpha=.2)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
        save(fig, "external_" + model)


def fidelity():
    d = pd.read_csv(DATA / "powered_fidelity/rows.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    for power, color in [(1, "#0072B2"), (3, "#D55E00")]:
        curve = d[d.power == power].groupby("rank").abs_cut_error_over_mass.max()
        ax.plot(curve.index, curve, "o-", ms=8, color=color, label="Kernel power " + str(power))
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
    ax.set_xlabel("Factor rank")
    ax.set_ylabel("Cut error / total mass")
    ax.set_title("Fidelity on six full datasets")
    ax.set_ylim(-.005, .335)
    ax.set_yticks([0, .1, .2, .3])
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1, frameon=False)
    save(fig, "powered_fidelity")


def rank_sensitivity():
    d = pd.read_csv(DATA / "powered_fidelity/quality.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#333333"]
    for (dataset, group), color in zip(d[d.power == 3].groupby("dataset"), palette):
        curve = group.groupby("rank").nnl_rel.mean()
        label = dataset.replace("moleculenet_", "")
        label = label.upper() if len(label) <= 5 else label.capitalize()
        ax.plot(curve.index, curve, "o-", ms=6, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
    ax.set_xlabel("Factor rank")
    ax.set_ylabel("Local leakage / random")
    ax.set_title("Combined setting: rank sensitivity")
    ax.set_ylim(-.004, .105)
    ax.set_yticks([0, .05, .1])
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    save(fig, "combined_rank")


def concept():
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 7.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.7]})
    fig.supxlabel("Fold sum p(c) = Σ B(i) for i in fold c\n"
                  "Cut = ½ [ ‖Σ p(c)‖² − Σ ‖p(c)‖² ]", fontsize=18)
    rng = np.random.default_rng(4)
    B = rng.uniform(.1, 1, (15, 3))
    B /= np.sqrt((B @ B.T).max())
    S = B @ B.T
    coarsened = np.array([[S[5*a:5*a+5, 5*b:5*b+5].mean() for b in range(3)] for a in range(3)])
    nodes = np.array([[.15, .2], [.85, .2], [.5, .8]])
    ax = axes[0, 0]
    for a in range(3):
        for b in range(a+1, 3):
            ax.plot(nodes[[a,b], 0], nodes[[a,b], 1], color="#0072B2", lw=6*coarsened[a,b], zorder=1)
    for i, (x,y) in enumerate(nodes):
        ax.add_patch(Circle((x,y), .12, facecolor="#D9EDF7", edgecolor="#0072B2", lw=2, zorder=2))
        ax.text(x,y,str(i+1), ha="center", va="center", fontsize=18, zorder=3)
    ax.set(xlim=(-.05,1.05), ylim=(0,1))
    ax.set_aspect("equal")
    candidate = S.copy()
    np.fill_diagonal(candidate, -np.inf)
    keep = np.argsort(candidate, axis=1)[:, -3:]
    sparse = np.zeros_like(S)
    np.put_along_axis(sparse, keep, np.take_along_axis(S, keep, axis=1), axis=1)
    sparse = np.maximum(sparse, sparse.T)
    for row, matrix, title, text in [(0, None, "Coarsen", "Weighted supernodes\nand inter-group edges"),
            (1, sparse, "Sparsify", "Select nearby pairs"),
            (2, B, "Compress", "Shared coordinates\napproximate all pairs")]:
        if matrix is not None:
            axes[row, 0].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axes[row, 0].set_axis_off()
        axes[row, 1].set_axis_off()
        axes[row, 1].text(0, .65, title, fontsize=18, weight="bold")
        axes[row, 1].text(0, .35, text, fontsize=18)
    save(fig, "concept")


def robustness():
    if not (DATA / "robustness_frontier.csv").exists():
        return
    d = pd.read_csv(DATA / "robustness_frontier.csv")
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.5, 5.4), layout="constrained")
    colors = {2: "#0072B2", 3: "#D55E00", 4: "#009E73"}
    markers = {0.3: "o", 0.4: "s", 0.5: "^", 0.6: "D"}
    for power, group in d.groupby("power"):
        group = group.sort_values("tau")
        ax.plot(group.morgan_global_ratio, group.morgan_nnl_rel, color=colors[power], alpha=.6)
        for _, row in group.iterrows():
            ax.scatter(row.morgan_global_ratio, row.morgan_nnl_rel, color=colors[power],
                       marker=markers[round(row.tau, 1)], s=65)
    ax.set_xlabel("Global leakage / random")
    ax.set_ylabel("Local leakage / random")
    ax.set_title("Power and grouping threshold")
    lo, hi = d.morgan_global_ratio.min(), d.morgan_global_ratio.max()
    pad = max(.02, (hi - lo) * .1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xticks(np.linspace(lo, hi, 4), [f"{v:.2f}" for v in np.linspace(lo, hi, 4)])
    top = float(d.morgan_nnl_rel.max()) * 1.15
    ax.set_ylim(0, top)
    ax.set_yticks(np.linspace(0, top, 4), [f"{v:.2f}" for v in np.linspace(0, top, 4)])
    ax.grid(alpha=.2)
    fig.legend([Line2D([], [], color=colors[p]) for p in (2, 3, 4)],
               ["Power " + str(p) for p in (2, 3, 4)], loc="outside upper center", ncol=3, frameon=False)
    fig.legend([Line2D([], [], color="black", marker=markers[t], linestyle="") for t in (.3, .4, .5, .6)],
               ["0.3", "0.4", "0.5", "0.6"], loc="outside lower center", ncol=4, frameon=False)
    save(fig, "power_threshold_frontier")

    raw = pd.read_csv(DATA / "robustness/rows.csv")
    keys = ["dataset", "seed"]
    plain = raw[raw.setting == "plain"].set_index(keys).maccs_nn_mean
    combined = raw[(raw.power == 3) & np.isclose(raw.tau, .4)].set_index(keys).maccs_nn_mean
    delta = (combined - plain).rename("delta").reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
    names = sorted(delta.dataset.unique())
    labels = {"bace": "BACE", "bbbp": "BBBP", "clintox": "ClinTox", "esol": "ESOL",
              "freesolv": "FreeSolv", "lipophilicity": "Lipophilicity"}
    for i, name in enumerate(names):
        values = delta[delta.dataset == name].delta.to_numpy()
        ax.scatter(values, np.full(len(values), i), s=35, color="#777777", label="Split seeds" if i == 0 else None)
        ax.scatter(values.mean(), i, s=80, marker="D", color="#D55E00", label="Dataset mean" if i == 0 else None)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(names)), [labels[n.replace("moleculenet_", "")] for n in names])
    ax.set_xlabel("MACCS nearest-similarity change")
    ax.set_title("Combined minus plain")
    lo, hi = min(float(delta.delta.min()), 0.), max(float(delta.delta.max()), 0.)
    pad = max(.005, (hi - lo) * .08)
    ax.set_xlim(lo - pad, hi + pad)
    ticks = np.linspace(lo, hi, 4)
    ax.set_xticks(ticks, [f"{v:.2f}" for v in ticks])
    ax.grid(axis="x", alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    save(fig, "independent_fingerprint")


def scaling():
    if not (DATA / "scaling_summary.csv").exists():
        return
    d = pd.read_csv(DATA / "scaling_summary.csv")
    if len(d) != 10 or set(d.device) != {"NVIDIA H100 NVL"}:
        raise ValueError("scaling figure requires ten H100 cases")
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout="constrained")
    for method, label in [("plain", "Plain low rank"), ("combined", "Combined")]:
        a = d[d.method == method].sort_values("n")
        ax.plot(a.n, a.split_s, "o-", ms=7, color=COLORS[method], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(800, 2_500_000)
    ax.set_xticks([1000, 10000, 100000, 1000000, 2000000], ["1k", "10k", "100k", "1M", "2M"])
    lo = float(d.split_s.min()) * .7
    ax.set_ylim(lo, float(d.split_s.max()) * 1.3)
    ticks = [v for v in [.01, .1, 1, 10, 100] if lo <= v <= ax.get_ylim()[1]]
    ax.set_yticks(ticks, [str(v) for v in ticks])
    ax.set_xlabel("Molecules")
    ax.set_ylabel("Split time (seconds)")
    ax.set_title("GPU split time on H100")
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    save(fig, "scaling")


def benchmark():
    path=DATA/'benchmark_rows.csv'
    if not path.exists():return
    d=pd.read_csv(path)
    d=d[(d.status=='ok')&(d.epsilon==.01)&d.dataset.str.startswith('moleculenet_')]
    datasets=d.groupby('dataset').n.first().sort_values().index.tolist()
    if len(datasets)!=11:raise ValueError('benchmark figures require all eleven molecular datasets')
    names={'freesolv':'FreeSolv','esol':'ESOL','bace':'BACE','bbbp':'BBBP','sider':'SIDER',
           'clintox':'ClinTox','lipophilicity':'Lipophilicity','tox21':'Tox21','qm8':'QM8','hiv':'HIV','muv':'MUV'}
    labels=[names[x.replace('moleculenet_','')] for x in datasets]
    for metric,name,title in [('leak_ratio','benchmark_global','Dense global overlap'),
                               ('test_ramp_default_rel','benchmark_local','Nearest-training-neighbour overlap')]:
        fig,ax=plt.subplots(figsize=(6.5,6.8),layout='constrained')
        for offset,(method,label,marker) in zip((-.18,0,.18),[('plain','Plain','o'),('combined','Combined','D'),('knn_graph','k-NN graph','s')]):
            stats=d[d.method==method].groupby('dataset')[metric].agg(['mean','std']).reindex(datasets)
            mean=stats['mean'].to_numpy();sd=stats['std'].to_numpy()
            if not np.isfinite(mean).all() or not np.isfinite(sd).all():raise ValueError('incomplete three-seed benchmark')
            ax.errorbar(mean,np.arange(11)+offset,xerr=np.vstack([np.minimum(mean,sd),sd]),fmt=marker,
                        ms=6,capsize=2,color=COLORS[method],label=label)
        ax.set_yticks(range(11),labels);ax.invert_yaxis();ax.set_ylim(10.6,-.6)
        ax.set_xlabel('Leakage / count-matched random');ax.set_title(title)
        if metric=='leak_ratio':ax.set_xlim(.30,1.08);ax.set_xticks([.4,.6,.8,1.])
        else:ax.set_xlim(-.015,1.08);ax.set_xticks([0,.25,.5,.75,1.])
        ax.axvline(1,color=COLORS['random'],ls=':',lw=1);ax.grid(axis='x',alpha=.2)
        fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        save(fig,name)


def runtime():
    path=DATA/'runtime_summary.csv'
    if not path.exists():return
    from matplotlib.lines import Line2D
    d=pd.read_csv(path)
    if len(d)!=12 or d.dataset.nunique()!=4:raise ValueError('runtime figure requires four complete exact-count cases')
    fig,ax=plt.subplots(figsize=(6.5,7.0),layout='constrained')
    markers={'moleculenet_bace':'o','moleculenet_bbbp':'s','moleculenet_clintox':'^','moleculenet_lipophilicity':'D'}
    names={'moleculenet_bace':'BACE','moleculenet_bbbp':'BBBP','moleculenet_clintox':'ClinTox','moleculenet_lipophilicity':'Lipophilicity'}
    for row in d.itertuples():
        ax.scatter(row.split_s_median,row.local_mean,s=80,marker=markers[row.dataset],color=COLORS[row.method])
    handles=[Line2D([],[],marker='o',ls='',color=COLORS[m],label=label) for m,label in [('plain','Plain'),('combined','Combined'),('datasail','DataSAIL')]]
    handles += [Line2D([],[],marker=marker,ls='',color='black',label=names[dataset]) for dataset,marker in markers.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    ax.set_xscale('log');ax.set_xlim(.08,20000);ax.set_xticks([.1,1,10,100,1000,10000],['0.1','1','10','100','1k','10k'])
    maximum=max(.1,float(d.local_mean.max())*1.1);ax.set_ylim(-.01,maximum)
    ax.set_yticks([x for x in (0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.) if x<=maximum])
    ax.set_xlabel('CPU split time (seconds)');ax.set_ylabel('Local leakage / random')
    ax.set_title('CPU cost at exactly matched counts');ax.grid(alpha=.2)
    save(fig,'runtime_matched')


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    external()
    fidelity()
    rank_sensitivity()
    concept()
    robustness()
    scaling()
    benchmark()
    runtime()
    (DATA / "figure_layout_checks.json").write_text(json.dumps(AUDIT, indent=2) + "\n")
