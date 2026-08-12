"""Poster figures for the 42x36 landscape low-rank poster.

Graph-forward, minimal-text: every figure carries its own message via a
takeaway title + direct value labels, so the poster body needs almost no prose.
Reads the committed result CSVs/JSON (no recompute) and writes 300-ppi PNGs to
poster/figures/.

    python poster/make_figures.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "lowrank_split", "results")
ORES = os.path.join(ROOT, "lowrank_split", "omol25", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette ---------------------------------------------------------------
LOWRANK = "#2563EB"   # hero blue
DATASAIL = "#9AA3B2"  # gray baseline
HYPER = "#E8710A"     # orange (k-NN hypergraph)
NATIVE = "#C0392B"    # red (existing native split)
RANDOM = "#C7CDD6"    # light gray (random baseline)
MAROON = "#7A0019"    # poster header accent
UMA_COL = "#7C3AED"       # violet -- UMA embedding
NONUMA_COL = "#94A3B8"    # slate -- non-UMA structural descriptor
INK = "#14243B"
GRID = "#D5DCE6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 22,
    "axes.titlesize": 26, "axes.titleweight": "bold",
    "axes.labelsize": 24, "xtick.labelsize": 19, "ytick.labelsize": 20,
    "legend.fontsize": 21, "axes.linewidth": 1.5,
    "lines.linewidth": 4.0, "lines.markersize": 15,
    "figure.dpi": 300, "savefig.dpi": 300,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
})


# fig_omol25_scaling_to_9-55M is placed at less than full column width on the
# poster, so the base rcParams above print slightly under their nominal point
# size. This multiplier bumps just that one call (via rc_context, not the
# shared rcParams) so its smallest on-poster text is >=18pt; tuned against
# poster/lowrank_poster.tex's actual placed width.
_BASE_RC = {k: plt.rcParams[k] for k in
            ("font.size", "axes.titlesize", "axes.labelsize",
             "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
SCALE_FONT_SCALE = 1.064
EMBED_FONT_SCALE = 1.3   # initial guess; will be retuned once placed width is known


def _scaled_rc(factor):
    return {k: v * factor for k, v in _BASE_RC.items()}


def _despine(ax, y=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)   # no gridlines (kept clean per design)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("saved", p)


# ===================== 3. Scale to full OMol25 (9.55M) ======================
def scale():
    df = pd.read_csv(os.path.join(ORES, "omol25_scaling.csv"))
    # Real OMol25 DataSAIL points (same 115-d structural-descriptor similarity
    # as the low-rank curve, same nested subsamples), NOT the old MoleculeNet/
    # ECFP points this plot used to borrow "for scale" -- those were a
    # different dataset entirely, plotted on this axis without saying so.
    dsf = pd.read_csv(os.path.join(ORES, "omol25_datasail_scaling.csv"))
    dsf = dsf[dsf["status"] == "ok"]
    with plt.rc_context(_scaled_rc(SCALE_FONT_SCALE)):
        fig, ax = plt.subplots(figsize=(11.6, 6.8))
        ax.loglog(df["n"], df["lowrank_time_s"], "o-", color=LOWRANK,
                  label="Low-rank", markeredgecolor="white",
                  markeredgewidth=1.5, zorder=5)
        # Unlike the MoleculeNet cross-dataset comparison (different datasets,
        # genuinely non-monotonic ILP hardness), this is the SAME dataset/
        # embedding at increasing n -- the trend here is close to monotonic
        # (9.2s->8.7s->44s->364s->5422s), so a connecting line is honest.
        # DataSAIL stops at n=100k: n=300k OOMs (671 GiB for the dense
        # distance matrix alone), confirming it structurally cannot reach
        # the millions-scale this plot's low-rank curve already covers.
        ax.loglog(dsf["n"], dsf["datasail_s"], "o-", color=DATASAIL,
                  label="DataSAIL (fast mode)",
                  markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        # DataSAIL here is its fastest (coarse, e_clusters=100) mode, which does
        # NOT hold 80/20 -- it collapses the test set to ~12% (n=3k) down to
        # ~4.6% (n=100k). Forcing a real ~20% test needs fine clustering, which
        # is 45-160x slower even at n=3k, so these are DataSAIL's BEST-case times.
        ax.annotate("fast/coarse mode: ~5–12% test\n(exact 20% is 45–160× slower)",
                    (dsf["n"].iloc[-1], dsf["datasail_s"].iloc[-1]),
                    xytext=(-8, -46), textcoords="offset points", ha="right",
                    va="top", fontsize=13.5 * SCALE_FONT_SCALE, color=DATASAIL)
        n_full = df["n"].max()
        t_full = float(df.loc[df["n"].idxmax(), "lowrank_time_s"])
        ax.annotate(f"full 9.55M\n→ {t_full:.0f} s", (n_full, t_full),
                    xytext=(-30, 44), textcoords="offset points", ha="right",
                    va="bottom", fontsize=18 * SCALE_FONT_SCALE, color=LOWRANK,
                    arrowprops=dict(arrowstyle="-|>", color=LOWRANK, lw=2.4),
                    zorder=6)
        ax.set_xlabel("dataset size  (n structures)")
        ax.set_ylabel("split time  (s)")
        ax.legend(loc="upper left", frameon=True, framealpha=0.96)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        save(fig, "fig_omol25_scaling_to_9-55M.png")


# ===================== 4. Learned UMA embeddings: native-split quality =====
def uma_native():
    """3-way quality (n=100k): random vs OMol25's shipped native split vs
    low-rank, all under UMA similarity. Single panel (was panel A of the old
    2-panel fig_omol25_uma_embedding_leakage.png).

    Similarity kernel = COSINE, not the original RBF (median-distance sigma).
    RBF was checked against ground truth (does "close under this kernel"
    actually predict UMA's own energy-prediction error on a held-out set) and
    lost decisively to cosine (Spearman rho 0.47 vs 0.55, n=~280 test points,
    p<1e-16) -- RBF's bandwidth has no principled value here, and every
    tested choice (median heuristic, Silverman's rule, k-NN-based, self-
    tuning) disagreed with the others by up to 30x in the resulting L(pi).
    Cosine has no free bandwidth parameter and is what the rest of this
    poster already uses (non-UMA descriptor, MoleculeNet ECFP/Tanimoto).
    """
    s = json.load(open(os.path.join(ORES, "omol25_uma_lowrank_split_summary_cosine.json")))
    labels = ["random\nsplit", "native\nsplit", "low-rank"]
    vals = [s["lpi_random"], s["lpi_native"], s["lpi_lowrank_3way"]]
    cols = [RANDOM, NATIVE, LOWRANK]
    with plt.rc_context(_scaled_rc(SCALE_FONT_SCALE)):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))   # wide+short: keeps column 3 clear of the footer
        bars = ax.bar(labels, vals, color=cols, width=0.62, edgecolor=INK,
                      linewidth=1.3, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x()+b.get_width()/2, v), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=21 * SCALE_FONT_SCALE, zorder=5)
        ax.set_ylabel(r"$L(\pi)$")
        ax.set_ylim(0.58, 0.665)
        _despine(ax)
        fig.tight_layout()
        save(fig, "fig_omol25_uma_native_leakage.png")


# ===================== 4b. UMA vs non-UMA embedding, matched n=2000 =========
def embedding_comparison():
    """Does the physically-informed UMA embedding actually change what counts
    as leakage, or would any embedding do? The SAME 2,000 physical OMol25
    structures (UMA panel's local indices mapped through orig_row.npy to
    global rows, then the non-UMA descriptor pulled for those exact rows --
    NOT independently seeded subsamples, which would pick different molecules
    since the UMA cache's local index space isn't the global one).

    UMA side uses COSINE similarity, not RBF -- see uma_native()'s docstring
    for why (RBF's bandwidth is unvalidated and loses to cosine when checked
    against UMA's own prediction error). Non-UMA already used cosine.

    Only low-rank's comparison is ratio-clean: it hits the exact 20% test
    fraction under both embeddings, so its UMA-vs-descriptor gap is a real
    embedding effect. DataSAIL's ILP relaxation lands at a DIFFERENT realized
    test fraction under each embedding (16.0% UMA, 14.2% non-UMA), both below
    the 20% target -- and a smaller test block mechanically lowers L(pi). That
    confound is large enough here that DataSAIL's own UMA-vs-descriptor gap
    can't be cleanly attributed to the embedding; the plot says so rather than
    implying a conclusion the data doesn't support.
    """
    d_uma = json.load(open(os.path.join(ORES, "omol25_uma_datasail_cosine.json")))
    d_non = json.load(open(os.path.join(ORES, "omol25_nonuma_n2000.json")))

    # Random's L(pi) is ~embedding-invariant (0.321 UMA vs 0.319 non-UMA, same
    # 2000 molecules): a uniformly random assignment's expected leakage is
    # mostly a function of the split ratio, not which kernel scores it -- so
    # it gets ONE bar, not a UMA/non-UMA pair implying an effect that isn't there.
    random_val = (d_uma["lpi_random"] + d_non["lpi_random"]) / 2

    paired = ["DataSAIL", "low-rank"]
    uma_vals = [d_uma["datasail"]["lpi"], d_uma["lowrank"]["lpi"]]
    non_vals = [d_non["datasail"]["lpi"], d_non["lpi_lowrank"]]

    x0 = 0                              # random, single bar
    xp = np.arange(1, 1 + len(paired))  # DataSAIL, low-rank
    w = 0.32
    with plt.rc_context(_scaled_rc(EMBED_FONT_SCALE)):
        fig, ax = plt.subplots(figsize=(9.5, 6.8))
        ax.bar([x0], [random_val], w * 1.7, color=RANDOM, edgecolor=INK,
               linewidth=1.3, zorder=3)
        ax.annotate(f"{random_val:.2f}", (x0, random_val), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=17 * EMBED_FONT_SCALE, zorder=5)
        b1 = ax.bar(xp - w/2, uma_vals, w, label="UMA embedding", color=UMA_COL,
                    edgecolor=INK, linewidth=1.3, zorder=3)
        b2 = ax.bar(xp + w/2, non_vals, w, label="structural descriptor\n(non-UMA)",
                    color=NONUMA_COL, edgecolor=INK, linewidth=1.3, zorder=3)
        for bars, vals in ((b1, uma_vals), (b2, non_vals)):
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x()+b.get_width()/2, v), xytext=(0, 6),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=17 * EMBED_FONT_SCALE, zorder=5)
        ax.set_ylabel(r"$L(\pi)$")
        ax.set_xticks([x0, *xp]); ax.set_xticklabels(["random\nsplit", *paired])
        ax.set_ylim(0, max([random_val] + uma_vals + non_vals) * 1.28)
        ax.legend(loc="upper right", frameon=True, framealpha=0.96,
                  fontsize=15 * EMBED_FONT_SCALE)
        _despine(ax)
        fig.text(0.01, -0.02,
                 "Same 2,000 physical OMol25 structures under both embeddings; UMA\n"
                 "scored with cosine similarity (validated against UMA's own prediction\n"
                 "error, not the original unvalidated RBF bandwidth). Random is one bar\n"
                 "(L(pi) is ~embedding-invariant). Low-rank hits the exact 20% test\n"
                 "target under both embeddings, so its gap is a real embedding effect.\n"
                 "DataSAIL's realized test_frac differs by embedding (16.0% UMA vs\n"
                 "14.2% non-UMA, both below the 20% target), and a smaller test block\n"
                 "mechanically lowers L(pi) -- so DataSAIL's own UMA-vs-descriptor gap\n"
                 "is confounded and not a clean comparison.",
                 fontsize=9 * EMBED_FONT_SCALE, ha="left", va="top", color="#555")
        fig.tight_layout()
        save(fig, "fig_omol25_embedding_comparison.png")


# ===================== 5. Exact balance + determinism (NEW) =================
def balance():
    df = pd.read_csv(os.path.join(ORES, "omol25_scaling.csv"))
    hg = df.dropna(subset=["hypergraph_test_frac"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 5.8))

    # --- panel A: realized test fraction (exact balance) ---
    a1.axhline(0.20, color=INK, ls=":", lw=2.2, zorder=1)
    a1.text(df["n"].min(), 0.205, "target 20%", fontsize=15, color=INK, va="bottom")
    a1.semilogx(df["n"], df["lowrank_test_frac"], "o-", color=LOWRANK,
                label="Low-rank", markeredgecolor="white",
                markeredgewidth=1.5, zorder=4)
    a1.semilogx(hg["n"], hg["hypergraph_test_frac"], "s--", color=HYPER,
                label="k-NN hypergraph", markeredgecolor="white",
                markeredgewidth=1.5, zorder=3)
    a1.set_ylim(0.10, 0.24)
    a1.set_xlabel("dataset size  (n)")
    a1.set_ylabel("realized test fraction")
    a1.set_title("Exact target ratios —\nhypergraph drifts off balance", fontsize=20, pad=8)
    a1.legend(loc="lower right", frameon=True, framealpha=0.96, fontsize=17)
    a1.grid(True, which="both", color=GRID, alpha=0.6, linewidth=1.0)
    a1.set_axisbelow(True)
    a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)

    # --- panel B: run-to-run spread of L(pi) (determinism) ---
    lr_std = np.zeros(len(hg))
    hg_std = hg["hypergraph_lpi_std"].to_numpy(float)
    xx = np.arange(len(hg)); w = 0.38
    a2.bar(xx - w/2, lr_std + 1e-4, w, color=LOWRANK, edgecolor=INK, linewidth=1.2,
           label="Low-rank", zorder=3)
    a2.bar(xx + w/2, hg_std, w, color=HYPER, edgecolor=INK, linewidth=1.2,
           label="k-NN hypergraph", zorder=3)
    for i, v in enumerate(hg_std):
        a2.annotate(f"±{v:.3f}", (xx[i]+w/2, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=13, color="#8a4b12", zorder=5)
    a2.annotate("std = 0\n(deterministic)", (xx[0]-w/2, 1e-4), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=15, color=LOWRANK, fontweight="bold", zorder=5)
    a2.set_xticks(xx)
    a2.set_xticklabels([f"{int(n/1000)}k" if n < 1e6 else f"{n/1e6:.0f}M"
                        for n in hg["n"]], fontsize=16)
    a2.set_xlabel("dataset size  (n)")
    a2.set_ylabel(r"$L(\pi)$ std over 5 seeds")
    a2.set_title("Fully deterministic —\nno run-to-run variance", fontsize=20, pad=8)
    a2.legend(loc="upper right", frameon=True, framealpha=0.96, fontsize=17)
    _despine(a2)
    fig.tight_layout()
    save(fig, "fig_omol25_split_balance_and_determinism.png")


if __name__ == "__main__":
    scale(); uma_native(); embedding_comparison()   # balance() dropped: DataSAIL-only poster
    # leakage() and speed() removed: superseded by the combined DataSAIL+
    # baselines charts in make_baseline_figures.py (leakage_vs_baselines(),
    # time_vs_baselines())
    print("\nPoster figures ->", OUT)
