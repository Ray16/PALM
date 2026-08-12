"""Bar chart of the UMA-embedding leakage result (100k, 3-way native proportions):
random vs native vs low-rank, on the learned UMA-RBF similarity. 300 DPI.
Separate experiment from the scaling plots (which use the hand descriptor vs n)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "omol25", "results")
s = json.load(open(os.path.join(RESULTS, "omol25_uma_lowrank_split_summary.json")))

labels = ["random\nbaseline", "existing\nnative split", "low-rank\nre-split"]
vals = [s["lpi_random"], s["lpi_native"], s["lpi_lowrank_3way"]]
colors = ["#9ca3af", "#dc2626", "#2563eb"]

fig, ax = plt.subplots(figsize=(5.6, 4.4))
bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# annotate the two reductions
ax.annotate("", xy=(1, s["lpi_native"]), xytext=(0, s["lpi_random"]),
            arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.2))
ax.text(0.5, (s["lpi_random"] + s["lpi_native"]) / 2 + 0.004,
        f"native −{s['lpi_random']-s['lpi_native']:.3f}", color="#6b7280", fontsize=8.5, ha="center")
ax.annotate("", xy=(2, s["lpi_lowrank_3way"]), xytext=(1, s["lpi_native"]),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.4))
ax.text(1.5, (s["lpi_native"] + s["lpi_lowrank_3way"]) / 2,
        f"low-rank −{s['lpi_native']-s['lpi_lowrank_3way']:.3f}",
        color="#2563eb", fontsize=9, ha="center", fontweight="bold")

ax.set_ylabel("L(π)   (lower = less leakage)")
ax.set_ylim(0.55, 0.675)
ax.set_title(f"OMol25 leakage on learned UMA-RBF similarity\n(n={s['n']:,}, 3-way native proportions)",
             fontsize=10.5)
ax.grid(True, axis="y", alpha=0.3, zorder=0)
cap = ("Similarity = RBF over mean-pooled UMA-small backbone embeddings "
       "(sigma = median pairwise dist).\nFactorized L(π) validated exact vs O(n²) "
       "(|diff|<1e-4). Low-rank cuts leakage ~5× more than the native composition split.")
fig.text(0.01, -0.02, cap, fontsize=6.5, ha="left", va="top", color="#555")
fig.tight_layout()
out = os.path.join(RESULTS, "omol25_uma_leakage.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
