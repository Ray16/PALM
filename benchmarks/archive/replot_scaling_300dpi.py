"""Re-render the two OMol25 scaling plots from omol25_scaling.csv at 300 DPI.
Styling/captions mirror omol25_scaling.py exactly; only the data source (CSV
instead of a fresh sweep) and the DPI differ."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
REPS_SMALL, SMALL_N = 5, 100_000
df = pd.read_csv(os.path.join(RESULTS, "omol25_scaling.csv"))

# ---- plot 1: split time vs n (log-log) ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(df["n"], df["lowrank_time_s"], "o-", color="#2563eb", label="low-rank")
hg = df.dropna(subset=["hypergraph_time_s"])
ax.loglog(hg["n"], hg["hypergraph_time_s"], "s--", color="#dc2626", label="hypergraph")
if len(hg):
    ax.axvline(hg["n"].max(), color="#dc2626", ls=":", alpha=0.5)
    ax.text(hg["n"].max(), ax.get_ylim()[0] * 1.5,
            "  hypergraph\n  infeasible →", color="#dc2626", fontsize=8, va="bottom")
ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("split time (s)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
cap = (f"low-rank time = Nystrom factor + one balanced assignment "
       f"(median of {REPS_SMALL} runs for n<={SMALL_N:,}, single run above).\n"
       "Full run_lowrank_split (k-means++ landmarks, 4 restarts, FM polish) is a "
       "constant factor higher, still O(n);\nuniform landmarks used here vs k-means++ "
       "in the library (minor fidelity gap).")
fig.text(0.01, -0.02, cap, fontsize=6.5, ha="left", va="top", color="#555")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "omol25_scaling_time.png"), dpi=300, bbox_inches="tight")

# ---- plot 2: L(pi) vs n ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.semilogx(df["n"], df["lowrank_lpi"], "o-", color="#2563eb", label="low-rank")
hgl = df.dropna(subset=["hypergraph_lpi"])
ax.errorbar(hgl["n"], hgl["hypergraph_lpi"], yerr=hgl["hypergraph_lpi_std"],
            fmt="s--", color="#dc2626", label="hypergraph (median±std)", capsize=3)
ax.semilogx(df["n"], df["random_lpi"], "^:", color="#6b7280", label="random")
ax.set_xlabel("dataset size (n structures)"); ax.set_ylabel("L(π)  (lower = less leakage)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "omol25_scaling_lpi.png"), dpi=300, bbox_inches="tight")

print("re-rendered at 300 dpi:")
for f in ("omol25_scaling_time.png", "omol25_scaling_lpi.png"):
    print("  ", os.path.join(RESULTS, f))
