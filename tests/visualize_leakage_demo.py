#!/usr/bin/env python
"""Generate professional visualizations demonstrating PALM's effectiveness
in reducing data leakage across splitting techniques."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(TESTS_DIR, "output")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
COLORS = {
    "R": "#94a3b8",       # gray - baseline
    "I1e": "#3b82f6",     # blue
    "I1f": "#8b5cf6",     # purple
    "I2": "#f59e0b",      # amber
    "C1e": "#10b981",     # green
    "C1f": "#6366f1",     # indigo
    "C2": "#ef4444",      # red
}
TECHNIQUE_LABELS = {
    "R": "Random",
    "I1e": "Identity (e1)",
    "I1f": "Identity (e2)",
    "I2": "Identity (both)",
    "C1e": "Cluster (e1)",
    "C1f": "Cluster (e2)",
    "C2": "Cluster (both)",
}


def load_metrics(dataset):
    """Load all technique metrics for a dataset."""
    metrics_dir = os.path.join(OUTPUT_DIR, dataset, "metrics")
    results = {}
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.endswith(".json"):
            continue
        technique = fname.split("_")[0]
        with open(os.path.join(metrics_dir, fname)) as f:
            results[technique] = json.load(f)
    return results


def plot_bbbp_leakage(metrics, save_path):
    """BBBP (1D): Show how C1e reduces leakage vs R and I1e."""
    techniques = ["R", "I1e", "C1e"]
    labels = [TECHNIQUE_LABELS[t] for t in techniques]
    colors = [COLORS[t] for t in techniques]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Mean NN Distance (higher = better separation)
    ax = axes[0]
    nn_dists = [metrics[t]["nn_leakage"]["mean_nn_dist"] for t in techniques]
    bars = ax.bar(labels, nn_dists, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, nn_dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean NN Distance")
    ax.set_title("Train-Test Separation", fontweight="bold")

    # Panel 2: Zero-distance pairs (lower = less leakage)
    ax = axes[1]
    zero_counts = [metrics[t]["nn_leakage"]["zero_dist_count"] for t in techniques]
    bars = ax.bar(labels, zero_counts, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, zero_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Identical Train-Test Pairs", fontweight="bold")

    # Panel 3: Distribution shift (lower = more representative)
    ax = axes[2]
    shifts = [metrics[t]["distribution_shift"]["mean_normalized_shift"] for t in techniques]
    bars = ax.bar(labels, shifts, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, shifts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean Normalized Shift")
    ax.set_title("Distribution Shift", fontweight="bold")

    fig.suptitle("BBBP Dataset — 2,050 Molecules — Data Leakage Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_davis_leakage(metrics, save_path):
    """Davis (2D): Show entity overlap reduction across techniques."""
    techniques = ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]
    labels = [TECHNIQUE_LABELS[t] for t in techniques]
    colors = [COLORS[t] for t in techniques]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Drug (e1) overlap
    ax = axes[0]
    e1_overlaps = []
    for t in techniques:
        eo = metrics[t].get("entity_overlap", {})
        e1_overlaps.append(eo.get("e1_overlap", 0))
    bars = ax.bar(labels, e1_overlaps, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, e1_overlaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Overlapping Drugs")
    ax.set_title("Drug Entity Overlap", fontweight="bold")
    ax.set_ylim(0, max(e1_overlaps) * 1.15)
    ax.axhline(y=0, color="#e2e8f0", linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Panel 2: Target (e2) overlap
    ax = axes[1]
    e2_overlaps = []
    for t in techniques:
        eo = metrics[t].get("entity_overlap", {})
        e2_overlaps.append(eo.get("e2_overlap", 0))
    bars = ax.bar(labels, e2_overlaps, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, e2_overlaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Overlapping Targets")
    ax.set_title("Target Entity Overlap", fontweight="bold")
    ax.set_ylim(0, max(e2_overlaps) * 1.15)
    ax.axhline(y=0, color="#e2e8f0", linewidth=0.5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Panel 3: Coverage
    ax = axes[2]
    coverages = [metrics[t].get("coverage", 0) * 100 for t in techniques]
    bars = ax.bar(labels, coverages, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Data Coverage", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="#e2e8f0", linewidth=0.8, linestyle="--")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Davis DTI Dataset — 4,000 Interactions (40 Drugs × 100 Targets) — Entity Overlap Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined_summary(bbbp_metrics, davis_metrics, save_path):
    """Combined figure: 2-row layout showing both datasets."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: BBBP (1D) ──────────────────────────────────────────────────
    techniques_1d = ["R", "I1e", "C1e"]
    labels_1d = [TECHNIQUE_LABELS[t] for t in techniques_1d]
    colors_1d = [COLORS[t] for t in techniques_1d]

    # 1a: NN Distance
    ax = fig.add_subplot(gs[0, 0])
    nn_dists = [bbbp_metrics[t]["nn_leakage"]["mean_nn_dist"] for t in techniques_1d]
    bars = ax.bar(labels_1d, nn_dists, color=colors_1d, edgecolor="white", linewidth=1, width=0.55)
    for bar, val in zip(bars, nn_dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Mean NN Distance")
    ax.set_title("Train-Test Separation", fontweight="bold", fontsize=11)

    # 1b: Zero-distance pairs
    ax = fig.add_subplot(gs[0, 1])
    zero_counts = [bbbp_metrics[t]["nn_leakage"]["zero_dist_count"] for t in techniques_1d]
    bars = ax.bar(labels_1d, zero_counts, color=colors_1d, edgecolor="white", linewidth=1, width=0.55)
    for bar, val in zip(bars, zero_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Identical Pairs")
    ax.set_title("Data Leakage (Duplicates)", fontweight="bold", fontsize=11)

    # 1c: Distribution shift
    ax = fig.add_subplot(gs[0, 2])
    shifts = [bbbp_metrics[t]["distribution_shift"]["mean_normalized_shift"]
              for t in techniques_1d]
    bars = ax.bar(labels_1d, shifts, color=colors_1d, edgecolor="white", linewidth=1, width=0.55)
    for bar, val in zip(bars, shifts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Normalized Shift")
    ax.set_title("Distribution Shift", fontweight="bold", fontsize=11)

    # 1d: Summary text box
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    summary_text = (
        "BBBP — 2,050 Molecules\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Cluster splitting (C1e)\n"
        "eliminates identical\n"
        "train-test pairs and\n"
        f"increases separation by\n"
        f"{nn_dists[2]/nn_dists[0]:.1f}× vs Random.\n\n"
        "Trade-off: higher\n"
        "distribution shift\n"
        "(expected for rigorous\n"
        "splitting)."
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9",
                      edgecolor="#cbd5e1", linewidth=1))

    # Row 1 title
    fig.text(0.5, 0.96, "1D Splitting: BBBP Drug Permeability Dataset",
             ha="center", fontsize=13, fontweight="bold", color="#1e293b")

    # ── Row 2: Davis (2D) ─────────────────────────────────────────────────
    techniques_2d = ["R", "I1e", "I1f", "I2", "C1e", "C2"]
    labels_2d = [t for t in techniques_2d]
    colors_2d = [COLORS[t] for t in techniques_2d]

    # 2a: Drug overlap
    ax = fig.add_subplot(gs[1, 0])
    e1_overlaps = [davis_metrics[t].get("entity_overlap", {}).get("e1_overlap", 0)
                   for t in techniques_2d]
    bars = ax.bar(labels_2d, e1_overlaps, color=colors_2d, edgecolor="white", linewidth=1, width=0.55)
    for bar, val in zip(bars, e1_overlaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Overlapping Drugs")
    ax.set_title("Drug Entity Overlap", fontweight="bold", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 2b: Target overlap
    ax = fig.add_subplot(gs[1, 1])
    e2_overlaps = [davis_metrics[t].get("entity_overlap", {}).get("e2_overlap", 0)
                   for t in techniques_2d]
    bars = ax.bar(labels_2d, e2_overlaps, color=colors_2d, edgecolor="white", linewidth=1, width=0.55)
    for bar, val in zip(bars, e2_overlaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Overlapping Targets")
    ax.set_title("Target Entity Overlap", fontweight="bold", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 2c: Coverage
    ax = fig.add_subplot(gs[1, 2])
    coverages = [davis_metrics[t].get("coverage", 0) * 100 for t in techniques_2d]
    bars = ax.bar(labels_2d, coverages, color=colors_2d, edgecolor="white", linewidth=1, width=0.55)
    for bar, val in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Data Coverage", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 120)
    ax.axhline(y=100, color="#e2e8f0", linewidth=0.8, linestyle="--")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 2d: Summary text box
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    summary_text = (
        "Davis DTI — 4,000 Pairs\n"
        "(40 Drugs × 100 Targets)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Random: {e1_overlaps[0]}/{40} drug +\n"
        f"  {e2_overlaps[0]}/{84} target overlap\n"
        f"  (100% leakage)\n\n"
        f"I2: 0 drug + 0 target\n"
        f"  overlap ({coverages[3]:.0f}% coverage)\n\n"
        f"C1e: 0 drug overlap +\n"
        f"  structural separation\n"
        f"  ({coverages[4]:.0f}% coverage)"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f1f5f9",
                      edgecolor="#cbd5e1", linewidth=1))

    # Row 2 title
    fig.text(0.5, 0.48, "2D Splitting: Davis Drug-Target Interaction Dataset",
             ha="center", fontsize=13, fontweight="bold", color="#1e293b")

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_mp_leakage(metrics, save_path):
    """MP materials (1D): Show how C1e reduces leakage vs R."""
    techniques = ["R", "C1e"]
    labels = [TECHNIQUE_LABELS[t] for t in techniques]
    colors = [COLORS[t] for t in techniques]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Mean NN Distance
    ax = axes[0]
    nn_dists = [metrics[t]["nn_leakage"]["mean_nn_dist"] for t in techniques]
    bars = ax.bar(labels, nn_dists, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, nn_dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean NN Distance")
    ax.set_title("Train-Test Separation", fontweight="bold")

    # Panel 2: Min NN Distance
    ax = axes[1]
    min_dists = [metrics[t]["nn_leakage"]["min_nn_dist"] for t in techniques]
    bars = ax.bar(labels, min_dists, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, min_dists):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Min NN Distance")
    ax.set_title("Worst-Case Separation", fontweight="bold")

    # Panel 3: Distribution shift
    ax = axes[2]
    shifts = [metrics[t]["distribution_shift"]["mean_normalized_shift"] for t in techniques]
    bars = ax.bar(labels, shifts, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, shifts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean Normalized Shift")
    ax.set_title("Distribution Shift", fontweight="bold")

    fig.suptitle("Materials Project — 500 Materials — Data Leakage Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_leakage_heatmap(davis_metrics, save_path):
    """Heatmap showing entity overlap as a fraction for all techniques."""
    techniques = ["R", "I1e", "I1f", "I2", "C1e", "C1f", "C2"]
    present = [t for t in techniques if t in davis_metrics]

    n_drugs = 40
    n_targets = 84

    data = np.zeros((len(present), 2))
    for i, t in enumerate(present):
        eo = davis_metrics[t].get("entity_overlap", {})
        data[i, 0] = eo.get("e1_overlap", 0) / n_drugs * 100
        data[i, 1] = eo.get("e2_overlap", 0) / n_targets * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    # Custom colormap: green (0%) to red (100%)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("leakage",
        ["#10b981", "#fbbf24", "#ef4444"])

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Drug Overlap (%)", "Target Overlap (%)"], fontsize=11)
    ax.set_yticks(range(len(present)))
    labels = [f"{t} ({TECHNIQUE_LABELS[t]})" for t in present]
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(len(present)):
        for j in range(2):
            val = data[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontweight="bold", fontsize=12, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Entity Overlap (%)")
    ax.set_title("Entity Overlap Across Splitting Techniques\n"
                 "Davis DTI (40 Drugs × 100 Targets)",
                 fontweight="bold", fontsize=13, pad=15)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    out_dir = os.path.join(OUTPUT_DIR, "leakage_demo")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating leakage demonstration figures...")
    print()

    # Load metrics
    print("[1/5] Loading metrics...")
    bbbp_metrics = load_metrics("bbbp")
    davis_metrics = load_metrics("davis_large")
    mp_metrics = load_metrics("mp_regression")
    print(f"  BBBP techniques: {list(bbbp_metrics.keys())}")
    print(f"  Davis techniques: {list(davis_metrics.keys())}")
    print(f"  MP materials techniques: {list(mp_metrics.keys())}")

    # Generate figures
    print()
    print("[2/5] BBBP leakage analysis...")
    plot_bbbp_leakage(bbbp_metrics,
                      os.path.join(out_dir, "bbbp_leakage_analysis.png"))

    print("[3/5] Davis entity overlap analysis...")
    plot_davis_leakage(davis_metrics,
                       os.path.join(out_dir, "davis_leakage_analysis.png"))

    print("[4/5] MP materials leakage analysis...")
    plot_mp_leakage(mp_metrics,
                    os.path.join(out_dir, "mp_leakage_analysis.png"))

    print("[5/5] Combined summary + heatmap...")
    plot_combined_summary(bbbp_metrics, davis_metrics,
                          os.path.join(out_dir, "combined_leakage_summary.png"))
    plot_leakage_heatmap(davis_metrics,
                         os.path.join(out_dir, "davis_overlap_heatmap.png"))

    print()
    print(f"All figures saved to {out_dir}/")
