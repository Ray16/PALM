"""Split visualization: 2D scatter plots colored by split assignment."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def generate_split_plots(feature_data, split_assignments, output_dir,
                         dataset_name, entity_name, technique,
                         split_names=("train", "test"), method="tsne",
                         precomputed_coords=None):
    """Generate a 2D scatter plot of entities colored by split assignment.

    Args:
        feature_data: dict {entity_id: feature_vector}
        split_assignments: dict {entity_id: split_name}
        output_dir: directory to save plots
        dataset_name: name of the dataset
        entity_name: name of the entity (e.g. "drugs", "targets")
        technique: splitting technique name (e.g. "R", "I2")
        split_names: ordered list of split names
        method: "tsne" or "umap"
        precomputed_coords: optional (names, coords) tuple from a previous
            dimensionality reduction to avoid recomputing t-SNE/UMAP.

    Returns:
        Path to the saved PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping visualization")
        return None

    names = sorted(feature_data.keys())
    X = np.array([feature_data[n] for n in names])

    if X.shape[0] < 3:
        logger.warning("  Too few entities (%d) for visualization, skipping", X.shape[0])
        return None

    # Dimensionality reduction (reuse precomputed if available)
    if precomputed_coords is not None:
        pre_names, pre_xy = precomputed_coords
        if list(pre_names) == names:
            coords = pre_xy
        else:
            coords = _reduce(X, method)
    else:
        coords = _reduce(X, method)
    if coords is None:
        logger.warning("  Dimensionality reduction returned None for %s/%s, skipping plot",
                        technique, entity_name)
        return None

    # Map entities to split labels
    labels = [split_assignments.get(n, "unassigned") for n in names]

    # Color palette
    colors_map = {
        "train": "#2563eb",
        "test": "#dc2626",
        "val": "#d97706",
        "validation": "#d97706",
        "unassigned": "#94a3b8",
        "not selected": "#94a3b8",
    }
    default_colors = ["#2563eb", "#dc2626", "#d97706", "#16a34a", "#7c3aed"]
    for i, sn in enumerate(split_names):
        if sn not in colors_map:
            colors_map[sn] = default_colors[i % len(default_colors)]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for sn in list(split_names) + ["unassigned", "not selected"]:
        mask = [l == sn for l in labels]
        if not any(mask):
            continue
        pts = coords[mask]
        color = colors_map.get(sn, "#94a3b8")
        ax.scatter(pts[:, 0], pts[:, 1], c=color, label=sn,
                   alpha=0.7, s=30, edgecolors="white", linewidths=0.3)

    method_label = "t-SNE" if method == "tsne" else method.upper()
    ax.set_xlabel(f"{method_label}-1", fontsize=16)
    ax.set_ylabel(f"{method_label}-2", fontsize=16)
    ax.set_title(f"{dataset_name} — {entity_name} — {technique}", fontsize=18)
    ax.legend(framealpha=0.9, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fname = f"{technique}_{entity_name}_{method}.png"
    path = os.path.join(plot_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved plot: {path}")
    return path


def _reduce(X, method):
    """Reduce feature matrix to 2D."""
    from sklearn.preprocessing import StandardScaler

    # Handle constant/zero features
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Remove zero-variance columns
    var = X_clean.var(axis=0)
    X_clean = X_clean[:, var > 0] if (var > 0).any() else X_clean

    if X_clean.shape[1] == 0:
        logger.warning("  All features are constant — skipping visualization")
        return None

    # Scale
    X_scaled = StandardScaler().fit_transform(X_clean)

    # PCA pre-reduction if high-dimensional
    if X_scaled.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_comp = min(50, X_scaled.shape[0] - 1)
        X_scaled = PCA(n_components=n_comp).fit_transform(X_scaled)

    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, X_scaled.shape[0] - 1))
            return reducer.fit_transform(X_scaled)
        except ImportError:
            logger.info("  umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, X_scaled.shape[0] // 4))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                     max_iter=1000)
        return tsne.fit_transform(X_scaled)

    logger.warning(f"  Unknown reduction method: {method}")
    return None


def generate_comparison_chart(all_metrics, output_dir, dataset_name):
    """Generate a horizontal bar chart comparing metrics across techniques.

    Args:
        all_metrics: dict {technique_name: metrics_dict} from compute_split_metrics
        output_dir: directory to save the chart
        dataset_name: name of the dataset

    Returns:
        Path to the saved PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping comparison chart")
        return None

    if not all_metrics:
        logger.warning("  No metrics to compare, skipping comparison chart")
        return None

    techniques = list(all_metrics.keys())

    # Collect metric values per technique
    metric_defs = [
        ("Coverage %", lambda m: m.get("coverage", 0) * 100, True),
        ("Entity Overlap", lambda m: (
            m.get("entity_overlap", {}).get("e1_overlap", 0)
        ), False),
        ("NN Separation (mean dist)", lambda m: (
            m.get("nn_leakage", {}).get("mean_nn_dist", 0)
        ), True),
        ("Distribution Shift", lambda m: (
            m.get("distribution_shift", {}).get("mean_normalized_shift", 0)
        ), False),
    ]

    # Filter to metrics that have nonzero data for at least one technique
    active_metrics = []
    for label, extractor, higher_is_better in metric_defs:
        values = [extractor(all_metrics[t]) for t in techniques]
        if any(v != 0 for v in values):
            active_metrics.append((label, values, higher_is_better))

    if not active_metrics:
        logger.warning("  All metrics are zero, skipping comparison chart")
        return None

    n_metrics = len(active_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, max(2.5, 0.6 * len(techniques) + 1.5)))
    if n_metrics == 1:
        axes = [axes]

    for ax, (label, values, higher_is_better) in zip(axes, active_metrics):
        max_val = max(values) if max(values) > 0 else 1
        best_idx = values.index(max(values)) if higher_is_better else values.index(min(values))

        best_val = max(values) if higher_is_better else min(values)
        colors = []
        for i, v in enumerate(values):
            if v == best_val:
                colors.append("#16a34a")  # green for best
            elif higher_is_better:
                colors.append("#d97706" if v >= max_val * 0.7 else "#dc2626")
            else:
                min_val = min(values) if min(values) > 0 else 0
                colors.append("#d97706" if v <= min_val * 1.5 or v == 0 else "#dc2626")

        y_pos = np.arange(len(techniques))
        bars = ax.barh(y_pos, values, color=colors, height=0.5, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(techniques, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.invert_yaxis()

        # Value labels on bars
        for i, (bar, v) in enumerate(zip(bars, values)):
            fmt = f"{v:.1f}" if v >= 1 else f"{v:.4f}"
            suffix = ""
            ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{fmt}{suffix}", va="center", fontsize=8,
                    fontweight="bold" if i == best_idx else "normal",
                    color="#16a34a" if i == best_idx else "#64748b")

        xlim = min(max_val * 1.35, 100) if "%" in label else max_val * 1.35
        ax.set_xlim(0, xlim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Technique Comparison — {dataset_name}", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"comparison_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved comparison chart: {path}")
    return path
