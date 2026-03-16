"""Split quality metrics: leakage detection, distribution comparison, balance."""

import logging
import json
import os

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def compute_split_metrics(feature_data, split_assignments, split_names,
                          entity_overlap=None):
    """Compute quality metrics for a split.

    Args:
        feature_data: dict {entity_id: feature_vector}
        split_assignments: dict {entity_id: split_name}
        split_names: list of split names (e.g. ["train", "test"])
        entity_overlap: optional dict with overlap info, e.g.
            {"e1_overlap": 5, "e1_total": 100, "e2_overlap": 3, "e2_total": 80}

    Returns:
        dict of metrics
    """
    names = sorted(feature_data.keys())
    X = np.array([feature_data[n] for n in names])
    labels = [split_assignments.get(n, "unassigned") for n in names]

    metrics = {}

    # 1. Split size balance
    counts = {}
    for sn in split_names:
        counts[sn] = sum(1 for l in labels if l == sn)
    total = sum(counts.values())
    metrics["split_counts"] = counts
    metrics["split_fractions"] = {sn: round(c / total, 4) if total > 0 else 0
                                   for sn, c in counts.items()}

    # 1b. Coverage: fraction of entities assigned to a named split
    n_assigned = sum(1 for l in labels if l in split_names)
    n_total = len(labels)
    metrics["coverage"] = round(n_assigned / n_total, 4) if n_total > 0 else 0

    # 1c. Entity overlap between splits (passed from pipeline)
    if entity_overlap:
        metrics["entity_overlap"] = entity_overlap

    # Clean feature matrix once for both NN leakage and distribution shift
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Nearest-neighbor leakage (for each test point, is its NN in train?)
    if len(split_names) >= 2 and "train" in split_names:
        test_name = "test" if "test" in split_names else split_names[-1]
        train_idx = [i for i, l in enumerate(labels) if l == "train"]
        test_idx = [i for i, l in enumerate(labels) if l == test_name]

        if train_idx and test_idx:
            X_train = X_clean[train_idx]
            X_test = X_clean[test_idx]

            # Use cosine distance for high-dimensional, euclidean otherwise
            # Fall back to euclidean if any vectors have zero norm (cosine undefined)
            metric = "cosine" if X_clean.shape[1] > 100 else "euclidean"
            if metric == "cosine":
                norms = np.linalg.norm(X_clean, axis=1)
                if (norms == 0).any():
                    metric = "euclidean"
            try:
                dists = cdist(X_test, X_train, metric=metric)
                nn_dists = dists.min(axis=1)
                metrics["nn_leakage"] = {
                    "metric": metric,
                    "mean_nn_dist": round(float(np.mean(nn_dists)), 6),
                    "median_nn_dist": round(float(np.median(nn_dists)), 6),
                    "min_nn_dist": round(float(np.min(nn_dists)), 6),
                    "max_nn_dist": round(float(np.max(nn_dists)), 6),
                    "zero_dist_count": int((nn_dists == 0).sum()),
                    "zero_dist_frac": round(float((nn_dists == 0).mean()), 4),
                }
            except Exception as exc:
                logger.warning(f"  NN leakage computation failed: {exc}")

    # 3. Feature distribution comparison (Wasserstein-like via mean/std shift)
    if len(split_names) >= 2:
        s1 = split_names[0]
        s2 = split_names[-1]
        idx1 = [i for i, l in enumerate(labels) if l == s1]
        idx2 = [i for i, l in enumerate(labels) if l == s2]
        if idx1 and idx2:
            X1 = X_clean[idx1]
            X2 = X_clean[idx2]
            # Per-feature mean absolute difference
            mean_shift = np.abs(X1.mean(axis=0) - X2.mean(axis=0))
            # Normalize by overall std
            overall_std = X_clean.std(axis=0)
            overall_std[overall_std == 0] = 1.0
            normalized_shift = mean_shift / overall_std
            metrics["distribution_shift"] = {
                "mean_normalized_shift": round(float(normalized_shift.mean()), 4),
                "max_normalized_shift": round(float(normalized_shift.max()), 4),
                "num_features": int(X_clean.shape[1]),
            }

    return metrics


def save_metrics(metrics, output_dir, technique, dataset_name):
    """Save metrics to a JSON file."""
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"{technique}_{dataset_name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved metrics: {path}")
    return path
