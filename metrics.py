"""Split quality metrics: leakage detection, distribution comparison, balance."""

import logging
import json
import os
import itertools

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _choose_metric(X):
    """Select a distance metric that mirrors the one used for splitting.

    This keeps the *audit* (leakage measurement) consistent with the space the
    split was actually built in:
      - binary fingerprints (all 0/1, >=128 dims) -> Jaccard (== Tanimoto)
      - sparse features (>50% zeros)              -> cosine
      - dense features                            -> euclidean

    Cosine is undefined for zero-norm rows, so we fall back to euclidean when
    any all-zero vector is present.
    """
    if X.shape[1] >= 128 and np.all((X == 0) | (X == 1)):
        return "jaccard"
    sparsity = (X == 0).sum() / X.size if X.size else 0.0
    if sparsity > 0.5:
        norms = np.linalg.norm(X, axis=1)
        if (norms == 0).any():
            return "euclidean"
        return "cosine"
    return "euclidean"


def _pair_leakage(X, ref_idx, query_idx, metric):
    """Nearest-neighbor leakage of a query split against a reference split.

    For every entity in ``query_idx`` we find the distance to its nearest
    neighbor in ``ref_idx``. Small distances (and especially zero distances)
    mean query entities are near-duplicates of reference entities -> leakage.
    """
    if not ref_idx or not query_idx:
        return None
    X_ref = X[ref_idx]
    X_query = X[query_idx]
    try:
        dists = cdist(X_query, X_ref, metric=metric)
        # Jaccard is NaN when both vectors are all-zero; treat as maximally distant.
        if metric == "jaccard":
            dists = np.nan_to_num(dists, nan=1.0)
        nn_dists = dists.min(axis=1)
        return {
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
        return None


def _pair_distribution_shift(X, idx_a, idx_b, overall_std):
    """Per-feature normalized mean shift between two splits."""
    if not idx_a or not idx_b:
        return None
    mean_shift = np.abs(X[idx_a].mean(axis=0) - X[idx_b].mean(axis=0))
    normalized_shift = mean_shift / overall_std
    return {
        "mean_normalized_shift": round(float(normalized_shift.mean()), 4),
        "max_normalized_shift": round(float(normalized_shift.max()), 4),
        "num_features": int(X.shape[1]),
    }


def compute_split_metrics(feature_data, split_assignments, split_names,
                          entity_overlap=None):
    """Compute quality metrics for a split.

    Leakage and distribution shift are computed for *every* ordered pair of
    named splits (train->val, train->test, val->test, ...), so validation sets
    are audited too — not just train vs test. The ``nn_leakage`` and
    ``distribution_shift`` keys carry the primary train-vs-test pair for
    backward compatibility, while ``nn_leakage_pairs`` /
    ``distribution_shift_pairs`` carry all pairs.

    Args:
        feature_data: dict {entity_id: feature_vector}
        split_assignments: dict {entity_id: split_name}
        split_names: list of split names (e.g. ["train", "val", "test"])
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

    # Clean feature matrix once for all downstream distance computations
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    metric = _choose_metric(X_clean)

    # Precompute per-split index lists
    split_idx = {sn: [i for i, l in enumerate(labels) if l == sn]
                 for sn in split_names}

    # 2. Nearest-neighbor leakage for every ordered pair of named splits.
    # Reference = the split listed first (train-like); query = listed later
    # (eval-like). Pairs follow the order in split_names.
    if len(split_names) >= 2:
        pair_leakage = {}
        for ref, query in itertools.combinations(split_names, 2):
            res = _pair_leakage(X_clean, split_idx[ref], split_idx[query], metric)
            if res is not None:
                pair_leakage[f"{ref}->{query}"] = res
        if pair_leakage:
            metrics["nn_leakage_pairs"] = pair_leakage
            # Primary pair for backward-compatible consumers
            if "train" in split_names:
                eval_name = "test" if "test" in split_names else split_names[-1]
                primary = pair_leakage.get(f"train->{eval_name}")
                if primary is not None:
                    metrics["nn_leakage"] = primary
            if "nn_leakage" not in metrics:
                # Fall back to the worst (smallest separation) pair
                worst = min(pair_leakage.values(), key=lambda d: d["mean_nn_dist"])
                metrics["nn_leakage"] = worst

    # 3. Feature distribution comparison for every pair of named splits.
    if len(split_names) >= 2:
        overall_std = X_clean.std(axis=0)
        overall_std[overall_std == 0] = 1.0
        pair_shift = {}
        for a, b in itertools.combinations(split_names, 2):
            res = _pair_distribution_shift(X_clean, split_idx[a], split_idx[b], overall_std)
            if res is not None:
                pair_shift[f"{a}->{b}"] = res
        if pair_shift:
            metrics["distribution_shift_pairs"] = pair_shift
            # Primary: first vs last named split (train vs test)
            primary_key = f"{split_names[0]}->{split_names[-1]}"
            metrics["distribution_shift"] = pair_shift.get(
                primary_key, next(iter(pair_shift.values()))
            )

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
