"""Hash-based feature caching to avoid recomputation."""

import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Increment this version whenever feature computation logic changes to
# invalidate all existing caches.  This ensures stale cached values are
# never reused after the underlying code has been updated.
CACHE_VERSION = "2"

CACHE_DIR = os.environ.get("PALM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".palm_cache"))


def _cache_key(entities, entity_type, feature_sets):
    """Generate a deterministic hash key from entity data and feature config.

    The key incorporates ``CACHE_VERSION`` so that bumping the version
    automatically invalidates every previously cached result.
    """
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}".encode())
    h.update(entity_type.encode())
    h.update(json.dumps(sorted(feature_sets) if feature_sets else [], sort_keys=True).encode())
    for eid in sorted(entities.keys()):
        h.update(f"{eid}={entities[eid]}".encode())
    return h.hexdigest()[:16]


def get_cached_features(entities, entity_type, feature_sets):
    """Return cached feature DataFrame if available, else None."""
    key = _cache_key(entities, entity_type, feature_sets)
    path = os.path.join(CACHE_DIR, f"features_{key}.parquet")
    if os.path.isfile(path):
        try:
            df = pd.read_parquet(path)
            logger.info(f"  Cache hit: {path}")
            return df
        except Exception:
            pass
    return None


def save_cached_features(entities, entity_type, feature_sets, df):
    """Save feature DataFrame to cache."""
    key = _cache_key(entities, entity_type, feature_sets)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"features_{key}.parquet")
    try:
        df.to_parquet(path)
        logger.info(f"  Cached features: {path}")
    except Exception as exc:
        logger.warning(f"  Failed to cache features: {exc}")


def get_cached_dist(entities, entity_type, feature_sets):
    """Return cached (names, dist_matrix) if available."""
    key = _cache_key(entities, entity_type, feature_sets)
    names_path = os.path.join(CACHE_DIR, f"dist_names_{key}.json")
    dist_path = os.path.join(CACHE_DIR, f"dist_matrix_{key}.npy")
    if os.path.isfile(names_path) and os.path.isfile(dist_path):
        try:
            with open(names_path) as f:
                names = json.load(f)
            dist = np.load(dist_path)
            logger.info(f"  Cache hit (dist): {dist_path}")
            return names, dist
        except Exception:
            pass
    return None


def save_cached_dist(entities, entity_type, feature_sets, names, dist):
    """Save distance matrix to cache."""
    key = _cache_key(entities, entity_type, feature_sets)
    os.makedirs(CACHE_DIR, exist_ok=True)
    names_path = os.path.join(CACHE_DIR, f"dist_names_{key}.json")
    dist_path = os.path.join(CACHE_DIR, f"dist_matrix_{key}.npy")
    try:
        with open(names_path, "w") as f:
            json.dump(names, f)
        np.save(dist_path, dist)
        logger.info(f"  Cached dist matrix: {dist_path}")
    except Exception as exc:
        logger.warning(f"  Failed to cache dist matrix: {exc}")
