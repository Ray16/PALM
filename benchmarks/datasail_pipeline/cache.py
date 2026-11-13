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
CACHE_VERSION = "3"

# Increment whenever distance-matrix preprocessing (scaling, PCA, metric
# selection, normalization in splitting.compute_dist_matrix) changes.
PREPROCESSING_VERSION = "1"

CACHE_DIR = os.environ.get("PALM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".palm_cache"))

# In-process memo of content hashes, keyed by (path, size, mtime_ns) so a
# large embedding file is only read once per run unless it changes.
_FINGERPRINT_MEMO = {}


def _hash_file(path, h):
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


def file_fingerprint(path):
    """Content fingerprint of a file or directory (None if path is unset).

    Files: sha256 of the bytes. Directories: sha256 over every file's relative
    path and bytes, in sorted order. Missing paths fingerprint as "missing".
    """
    if not path:
        return None
    path = os.path.abspath(str(path))
    if not os.path.exists(path):
        return f"missing:{path}"
    if os.path.isfile(path):
        st = os.stat(path)
        memo_key = (path, st.st_size, st.st_mtime_ns)
        if memo_key not in _FINGERPRINT_MEMO:
            h = hashlib.sha256()
            _hash_file(path, h)
            _FINGERPRINT_MEMO[memo_key] = h.hexdigest()
        return _FINGERPRINT_MEMO[memo_key]
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            h.update(os.path.relpath(fpath, path).encode())
            h.update(file_fingerprint(fpath).encode())
    return h.hexdigest()


def _update_value(h, value):
    """Hash an entity value; arrays are hashed by bytes (str() truncates)."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        h.update(f"{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
    else:
        h.update(str(value).encode())


def _cache_key(entities, entity_type, feature_sets, spec=None):
    """Generate a deterministic hash key from entity data and feature config.

    The key incorporates ``CACHE_VERSION`` so that bumping the version
    automatically invalidates every previously cached result. ``spec`` is a
    JSON-serializable dict of every other setting that changes the cached
    value (featurizer models, content fingerprints, distance preprocessing).
    """
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}".encode())
    h.update(entity_type.encode())
    h.update(json.dumps(sorted(feature_sets) if feature_sets else [], sort_keys=True).encode())
    if spec:
        h.update(json.dumps(spec, sort_keys=True, default=str).encode())
    for eid in sorted(entities.keys()):
        h.update(f"{eid}=".encode())
        _update_value(h, entities[eid])
        h.update(b"\x00")
    return h.hexdigest()[:16]


def get_cached_features(entities, entity_type, feature_sets, spec=None):
    """Return cached feature DataFrame if available, else None."""
    key = _cache_key(entities, entity_type, feature_sets, spec)
    path = os.path.join(CACHE_DIR, f"features_{key}.parquet")
    if os.path.isfile(path):
        try:
            df = pd.read_parquet(path)
            logger.info(f"  Cache hit: {path}")
            return df
        except Exception:
            pass
    return None


def save_cached_features(entities, entity_type, feature_sets, df, spec=None):
    """Save feature DataFrame to cache."""
    key = _cache_key(entities, entity_type, feature_sets, spec)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"features_{key}.parquet")
    try:
        df.to_parquet(path)
        logger.info(f"  Cached features: {path}")
    except Exception as exc:
        logger.warning(f"  Failed to cache features: {exc}")


def dist_spec(use_cosine, use_pca, pca_components, use_tanimoto):
    """Complete preprocessing spec for a distance matrix cache entry."""
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "use_cosine": bool(use_cosine),
        "use_pca": bool(use_pca),
        "pca_components": int(pca_components),
        # None means auto-detect (binary fingerprint -> Tanimoto); the result
        # is a function of the hashed feature values, so "auto" is sufficient.
        "use_tanimoto": "auto" if use_tanimoto is None else bool(use_tanimoto),
    }


def get_cached_dist(entities, entity_type, feature_sets, spec=None):
    """Return cached (names, dist_matrix) if available."""
    key = _cache_key(entities, entity_type, feature_sets, spec)
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


def save_cached_dist(entities, entity_type, feature_sets, names, dist, spec=None):
    """Save distance matrix to cache."""
    key = _cache_key(entities, entity_type, feature_sets, spec)
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
