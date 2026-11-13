"""DataSAIL integration for 1D and 2D dataset splitting."""

import logging
import os
import sys
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from .cache import get_cached_dist, save_cached_dist, dist_spec

# grakel (a DataSAIL dep) still does `from numpy import ComplexWarning`, removed
# in numpy>=1.25; restore the alias so the import succeeds.
if not hasattr(np, "ComplexWarning"):
    np.ComplexWarning = np.exceptions.ComplexWarning

# Suppress DataSAIL's "tool not found" shell messages during import.
# DataSAIL's settings.py runs os.system("cd-hit -h > /dev/null") etc. at import
# time, producing shell "not found" messages on fd 2 of child processes.
_saved_fd2 = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
try:
    from datasail.sail import datasail
finally:
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd2)
    os.close(_devnull_fd)

logger = logging.getLogger(__name__)


def _is_binary_fingerprint(X):
    """Check if feature matrix looks like a binary fingerprint (all 0/1 values)."""
    return X.shape[1] >= 128 and np.all((X == 0) | (X == 1))


PCA_COMPONENTS = 20


def _cached_dist_matrix(data_dict, label, use_cosine, use_pca,
                        pca_components=PCA_COMPONENTS, use_tanimoto=None):
    """compute_dist_matrix with a cache keyed on feature values + full spec."""
    spec = dist_spec(use_cosine, use_pca, pca_components, use_tanimoto)
    cached = get_cached_dist(data_dict, label, [], spec)
    if cached is not None:
        return cached
    names, dist = compute_dist_matrix(data_dict, use_cosine=use_cosine, use_pca=use_pca,
                                      pca_components=pca_components,
                                      use_tanimoto=use_tanimoto)
    save_cached_dist(data_dict, label, [], names, dist, spec)
    return names, dist


def _run_all(run_fn, techniques, n_jobs):
    """Run ``run_fn(t) -> (technique, result, elapsed, error)`` for each technique.

    Sequential when n_jobs == 1; otherwise a thread pool of n_jobs workers.
    """
    if n_jobs <= 1 or len(techniques) <= 1:
        return [run_fn(t) for t in techniques]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(run_fn, techniques))


def compute_dist_matrix(data_dict, use_cosine=False, use_pca=False,
                        pca_components=PCA_COMPONENTS, use_tanimoto=None):
    """Compute normalized distance matrix from entity embeddings.

    Args:
        data_dict: {entity_name: feature_vector}
        use_cosine: use cosine distance instead of Euclidean
        use_pca: apply PCA before distance computation
        pca_components: number of PCA components
        use_tanimoto: use Tanimoto distance (auto-detected from binary fingerprints
                      if None)

    Returns:
        (names, dist_matrix) tuple
    """
    from sklearn.decomposition import PCA

    names = sorted(data_dict.keys())
    X = np.array([data_dict[n] for n in names])

    # Auto-detect binary fingerprints and use Tanimoto distance
    if use_tanimoto is None:
        use_tanimoto = _is_binary_fingerprint(X)

    if use_tanimoto:
        logger.info("  Using Tanimoto distance (binary fingerprint detected)")
        # Tanimoto distance: 1 - |A & B| / |A | B|
        # Equivalent to Jaccard distance for binary vectors
        dist = squareform(pdist(X, metric="jaccard"))
        # Handle NaN (when both vectors are all-zero)
        dist = np.nan_to_num(dist, nan=1.0)
        return names, dist

    if use_pca and X.shape[1] > pca_components:
        n_components = min(pca_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

    if use_cosine:
        dist = squareform(pdist(X, metric="cosine"))
    else:
        X_scaled = StandardScaler().fit_transform(X)
        dist = squareform(pdist(X_scaled, metric="euclidean"))

    dmax = dist.max()
    if dmax > 0:
        dist = dist / dmax
    return names, dist


def _require_solution(splits, technique, kwargs):
    """DataSAIL returns ``None`` instead of raising when its solver finds no split."""
    if splits is None or technique not in splits or not splits[technique]:
        n_e = len(kwargs.get("e_data") or {})
        n_f = len(kwargs.get("f_data") or {})
        sizes = f"{n_e} e1" + (f" / {n_f} e2" if n_f else "") + " entities"
        raise RuntimeError(
            f"DataSAIL found no feasible {technique} split for ratio "
            f"{list(kwargs.get('splits', []))} with {sizes} (its solver returned no "
            "assignment). Try a larger test fraction tolerance (epsilon), more entities, "
            "or a different technique.")
    return splits[technique]


def run_technique(technique, common_kwargs):
    """Run a single DataSAIL technique. Module-level for multiprocessing."""
    t0 = time.time()
    try:
        e_s, f_s, i_s = datasail(techniques=[technique], **common_kwargs)
        return technique, _require_solution(i_s, technique, common_kwargs), time.time() - t0, None
    except (RuntimeError, ValueError, TypeError) as exc:
        return technique, None, time.time() - t0, str(exc)


def run_splitting(e1_data, e2_data, interactions, config, failures=None):
    """Run DataSAIL splitting with adaptive sparsity handling.

    Args:
        e1_data: dict {entity_name: feature_vector} for e1-entities
        e2_data: dict {entity_name: feature_vector} for e2-entities
        interactions: list of (e1_name, e2_name) tuples
        config: SplittingConfig object
        failures: optional dict, filled with {technique: error message} for
            techniques DataSAIL itself failed on

    Returns:
        dict mapping technique -> dict mapping (e1_id, e2_id) -> split_name
    """
    # Check feature sparsity
    e1_features = np.array([e1_data[n] for n in sorted(e1_data.keys())])
    e2_features = np.array([e2_data[n] for n in sorted(e2_data.keys())])
    e1_sparsity = (e1_features == 0).sum() / e1_features.size
    e2_sparsity = (e2_features == 0).sum() / e2_features.size
    logger.info(f"  Entity1 feature sparsity: {e1_sparsity * 100:.1f}%")
    logger.info(f"  Entity2 feature sparsity: {e2_sparsity * 100:.1f}%")

    # Adaptive distance metric selection
    use_pca_e1 = e1_sparsity > 0.9 and "C2" in config.techniques
    use_pca_e2 = e2_sparsity > 0.9 and "C2" in config.techniques
    use_cosine_e1 = e1_sparsity > 0.5 and not use_pca_e1
    use_cosine_e2 = e2_sparsity > 0.5 and not use_pca_e2

    if use_pca_e1 or use_pca_e2:
        logger.info(f"  Using PCA dimensionality reduction: {'E1' if use_pca_e1 else ''}{' E2' if use_pca_e2 else ''}")
    if use_cosine_e1 or use_cosine_e2:
        logger.info(f"  Using cosine distance for: {'E1' if use_cosine_e1 else ''}{' E2' if use_cosine_e2 else ''}")

    # Compute distance matrices (cached on feature values + preprocessing spec)
    e1_names, e1_dist = _cached_dist_matrix(e1_data, "e1", use_cosine_e1, use_pca_e1)
    e2_names, e2_dist = _cached_dist_matrix(e2_data, "e2", use_cosine_e2, use_pca_e2)

    logger.info(f"  Distance matrices: e1={e1_dist.shape}, e2={e2_dist.shape}")

    # Adaptive e2_clusters based on sparsity
    if e2_sparsity > 0.5:
        if e2_sparsity > 0.9:
            adaptive_e2_clusters = max(5, min(10, len(e2_data) // 50))
        else:
            adaptive_e2_clusters = min(20, len(e2_data) // 20)
        logger.info(f"  High sparsity detected! Reducing e2_clusters: {config.e2_clusters} -> {adaptive_e2_clusters}")
    else:
        adaptive_e2_clusters = config.e2_clusters
        logger.info(f"  Using e2_clusters: {adaptive_e2_clusters}")

    # Adaptive e1_clusters and relaxation for C2 with sparse features
    adaptive_e1_clusters = min(max(9, len(e1_data) // 20), len(e1_data))
    adaptive_delta = 0.1
    adaptive_epsilon = 0.1
    adaptive_max_sec = config.max_sec

    if "C2" in config.techniques and e2_sparsity > 0.5:
        adaptive_e1_clusters = max(2, min(3, len(e1_data) // 3))
        adaptive_delta = 0.4
        adaptive_epsilon = 0.4
        adaptive_max_sec = max(600, config.max_sec * 2)
        logger.info(f"  C2 constraint relaxation for sparse features:")
        logger.info(f"    e1_clusters: {min(9, len(e1_data))} -> {adaptive_e1_clusters}")
        logger.info(f"    delta/epsilon: 0.1 -> {adaptive_delta}")
        logger.info(f"    max_sec: {config.max_sec} -> {adaptive_max_sec}")
        logger.warning(f"  C2 may still fail with high-sparsity features!")

    # Build DataSAIL kwargs
    common_kwargs = dict(
        splits=config.splits,
        names=config.names,
        runs=1,
        solver=config.solver,
        max_sec=adaptive_max_sec,
        e_type="O",
        e_data=e1_data,
        e_dist=(e1_names, e1_dist),
        e_clusters=adaptive_e1_clusters,
        f_type="O",
        f_data=e2_data,
        f_dist=(e2_names, e2_dist),
        f_clusters=adaptive_e2_clusters,
        # Deduplicate: DataSAIL returns one label per (e1, e2) key, and its R
        # technique slices the shuffled list by position, so repeated pairs
        # (multiple rows per pair) land in several slices and the last one
        # wins — skewing R towards the final split (e.g. 20/80 instead of 80/20).
        inter=list(dict.fromkeys(interactions)),
        delta=adaptive_delta,
        epsilon=adaptive_epsilon,
        threads=config.threads,
    )

    # Sequential by default; n_jobs > 1 uses threads (ProcessPoolExecutor
    # causes hangs due to SCIP solver child processes not terminating cleanly)
    logger.info(f"  Running {len(config.techniques)} techniques...")
    all_inter_splits = {}
    t_start = time.time()

    outcomes = _run_all(lambda t: run_technique(t, common_kwargs),
                        list(config.techniques), config.n_jobs)
    for technique, result, elapsed, error in outcomes:
        if error is not None:
            logger.warning(f"    {technique} FAILED after {elapsed:.1f}s: {error}")
            if failures is not None:
                failures[technique] = f"DataSAIL error: {error}"
        else:
            all_inter_splits[technique] = result
            logger.info(f"    {technique} finished in {elapsed:.1f}s")

    logger.info(f"  All techniques finished in {time.time() - t_start:.1f}s")

    if not all_inter_splits:
        raise RuntimeError("All splitting techniques failed. Check logs for details.")

    # Extract first run results
    results = {}
    for technique, run_results in all_inter_splits.items():
        results[technique] = run_results[0]

    return results


# ── 1D splitting (single entity, no interactions) ────────────────────────

# Techniques that work for 1D (no e2-entity / no interactions)
TECHNIQUES_1D = {"R", "I1e", "C1e", "scaffold"}

# Map 2D techniques to 1D equivalents
TECHNIQUE_MAP_1D = {
    "R": "R", "I1e": "I1e", "C1e": "C1e",
    "I1f": "I1e", "I2": "I1e",
    "C1f": "C1e", "C2": "C1e",
    "scaffold": "scaffold",
}


def run_technique_1d(technique, common_kwargs, use_self_inter):
    """Run a single 1D DataSAIL technique."""
    t0 = time.time()
    try:
        e_s, f_s, i_s = datasail(techniques=[technique], **common_kwargs)
        if use_self_inter:
            # R technique: extract from i_splits, convert (e,e) -> e
            result = {k[0]: v for k, v in _require_solution(i_s, technique, common_kwargs)[0].items()}
        else:
            # C1e / I1e: extract from e_splits
            result = _require_solution(e_s, technique, common_kwargs)[0]
        return technique, result, time.time() - t0, None
    except (RuntimeError, ValueError, TypeError) as exc:
        return technique, None, time.time() - t0, str(exc)


def run_splitting_1d(e_data, config, failures=None):
    """Run DataSAIL splitting for 1D data (single entity, no interactions).

    Args:
        e_data: dict {entity_name: feature_vector}
        config: SplittingConfig object
        failures: optional dict, filled with {technique: error message} for
            techniques DataSAIL itself failed on

    Returns:
        dict mapping technique -> dict mapping entity_id -> split_name
    """
    # Map requested techniques to valid 1D equivalents (deduplicate)
    mapped = {}
    for t in config.techniques:
        t1d = TECHNIQUE_MAP_1D.get(t, t)
        if t1d not in TECHNIQUES_1D:
            logger.warning(f"  Skipping unsupported 1D technique: {t}")
            continue
        if t1d not in mapped:
            mapped[t1d] = t  # keep original name for first mapping

    if not mapped:
        raise ValueError("No valid 1D techniques selected")

    techniques_to_run = list(mapped.keys())
    logger.info(f"  1D techniques: {techniques_to_run}")

    # Check feature sparsity
    e_features = np.array([e_data[n] for n in sorted(e_data.keys())])
    e_sparsity = (e_features == 0).sum() / e_features.size
    logger.info(f"  Feature sparsity: {e_sparsity * 100:.1f}%")

    use_pca = e_sparsity > 0.9
    use_cosine = e_sparsity > 0.5 and not use_pca

    # Compute distance matrix (cached on feature values + preprocessing spec)
    e_names, e_dist = _cached_dist_matrix(e_data, "e1", use_cosine, use_pca)
    logger.info(f"  Distance matrix: {e_dist.shape}")

    adaptive_e_clusters = min(max(9, len(e_data) // 20), len(e_data))

    # Build separate kwargs for R (needs self-interactions) vs C1e/I1e (does not)
    base_kwargs = dict(
        splits=config.splits,
        names=config.names,
        runs=1,
        solver=config.solver,
        max_sec=config.max_sec,
        e_type="O",
        e_data=e_data,
        e_dist=(e_names, e_dist),
        e_clusters=adaptive_e_clusters,
        delta=0.1,
        epsilon=0.1,
        threads=config.threads,
    )

    # R technique needs self-interactions + f_data
    r_kwargs = dict(
        **base_kwargs,
        f_type="O",
        f_data=e_data,
        f_dist=(e_names, e_dist),
        f_clusters=adaptive_e_clusters,
        inter=[(n, n) for n in sorted(e_data.keys())],
    )

    results = {}
    t_start = time.time()

    def _run_one(t):
        use_self_inter = (t == "R")
        kwargs = r_kwargs if use_self_inter else base_kwargs
        return run_technique_1d(t, kwargs, use_self_inter)

    for technique, result, elapsed, error in _run_all(_run_one, techniques_to_run,
                                                      config.n_jobs):
        if error is not None:
            logger.warning(f"    {technique} FAILED after {elapsed:.1f}s: {error}")
            if failures is not None:
                failures[technique] = f"DataSAIL error: {error}"
        else:
            results[technique] = result
            logger.info(f"    {technique} finished in {elapsed:.1f}s")

    logger.info(f"  All techniques finished in {time.time() - t_start:.1f}s")

    if not results:
        raise RuntimeError("All splitting techniques failed. Check logs for details.")

    return results


def run_scaffold_splitting(entities, config):
    """Split molecules by Bemis-Murcko scaffold.

    Groups molecules by their generic scaffold, then assigns scaffold groups
    to splits proportionally. Ensures no scaffold appears in multiple splits.

    Args:
        entities: dict {entity_id: SMILES_string}
        config: SplittingConfig object

    Returns:
        dict mapping entity_id -> split_name
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        raise ImportError("Scaffold splitting requires RDKit. Install with: pip install rdkit")

    # Extract scaffolds
    scaffold_to_entities = {}  # scaffold_smiles -> [entity_ids]
    for eid, smiles in entities.items():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            scaffold_key = "_no_scaffold"
        else:
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                generic = MurckoScaffold.MakeScaffoldGeneric(core)
                scaffold_key = Chem.MolToSmiles(generic)
            except Exception:
                scaffold_key = "_no_scaffold"
        scaffold_to_entities.setdefault(scaffold_key, []).append(eid)

    logger.info(f"  Found {len(scaffold_to_entities)} unique scaffolds for {len(entities)} molecules")

    # Sort scaffolds by size (largest first) for greedy bin-packing
    sorted_scaffolds = sorted(scaffold_to_entities.items(), key=lambda x: len(x[1]), reverse=True)

    # Compute target sizes
    total = sum(len(eids) for _, eids in sorted_scaffolds)
    split_sum = sum(config.splits)
    targets = {name: total * s / split_sum for name, s in zip(config.names, config.splits)}
    current = {name: 0 for name in config.names}

    # Greedy assignment: assign each scaffold group to the split that is most under-target
    assignments = {}  # entity_id -> split_name
    for scaffold, eids in sorted_scaffolds:
        # Find the split with the largest remaining capacity
        best_split = max(config.names, key=lambda n: targets[n] - current[n])
        for eid in eids:
            assignments[eid] = best_split
        current[best_split] += len(eids)

    for name in config.names:
        logger.info(f"    {name}: {current[name]} entities ({100*current[name]/total:.1f}%)")

    return assignments
