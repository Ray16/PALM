"""DataSAIL integration for 2D dataset splitting."""

import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from datasail.sail import datasail


def compute_dist_matrix(data_dict, use_cosine=False, use_pca=False, pca_components=20):
    """Compute normalized distance matrix from entity embeddings.

    Args:
        data_dict: {entity_name: feature_vector}
        use_cosine: use cosine distance instead of Euclidean
        use_pca: apply PCA before distance computation
        pca_components: number of PCA components

    Returns:
        (names, dist_matrix) tuple
    """
    from sklearn.decomposition import PCA

    names = sorted(data_dict.keys())
    X = np.array([data_dict[n] for n in names])

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


def run_technique(technique, common_kwargs):
    """Run a single DataSAIL technique. Module-level for multiprocessing."""
    t0 = time.time()
    e_s, f_s, i_s = datasail(techniques=[technique], **common_kwargs)
    return technique, i_s[technique], time.time() - t0


def run_splitting(e_data, f_data, interactions, config):
    """Run DataSAIL splitting with adaptive sparsity handling.

    Args:
        e_data: dict {entity_name: feature_vector} for e-entities
        f_data: dict {entity_name: feature_vector} for f-entities
        interactions: list of (e_name, f_name) tuples
        config: SplittingConfig object

    Returns:
        dict mapping technique -> dict mapping (e_id, f_id) -> split_name
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Check feature sparsity
    e_features = np.array([e_data[n] for n in sorted(e_data.keys())])
    f_features = np.array([f_data[n] for n in sorted(f_data.keys())])
    e_sparsity = (e_features == 0).sum() / e_features.size
    f_sparsity = (f_features == 0).sum() / f_features.size
    print(f"  E-entity feature sparsity: {e_sparsity * 100:.1f}%")
    print(f"  F-entity feature sparsity: {f_sparsity * 100:.1f}%")

    # Adaptive distance metric selection
    use_pca_e = e_sparsity > 0.9 and "C2" in config.techniques
    use_pca_f = f_sparsity > 0.9 and "C2" in config.techniques
    use_cosine_e = e_sparsity > 0.5 and not use_pca_e
    use_cosine_f = f_sparsity > 0.5 and not use_pca_f

    if use_pca_e or use_pca_f:
        print(f"  Using PCA dimensionality reduction: {'E' if use_pca_e else ''}{' F' if use_pca_f else ''}")
    if use_cosine_e or use_cosine_f:
        print(f"  Using cosine distance for: {'E' if use_cosine_e else ''}{' F' if use_cosine_f else ''}")

    # Compute distance matrices
    e_names, e_dist = compute_dist_matrix(e_data, use_cosine=use_cosine_e, use_pca=use_pca_e)
    f_names, f_dist = compute_dist_matrix(f_data, use_cosine=use_cosine_f, use_pca=use_pca_f)
    print(f"  Distance matrices: e={e_dist.shape}, f={f_dist.shape}")

    # Adaptive f_clusters based on sparsity
    if f_sparsity > 0.5:
        if f_sparsity > 0.9:
            adaptive_f_clusters = max(5, min(10, len(f_data) // 50))
        else:
            adaptive_f_clusters = min(20, len(f_data) // 20)
        print(f"  High sparsity detected! Reducing f_clusters: {config.f_clusters} -> {adaptive_f_clusters}")
    else:
        adaptive_f_clusters = config.f_clusters
        print(f"  Using f_clusters: {adaptive_f_clusters}")

    # Adaptive e_clusters and relaxation for C2 with sparse features
    adaptive_e_clusters = min(9, len(e_data))
    adaptive_delta = 0.1
    adaptive_epsilon = 0.1
    adaptive_max_sec = config.max_sec

    if "C2" in config.techniques and f_sparsity > 0.5:
        adaptive_e_clusters = max(2, min(3, len(e_data) // 3))
        adaptive_delta = 0.4
        adaptive_epsilon = 0.4
        adaptive_max_sec = max(600, config.max_sec * 2)
        print(f"  C2 constraint relaxation for sparse features:")
        print(f"    e_clusters: {min(9, len(e_data))} -> {adaptive_e_clusters}")
        print(f"    delta/epsilon: 0.1 -> {adaptive_delta}")
        print(f"    max_sec: {config.max_sec} -> {adaptive_max_sec}")
        print(f"  WARNING: C2 may still fail with high-sparsity features!")

    # Build DataSAIL kwargs
    common_kwargs = dict(
        splits=config.splits,
        names=config.names,
        runs=1,
        solver=config.solver,
        max_sec=adaptive_max_sec,
        e_type="O",
        e_data=e_data,
        e_dist=(e_names, e_dist),
        e_clusters=adaptive_e_clusters,
        f_type="O",
        f_data=f_data,
        f_dist=(f_names, f_dist),
        f_clusters=adaptive_f_clusters,
        inter=interactions,
        delta=adaptive_delta,
        epsilon=adaptive_epsilon,
    )

    # Run techniques in parallel
    print(f"  Running {len(config.techniques)} techniques in parallel...")
    all_inter_splits = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=len(config.techniques)) as pool:
        futures = {pool.submit(run_technique, t, common_kwargs): t for t in config.techniques}
        for future in as_completed(futures):
            technique, result, elapsed = future.result()
            all_inter_splits[technique] = result
            print(f"    {technique} finished in {elapsed:.1f}s")

    print(f"  All techniques finished in {time.time() - t_start:.1f}s")

    # Extract first run results
    results = {}
    for technique, run_results in all_inter_splits.items():
        results[technique] = run_results[0]

    return results
