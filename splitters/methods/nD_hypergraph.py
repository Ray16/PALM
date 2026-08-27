"""n-D (multi-component / reaction) hypergraph splitters.

Each record (e.g. a reaction rA + rB + rC -> product) is a vertex with several
component *axes*. Cutting a similarity relationship on any axis is leakage on
that axis; minimizing the connectivity objective minimizes total cross-split
component leakage across all axes at once — the n-D generalization DataSAIL's
2-D engine cannot express.

- :class:`HypergraphNDSplitter` — one hyperedge per (axis, identity/similarity
  *cluster*); best when axes have recurring exact/near-exact values.
- :class:`HypergraphNDKnnSplitter` — per-axis record-level *k-NN* hyperedges (the
  1-D construction applied per axis); tracks the scaled ``L(pi)`` far better on
  high-cardinality, near-unique axes (e.g. diverse reactants).

Both take an :class:`NDInput` (``records`` + ``axis_feature_maps``) and return a
uniform :class:`~PALM.splitters.base.SplitResult` whose assignment is keyed by
record index.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..base import BaseSplitter, SplitResult, SplitSpec, register
from ..common.feature_preparation import choose_metric
from ..common.leakage_metrics import macro_axis_lpi
from ..common.nearest_neighbors import build_knn_hyperedges
from ..common.mtkahypar_partition import partition_hypergraph
from ..common.split_naming import assign_split_names

logger = logging.getLogger(__name__)

# Above this many records, skip the O(n^2) macro-leakage diagnostic.
_LEAKAGE_MAX_N = 100_000


@dataclass
class NDInput:
    """A multi-component dataset: records + per-axis featurization.

    ``records``: list of ``{axis_name: component_value}`` (one per record).
    ``axis_feature_maps``: ``{axis_name: {value: feature_vector or None}}``;
    None / all-zero features -> identity-only hyperedges on that axis.
    """

    records: List[dict]
    axis_feature_maps: Dict[str, dict]


def _as_nd(data) -> NDInput:
    if isinstance(data, NDInput):
        return data
    if isinstance(data, dict):
        return NDInput(data["records"], data["axis_feature_maps"])
    records, afm = data          # (records, axis_feature_maps) tuple
    return NDInput(records, afm)


def _cluster_axis(values, feat, sim_threshold):
    """Cluster a single axis's unique values by similarity.

    Returns ``{value: cluster_label}``. Values with no feature vector (or an
    all-zero one) fall back to their own identity cluster. ``sim_threshold`` is
    the minimum similarity to merge; 1.0 -> pure identity (no merging).
    """
    feats = [feat.get(v) for v in values]
    has_feat = [f is not None and np.any(f) for f in feats]
    idx = [i for i, h in enumerate(has_feat) if h]
    if sim_threshold >= 1.0 or len(idx) < 2:
        return {v: f"id::{v}" for v in values}

    X = np.asarray([feats[i] for i in idx], dtype=float)
    metric = choose_metric(X)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    if metric in ("tanimoto", "cosine"):
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(X, metric="jaccard" if metric == "tanimoto" else "cosine"))
        cl = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                     linkage="average", distance_threshold=1.0 - sim_threshold)
        lab = cl.fit_predict(D)
    else:
        Xs = StandardScaler().fit_transform(X)
        cl = AgglomerativeClustering(n_clusters=None, linkage="ward",
                                     distance_threshold=float(sim_threshold))
        lab = cl.fit_predict(Xs)

    labels = {}
    for j, i in enumerate(idx):
        labels[values[i]] = f"cl::{int(lab[j])}"
    for i, h in enumerate(has_feat):
        if not h:
            labels[values[i]] = f"id::{values[i]}"
    return labels


def _nd_leakage(records, afm, labels):
    if len(records) > _LEAKAGE_MAX_N:
        return None, None
    macro, per_axis = macro_axis_lpi(records, afm, labels)
    return round(macro, 6), {a: round(v, 6) for a, v in per_axis.items()}


@register("hypergraph_nd")
class HypergraphNDSplitter(BaseSplitter):
    description = "Per-axis identity/similarity-cluster hyperedges, KM1 cut (n-D records)"
    arity = "nd"

    @dataclass
    class Params:
        sim_threshold: float = 1.0
        axis_weights: Optional[dict] = None
        threads: int = 8
        preset: str = "quality"

    def split(self, data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        nd = _as_nd(data)
        records, afm = nd.records, nd.axis_feature_maps
        n = len(records)
        axes = list(afm.keys())
        aw = p.axis_weights or {}

        axis_clusters = {}
        for axis in axes:
            values = sorted({str(r[axis]) for r in records})
            axis_clusters[axis] = _cluster_axis(values, afm[axis], p.sim_threshold)
            n_identity = sum(1 for lab in axis_clusters[axis].values() if lab.startswith("id::"))
            if n_identity:
                logger.info("  Axis %r: %d/%d values fall back to identity", axis, n_identity, len(values))

        hyperedges, weights = [], []
        for axis in axes:
            v2c = axis_clusters[axis]
            groups = defaultdict(list)
            for i, r in enumerate(records):
                groups[v2c[str(r[axis])]].append(i)
            w = max(1, int(round(aw.get(axis, 1.0) * 1000)))
            for members in groups.values():
                if 2 <= len(members) < n:
                    hyperedges.append(members)
                    weights.append(w)
        if not hyperedges:
            raise ValueError("No non-trivial hyperedges; cannot split (check component variety)")

        block_of, km1, imbalance = partition_hypergraph(
            n, hyperedges, weights, spec.splits, seed=spec.seed,
            threads=p.threads, epsilon=spec.epsilon, preset=p.preset)
        labels = np.asarray(block_of)
        assignment = assign_split_names(list(range(n)), labels, spec.splits, spec.names)

        # per-axis identity overlap (values appearing in >1 split)
        overlap = {}
        for axis in axes:
            by_split = defaultdict(set)
            for i, r in enumerate(records):
                by_split[assignment[i]].add(str(r[axis]))
            shared = set.intersection(*by_split.values()) if len(by_split) > 1 else set()
            overlap[axis] = {"n_values": len({str(r[axis]) for r in records}),
                             "shared_across_splits": len(shared)}

        leak, per_axis = _nd_leakage(records, afm, labels)
        return self._result(assignment, spec, time.time() - t0, km1=int(km1),
                            imbalance=round(imbalance, 4), n_hyperedges=len(hyperedges),
                            axis_overlap=overlap, leakage=leak, axis_leakage=per_axis)


@register("hypergraph_nd_knn")
class HypergraphNDKnnSplitter(BaseSplitter):
    description = "Per-axis record-level k-NN hyperedges, KM1 cut (n-D, high-cardinality axes)"
    arity = "nd"

    @dataclass
    class Params:
        k: int = 25
        threads: int = 8
        preset: str = "quality"

    def split(self, data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        nd = _as_nd(data)
        records, afm = nd.records, nd.axis_feature_maps
        n = len(records)

        all_edges, all_w = [], []
        for axis in afm:
            vals = [str(r[axis]) for r in records]
            feat_map = afm[axis]
            dim = 0
            for v in vals:
                f = feat_map.get(v)
                if f is not None and np.any(f):
                    dim = len(np.asarray(f).ravel()); break
            if not dim:
                continue
            idx_keep, rows = [], []
            for i, v in enumerate(vals):
                f = feat_map.get(v)
                if f is not None and np.any(f):
                    idx_keep.append(i); rows.append(np.asarray(f, dtype=np.float32).ravel())
            if len(idx_keep) < 3:
                continue
            X = np.vstack(rows)
            edges, w = build_knn_hyperedges(X, k=min(p.k, len(idx_keep) - 1),
                                            metric=choose_metric(X), use_gpu=True)
            remap = np.asarray(idx_keep)
            for e, ww in zip(edges, w):
                all_edges.append([int(remap[j]) for j in e]); all_w.append(ww)
        if not all_edges:
            raise ValueError("No k-NN hyperedges built (no axis had usable features)")

        block_of, km1, imbalance = partition_hypergraph(
            n, all_edges, all_w, spec.splits, seed=spec.seed,
            threads=p.threads, epsilon=spec.epsilon, preset=p.preset)
        labels = np.asarray(block_of)
        assignment = assign_split_names(list(range(n)), labels, spec.splits, spec.names)
        leak, per_axis = _nd_leakage(records, afm, labels)
        return self._result(assignment, spec, time.time() - t0, km1=int(km1),
                            imbalance=round(imbalance, 4), n_hyperedges=len(all_edges),
                            leakage=leak, axis_leakage=per_axis)
