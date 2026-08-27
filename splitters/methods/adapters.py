"""Registry adapters for DataSAIL and Bemis–Murcko scaffold splitting.

These wrap PALM's existing splitting logic so the two baselines are selectable
through the same registry/CLI/tool surface as the native methods — **without**
rewriting ``splitting.py`` or touching the production pipeline. Both import their
heavy/optional dependencies lazily, so ``import PALM.splitters`` works even when
DataSAIL or RDKit is absent (the method simply raises a clear error when called).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..base import BaseSplitter, SplitResult, SplitSpec, register
from ..common.feature_preparation import choose_metric, feature_matrix_from_dict, to_scipy_metric
from ..common.leakage_metrics import scaled_lpi

logger = logging.getLogger(__name__)

_LEAKAGE_MAX_N = 100_000


def _distance_matrix(X, metric):
    """Normalized [0,1] distance matrix for DataSAIL's custom-similarity mode."""
    from scipy.spatial.distance import pdist, squareform
    from sklearn.preprocessing import StandardScaler

    scipy_metric = to_scipy_metric(metric)
    if scipy_metric == "jaccard":
        dist = squareform(pdist(X, metric="jaccard"))
        dist = np.nan_to_num(dist, nan=1.0)
        return dist
    if scipy_metric == "cosine":
        dist = squareform(pdist(X, metric="cosine"))
    else:
        dist = squareform(pdist(StandardScaler().fit_transform(X), metric="euclidean"))
    dmax = dist.max()
    return dist / dmax if dmax > 0 else dist


@register("datasail")
class DataSailSplitter(BaseSplitter):
    description = "DataSAIL cluster-based cold split (C1e) over a custom similarity"
    arity = "1d"

    @dataclass
    class Params:
        technique: str = "C1e"
        metric: Optional[str] = None
        solver: str = "SCIP"
        max_sec: int = 300
        e_clusters: Optional[int] = None

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        metric = p.metric or choose_metric(X)
        dist = _distance_matrix(X, metric)
        names = [str(i) for i in ids]
        e_clusters = p.e_clusters if p.e_clusters is not None else min(max(9, n // 20), n)

        # grakel (a DataSAIL dep) still does `from numpy import ComplexWarning`,
        # removed in numpy>=1.25; restore the alias so the import succeeds.
        if not hasattr(np, "ComplexWarning"):
            np.ComplexWarning = np.exceptions.ComplexWarning
        from datasail.sail import datasail   # lazy: heavy optional dependency
        e_splits, _f, _i = datasail(
            techniques=[p.technique], splits=list(spec.splits), names=list(spec.names),
            runs=1, solver=p.solver, max_sec=p.max_sec, e_type="O",
            e_data={s: s for s in names}, e_dist=(names, dist),
            e_clusters=e_clusters, delta=0.1, epsilon=0.1)
        raw = e_splits[p.technique][0]
        assignment = {ids[i]: raw.get(str(ids[i]), spec.names[-1]) for i in range(n)}

        labels = [assignment[i] for i in ids]
        leak = round(scaled_lpi(X, labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            technique=p.technique, leakage=leak)


@register("scaffold")
class ScaffoldSplitter(BaseSplitter):
    description = "Bemis–Murcko generic-scaffold grouping, greedy proportional assignment"
    arity = "1d"

    Params = None  # takes SMILES; no tunable params

    def split(self, entities, spec: SplitSpec) -> SplitResult:
        """``entities``: ``{entity_id: SMILES}``."""
        t0 = time.time()
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError as exc:   # pragma: no cover
            raise ImportError("scaffold splitting requires RDKit") from exc

        scaffold_to_entities = {}
        for eid, smiles in entities.items():
            mol = Chem.MolFromSmiles(str(smiles))
            key = "_no_scaffold"
            if mol is not None:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    key = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
                except Exception:
                    key = "_no_scaffold"
            scaffold_to_entities.setdefault(key, []).append(eid)

        sorted_scaffolds = sorted(scaffold_to_entities.items(),
                                  key=lambda x: len(x[1]), reverse=True)
        total = sum(len(e) for _, e in sorted_scaffolds)
        split_sum = sum(spec.splits)
        targets = {n: total * s / split_sum for n, s in zip(spec.names, spec.splits)}
        current = {n: 0 for n in spec.names}

        assignment = {}
        for _scaffold, eids in sorted_scaffolds:
            best = max(spec.names, key=lambda n: targets[n] - current[n])
            for eid in eids:
                assignment[eid] = best
            current[best] += len(eids)

        return self._result(assignment, spec, time.time() - t0,
                            n_scaffolds=len(scaffold_to_entities))
