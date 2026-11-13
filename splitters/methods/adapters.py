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
import warnings
from collections import Counter
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


def _examples(xs, k=5):
    xs = list(xs)
    return ", ".join(repr(x) for x in xs[:k]) + (" ..." if len(xs) > k else "")


def _validate_datasail_coverage(raw, names, split_names):
    """Check DataSAIL's ``{name: split}`` map covers ``names`` exactly once.

    DataSAIL may silently drop entities (or mark them ``"not selected"``); the
    old ``raw.get(id, last_split)`` turned every such omission into a test
    member. Missing, duplicate, unknown IDs and unknown split labels now raise.
    """
    problems = []
    dup = [k for k, c in Counter(names).items() if c > 1]
    if dup:
        problems.append(f"{len(dup)} duplicate input IDs (after str()): {_examples(dup)}")
    name_set = set(names)
    missing = [k for k in dict.fromkeys(names) if k not in raw]
    if missing:
        problems.append(f"{len(missing)} IDs missing from DataSAIL output: {_examples(missing)}")
    unknown = [k for k in raw if k not in name_set]
    if unknown:
        problems.append(f"{len(unknown)} unknown IDs in DataSAIL output: {_examples(unknown)}")
    allowed = set(split_names)
    bad = [k for k in dict.fromkeys(names) if k in raw and raw[k] not in allowed]
    if bad:
        labels = sorted({str(raw[k]) for k in bad})
        problems.append(f"{len(bad)} IDs assigned to unknown split labels {labels} "
                        f"(expected {sorted(allowed)}): {_examples(bad)}")
    if problems:
        raise RuntimeError(f"DataSAIL coverage error ({len(names)} input IDs, "
                           f"{len(raw)} returned): " + "; ".join(problems))


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
        _validate_datasail_coverage(raw, names, spec.names)
        assignment = {ids[i]: raw[names[i]] for i in range(n)}

        labels = [assignment[i] for i in ids]
        leak = round(scaled_lpi(X, labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            technique=p.technique, leakage=leak)


@register("scaffold")
class ScaffoldSplitter(BaseSplitter):
    """Bemis–Murcko generic-scaffold grouping with greedy proportional assignment.

    Grouping policy:

    * **Ring-bearing molecules** are grouped by their generic Murcko scaffold, so
      molecules sharing a ring system never straddle splits.
    * **Acyclic molecules** (empty scaffold) each form their *own singleton group*
      by default (``acyclic="singleton"``). An acyclic molecule has no ring
      system, so there is no scaffold relation — and hence no scaffold leakage —
      linking it to any other molecule. Pooling them all under the empty scaffold
      ``""`` (``acyclic="shared"``, the old behaviour) invents a relation that
      does not exist and creates one giant group that can swallow a whole split
      on acyclic-heavy data (e.g. ESOL, FreeSolv).
    * **Invalid / unparseable SMILES** raise by default (``invalid="raise"``);
      pass ``invalid="singleton"`` to give each its own singleton group instead
      of the old silent shared ``_no_scaffold`` bucket.

    After assignment, realized split fractions are checked against the targets:
    any deviation above ``size_tolerance`` (absolute fraction) emits a
    ``UserWarning`` and sets ``diagnostics["balance_ok"] = False``; if the
    largest group alone is bigger than the tolerance allows, the cause is flagged
    as ``diagnostics["giant_group"] = True``.
    """

    description = "Bemis–Murcko generic-scaffold grouping, greedy proportional assignment"
    arity = "1d"

    @dataclass
    class Params:
        acyclic: str = "singleton"      # "singleton" | "shared"
        invalid: str = "raise"          # "raise" | "singleton"
        size_tolerance: float = 0.05    # max |realized - target| split fraction

    def __init__(self, **params):
        super().__init__(**params)
        if self.params.acyclic not in ("singleton", "shared"):
            raise ValueError(f"acyclic must be 'singleton' or 'shared', got {self.params.acyclic!r}")
        if self.params.invalid not in ("raise", "singleton"):
            raise ValueError(f"invalid must be 'raise' or 'singleton', got {self.params.invalid!r}")

    def split(self, entities, spec: SplitSpec) -> SplitResult:
        """``entities``: ``{entity_id: SMILES}``."""
        p = self.params
        t0 = time.time()
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.Scaffolds import MurckoScaffold
        except ImportError as exc:   # pragma: no cover
            raise ImportError("scaffold splitting requires RDKit") from exc
        RDLogger.DisableLog("rdApp.*")

        groups, invalid, n_acyclic = {}, [], 0
        for eid, smiles in entities.items():
            mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
            scaf = None
            if mol is not None and mol.GetNumAtoms() > 0:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    scaf = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
                except Exception:
                    scaf = None
            if scaf is None:
                invalid.append((eid, smiles))
                key = ("invalid", eid)
            elif scaf == "":
                n_acyclic += 1
                key = ("acyclic", eid) if p.acyclic == "singleton" else ("scaffold", "")
            else:
                key = ("scaffold", scaf)
            groups.setdefault(key, []).append(eid)

        if invalid and p.invalid == "raise":
            ex = ", ".join(f"{e!r}: {s!r}" for e, s in invalid[:5])
            raise ValueError(f"{len(invalid)} of {len(entities)} SMILES could not be parsed "
                             f"into a scaffold (e.g. {ex}); pass invalid='singleton' to "
                             f"give each its own group")

        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        total = sum(len(e) for _, e in sorted_groups)
        split_sum = sum(spec.splits)
        targets = {n: total * s / split_sum for n, s in zip(spec.names, spec.splits)}
        current = {n: 0 for n in spec.names}

        assignment = {}
        for _key, eids in sorted_groups:
            best = max(spec.names, key=lambda n: targets[n] - current[n])
            for eid in eids:
                assignment[eid] = best
            current[best] += len(eids)

        # post-assignment balance validation
        denom = total or 1
        dev = {n: abs(current[n] - targets[n]) / denom for n in spec.names}
        max_dev = max(dev.values()) if dev else 0.0
        largest = len(sorted_groups[0][1]) if sorted_groups else 0
        largest_frac = largest / denom
        max_target_frac = max(spec.splits) / split_sum
        # a group bigger than every target (+ tolerance) cannot be placed anywhere
        # without breaking balance: no grouping-respecting assignment can succeed
        giant = largest_frac > max_target_frac + p.size_tolerance
        balance_ok = max_dev <= p.size_tolerance
        if not balance_ok:
            msg = (f"scaffold split off target: max |realized - target| fraction "
                   f"{max_dev:.3f} > tolerance {p.size_tolerance} "
                   f"(counts {current}, targets { {k: round(v, 1) for k, v in targets.items()} })")
            if giant:
                msg += (f"; largest scaffold group has {largest}/{total} "
                        f"({largest_frac:.1%}) members, which makes the target "
                        f"fractions unreachable")
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning(msg)

        return self._result(assignment, spec, time.time() - t0,
                            n_scaffolds=len(groups), n_acyclic=n_acyclic,
                            n_invalid=len(invalid), acyclic_policy=p.acyclic,
                            largest_group=largest, largest_group_frac=round(largest_frac, 4),
                            max_fraction_deviation=round(max_dev, 4),
                            size_tolerance=p.size_tolerance,
                            balance_ok=balance_ok, giant_group=bool(giant))


@register("astartes")
class AstartesSplitter(BaseSplitter):
    description = "astartes clustering cold-split (KMeans sampler) — external leakage-min baseline"
    arity = "1d"

    @dataclass
    class Params:
        sampler: str = "kmeans"
        metric: Optional[str] = None

    def split(self, feature_data, spec: SplitSpec) -> SplitResult:
        p = self.params
        t0 = time.time()
        ids, X = feature_matrix_from_dict(feature_data, min_rows=len(spec.splits))
        n = len(ids)
        metric = p.metric or choose_metric(X)
        train_frac = spec.splits[0] / sum(spec.splits)
        import astartes                                   # lazy optional dependency
        out = astartes.train_test_split(np.asarray(X, dtype=float), train_size=train_frac,
                                        sampler=p.sampler, random_state=spec.seed,
                                        return_indices=True)
        tr, te = out[-2], out[-1]                         # (..., idx_train, idx_test)
        name_tr, name_te = spec.names[0], spec.names[-1]
        assignment = {ids[int(i)]: name_tr for i in tr}
        assignment.update({ids[int(i)]: name_te for i in te})
        labels = [assignment[i] for i in ids]
        leak = round(scaled_lpi(X, labels, metric=metric), 6) if n <= _LEAKAGE_MAX_N else None
        return self._result(assignment, spec, time.time() - t0, metric=metric,
                            sampler=p.sampler, leakage=leak)


@register("lohi")
class LohiSplitter(BaseSplitter):
    description = "Lo-Hi 'Hi' split (no cross-split neighbours above a Tanimoto threshold)"
    arity = "1d"

    @dataclass
    class Params:
        similarity_threshold: float = 0.4

    def split(self, entities, spec: SplitSpec) -> SplitResult:
        """``entities``: ``{entity_id: SMILES}`` (molecule-only baseline)."""
        p = self.params
        t0 = time.time()
        ids = list(entities)
        smiles = [str(entities[i]) for i in ids]
        tot = sum(spec.splits)
        import lohi_splitter as lohi                      # lazy optional dependency
        parts = lohi.hi_train_test_split(
            smiles, similarity_threshold=p.similarity_threshold,
            train_min_frac=0.9 * spec.splits[0] / tot,
            test_min_frac=0.5 * spec.splits[-1] / tot, verbose=False)
        train_idx, test_idx = parts[0], parts[1]
        name_tr, name_te = spec.names[0], spec.names[-1]
        assignment = {ids[int(i)]: name_tr for i in train_idx}
        assignment.update({ids[int(i)]: name_te for i in test_idx})
        # leakage on molecule-native ECFP-1024 / Tanimoto over the assigned entities
        leak = None
        assigned = [i for i in ids if i in assignment]
        if 0 < len(assigned) <= _LEAKAGE_MAX_N:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import AllChem
            RDLogger.DisableLog("rdApp.*")
            rows, labs = [], []
            for i in assigned:
                m = Chem.MolFromSmiles(str(entities[i]))
                if m is None:
                    continue
                a = np.zeros(1024, dtype=np.float32)
                a[list(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024).GetOnBits())] = 1.0
                rows.append(a); labs.append(assignment[i])
            if rows:
                leak = round(scaled_lpi(np.asarray(rows), labs, metric="tanimoto"), 6)
        return self._result(assignment, spec, time.time() - t0,
                            similarity_threshold=p.similarity_threshold, leakage=leak)
