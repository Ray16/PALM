"""DataSAIL C1e wrappers — the one copy.

Two flavours of the single-technique DataSAIL cluster-based 1-D split (``C1e``):

- :func:`datasail_fingerprint` — the ECFP/Tanimoto ``e_type="M"`` variant that
  clusters molecules directly from SMILES (used by the MoleculeNet benchmarks).
- :func:`datasail_distance` — the ``e_type="O"`` variant that clusters from a
  precomputed distance matrix (used by the OMol25 / UMA-embedding studies, where
  the similarity is a learned kernel rather than a fingerprint).

Both return the raw ``{id: split_name}`` assignment DataSAIL produces, or raise
``RuntimeError`` if the ILP is infeasible.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence


def datasail_fingerprint(e_data: Dict[str, str], splits: Sequence[float] = (8, 2),
                         names: Sequence[str] = ("train", "test"),
                         max_sec: Optional[int] = None, **kw) -> Dict[str, str]:
    """DataSAIL C1e over SMILES (``e_type="M"``). ``e_data`` maps id -> SMILES."""
    from datasail.sail import datasail

    e_s, _, _ = datasail(techniques=["C1e"], splits=list(splits), names=list(names),
                         e_type="M", e_data=e_data, max_sec=max_sec, **kw)
    if e_s.get("C1e") is None:
        raise RuntimeError("DataSAIL C1e returned no assignment (infeasible ILP)")
    return e_s["C1e"][0]


def datasail_distance(names_ids: Sequence[str], dist, splits: Sequence[float] = (8, 2),
                      split_names: Sequence[str] = ("train", "test"),
                      max_sec: Optional[int] = None, epsilon: float = 0.2,
                      e_clusters: int = 100, **kw) -> Dict[str, str]:
    """DataSAIL C1e over a precomputed distance matrix (``e_type="O"``).

    ``names_ids`` is the list of entity ids; ``dist`` the (n, n) float64 distance
    matrix aligned to it. ``epsilon`` / ``e_clusters`` default to the
    feasible-at-scale settings the OMol25 studies use (the default epsilon=0.05
    with 50 clusters is infeasible past a few thousand rows).
    """
    from datasail.sail import datasail

    names_ids = list(names_ids)
    e_data = {n: n for n in names_ids}
    e_s, _, _ = datasail(techniques=["C1e"], splits=list(splits), names=list(split_names),
                         e_type="O", e_data=e_data, e_dist=(names_ids, dist),
                         max_sec=max_sec, epsilon=epsilon, e_clusters=e_clusters, **kw)
    if e_s.get("C1e") is None:
        raise RuntimeError("DataSAIL C1e returned no assignment (infeasible ILP) -- "
                           "try relaxing epsilon further or increasing e_clusters")
    return e_s["C1e"][0]
