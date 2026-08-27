"""Shared numerical kernels for the PALM splitters.

These modules are the single home for logic that used to be copy-pasted across
``hypergraph.py``, ``lowrank_split.py``, ``metrics.py`` and the benchmark tree:

- :mod:`.feature_preparation`  — feature-matrix prep, binary-fingerprint /
  metric detection
- :mod:`.pairwise_similarity`  — the tanimoto / cosine / euclidean kernel (torch)
- :mod:`.nearest_neighbors`    — GPU/CPU k-NN neighbours and weighted graph edges
- :mod:`.balanced_assignment`  — target sizes, capacity corridor, balanced assign
- :mod:`.split_naming`         — block-index → split-name mapping
- :mod:`.mtkahypar_partition`  — Mt-KaHyPar context/partition helpers (KM1 & CUT)
- :mod:`.fiduccia_mattheyses`  — Fiduccia–Mattheyses single-move polish loop
- :mod:`.leakage_metrics`      — canonical scaled ``L(pi)`` leakage scorers
"""
