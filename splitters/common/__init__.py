"""Shared numerical kernels for the PALM splitters.

These modules are the single home for logic shared across the splitter methods
(the standalone ``PALM.hypergraph`` / ``PALM.lowrank`` packages and the adapters)
and the benchmark tree:

- :mod:`.feature_preparation`  — feature-matrix prep, binary-fingerprint /
  metric detection
- :mod:`.pairwise_similarity`  — the tanimoto / cosine / euclidean kernel (torch)
- :mod:`.balanced_assignment`  — target sizes, capacity corridor, balanced assign
- :mod:`.split_naming`         — block-index → split-name mapping
- :mod:`.fiduccia_mattheyses`  — Fiduccia–Mattheyses single-move polish loop
- :mod:`.leakage_metrics`      — canonical scaled ``L(pi)`` leakage scorers

Method-specific kernels live with their package: the k-NN construction and
Mt-KaHyPar partitioning in :mod:`PALM.hypergraph`, the Nyström factorization and
balanced-Lloyd optimizer in :mod:`PALM.lowrank`.
"""
