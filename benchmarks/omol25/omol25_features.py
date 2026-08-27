"""Dependency-free featurizer for OMol25 structures (ase + numpy only).

OMol25 is chemically heterogeneous (small organics, metal/transition-metal
complexes, electrolytes, biomolecules) with 3D geometry, and each system also
carries a total CHARGE and SPIN multiplicity that are part of its identity. A
similarity split therefore needs a per-structure vector that encodes:

  - composition  : which elements, in what proportion (covers all 83 elements,
                   incl. transition metals)  -> fractional element histogram
  - elemental    : mass / covalent-radius statistics (a MAGPIE-lite summary)
  - 3D geometry  : element-agnostic radial distance distribution (a mini-RDF)
  - state        : total charge and spin multiplicity

This needs no gated model download and no heavy deps, so it runs today for a
first end-to-end low-rank split. For the QUALITY run, swap this for a pooled
UMA/eSEN embedding (see ``omol25_embed.py`` / the README) — the split code is
unchanged, only the feature matrix differs.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, covalent_radii

Z_MAX = 83                 # OMol25 spans elements up to Z~83
RDF_BINS = 24
RDF_MAX_A = 6.0            # Angstrom; interatomic distances beyond this add little


def featurize_atoms(atoms: Atoms, z_max: int = Z_MAX,
                    rdf_bins: int = RDF_BINS, rdf_max: float = RDF_MAX_A) -> np.ndarray:
    """One structure -> a fixed-length descriptor (composition | elemental | RDF | state)."""
    Z = atoms.get_atomic_numbers()
    n_atoms = len(Z)

    # composition: fractional element histogram over Z = 1..z_max
    comp = np.bincount(Z, minlength=z_max + 1)[1:z_max + 1].astype(np.float64)
    comp /= max(comp.sum(), 1.0)

    # elemental property statistics (built into ase.data)
    masses = atomic_masses[Z]
    radii = covalent_radii[Z]
    elemental = np.array([masses.mean(), masses.std(), radii.mean(), radii.std(),
                          float(n_atoms)])

    # mini-RDF: normalized histogram of interatomic distances
    if n_atoms > 1:
        d = atoms.get_all_distances(mic=False)
        d = d[np.triu_indices(n_atoms, k=1)]
        rdf, _ = np.histogram(d, bins=rdf_bins, range=(0.0, rdf_max))
        rdf = rdf.astype(np.float64)
        rdf /= max(rdf.sum(), 1.0)
    else:
        rdf = np.zeros(rdf_bins)

    # state: charge and spin multiplicity (stored in atoms.info for OMol25)
    charge = float(atoms.info.get("charge", 0))
    spin = float(atoms.info.get("spin", atoms.info.get("spin_multiplicity", 1)))

    return np.concatenate([comp, elemental, rdf, [charge, spin]]).astype(np.float32)


def featurize_dataset(structures: Iterable[Atoms], standardize: bool = True
                      ) -> Tuple[np.ndarray, List[str]]:
    """Featurize an iterable of ASE Atoms.

    Returns (X, formulas). If ``standardize``, z-score each column (so the
    different feature blocks are comparable under cosine/euclidean similarity).
    ``formulas`` are the Hill chemical formulas — used to build the composition
    (formula) baseline split that OMol25 ships with.
    """
    feats, formulas = [], []
    for atoms in structures:
        feats.append(featurize_atoms(atoms))
        formulas.append(atoms.get_chemical_formula())
    X = np.vstack(feats).astype(np.float32)
    if standardize:
        mu = X.mean(0)
        sigma = X.std(0)
        sigma[sigma == 0] = 1.0
        X = ((X - mu) / sigma).astype(np.float32)
    return X, formulas
