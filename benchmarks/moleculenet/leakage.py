"""MoleculeNet leakage scoring — thin wrapper over the canonical scorer.

The GPU, chunked scaled-L(pi) that matches DataSAIL's ``eval_split`` now lives in
``PALM.splitters.common.leakage_metrics.scaled_lpi_smiles`` (ECFP-1024 /
Tanimoto, whole molecule). This module re-exports it under the historical
``scaled_lpi`` name and keeps the ``eval_split`` cross-check the ``--validate``
path uses.
"""

from __future__ import annotations

from PALM.splitters.common.leakage_metrics import scaled_lpi_smiles as scaled_lpi


def validate_against_eval_split(smiles, split, tol=1e-3):
    """Confirm ``scaled_lpi`` matches DataSAIL's ``eval_split`` on this split.

    Both are the scaled L(pi) over ECFP/Tanimoto pairs; ``scaled_lpi`` is the
    GPU, chunked reimplementation. Returns ``(ours, theirs, abs_diff, ok)``. Only
    feasible on small datasets (eval_split builds the full n x n matrix on CPU).
    Raises if DataSAIL is unavailable.
    """
    from datasail.eval import eval_split

    data = {s: s for s in smiles}
    theirs, _, _ = eval_split("M", data, None, "ecfp", None, None, split)
    ours, _ = scaled_lpi(list(smiles), split)
    diff = abs(ours - theirs)
    return ours, theirs, diff, diff <= tol
