"""Quality featurizer for OMol25: pooled embeddings from the pretrained UMA/eSEN
model released with the dataset.

The learned model is the only single representation that natively spans
organics, metal/transition-metal complexes, electrolytes and biomolecules AND
encodes 3D geometry, charge and spin. Per the OMol25 / representation studies,
a good per-structure vector is the **mean-pooled pre-readout node embedding**
(the node features just before the energy/force output heads).

This module is a thin, guarded wrapper: it imports ``fairchem`` lazily so the
rest of the pipeline runs without it. Install for the quality run:

    pip install fairchem-core        # provides the model + AseDBDataset

then feed the returned matrix straight into ``run_lowrank_split(..., metric="cosine")``.
Extraction hooks the pre-readout node features; the exact attribute path depends
on the checkpoint, so ``NODE_FEATURE_HOOK`` centralizes it for easy adjustment.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# name of the submodule whose output we mean-pool (pre-readout node features).
# Adjust to match the loaded checkpoint's architecture if needed.
NODE_FEATURE_HOOK = "backbone"


def embed_structures(structures: List, checkpoint: str = "uma-sm",
                     batch_size: int = 64, device: Optional[str] = None) -> np.ndarray:
    """Return (n, d) mean-pooled pre-readout embeddings for the given ASE Atoms.

    Args:
        structures: list of ASE Atoms (charge/spin in ``atoms.info``).
        checkpoint: fairchem model name / path (e.g. a UMA or eSEN OMol25 ckpt).
        batch_size: forward-pass batch size.
        device: 'cuda' | 'cpu'; autodetected if None.
    """
    import torch
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets.atomic_data import AtomicData

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = pretrained_mlip.get_predict_unit(checkpoint, device=device)
    model = predictor.model.eval()

    captured = {}

    def _hook(_module, _inp, output):
        # output is expected to carry per-node features; keep the tensor
        captured["node"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = dict(model.named_modules()).get(NODE_FEATURE_HOOK)
    if handle is None:
        raise RuntimeError(
            f"Could not find submodule {NODE_FEATURE_HOOK!r} to hook; inspect "
            f"model.named_modules() and set NODE_FEATURE_HOOK accordingly.")
    hook = handle.register_forward_hook(_hook)

    embeddings = []
    try:
        for start in range(0, len(structures), batch_size):
            batch = structures[start:start + batch_size]
            data = AtomicData.from_ase(batch) if hasattr(AtomicData, "from_ase") else batch
            with torch.no_grad():
                predictor.predict(data)
            node = captured["node"]                       # (sum_atoms, d)
            # mean-pool per structure using atom counts
            sizes = [len(a) for a in batch]
            idx = 0
            for s in sizes:
                embeddings.append(node[idx:idx + s].mean(0).cpu().numpy())
                idx += s
    finally:
        hook.remove()
    return np.vstack(embeddings).astype(np.float32)
