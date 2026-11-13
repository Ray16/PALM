"""Near-duplicate ("tail") leakage report for a split — a splitter-agnostic diagnostic.

The leakage *number* a splitter minimizes is the SUM of cross-split similarity; a small
sum can still hide a few **near-duplicate** pairs (similarity > τ) straddling the split,
which is the contamination that most corrupts an OOD test. This module measures that tail
on the **true** similarity (not any low-rank approximation) for *any* split — random,
scaffold, DataSAIL, lowrank — so it works as a benchmark QC check independent of how the
split was produced.

Headline number: ``held_out_contam_frac`` — the fraction of held-out (test) entities that
have a near-duplicate (sim > τ) in another split, i.e. "how many test entities have an
analog they could have memorised."

Findings context (see ``docs/FINDINGS.md`` Step 8): forcing near-twins onto one side
only widens the realized generalization gap when this tail is large (e.g. congeneric
series like bace, ~11%); when it is already small, constraining it is neutral/harmful.
So use this to *decide whether* to apply a tail constraint, not to apply one blindly.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from lowrank.splitters.common.feature_preparation import choose_metric
from lowrank.splitters.common.pairwise_similarity import pairwise_similarity


def _as_assignment(assignment) -> Mapping[Any, str]:
    """Accept a raw ``{id: split_name}`` mapping or a SplitResult-like object."""
    return getattr(assignment, "assignment", assignment)


def tail_report(feature_data: Mapping[Any, np.ndarray],
                assignment,
                *,
                tau: float = 0.7,
                metric: Optional[str] = None,
                held_out: str = "test",
                chunk: int = 512) -> Dict[str, Any]:
    """Near-duplicate leakage across the splits of ``assignment``.

    ``feature_data`` maps entity id -> feature vector (same dict passed to ``split``);
    ``assignment`` maps id -> split name (or is a ``SplitResult``). Similarities above
    ``tau`` between entities in *different* splits are the tail. Computation is chunked
    over rows (O(n·chunk) memory) and uses the true kernel (auto-selected if ``metric``
    is None), so it is exact, not a Nyström estimate.

    Returns a dict with: ``tau``, ``metric``, ``n``, ``splits`` (sizes),
    ``n_cross_pairs`` (unordered different-split pairs with sim > τ), ``max_cross_sim``,
    ``contam_frac`` (per split: fraction with a cross-split near-twin), and the
    held-out convenience fields ``held_out``, ``held_out_contam_frac``,
    ``held_out_mean_nn_cross_sim`` (None if ``held_out`` is not a split name).
    """
    import torch

    assign = _as_assignment(assignment)
    ids = list(feature_data)
    X = np.stack([np.asarray(feature_data[i]) for i in ids])
    labels = np.array([assign[i] for i in ids])
    n = len(ids)
    if metric is None:
        metric = choose_metric(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    lab_codes, _ = _codes(labels)
    lab_t = torch.as_tensor(lab_codes, device=device)

    row_max = torch.full((n,), -1.0, device=device)     # max cross-split sim per row
    directed_pairs = 0                                    # sum over i of #cross j>τ
    global_max = 0.0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        S = pairwise_similarity(Xt[start:end], Xt, metric)      # (c, n), true kernel
        same = lab_t[start:end, None] == lab_t[None, :]         # same-split (incl. self)
        S = S.masked_fill(same, -1.0)                           # drop same-split + self
        row_max[start:end] = S.max(dim=1).values
        over = S > tau
        directed_pairs += int(over.sum().item())
        if over.any():
            global_max = max(global_max, float(S[over].max().item()))

    row_max_np = row_max.cpu().numpy()
    split_names = sorted(set(labels.tolist()))
    contam_frac = {
        s: (float((row_max_np[labels == s] > tau).mean()) if (labels == s).any() else 0.0)
        for s in split_names
    }

    report: Dict[str, Any] = {
        "tau": tau,
        "metric": metric,
        "n": n,
        "splits": {s: int((labels == s).sum()) for s in split_names},
        "n_cross_pairs": directed_pairs // 2,               # unordered
        "max_cross_sim": round(global_max, 4),
        "contam_frac": {s: round(v, 4) for s, v in contam_frac.items()},
        "held_out": held_out,
        "held_out_contam_frac": None,
        "held_out_mean_nn_cross_sim": None,
    }
    if held_out in contam_frac:
        ho = row_max_np[labels == held_out]
        report["held_out_contam_frac"] = round(contam_frac[held_out], 4)
        # nearest cross-split similarity averaged over held-out entities (clamp the
        # -1 sentinel for any held-out entity that has no other-split rows at all)
        report["held_out_mean_nn_cross_sim"] = round(float(np.clip(ho, 0.0, 1.0).mean()), 4)
    return report


def _codes(labels: np.ndarray):
    """Integer-encode split-name labels (stable, for cheap == masking)."""
    uniq, codes = np.unique(labels, return_inverse=True)
    return codes.astype(np.int64), uniq


def format_report(rep: Dict[str, Any]) -> str:
    """One-line human summary of a :func:`tail_report` result."""
    ho = rep.get("held_out")
    hof = rep.get("held_out_contam_frac")
    hof_s = f"{hof:.1%}" if isinstance(hof, float) else "n/a"
    return (f"tail@τ={rep['tau']} ({rep['metric']}): "
            f"{ho} contam={hof_s}, cross-pairs>τ={rep['n_cross_pairs']}, "
            f"max cross-sim={rep['max_cross_sim']}")
