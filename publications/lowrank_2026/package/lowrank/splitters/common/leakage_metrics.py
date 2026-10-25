"""Canonical scaled ``L(pi)`` leakage scorers.

The scaled leakage of a split is the cross-split share of total pairwise
similarity (diagonal excluded):

    L(pi) = ( sum sim(i,j) over i,j in different splits ) / ( sum over all pairs )

Lower is better; 0 = no cross-split similarity. This matches DataSAIL's
``eval_split`` when scored over ECFP/Tanimoto.

- :func:`scaled_lpi` — generic, on a feature matrix + integer labels + metric.
  Used both for benchmark scoring and for ``SplitResult`` diagnostics.
- :func:`scaled_lpi_smiles` — the ECFP-1024/Tanimoto convenience (DataSAIL-exact).
- :func:`macro_axis_lpi` — macro-average over the axes of an n-D record set.
- :func:`nn_leakage` — nearest-neighbour leakage: per held-out point, background-
  ramped similarity to its closest train neighbour, normalized to a random split.
  Tail-sensitive where ``scaled_lpi`` (a mean over all pairs) is not.
- :func:`factor_leakage` — the low-rank factor-space proxy (the optimizer's cheap
  objective; correlates ~0.995–1.0 with the ECFP metric at rank 128–256).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .pairwise_similarity import pairwise_similarity, standardize_for_metric


def scaled_lpi(
    X,
    labels: Sequence[int],
    metric: str = "tanimoto",
    block: int = 4096,
    fold_weights=None,
) -> Optional[float]:
    """Scaled L(pi) over the pairwise ``metric`` similarity of feature rows ``X``.

    ``labels`` is any per-row split id (ints or strings); the diagonal is
    excluded. Non-negative unweighted kernels yield a ratio in [0, 1]. Signed
    cosine and fold weights need not; nonpositive normalization mass is undefined.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from lowrank.diagnostics import prepare_reference

    ref = prepare_reference(X, metric, device)
    Xt = ref
    if fold_weights is None:
        _, lab = np.unique(np.asarray(labels), return_inverse=True)
        weights_t = None
    else:
        from lowrank.factors import validate_fold_weights

        W = validate_fold_weights(fold_weights, len(fold_weights))
        lab = np.asarray(labels)
        if not np.issubdtype(lab.dtype, np.integer) or np.any(
            (lab < 0) | (lab >= len(W))
        ):
            raise ValueError(
                "weighted scores require integer labels aligned to fold_weights"
            )
        weights_t = torch.as_tensor(W, device=device)
    lab_t = torch.as_tensor(lab, dtype=torch.long, device=device)
    n = Xt.shape[0]

    total = torch.zeros((), device=device, dtype=torch.float64)
    leak = torch.zeros((), device=device, dtype=torch.float64)
    diag = torch.zeros((), device=device, dtype=torch.float64)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = pairwise_similarity(ref[s:e], ref, metric)
        rg = torch.arange(s, e, device=device)
        diag += sim[torch.arange(e - s, device=device), rg].sum(dtype=torch.float64)
        total += sim.sum(dtype=torch.float64)
        cross = (
            (lab_t[s:e, None] != lab_t[None, :]).float()
            if weights_t is None
            else weights_t[lab_t[s:e, None], lab_t[None, :]]
        )
        leak += (sim * cross).sum(dtype=torch.float64)
    denom = total - diag  # drop self-similarity
    return float((leak / denom).item()) if float(denom) > 0 else None


def scaled_lpi_smiles(
    smiles: Sequence[str], split: Dict[str, str], block: int = 4096
) -> Tuple[float, int]:
    """DataSAIL-exact scaled L(pi) over whole-molecule ECFP-1024/Tanimoto.

    ``split`` maps SMILES -> split name. Unparseable SMILES are dropped. Returns
    ``(leakage_ratio, n_scored)``.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    def _ecfp1024(s):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        a = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), a
        )
        return a

    fps, labels = [], []
    for s in smiles:
        fp = _ecfp1024(s)
        if fp is None:
            continue
        fps.append(fp)
        labels.append(split[s])
    if not fps:
        return 0.0, 0
    return scaled_lpi(np.asarray(fps), labels, metric="tanimoto", block=block), len(fps)


def axis_features(records, fmap, axis, metric=None):
    """Shared per-record view of one n-D axis: ``(vals, F, has, metric)``.

    The single definition of an axis's similarity, used by both the scorer
    (:func:`macro_axis_lpi`) and the ``lowrank_nd`` optimizer so the two can never
    disagree. ``has[i]`` marks records whose value carries a usable (non-zero)
    feature vector; ``F`` is ``(n, d)`` with zeros where ``has`` is False (``None``
    for identity-only axes). Similarity is:

    - both featured: ``metric`` similarity of the (for euclidean, standardized)
      feature rows;
    - both featureless: identity (1 iff same value);
    - mixed: 0 (the values necessarily differ).

    ``metric=None`` auto-selects from the featured rows (:func:`choose_metric`), so
    dense descriptors get euclidean rather than Tanimoto. Tanimoto is rejected on
    negative-valued features (its self-similarity is not 1 there).
    """
    from .feature_preparation import choose_metric
    from .pairwise_similarity import check_metric

    vals = [str(r[axis]) for r in records]
    n = len(vals)
    has = np.zeros(n, dtype=bool)
    dim = 0
    for v in vals:
        f = fmap.get(v)
        if f is not None and np.any(f):
            dim = int(np.asarray(f).ravel().shape[0])
            break
    if not dim:
        return vals, None, has, None
    F = np.zeros((n, dim), dtype=np.float32)
    for i, v in enumerate(vals):
        f = fmap.get(v)
        if f is not None and np.any(f):
            F[i] = np.asarray(f, np.float32).ravel()
            has[i] = True
    if not np.isfinite(F).all():
        raise ValueError(f"axis {axis!r}: impute non-finite features explicitly")
    if metric is None:
        metric = choose_metric(F[has])
    check_metric(metric)
    if metric == "tanimoto" and not np.all((F[has] == 0) | (F[has] == 1)):
        raise ValueError(
            f"axis {axis!r}: tanimoto needs binary, non-negative features; "
            "pass a different metric for this axis"
        )
    return vals, F, has, metric


def _identity_mass(vals, labels, fold_weights=None):
    """``(total, cross)`` off-diagonal identity-similarity mass among ``vals``."""
    if len(vals) == 0:
        return 0.0, 0.0
    _, codes = np.unique(np.asarray(vals), return_inverse=True)
    c = np.bincount(codes).astype(np.float64)
    if fold_weights is not None:
        counts = np.zeros((len(c), len(fold_weights)))
        np.add.at(counts, (codes, labels), 1.0)
        total = float((c * c).sum() - len(vals))
        return total, float(np.sum((counts @ fold_weights) * counts))
    _, joint = np.unique(np.stack([codes, labels]), axis=1, return_inverse=True)
    cl = np.bincount(np.ravel(joint)).astype(np.float64)
    total = float((c * c).sum() - len(vals))
    within = float((cl * cl).sum() - len(vals))
    return total, total - within


def macro_axis_lpi(
    records,
    axis_feature_maps,
    labels,
    block: int = 2048,
    axis_metrics=None,
    fold_weights=None,
) -> Tuple[float, Dict[str, float]]:
    """Macro-average scaled L(pi) over the component axes of an n-D record set.

    Per axis, similarity is defined by :func:`axis_features` (feature ``metric``
    between featured values, identity between featureless values); leakage is the
    cross-split share of total similarity (diagonal removed). ``axis_metrics``
    optionally maps axis -> metric (default: auto per axis). Returns
    ``(macro_lpi, {axis: lpi})``.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = np.asarray(labels)
    if fold_weights is None:
        _, lab = np.unique(labels, return_inverse=True)
        W = None
    else:
        from lowrank.factors import validate_fold_weights

        W = validate_fold_weights(fold_weights, len(fold_weights))
        if not np.issubdtype(labels.dtype, np.integer) or np.any(
            (labels < 0) | (labels >= len(W))
        ):
            raise ValueError(
                "weighted scores require integer labels aligned to fold_weights"
            )
        lab = labels
    axis_metrics = axis_metrics or {}

    per_axis = {}
    for axis in axis_feature_maps:
        vals, F, has, metric = axis_features(
            records, axis_feature_maps[axis], axis, axis_metrics.get(axis)
        )
        miss = ~has
        total, leak = _identity_mass([v for v, m in zip(vals, miss) if m], lab[miss], W)
        if F is not None and has.any():
            Xt = torch.as_tensor(F[has], device=device)
            ref = standardize_for_metric(Xt, metric)
            lab_t = torch.as_tensor(lab[has], dtype=torch.long, device=device)
            m = ref.shape[0]
            ft = torch.zeros((), device=device, dtype=torch.float64)
            fl = torch.zeros((), device=device, dtype=torch.float64)
            for s in range(0, m, block):
                e = min(s + block, m)
                sim = pairwise_similarity(ref[s:e], ref, metric)
                sim[
                    torch.arange(e - s, device=device),
                    torch.arange(s, e, device=device),
                ] = 0.0
                cross = (
                    (lab_t[s:e, None] != lab_t[None, :])
                    if W is None
                    else torch.as_tensor(W, device=device)[
                        lab_t[s:e, None], lab_t[None, :]
                    ]
                )
                ft += sim.sum(dtype=torch.float64)
                fl += (sim * cross).sum(dtype=torch.float64)
            total += float(ft)
            leak += float(fl)
        if metric == "cosine" and total < -1e-12:
            raise ValueError(
                f"axis {axis!r}: signed cosine has nonpositive normalization mass"
            )
        per_axis[axis] = leak / total if total > 0 else 0.0

    macro = float(np.mean(list(per_axis.values()))) if per_axis else 0.0
    return macro, per_axis


# ---------------------------------------------------------------------------
# Nearest-neighbour leakage (tail-sensitive complement to ``scaled_lpi``)
# ---------------------------------------------------------------------------
#
# ``scaled_lpi`` sums *every* cross-split pair, so on clustered / long-tailed
# data it is dominated by the O(n^2) mass of weak background similarity (on BACE
# ~87% of the total comes from pairs with Tanimoto < 0.4) and mostly reports the
# split proportions; congeneric series also enter quadratically in their size.
# Memorization leakage is a nearest-neighbour effect, so ``nn_leakage`` scores
# each held-out point by its closest reference (train) neighbours, passes that
# through a ramp that zeroes background similarity, and averages per point.

# ECFP/Tanimoto has an absolute scale: below ~0.4 structures are unrelated,
# above ~0.8 they are close analogs / near-duplicates.
NNL_TANIMOTO_RAMP = (0.4, 0.8)
# Other metrics have no absolute scale; the ramp is placed at these quantiles of
# the dataset's own pairwise-similarity distribution.
NNL_QUANTILE_RAMP = (0.99, 0.999)


def _pair_sim_quantiles(ref, metric, qs, max_rows=1024, max_cols=8192, seed=0):
    import torch

    n = ref.shape[0]
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(n, generator=g)[: min(n, max_rows)].to(ref.device)
    cols = torch.randperm(n, generator=g)[: min(n, max_cols)].to(ref.device)
    sim = pairwise_similarity(ref[rows], ref[cols], metric)
    off = rows[:, None] != cols[None, :]
    vals = sim[off].double().cpu().numpy()
    return tuple(float(np.quantile(vals, q)) for q in qs)


def _topk_ref_similarity(ref, q_idx, r_idx, metric, k, block):
    import torch

    kk = min(k, int(r_idx.numel()))
    out = []
    for s in range(0, int(q_idx.numel()), block):
        sim = pairwise_similarity(ref[q_idx[s : s + block]], ref[r_idx], metric)
        out.append(sim.topk(kk, dim=1).values.mean(1))
    return torch.cat(out)


def nn_leakage(
    X,
    labels: Sequence,
    metric: str = "tanimoto",
    reference="train",
    k: int = 1,
    ramp: Tuple[float, float] = None,
    n_perm: int = 5,
    seed: int = 0,
    block: int = 4096,
    return_per_point: bool = False,
    device: str = "cpu",
) -> Dict:
    """Nearest-neighbour leakage: raw ``nnl`` in [0, 1], relative ``nnl_rel`` >= 0.

    For every held-out row ``i`` (any split other than ``reference``), let
    ``s_i`` be the mean of its top-``k`` similarities to the reference split and
    ``h(s) = clip((s - lo) / (hi - lo), 0, 1)``. Then

        nnl     = mean_i h(s_i)              (each held-out point weighs 1)
        nnl_rel = nnl / E[nnl | random split with the same sizes]

    ``nnl_rel`` is 1 for a split as leaky as random and 0 when no held-out point
    has a reference neighbour above ``lo``; it is comparable across datasets and
    split proportions; the relative value can exceed one. ``device`` defaults to
    CPU; CUDA requires an explicit request. ``ramp=(lo, hi)`` defaults to :data:`NNL_TANIMOTO_RAMP`
    for tanimoto and to the :data:`NNL_QUANTILE_RAMP` quantiles of the pairwise
    similarity otherwise (estimated from at most 1024 by 8192 sampled pairs).
    ``reference`` is the split name held-out rows are
    compared against; if absent from ``labels`` the largest split is used.

    Returns a dict with ``nnl``, ``nnl_rel``, ``nnl_random``, ``per_split``
    (``nnl`` per held-out split), ``mean_nn_sim`` (raw mean ``s_i``),
    ``frac_above_hi`` (share of held-out rows with ``s_i >= hi``), and the
    ``ramp``/``k`` used; plus ``per_point`` ``(held_out_row_idx, s_i)`` when
    ``return_per_point``.
    """
    import torch

    from lowrank.diagnostics import prepare_reference, positive_integer

    for name, value in (("k", k), ("block", block), ("n_perm", n_perm)):
        positive_integer(name, value)
    ref = prepare_reference(X, metric, device)
    labels = np.asarray(labels).astype(str)
    names, counts = np.unique(labels, return_counts=True)
    reference = (
        str(reference) if str(reference) in names else str(names[np.argmax(counts)])
    )
    if len(names) < 2:
        raise ValueError(
            "nn_leakage needs at least one held-out split besides the reference"
        )

    estimated_ramp = ramp is None
    if estimated_ramp:
        ramp = (
            NNL_TANIMOTO_RAMP
            if metric == "tanimoto"
            else _pair_sim_quantiles(ref, metric, NNL_QUANTILE_RAMP, seed=seed)
        )
    lo, hi = float(ramp[0]), float(ramp[1])
    if not np.isfinite([lo, hi]).all() or hi < lo or (not estimated_ramp and hi == lo):
        raise ValueError("ramp must have finite increasing bounds")

    def _score(lab):
        is_ref = lab == reference
        q_idx = torch.as_tensor(np.flatnonzero(~is_ref), device=device)
        r_idx = torch.as_tensor(np.flatnonzero(is_ref), device=device)
        s = _topk_ref_similarity(ref, q_idx, r_idx, metric, k, block)
        h = ((s - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0)
        return q_idx.cpu().numpy(), s.cpu().numpy(), h.cpu().numpy()

    q_idx, s, h = _score(labels)
    rng = np.random.default_rng(seed)
    base = float(
        np.mean(
            [_score(rng.permutation(labels))[2].mean() for _ in range(max(n_perm, 1))]
        )
    )
    nnl = float(h.mean())
    held = labels[q_idx]
    out = {
        "nnl": nnl,
        "nnl_rel": nnl / base if base > 0 else None,
        "relative_status": "estimated"
        if base > 0
        else "undefined_zero_reference_baseline",
        "ramp_status": "degenerate_quantiles" if hi == lo else "defined",
        "nnl_random": base,
        "per_split": {
            str(n): float(h[held == n].mean()) for n in names if n != reference
        },
        "mean_nn_sim": float(s.mean()),
        "frac_above_hi": float((s >= hi).mean()),
        "ramp": (lo, hi),
        "k": int(k),
        "reference": reference,
    }
    if return_per_point:
        out["per_point"] = (q_idx, s)
    return out


# NOTE: ``factor_leakage`` (the low-rank factor-space objective) moved to the
# standalone package: ``lowrank.objective.factor_leakage``.
