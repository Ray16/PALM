"""Hypergraph (PALM) and low-rank (low_rank) splitters as pipeline techniques.

Importing this module registers two techniques with the DataSAIL pipeline:

  hypergraph  1D: ``hypergraph`` — k-NN similarity hyperedges, Mt-KaHyPar KM1 cut
              2D: ``hypergraph_nd_knn`` — per-axis record-level k-NN hyperedges
  lowrank     1D: ``lowrank`` — Nyström factor + balanced Lloyd + FM polish
              2D: ``lowrank_nd`` — per-axis Nyström/identity factors

Both run on the featurized entities the pipeline already computed, so their
splits get the same validation, metrics, plots and exports as DataSAIL's.

Low-rank uses the namespaced ``lowrank.splitters`` API of the ``lowrank-split``
package. An installed copy (``pip install ./PALM/low_rank``) always wins; otherwise
the copy bundled in ``PALM/low_rank`` is used. PALM is imported as ``PALM.splitters``.
"""

import os
import sys
from collections import Counter
from pathlib import Path

from PALM.benchmarks.datasail_pipeline.config import register_technique

PALM_DIR = Path(__file__).resolve().parents[1]          # the PALM checkout
ROOT_DIR = PALM_DIR.parent                               # directory containing it
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Solver threads for Mt-KaHyPar (shared nodes: keep it modest).
THREADS = int(os.environ.get("WEBAPP_SPLITTER_THREADS", "2"))


def _epsilon(splitting):
    """Balance slack handed to the solvers: their 5% default, kept inside the
    pipeline's own size check (``size_tolerance``) so a solver-feasible split
    is never rejected afterwards as off-size."""
    return min(0.05, splitting.size_tolerance)


def _nd_input(ctx):
    """One n-D record per interaction row, plus the per-axis feature maps.

    Records are rows (not unique pairs) so the solvers balance the same row
    fractions the pipeline validates against.
    """
    records = [{"e1": a, "e2": b} for a, b in ctx.interactions]
    afm = {"e1": ctx.e1_data, "e2": ctx.e2_data}
    return {"records": records, "axis_feature_maps": afm}


def _pair_assignment(ctx, row_assignment):
    """Collapse per-row labels to one label per (e1, e2) pair by majority vote.

    Repeated rows of a pair share both components, so the solvers almost always
    co-locate them already; the vote only resolves rare disagreements so that a
    pair never straddles splits (ties go to the earlier split name).
    """
    votes = {}
    for i, pair in enumerate(ctx.interactions):
        votes.setdefault(pair, Counter())[row_assignment[i]] += 1
    order = {n: k for k, n in enumerate(ctx.splitting.names)}
    return {pair: min(c, key=lambda n: (-c[n], order.get(n, len(order))))
            for pair, c in votes.items()}


def _split_hypergraph(ctx):
    from PALM.splitters import SplitSpec, split
    s = ctx.splitting
    spec = SplitSpec(splits=list(s.splits), names=list(s.names), seed=s.seed,
                     epsilon=_epsilon(s))
    if ctx.is_1d:
        res = split("hypergraph", ctx.e1_data, spec, threads=THREADS,
                    use_gpu=_gpu_visible())
        return dict(res.assignment)
    res = split("hypergraph_nd_knn", _nd_input(ctx), spec, threads=THREADS)
    return _pair_assignment(ctx, res.assignment)


def _split_lowrank(ctx):
    try:
        from lowrank.splitters import SplitSpec, split
    except ModuleNotFoundError as exc:
        if exc.name == "lowrank.splitters":
            # Some other `lowrank` won the import, typically PALM's own
            # PALM/lowrank/ because the server was started from inside the
            # checkout. Retrying cannot replace a module already imported.
            found = getattr(sys.modules.get("lowrank"), "__file__", "?")
            raise ImportError(
                f"`lowrank` resolved to {found}, not the lowrank-split package. "
                "Start the app from the directory that contains PALM/ "
                "(see PALM/webapp/run.sh) or `pip install ./PALM/low_rank`.") from exc
        if exc.name != "lowrank":
            raise
        # Not installed: use the copy bundled in PALM/low_rank. Appended so an
        # installed package always takes precedence.
        bundled = str(PALM_DIR / "low_rank")
        if bundled not in sys.path:
            sys.path.append(bundled)
        from lowrank.splitters import SplitSpec, split
    s = ctx.splitting
    eps = _epsilon(s)
    spec = SplitSpec(splits=list(s.splits), names=list(s.names), seed=s.seed,
                     epsilon=eps)
    if ctx.is_1d:
        res = split("lowrank", ctx.e1_data, spec, allowed_size_imbalance=eps,
                    score_exact_leakage=False)
        return dict(res.assignment)
    res = split("lowrank_nd", _nd_input(ctx), spec, allowed_size_imbalance=eps,
                score_exact_leakage=False)
    return _pair_assignment(ctx, res.assignment)


def _gpu_visible():
    """GPU only when the server was started on a reserved GPU (run.sh --gpu)."""
    return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


register_technique(
    "hypergraph", _split_hypergraph,
    description="PALM hypergraph: k-NN similarity hyperedges cut with Mt-KaHyPar")
register_technique(
    "lowrank", _split_lowrank,
    description="Low-rank: Nyström similarity factor, balanced Lloyd + FM polish")
