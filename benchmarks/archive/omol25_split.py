"""Run a low-rank leakage-minimizing split on OMol25 and compare it against the
dataset's native composition (formula) split.

OMol25 ships a *formula-based* split: every conformer of a chemical formula is
kept in one partition. That removes conformer leakage but NOT cross-formula
chemical similarity (a molecule and its methyl homolog, or a complex differing
by one ligand, can sit in different formulas yet be highly similar). The
low-rank splitter works on a continuous similarity, so it can suppress that
residual leakage. This script quantifies the difference.

Usage (palm env; requires OMol25 data + the featurizer's deps):
    python -m PALM.lowrank_split.omol25.omol25_split --src /path/to/omol25/train_4M --limit 50000
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List, Optional

import numpy as np

sys.path.insert(0, "/nfs/lambda_stor_01/homes/rzhu")
from PALM.lowrank_split.lowrank_split import run_lowrank_split
from PALM.lowrank_split.omol25.omol25_features import featurize_dataset


# ── data loading ───────────────────────────────────────────────────────────

def load_omol25(src: str, limit: Optional[int] = None) -> List:
    """Load OMol25 structures as a list of ASE Atoms.

    Prefers fairchem's AseDBDataset (native *.aselmdb reader); falls back to an
    ASE database / trajectory the caller points at. Charge and spin live in
    ``atoms.info`` for OMol25.
    """
    try:
        from fairchem.core.datasets import AseDBDataset
        ds = AseDBDataset({"src": src})
        n = len(ds) if limit is None else min(limit, len(ds))
        return [ds.get_atoms(i) for i in range(n)]
    except ImportError:
        pass
    # fallback: anything ASE can read (.db, .json, .traj, .xyz, directory)
    from ase.io import iread
    out = []
    for i, atoms in enumerate(iread(src)):
        if limit is not None and i >= limit:
            break
        out.append(atoms)
    return out


# ── leakage metrics on dense features (cosine) ─────────────────────────────

def _cosine_leakage(X: np.ndarray, labels: np.ndarray, block: int = 4096):
    """Scaled cross-split cosine leakage + per-test nearest-train cosine summary."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.nn.functional.normalize(torch.as_tensor(X, dtype=torch.float32, device=dev), dim=1)
    lab = torch.as_tensor(labels, dtype=torch.long, device=dev)
    n = Xt.shape[0]
    total = torch.zeros((), device=dev)
    leak = torch.zeros((), device=dev)
    tr = Xt[lab == 0]
    nn_max = []
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = Xt[s:e] @ Xt.T
        total += sim.sum()
        cross = (lab[s:e][:, None] != lab[None, :]).float()
        leak += (sim * cross).sum()
    for s in range(0, tr.shape[0], block):    # NN leakage: test rows vs train
        pass
    te = Xt[lab == 1]
    for s in range(0, te.shape[0], block):
        e = min(s + block, te.shape[0])
        sim = te[s:e] @ tr.T
        nn_max.append(sim.max(1).values)
    nn = torch.cat(nn_max) if nn_max else torch.zeros(1, device=dev)
    return dict(lpi=float((leak / total).item()),
                nn_mean=float(nn.mean().item()),
                nn_p90=float((nn >= 0.9).float().mean().item()))


def formula_split(formulas: List[str], test_frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """OMol25-style split: assign whole formulas to train/test (0=train, 1=test)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(formulas))
    rng.shuffle(uniq)
    counts = {f: 0 for f in uniq}
    for f in formulas:
        counts[f] += 1
    n = len(formulas)
    test_formulas, acc = set(), 0
    for f in uniq:
        if acc >= test_frac * n:
            break
        test_formulas.add(f)
        acc += counts[f]
    return np.array([1 if f in test_formulas else 0 for f in formulas])


# ── driver ─────────────────────────────────────────────────────────────────

def compare(structures: Iterable, rank: int = 512, seed: int = 0) -> dict:
    """Featurize, run both splits, return their leakage metrics."""
    X, formulas = featurize_dataset(structures, standardize=True)
    n = X.shape[0]
    fd = {i: X[i] for i in range(n)}

    lr = run_lowrank_split(fd, [8, 2], ["train", "test"], rank=rank,
                           metric="cosine", seed=seed)
    lr_lab = np.array([0 if lr[i] == "train" else 1 for i in range(n)])
    fm_lab = formula_split(formulas, test_frac=0.2, seed=seed)

    return {
        "n": n, "n_formulas": len(set(formulas)),
        "lowrank": _cosine_leakage(X, lr_lab),
        "formula_split": _cosine_leakage(X, fm_lab),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="OMol25 source (aselmdb dir / ASE-readable file)")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--rank", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading up to {args.limit} structures from {args.src} ...", flush=True)
    structures = load_omol25(args.src, limit=args.limit)
    res = compare(structures, rank=args.rank)
    print(f"\nn={res['n']}  unique_formulas={res['n_formulas']}\n")
    print(f"{'split':<16}{'cosine_lpi':>12}{'NN_mean':>10}{'%NN>=0.9':>10}")
    print("-" * 48)
    for name in ("formula_split", "lowrank"):
        m = res[name]
        print(f"{name:<16}{m['lpi']:>12.4f}{m['nn_mean']:>10.3f}{m['nn_p90']*100:>9.1f}%")


if __name__ == "__main__":
    main()
