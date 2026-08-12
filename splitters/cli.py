"""Command-line interface: ``python -m PALM.splitters``.

    python -m PALM.splitters list
    python -m PALM.splitters describe [--method lowrank]
    python -m PALM.splitters split --method lowrank --features feats.npz \
        --splits 8 2 --names train test --out split.csv

``--features`` accepts:
  - ``.npz`` with arrays ``ids`` (n,) and ``X`` (n, d), or just ``X`` (ids = 0..n-1)
  - ``.json`` mapping ``{id: [vector]}`` (or ``{id: "SMILES"}`` for --method scaffold)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .base import SplitSpec
from .dispatch import split
from .registry import describe_splitters, list_splitters


def _load_features(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        X = z["X"]
        ids = z["ids"] if "ids" in z else np.arange(len(X))
        return {ids[i].item() if hasattr(ids[i], "item") else ids[i]: X[i]
                for i in range(len(X))}
    if path.endswith(".json"):
        with open(path) as fh:
            raw = json.load(fh)
        if raw and all(isinstance(v, str) for v in raw.values()):
            return raw                                    # SMILES (scaffold)
        return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
    raise ValueError(f"unsupported --features format: {path} (use .npz or .json)")


def _parse_params(items):
    """``["rank=256", "fm=false"]`` -> ``{"rank": 256, "fm": False}`` (typed)."""
    out = {}
    for it in items or []:
        key, _, val = it.partition("=")
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m PALM.splitters",
                                     description="PALM leakage-minimizing dataset splitters")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered splitter names")

    pd = sub.add_parser("describe", help="show splitter descriptions + param schemas")
    pd.add_argument("--method", default=None)

    ps = sub.add_parser("split", help="run a split")
    ps.add_argument("--method", required=True)
    ps.add_argument("--features", required=True, help=".npz or .json feature file")
    ps.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    ps.add_argument("--names", nargs="+", default=["train", "test"])
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--epsilon", type=float, default=0.05)
    ps.add_argument("--param", action="append", default=[],
                    help="method param as key=value (repeatable)")
    ps.add_argument("--out", default=None, help="CSV output (id,split); default stdout JSON")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(list_splitters()))
        return 0

    if args.cmd == "describe":
        rows = describe_splitters()
        if args.method:
            rows = [r for r in rows if r["name"] == args.method]
        print(json.dumps(rows, indent=2))
        return 0

    # split
    feature_data = _load_features(args.features)
    spec = SplitSpec(splits=args.splits, names=args.names, seed=args.seed, epsilon=args.epsilon)
    result = split(args.method, feature_data, spec, **_parse_params(args.param))

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "split"])
            for k, v in result.assignment.items():
                w.writerow([k, v])
        print(f"wrote {len(result.assignment)} assignments -> {args.out}")
        print(json.dumps(result.diagnostics, indent=2))
    else:
        print(json.dumps(result.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
