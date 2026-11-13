"""Installed CLI: ``lowrank-split`` or ``python -m lowrank.splitters``.

Feature inputs are a JSON mapping from IDs to vectors, or an NPZ archive with
``X`` (n, d) and optional ``ids`` (n,). NPZ object IDs remain supported for
compatibility; load only trusted NPZ files because NumPy pickle loading is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from .base import SplitSpec
from .dispatch import split
from .registry import describe_splitters, list_splitters


def _validate_vectors(vectors: Any) -> None:
    """Reject malformed feature geometry before constructing an assignment map."""
    try:
        array = np.asarray(vectors)
        if not np.issubdtype(array.dtype, np.number):
            array = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must contain numeric vectors of equal length") from exc
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("features must be a nonempty two-dimensional matrix (n, d)")
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start + 4096]).all():
            raise ValueError("features contain NaN or infinity; impute explicitly before splitting")


def _load_features(path: str | Path) -> dict:
    """Load trusted feature input without changing its numerical representation.

    JSON vectors retain the historical float32 conversion. NPZ feature arrays
    retain their stored dtype. IDs must be unique and hashable; otherwise a
    Python dictionary would silently discard entities.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            if "X" not in archive:
                raise ValueError("NPZ feature input must contain an array named 'X'")
            vectors = archive["X"]
            _validate_vectors(vectors)
            ids = archive["ids"] if "ids" in archive else np.arange(len(vectors))
            if ids.ndim != 1 or len(ids) != len(vectors):
                raise ValueError("NPZ 'ids' must be one-dimensional with one ID per row of X")
            keys = [value.item() if hasattr(value, "item") else value for value in ids]
            try:
                unique = len(set(keys)) == len(keys)
            except TypeError as exc:
                raise ValueError("NPZ IDs must be hashable scalar values") from exc
            if not unique:
                raise ValueError("NPZ IDs must be unique; duplicate IDs would discard entities")
            return dict(zip(keys, vectors))
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("JSON feature input must be a nonempty object mapping IDs to vectors")
        if all(isinstance(value, str) for value in raw.values()):
            return raw  # Retain string-valued inputs for compatible registered methods.
        _validate_vectors(list(raw.values()))
        return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}
    raise ValueError(f"unsupported --features format: {path} (use .npz or .json)")


def _parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeatable ``key=value`` arguments using the historical scalar types."""
    result = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value:
            raise ValueError(f"invalid --param {item!r}; expected a nonempty key=value")
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run a subcommand, returning zero on success and argparse errors on bad input."""
    parser = argparse.ArgumentParser(
        prog="lowrank-split", description="Capacity-controlled similarity splitting"
    )
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("list", help="list registered splitter names")
    describe = commands.add_parser("describe", help="show descriptions and parameter schemas")
    describe.add_argument("--method", default=None)
    run = commands.add_parser("split", help="split JSON or trusted NPZ features")
    run.add_argument("--method", required=True)
    run.add_argument("--features", required=True, help="JSON mapping or trusted NPZ archive")
    run.add_argument("--splits", nargs="+", type=float, default=[8, 2])
    run.add_argument("--names", nargs="+", default=["train", "test"])
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--epsilon", type=float, default=0.05)
    run.add_argument("--param", action="append", default=[], help="key=value (repeatable)")
    run.add_argument("--out", default=None, help="CSV path (id,split); default stdout JSON")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("\n".join(list_splitters()))
        return 0
    if args.cmd == "describe":
        rows = describe_splitters()
        if args.method:
            rows = [row for row in rows if row["name"] == args.method]
            if not rows:
                parser.error(f"unknown method {args.method!r}; use 'lowrank-split list'")
        print(json.dumps(rows, indent=2))
        return 0

    try:
        params = _parse_params(args.param)
        feature_data = _load_features(args.features)
        spec = SplitSpec(splits=args.splits, names=args.names, seed=args.seed, epsilon=args.epsilon)
        result = split(args.method, feature_data, spec, **params)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "split"])
                writer.writerows(result.assignment.items())
            print(f"wrote {len(result.assignment)} assignments -> {output}")
            print(json.dumps(result.diagnostics, indent=2))
        else:
            print(json.dumps(result.to_json(), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
