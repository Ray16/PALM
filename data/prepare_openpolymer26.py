"""Prepare an OpenPolymers 2026 (OPoly26) subsample for splitting.

OPoly26 (arXiv:2512.23117) is 6.57M DFT calculations on polymer clusters. The
public colabfit mirror ``colabfit/OPoly26-train`` stores atomistic records
(formula, positions, energies) as ~2.45GB parquet shards — no SMILES. To respect
the disk budget we **stream** the first ``LIMIT`` rows of just two columns from a
single train shard (a few MB transferred, not the full shard) and write
``data/openpolymer26/records.csv`` (id, formula, y) that the loader featurizes via
MAGPIE composition.

Run:  python -m PALM.data.prepare_openpolymer26
"""

import os

import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "openpolymer26")
LIMIT = 10_000
SHARD = "datasets/colabfit/OPoly26-train/co/co_0.parquet"   # OPoly26 train split


def prepare(limit=LIMIT):
    os.makedirs(OUT, exist_ok=True)
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open(SHARD))
    batch = next(pf.iter_batches(
        batch_size=limit, columns=["chemical_formula_reduced", "energy"]))
    df = batch.to_pandas().rename(
        columns={"chemical_formula_reduced": "formula", "energy": "y"})
    df.insert(0, "id", [f"opoly_{i}" for i in range(len(df))])
    df = df.dropna(subset=["formula"]).reset_index(drop=True)
    path = os.path.join(OUT, "records.csv")
    df[["id", "formula", "y"]].to_csv(path, index=False)
    print(f"[OPoly26] {len(df)} rows (streamed from train) -> {path}")
    return path


if __name__ == "__main__":
    prepare()
