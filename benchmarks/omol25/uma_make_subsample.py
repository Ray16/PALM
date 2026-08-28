"""Step 1 (palm env): draw a seeded stratified subsample index from meta.parquet.

Writes a CSV the uma-env embedder can read without pyarrow:
    columns = row, split, shard, db_id, native, data_id
`native` is the 0/1/2 native split (0=train_4M,1=val,2=test); we keep native
proportions so the 100k mirrors the full 3-way experiment.
"""
import argparse, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "..", "..", "data", "omol25",
                    "_cache", "meta.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "_cache_uma", "subsample.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    m = pd.read_parquet(META, columns=["split", "shard", "db_id", "data_id", "natoms"])
    n_total = len(m)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_total, size=min(args.n, n_total), replace=False))
    sub = m.iloc[idx].copy()
    sub.insert(0, "row", idx)                      # original position in meta / features.npy
    sub = sub.rename(columns={"split": "native"})
    sub.to_csv(args.out, index=False)
    print(f"wrote {len(sub):,} rows -> {args.out}")
    print("native proportions:", (sub['native'].value_counts(normalize=True)
                                   .round(4).to_dict()))
    print("data_id spread:\n", sub['data_id'].value_counts().to_string())
    print("natoms: mean %.1f  p50 %d  p95 %d  max %d" %
          (sub['natoms'].mean(), sub['natoms'].median(),
           sub['natoms'].quantile(0.95), sub['natoms'].max()))


if __name__ == "__main__":
    main()
