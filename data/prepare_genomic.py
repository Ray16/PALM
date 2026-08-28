"""Prepare a DNA / genomic sequence-classification dataset for PALM.

Source: Genomic Benchmarks (Gresova et al. 2023, BMC Genomic Data) via the
HuggingFace mirror. Default = ``human_enhancers_cohn`` (binary: enhancer vs not,
500 bp). Writes a tidy ``data/genomic/records.csv`` with columns:
    id, sequence, label, split

Entity = DNA sequence; a genomic-sequence modality distinct from RNA (Rfam).
Featurize by nucleotide k-mer composition (DNA4 = ACGT) — the DNA analog of the
Rfam loader — or hand to a genomic embedding model.

Run (palm env has HF `datasets`; keep caches off the full root disk):
    HF_HOME=/nfs/.../.hf_cache TMPDIR=/nfs/.../.tmp \
        /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.data.prepare_genomic
"""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "genomic")
OUT = os.path.join(OUT_DIR, "records.csv")

# HF dataset id -> friendly note. human_enhancers_cohn: 27,791 seqs, 500 bp, binary.
DEFAULT_HF = "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn"


def prepare(hf_id: str = DEFAULT_HF) -> str:
    from datasets import load_dataset

    ds = load_dataset(hf_id)
    seq_col = "seq" if "seq" in ds["train"].column_names else "sequence"
    rows = []
    for split, part in ds.items():
        for i, r in enumerate(part):
            rows.append({
                "id": f"{split}_{i}",
                "sequence": str(r[seq_col]).upper(),
                "label": int(r["label"]),
                "split": split,
            })
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[genomic] {hf_id}: {len(df)} rows "
          f"({df.label.nunique()} classes) -> {OUT}")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=DEFAULT_HF)
    args = ap.parse_args()
    prepare(args.hf_id)
