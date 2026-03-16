#!/usr/bin/env python
"""Download larger datasets for PALM data leakage demonstration."""

import csv
import json
import os
import pickle
import urllib.request

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, "data")


def download_bbbp():
    """Download BBBP (Blood-Brain Barrier Penetration) dataset (~2050 molecules)."""
    out_dir = os.path.join(DATA_DIR, "bbbp")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "bbbp.csv")
    if os.path.exists(out_file):
        with open(out_file) as f:
            n = sum(1 for _ in f) - 1
        print(f"  Already exists: {out_file} ({n} molecules)")
        return

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    print(f"  Downloading BBBP from {url}...")
    urllib.request.urlretrieve(url, out_file)
    with open(out_file) as f:
        n = sum(1 for _ in f) - 1
    print(f"  Downloaded {n} molecules to {out_file}")


def prepare_davis_large():
    """Prepare expanded Davis DTI dataset (40 drugs x 100 targets)."""
    out_dir = os.path.join(DATA_DIR, "davis_large")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "interactions.csv")
    if os.path.exists(out_file):
        with open(out_file) as f:
            n = sum(1 for _ in f) - 1
        print(f"  Already exists: {out_file} ({n} interactions)")
        return

    base_url = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/"

    print("  Downloading Davis ligands...")
    lig_text = urllib.request.urlopen(base_url + "ligands_can.txt").read().decode()
    drugs = json.loads(lig_text)

    print("  Downloading Davis proteins...")
    prot_text = urllib.request.urlopen(base_url + "proteins.txt").read().decode()
    targets = json.loads(prot_text)

    print("  Downloading Davis affinity matrix...")
    y_bytes = urllib.request.urlopen(base_url + "Y").read()
    Y = pickle.loads(y_bytes, encoding="latin1")

    drug_ids = list(drugs.keys())
    target_ids = list(targets.keys())
    print(f"  Full dataset: {len(drug_ids)} drugs x {len(target_ids)} targets")

    # Use subset: 40 drugs x 100 targets
    n_drugs = min(40, len(drug_ids))
    n_targets = min(100, len(target_ids))
    sub_d = drug_ids[:n_drugs]
    sub_t = target_ids[:n_targets]

    rows = []
    for di, did in enumerate(sub_d):
        for ti, tid in enumerate(sub_t):
            kd = float(Y[di, ti])
            rows.append({
                "drug_id": did,
                "smiles": drugs[did],
                "target_id": tid,
                "sequence": targets[tid],
                "Kd_nM": kd,
            })

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["drug_id", "smiles", "target_id", "sequence", "Kd_nM"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Created {len(rows)} interactions ({n_drugs} drugs x {n_targets} targets)")
    print(f"  Saved: {out_file}")


if __name__ == "__main__":
    print("Downloading larger datasets for leakage demonstration...")
    print()
    print("[1/2] BBBP (Blood-Brain Barrier Penetration)")
    download_bbbp()
    print()
    print("[2/2] Davis DTI (expanded)")
    prepare_davis_large()
    print()
    print("Done!")
