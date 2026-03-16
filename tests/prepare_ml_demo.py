#!/usr/bin/env python
"""Download molecular property datasets for ML leakage demonstration."""

import os
import urllib.request

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, "data")


def download_esol():
    """Download ESOL (aqueous solubility) dataset (~1128 molecules)."""
    out_dir = os.path.join(DATA_DIR, "esol")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "esol.csv")
    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}")
        return
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    print(f"  Downloading ESOL from {url}...")
    urllib.request.urlretrieve(url, out_file)
    import pandas as pd
    df = pd.read_csv(out_file)
    print(f"  Downloaded {len(df)} molecules")
    print(f"  Columns: {list(df.columns)}")


def download_lipophilicity():
    """Download Lipophilicity dataset (~4200 molecules)."""
    out_dir = os.path.join(DATA_DIR, "lipophilicity")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "lipophilicity.csv")
    if os.path.exists(out_file):
        print(f"  Already exists: {out_file}")
        return
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    print(f"  Downloading Lipophilicity from {url}...")
    urllib.request.urlretrieve(url, out_file)
    import pandas as pd
    df = pd.read_csv(out_file)
    print(f"  Downloaded {len(df)} molecules")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    print("Downloading molecular property datasets...")
    print()
    print("[1/2] ESOL (aqueous solubility)")
    download_esol()
    print()
    print("[2/2] Lipophilicity")
    download_lipophilicity()
    print()
    print("Done!")
