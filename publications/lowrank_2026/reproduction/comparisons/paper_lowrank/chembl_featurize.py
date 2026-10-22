"""E2 prep — ChEMBL 37 -> ECFP4 (radius 2, 2048 bits), bit-packed, for the scaling study.

    python -I chembl_featurize.py <chemreps.txt.gz> <out_dir> [--procs 16]
Writes <out_dir>/ecfp2048_packed.npy (uint8, n x 256), chembl_ids.npy, smiles.txt.gz.
Unparseable SMILES are dropped. The input is an untrusted download, so run with ``python -I``.
"""
import argparse
import gzip
import os
from multiprocessing import Pool

import numpy as np

NBITS = 2048


def _fp(smi):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = _fp.gen.GetFingerprint(m)
    a = np.zeros((NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.packbits(a)


def _init():
    from rdkit.Chem import rdFingerprintGenerator
    _fp.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids, smis = [], []
    with gzip.open(a.src, "rt") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                ids.append(p[0])
                smis.append(p[1])
    with Pool(a.procs, initializer=_init) as pool:
        fps = pool.map(_fp, smis, chunksize=2000)
    keep = [i for i, f in enumerate(fps) if f is not None and f.any()]
    np.save(os.path.join(a.out, "ecfp2048_packed.npy"), np.stack([fps[i] for i in keep]))
    np.save(os.path.join(a.out, "chembl_ids.npy"), np.array([ids[i] for i in keep]))
    with gzip.open(os.path.join(a.out, "smiles.txt.gz"), "wt") as fh:
        fh.write("\n".join(smis[i] for i in keep))
    print(f"kept {len(keep)} / {len(smis)}")


if __name__ == "__main__":
    main()
