"""Prepare OC22 (Open Catalyst 2022) as a composition-level PALM dataset.

Source: OC22 IS2RE-total LMDBs (Tran et al. 2023, ACS Catalysis; oxide
electrocatalysis). Each record is an adsorbate+oxide-slab system; we keep the
*composition* (chemical formula from atomic numbers) and the DFT relaxed energy,
mirroring the Materials Project / OMol25 loaders — no 3D structures retained.

Writes ``data/oc22/records.csv`` with columns:
    id, formula, energy, natoms, nads, split

Entity = catalytic system; a distinct materials modality (interfacial /
adsorption) vs the bulk-crystal composition sets. Featurize by formula -> MAGPIE.

The IS2RE LMDBs are pickled PyTorch-Geometric ``Data`` objects; this script
stubs the PyG classes so it reads them WITHOUT installing torch_geometric (only
``torch`` + ``lmdb`` + ``ase``, all present in the ``uma`` env).

Run:
    TMPDIR=/nfs/.../.tmp /homes/rzhu/miniforge3/envs/uma/bin/python -m PALM.data.prepare_oc22
"""

import csv
import glob
import os
import pickle
import sys
import types
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "oc22")
OUT = os.path.join(OUT_DIR, "records.csv")
BASE = os.path.join(OUT_DIR, "is2res_total_train_val_test_lmdbs",
                    "data", "oc22", "is2re-total")
LABELED_SPLITS = ["train", "val_id", "val_ood"]     # test_* are unlabeled

# IS2RE-total LMDBs (114 MB; the S2EF tarball is 20 GB — not needed here).
OC22_URL = ("https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/"
            "is2res_total_train_val_test_lmdbs.tar.gz")


def _ensure_data():
    """Download + extract the OC22 IS2RE LMDBs if not already present."""
    if os.path.isdir(BASE):
        return
    import tarfile
    import urllib.request
    os.makedirs(OUT_DIR, exist_ok=True)
    tgz = os.path.join(OUT_DIR, "is2res.tar.gz")
    print(f"[oc22] downloading IS2RE LMDBs (114 MB) -> {tgz}")
    urllib.request.urlretrieve(OC22_URL, tgz)
    print("[oc22] extracting ...")
    with tarfile.open(tgz) as t:
        t.extractall(OUT_DIR)
    os.remove(tgz)


def _install_pyg_stub():
    """Register minimal torch_geometric modules so the pickles unpickle."""
    def mk(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    tg, tgd, tgdd, tgds = (mk("torch_geometric"), mk("torch_geometric.data"),
                           mk("torch_geometric.data.data"),
                           mk("torch_geometric.data.storage"))

    class Cap:
        def __init__(self, *a, **k): self.__dict__.update(k)
        def __setstate__(self, s): self.__dict__["_state"] = s
        def __setitem__(self, k, v): self.__dict__.setdefault("_items", {})[k] = v

    class Data(Cap): pass
    for c in ("BaseStorage", "GlobalStorage", "NodeStorage", "EdgeStorage"):
        setattr(tgds, c, type(c, (Cap,), {}))
    tgdd.Data = tgd.Data = Data
    tg.data = tgd


def _fields(v):
    """Pull the attribute dict out of a stubbed PyG Data object."""
    st = getattr(v, "_state", None)
    if isinstance(st, dict):
        # fields are usually directly in _state; occasionally nested under _store
        if "atomic_numbers" not in st and isinstance(st.get("_store"), dict):
            return st["_store"]
        store = st.get("_store")
        inner = getattr(store, "_state", None)
        if isinstance(inner, dict):
            return inner
        return st
    return v.__dict__


def prepare(limit_per_split=None) -> str:
    import lmdb
    from ase.data import chemical_symbols

    _ensure_data()
    _install_pyg_stub()
    rows = []
    for split in LABELED_SPLITS:
        shards = sorted(glob.glob(os.path.join(BASE, split, "*.lmdb")))
        if not shards:
            print(f"[oc22] {split}: no shards found, skipping")
            continue
        kept = 0
        for shard in shards:
            env = lmdb.open(shard, readonly=True, lock=False, subdir=False)
            with env.begin() as txn:
                for key, raw in txn.cursor():
                    if not key.decode(errors="ignore").isdigit():
                        continue                             # skip 'length'/metadata keys
                    obj = pickle.loads(raw)
                    if not hasattr(obj, "_state"):
                        continue
                    d = _fields(obj)
                    an = d.get("atomic_numbers")
                    y = d.get("y_relaxed")
                    if an is None or y is None:
                        continue
                    zs = [int(z) for z in an.tolist()]
                    counts = Counter(chemical_symbols[z] for z in zs)
                    formula = "".join(f"{el}{counts[el]}" for el in sorted(counts))
                    rows.append({
                        "id": f"oc22_{split}_{d.get('sid')}",
                        "formula": formula,
                        "energy": float(y),
                        "natoms": int(d.get("natoms", len(zs))),
                        "nads": int(d.get("nads", 0)),
                        "split": split,
                    })
                    kept += 1
            env.close()
            if limit_per_split and kept >= limit_per_split:
                break
        print(f"[oc22] {split}: {kept} systems")
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["id", "formula", "energy", "natoms", "nads", "split"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[oc22] wrote {len(rows)} rows -> {OUT}")
    return OUT


if __name__ == "__main__":
    prepare()
