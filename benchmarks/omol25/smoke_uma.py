"""Smoke test: confirm the fixed AtomicData batching path runs end-to-end on a
small, deliberately heterogeneous subset (mixes structures that carry an energy
attribute with those that don't — the exact condition that crashed the full run).
"""
import os, csv, time
import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "DataSAIL_data", "1D", "omol25")
SPLIT_DIR = {0: "train_4M", 1: "val", 2: "test"}
N = 150

rows = []
with open(os.path.join(HERE, "_cache_uma", "subsample.csv")) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["row"]), int(r["native"]), int(r["shard"]),
                     int(r["db_id"]), int(float(r["natoms"]))))
rows = rows[:N]

# read atoms exactly as uma_embed.read_atoms does (fresh info; drop calc)
atoms, had_energy = [], 0
groups = {}
for i, (row, nat, shard, db_id, natoms) in enumerate(rows):
    groups.setdefault((nat, shard), []).append((i, db_id))
buf = [None] * len(rows)
for (nat, shard), items in sorted(groups.items()):
    db = connect(os.path.join(DATA_DIR, SPLIT_DIR[nat], f"data{shard:04d}.aselmdb"))
    for i, db_id in items:
        r = db.get(id=db_id)
        a = r.toatoms()
        a.calc = None
        a.info = {"charge": float(r.data.get("charge", 0.0)),
                  "spin": float(r.data.get("spin", 1.0))}
        buf[i] = a
atoms = buf
print(f"read {len(atoms)} structures", flush=True)

# how many produce an 'energy' key via from_ase? (this is what caused the crash)
for a in atoms:
    d = AtomicData.from_ase(a, task_name="omol", r_edges=False,
                            r_energy=False, r_forces=False, r_stress=False)
    if "energy" in d:
        had_energy += 1
print(f"structures that from_ase gives an 'energy' key: {had_energy}/{len(atoms)}", flush=True)

print("loading uma-s-1p2 ...", flush=True)
pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
core = pu.model.module
cap = {}
core.backbone.norm.register_forward_hook(
    lambda m, i, o: cap.__setitem__("n", (o[0] if isinstance(o, (tuple, list)) else o).detach()))

# one mixed batch through the FIXED path
datas = []
for a in atoms:
    d = AtomicData.from_ase(a, task_name="omol", r_edges=False,
                            r_energy=False, r_forces=False, r_stress=False)
    for k in ("energy", "forces", "stress"):
        if k in d:
            del d[k]
    datas.append(d)
t0 = time.time()
batch = atomicdata_list_to_batch(datas).to("cuda")
pu.predict(batch)
node = cap["n"]
scal = node[:, 0, :].float()
bidx = batch.batch.to(scal.device)
acc = torch.zeros(len(atoms), scal.shape[1], device=scal.device)
acc.index_add_(0, bidx, scal)
cnt = torch.bincount(bidx, minlength=len(atoms)).clamp(min=1).unsqueeze(1)
pooled = (acc / cnt).cpu().numpy()
print(f"OK: pooled embedding {pooled.shape}, finite={np.isfinite(pooled).all()}, "
      f"norm mean={np.linalg.norm(pooled, axis=1).mean():.3f}, {time.time()-t0:.1f}s", flush=True)
