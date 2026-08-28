"""Step 2 (uma env): compute UMA per-structure embeddings for the subsample.

Embedding = mean-pooled l=0 (rotation-invariant) slice of the pre-readout node
features (`backbone.norm`, [n_atoms, 9, 128])  ->  128-d per structure.

Reads subsample.csv (row,native,shard,db_id,data_id,natoms), random-accesses the
aselmdb shards, batches by atom budget through uma-s-1p2 with task=omol, and
saves embeddings aligned to CSV order.
"""
import argparse, os, time, csv
import numpy as np
import torch
from ase.db import connect
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "omol25")
SPLIT_DIR = {0: "train_4M", 1: "val", 2: "test"}


def load_subsample(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["row"]), int(r["native"]), int(r["shard"]),
                         int(r["db_id"]), int(float(r["natoms"]))))
    return rows


def read_atoms(rows, cache_path):
    """Return list of ASE Atoms in CSV order (charge/spin in .info).

    Caches numbers/positions/charge/spin to an npz so reruns skip the ~10-min
    random-access read.
    """
    from ase import Atoms
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        off = z["off"]
        nums, pos, ch, sp = z["numbers"], z["positions"], z["charge"], z["spin"]
        atoms = []
        for i in range(len(off) - 1):
            a = Atoms(numbers=nums[off[i]:off[i+1]], positions=pos[off[i]:off[i+1]])
            a.info["charge"] = float(ch[i]); a.info["spin"] = float(sp[i])
            atoms.append(a)
        print(f"  loaded {len(atoms):,} structures from cache", flush=True)
        return atoms

    atoms = [None] * len(rows)
    groups = {}
    for i, (row, nat, shard, db_id, natoms) in enumerate(rows):
        groups.setdefault((nat, shard), []).append((i, db_id))
    t0 = time.time()
    for k, (nat, shard) in enumerate(sorted(groups)):
        path = os.path.join(DATA_DIR, SPLIT_DIR[nat], f"data{shard:04d}.aselmdb")
        db = connect(path)
        for i, db_id in groups[(nat, shard)]:
            r = db.get(id=db_id)
            a = r.toatoms()
            a.calc = None                                  # drop any SinglePoint calc
            a.info = {"charge": float(r.data.get("charge", 0.0)),
                      "spin": float(r.data.get("spin", 1.0))}
            atoms[i] = a
        if (k + 1) % 40 == 0 or k + 1 == len(groups):
            print(f"  read {k+1}/{len(groups)} shards, {time.time()-t0:.0f}s", flush=True)
    assert all(a is not None for a in atoms), "missing structures"
    # write cache
    off = np.zeros(len(atoms) + 1, dtype=np.int64)
    for i, a in enumerate(atoms):
        off[i+1] = off[i] + len(a)
    nums = np.concatenate([a.numbers for a in atoms]).astype(np.int16)
    pos = np.concatenate([a.positions for a in atoms]).astype(np.float32)
    ch = np.array([a.info["charge"] for a in atoms], dtype=np.float32)
    sp = np.array([a.info["spin"] for a in atoms], dtype=np.float32)
    np.savez(cache_path, off=off, numbers=nums, positions=pos, charge=ch, spin=sp)
    print(f"  cached structures -> {cache_path}", flush=True)
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "_cache_uma", "subsample.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "_cache_uma", "uma_emb.npy"))
    ap.add_argument("--model", default="uma-s-1p2")
    ap.add_argument("--atom-budget", type=int, default=8000, help="max atoms per batch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-i", type=int, default=0, help="this shard index [0,shard_n)")
    ap.add_argument("--shard-n", type=int, default=1, help="total number of shards")
    args = ap.parse_args()

    rows = load_subsample(args.sub)
    if args.shard_n > 1:
        # contiguous block (rows are sorted by original meta position, so a block
        # touches a contiguous range of aselmdb files -> minimal redundant I/O)
        per = (len(rows) + args.shard_n - 1) // args.shard_n
        lo, hi = args.shard_i * per, min((args.shard_i + 1) * per, len(rows))
        rows = rows[lo:hi]
        print(f"[shard {args.shard_i}/{args.shard_n}] rows [{lo}:{hi}) = {len(rows):,}",
              flush=True)
    natoms = np.array([r[4] for r in rows])
    native = np.array([r[1] for r in rows], dtype=np.int64)
    orig_row = np.array([r[0] for r in rows], dtype=np.int64)
    print(f"{len(rows):,} structures; natoms mean {natoms.mean():.1f} max {natoms.max()}",
          flush=True)

    print(f"loading {args.model} ...", flush=True)
    pu = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    core = pu.model.module

    cap = {}
    def hook(m, i, o):
        cap["n"] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
    h = core.backbone.norm.register_forward_hook(hook)

    print("reading structures ...", flush=True)
    atoms = read_atoms(rows, args.out.replace(".npy", ".atoms.npz"))

    # build atom-budgeted batches (preserve order)
    batches, cur, cur_atoms = [], [], 0
    for i, a in enumerate(atoms):
        na = len(a)
        if cur and cur_atoms + na > args.atom_budget:
            batches.append(cur); cur, cur_atoms = [], 0
        cur.append(i); cur_atoms += na
    if cur:
        batches.append(cur)
    print(f"{len(batches)} batches (budget {args.atom_budget} atoms)", flush=True)

    D = None
    emb = None
    t0 = time.time()
    for bi, idxs in enumerate(batches):
        datas = []
        for i in idxs:
            d = AtomicData.from_ase(atoms[i], task_name="omol", r_edges=False,
                                    r_energy=False, r_forces=False, r_stress=False)
            # from_ase ignores r_energy/r_forces/r_stress and attaches these target
            # keys only to structures that carry them; atomicdata_list_to_batch takes
            # its key set from data_list[0] alone, so a batch mixing has-energy and
            # no-energy structures crashes. Strip the targets (they are prediction
            # outputs, not inputs to the backbone features we extract) so every
            # structure has an identical key set.
            for k in ("energy", "forces", "stress"):
                if k in d:
                    del d[k]
            datas.append(d)
        batch = atomicdata_list_to_batch(datas).to(args.device)
        pu.predict(batch)                       # fills hook
        node = cap["n"]                         # [tot_atoms, 9, 128]
        scal = node[:, 0, :].float()            # l=0 invariant -> [tot_atoms, 128]
        bidx = batch.batch.to(scal.device)
        ns = len(idxs)
        acc = torch.zeros(ns, scal.shape[1], device=scal.device)
        acc.index_add_(0, bidx, scal)
        cnt = torch.bincount(bidx, minlength=ns).clamp(min=1).unsqueeze(1)
        pooled = (acc / cnt).cpu().numpy()
        if emb is None:
            D = pooled.shape[1]; emb = np.zeros((len(atoms), D), dtype=np.float32)
        for j, i in enumerate(idxs):
            emb[i] = pooled[j]
        if (bi + 1) % 50 == 0 or bi + 1 == len(batches):
            done = sum(len(b) for b in batches[:bi+1])
            print(f"  {bi+1}/{len(batches)} batches, {done:,} structs, "
                  f"{time.time()-t0:.0f}s", flush=True)
    h.remove()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, emb)
    np.save(args.out.replace("uma_emb", "native"), native)
    np.save(args.out.replace("uma_emb", "orig_row"), orig_row)
    print(f"saved {emb.shape} -> {args.out}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
