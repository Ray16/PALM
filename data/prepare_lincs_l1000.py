"""Prepare LINCS L1000 (omics / perturbation) as a PALM dataset.

Source: LINCS L1000 Phase II (GEO GSE70138) Level-5 MODZ signatures. Each
signature is a differential gene-expression profile induced by perturbing a cell
line with a compound. We keep compound perturbations (``trt_cp``), the 978
landmark genes, and attach each compound's canonical SMILES.

This is a *precomputed-feature* modality with two similarity axes: expression
(the 978-gene vector) AND chemical structure (SMILES) — useful for comparing an
expression-space split against a structure-space split. Outputs (in data/lincs_l1000/):
    records.csv      id(sig_id), smiles, pert_iname, cell_id, dose, time
    expression.npy   float32 (N x 978), row-aligned to records.csv
    landmark_genes.txt   the 978 gene ids (feature order)

Run (palm env has h5py + pandas; keep TMPDIR off the full root disk):
    TMPDIR=/nfs/.../.tmp /homes/rzhu/miniforge3/envs/palm/bin/python -m PALM.data.prepare_lincs_l1000 --limit 20000
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "lincs_l1000")
GCTX = os.path.join(D, "Level5.gctx")

# LINCS L1000 Phase II (GEO GSE70138) — Level-5 GCTX (5 GB gz -> ~12 GB) + metadata.
_B70 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl"
_B92 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"
_FILES = {
    "sig_info.txt.gz": f"{_B70}/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
    "pert_info.txt.gz": f"{_B70}/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
    "gene_info.txt.gz": f"{_B92}/GSE92742_Broad_LINCS_gene_info.txt.gz",
}
_GCTX_GZ_URL = (f"{_B70}/GSE70138_Broad_LINCS_Level5_COMPZ_"
                "n118050x12328_2017-03-06.gctx.gz")


def _ensure_data():
    """Download the GCTX (+ metadata) and decompress it if not already present."""
    import gzip
    import shutil
    import urllib.request
    os.makedirs(D, exist_ok=True)
    for fn, url in _FILES.items():
        dst = os.path.join(D, fn)
        if not os.path.exists(dst):
            print(f"[l1000] downloading {fn}")
            urllib.request.urlretrieve(url, dst)
    if not os.path.exists(GCTX):
        gz = GCTX + ".gz"
        if not os.path.exists(gz):
            print("[l1000] downloading Level5 GCTX (5 GB) ...")
            urllib.request.urlretrieve(_GCTX_GZ_URL, gz)
        print("[l1000] decompressing GCTX (-> ~12 GB) ...")
        with gzip.open(gz, "rb") as fi, open(GCTX, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(gz)


def _lm_column(gi):
    for c in ("pr_is_lm", "pr_is_landmark", "pr_is_lmark"):
        if c in gi.columns:
            return c
    raise KeyError("no landmark flag column in gene_info")


def prepare(limit=20000) -> str:
    import h5py

    _ensure_data()
    gi = pd.read_csv(os.path.join(D, "gene_info.txt.gz"), sep="\t", dtype=str)
    lm_col = _lm_column(gi)
    landmark = set(gi.loc[gi[lm_col].isin(["1", "Y", "true", "True"]), "pr_gene_id"])
    print(f"[l1000] {len(landmark)} landmark genes")

    si = pd.read_csv(os.path.join(D, "sig_info.txt.gz"), sep="\t", dtype=str)
    si = si[si["pert_type"] == "trt_cp"]                      # compound perturbations
    pi = pd.read_csv(os.path.join(D, "pert_info.txt.gz"), sep="\t", dtype=str)
    smi_col = next((c for c in ("canonical_smiles", "canonical_smile", "smiles")
                    if c in pi.columns), None)
    smi = dict(zip(pi["pert_id"], pi[smi_col])) if smi_col else {}

    with h5py.File(GCTX, "r") as f:
        rid = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["/0/META/ROW/id"][:]]              # gene ids
        cid = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["/0/META/COL/id"][:]]              # signature ids
        mat = f["/0/DATA/0/matrix"]
        # orientation: matrix is (n_col, n_row) = (sigs, genes) in GCTX
        sig_axis0 = mat.shape[0] == len(cid)
        print(f"[l1000] matrix {mat.shape}; sig-major={sig_axis0}; "
              f"{len(rid)} genes x {len(cid)} sigs")

        lm_rows = [j for j, g in enumerate(rid) if g in landmark]
        # signatures we can use: compound perturbation WITH a usable SMILES
        bad_smi = {"", "-666", "restricted", None}
        pert_of_sig = dict(zip(si["sig_id"], si["pert_id"]))
        usable = {s for s, p in pert_of_sig.items() if smi.get(p, "") not in bad_smi}
        sig_of_pos = {s: i for i, s in enumerate(cid)}
        cand = [sig_of_pos[s] for s in si["sig_id"] if s in sig_of_pos and s in usable]
        rng = np.random.default_rng(0)
        if limit and len(cand) > limit:
            cand = sorted(rng.choice(cand, limit, replace=False).tolist())
        else:
            cand = sorted(cand)
        print(f"[l1000] selecting {len(cand)} compound signatures")

        # read selected signature rows, then landmark gene columns
        block = mat[cand, :] if sig_axis0 else mat[:, cand].T   # (N, n_genes)
        X = block[:, lm_rows].astype(np.float32)                # (N, 978)

    sel_sigs = [cid[i] for i in cand]
    sig_meta = si.set_index("sig_id")
    rows, keep = [], []
    for k, s in enumerate(sel_sigs):
        m = sig_meta.loc[s]
        pid = m["pert_id"]
        rows.append({
            "id": s, "smiles": smi.get(pid, ""),
            "pert_iname": m.get("pert_iname", ""), "cell_id": m.get("cell_id", ""),
            "dose": m.get("pert_idose", m.get("pert_dose", "")),
            "time": m.get("pert_itime", m.get("pert_time", "")),
        })
        keep.append(k)
    df = pd.DataFrame(rows)
    X = X[keep]

    os.makedirs(D, exist_ok=True)
    df.to_csv(os.path.join(D, "records.csv"), index=False)
    np.save(os.path.join(D, "expression.npy"), X)
    with open(os.path.join(D, "landmark_genes.txt"), "w") as fh:
        fh.write("\n".join(rid[j] for j in lm_rows))
    print(f"[l1000] wrote {len(df)} rows; expression {X.shape} -> {D}")
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()
    prepare(args.limit)
