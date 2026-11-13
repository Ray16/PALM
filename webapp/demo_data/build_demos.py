"""Build the bundled demo datasets from the local PALM data copies.

Run from anywhere (deterministic; re-running reproduces the same files):

    python PALM/webapp/demo_data/build_demos.py

Each demo is a small CSV next to this script, described in demos.json (title,
entity columns/types, the column used to colour the embedding plot, ...).
Sources are public benchmark datasets; see PALM/data/DataSAIL_data/source.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PALM_DIR = HERE.parents[1]                      # the PALM checkout
PALM_DATA = PALM_DIR / "data"
SEED = 0


def esol():
    d = pd.read_csv(PALM_DATA / "DataSAIL_data/1D/moleculenet/esol.csv")
    return pd.DataFrame({
        "compound": d["Compound ID"],
        "smiles": d["smiles"],
        "logS": d["measured log solubility in mols per litre"].round(3),
    })


def bbbp():
    d = pd.read_csv(PALM_DATA / "DataSAIL_data/1D/moleculenet/bbbp.csv")
    d = d.drop_duplicates("smiles")
    return pd.DataFrame({
        "name": d["name"],
        "smiles": d["smiles"],
        "permeable": d["p_np"].map({1: "yes", 0: "no"}),
    })


PROTEIN_CLASSES = ["transferase", "hydrolase", "oxidoreductase", "lyase",
                   "ligase", "isomerase", "transcription", "immune system"]


def lp_pdbbind_proteins(per_class=75, max_len=1000):
    d = pd.read_csv(PALM_DATA / "DataSAIL_data/2D/lp_pdbbind/LP_PDBBind.csv")
    d = d.rename(columns={d.columns[0]: "pdb_id"})
    # "hydrolase/hydrolase inhibitor" -> "hydrolase"
    d["protein_class"] = d["header"].str.lower().str.split("/").str[0].str.strip()
    d = d[d["protein_class"].isin(PROTEIN_CLASSES)]
    d = d[d["seq"].str.len().between(50, max_len)]
    d = d[d["seq"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")]
    d = d.drop_duplicates("seq").sort_values("pdb_id")
    d = d.sample(frac=1, random_state=SEED).groupby("protein_class").head(per_class)
    return d[["pdb_id", "protein_class", "seq"]].rename(
        columns={"seq": "sequence"}).sort_values("pdb_id").reset_index(drop=True)


def davis():
    d = pd.read_csv(ROOT / "PALM/tests/data/davis_large/interactions.csv")
    d["pKd"] = (9 - np.log10(d["Kd_nM"])).round(3)
    return d[["drug_id", "smiles", "target_id", "sequence", "Kd_nM", "pKd"]]


def materials_project(n=1500):
    d = pd.read_csv(PALM_DATA / "materials_project/summary.csv")
    # one entry per formula (lowest energy above hull), then a seeded sample
    d = d.sort_values("energy_above_hull").drop_duplicates("formula_pretty")
    d = d.sample(n, random_state=SEED).sort_values("material_id")
    return pd.DataFrame({
        "material_id": d["material_id"],
        "formula": d["formula_pretty"],
        "band_gap_eV": d["band_gap"].round(4),
        "formation_energy_eV_atom": d["formation_energy_per_atom"].round(4),
        "stable": d["is_stable"].map({True: "yes", False: "no"}),
    }).reset_index(drop=True)


def rfam(per_family=80):
    path = PALM_DATA / "DataSAIL_data/1D/rfam_rna/dataset_Rfam_6320_13classes.fasta"
    recs, rid, seq = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if rid:
                recs.append((rid, "".join(seq)))
            rid, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if rid:
        recs.append((rid, "".join(seq)))
    d = pd.DataFrame(recs, columns=["rna_id", "sequence"])
    # header: RF00001_AF095839_1_346-228_5S_rRNA -> family "5S_rRNA"
    d["family"] = d["rna_id"].str.split("_", n=4).str[4]
    d = d.drop_duplicates("sequence")
    d = d.sample(frac=1, random_state=SEED).groupby("family").head(per_family)
    return d[["rna_id", "family", "sequence"]].sort_values("rna_id").reset_index(drop=True)


BUILDERS = {
    "esol": esol,
    "bbbp": bbbp,
    "lp_pdbbind_proteins": lp_pdbbind_proteins,
    "davis": davis,
    "materials_project": materials_project,
    "rfam_rna": rfam,
}


def main():
    manifest = {d["id"]: d for d in json.loads((HERE / "demos.json").read_text())}
    for name, build in BUILDERS.items():
        df = build()
        out = HERE / manifest[name]["file"]
        df.to_csv(out, index=False)
        print(f"{name:22s} {len(df):6d} rows -> {out.name}")


if __name__ == "__main__":
    main()
