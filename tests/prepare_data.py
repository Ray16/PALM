"""Download and prepare example datasets for each entity type.

Datasets:
  1. Davis DTI         — molecule + biomolecule (drug SMILES + protein sequences)
  2. Adsorbate-Surface — molecule + material    (SMILES + formulas, synthetic)
  3. Gene-Drug         — gene + molecule        (CDS sequences + drug SMILES)

Run:
    python PALM/tests/prepare_data.py
"""

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import requests

# PALM/tests/ → PALM root is one level up
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PALM_DIR  = os.path.dirname(TESTS_DIR)
DATA_DIR  = os.path.join(TESTS_DIR, "data")
CFG_DIR   = os.path.join(TESTS_DIR, "configs")
OUT_DIR   = os.path.join(TESTS_DIR, "output")

for d in (DATA_DIR, CFG_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)


# ── Dataset 1: Davis Drug-Target (molecule + biomolecule) ─────────────────
# Source: DeepDTA GitHub (hkmztrk/DeepDTA)
#   ligands_can.txt  → JSON {pubchem_id: canonical_smiles}
#   proteins.txt     → JSON {gene_name: amino_acid_sequence}
#   Y                → numpy binary, shape (68 drugs, 442 proteins), Kd in nM

_DEEPDTA = ("https://raw.githubusercontent.com/hkmztrk"
            "/DeepDTA/master/data/davis")


def prepare_davis():
    out_dir  = os.path.join(DATA_DIR, "davis")
    out_path = os.path.join(out_dir, "interactions.csv")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return out_path

    print("  Downloading Davis dataset from DeepDTA/GitHub...")

    ligands  = json.loads(requests.get(f"{_DEEPDTA}/ligands_can.txt", timeout=30).text)
    proteins = json.loads(requests.get(f"{_DEEPDTA}/proteins.txt",    timeout=30).text)

    # Y is a Python-2-pickled numpy array; needs encoding='latin1' in Python 3
    y_bytes  = requests.get(f"{_DEEPDTA}/Y", timeout=30).content
    affinity = pickle.loads(y_bytes, encoding="latin1")   # (68, 442)

    drug_ids   = list(ligands.keys())
    target_ids = list(proteins.keys())

    # Keep first 10 drugs × 20 targets → 200 rows for a fast smoke test
    n_drugs, n_targets = 10, 20
    drug_ids   = drug_ids[:n_drugs]
    target_ids = target_ids[:n_targets]
    affinity   = affinity[:n_drugs, :n_targets]

    records = []
    for di, did in enumerate(drug_ids):
        for ti, tid in enumerate(target_ids):
            records.append({
                "drug_id":   did,
                "smiles":    ligands[did],
                "target_id": tid,
                "sequence":  proteins[tid],
                "kd_nM":     float(affinity[di, ti]),
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} interactions → {out_path}")
    return out_path


# ── Dataset 2: Adsorbate-Surface (molecule + material, synthetic) ──────────

def prepare_adsorbate_surface():
    out_dir  = os.path.join(DATA_DIR, "adsorbate_surface")
    out_path = os.path.join(out_dir, "interactions.csv")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return out_path

    adsorbates = [
        ("CO",    "[C-]#[O+]"),
        ("O2",    "O=O"),
        ("H2O",   "O"),
        ("OH",    "[OH]"),
        ("H",     "[H]"),
        ("NO",    "[N]=O"),
        ("NH3",   "N"),
        ("CH4",   "C"),
        ("CO2",   "O=C=O"),
        ("N2",    "N#N"),
        ("HCO",   "[CH]=O"),
        ("COOH",  "OC=O"),
        ("CH3OH", "CO"),
        ("O",     "[O]"),
        ("C",     "[C]"),
    ]
    surfaces = [
        "Fe", "Cu", "Ni", "Pt", "Au",
        "Fe2O3", "CuO", "NiO", "TiO2", "ZnO",
    ]

    np.random.seed(42)
    records = [
        {
            "adsorbate": sym,
            "smiles":    smi,
            "surface":   surf,
            "adsorption_energy": round(float(np.random.uniform(-3.5, 0.5)), 3),
        }
        for sym, smi in adsorbates
        for surf in surfaces
    ]

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} pairs → {out_path}")
    return out_path


# ── Dataset 3: Gene-Drug (gene + molecule) ────────────────────────────────

_GENE_ACCESSIONS = {
    "CYP3A4": "NM_017460",
    "CYP2D6": "NM_000106",
    "CYP2C9": "NM_000771",
    "CYP1A2": "NM_000761",
    "ABCB1":  "NM_000927",
    "UGT1A1": "NM_000463",
    "DPYD":   "NM_000110",
    "TPMT":   "NM_000367",
    "VKORC1": "NM_024006",
    "G6PD":   "NM_000402",
}

_DRUG_SMILES = {
    "warfarin":       "OC1=CC2=CC=CC=C2C(=O)C1CC(=O)C1=CC=CC=C1",
    "simvastatin":    "CCC(C)(C)C(=O)OC1CC(CC2CC(O)CC(=O)O2)C=C1C",
    "ibuprofen":      "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "omeprazole":     "COC1=CC2=C(NC(=O)C3=CN=CN3C)N=CC2=CC1=O",
    "tamoxifen":      "CCC(=C1C=CC(=CC1)OCCN(C)C)C1=CC=CC=C1",
    "codeine":        "CN1CC[C@]23C=C[C@@H](O)C[C@H]2OC4=C3C1=CC=C4OC",
    "azathioprine":   "Cn1cnc2c(N)ncnc12",
    "mercaptopurine": "S=c1[nH]cnc2c1ncn2",
    "fluorouracil":   "O=C1C=C(F)C(=O)N1",
    "capecitabine":   "CCCC(=O)Oc1nc(N)c(F)cn1",
    "methotrexate":   "CN(CC1=CN=C2C(=N1)C(=NC(N)=N2)N)C3=CC=C(C=C3)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
    "digoxin":        "O[C@@H]1C[C@@H](O)[C@H](O[C@@H]2CC[C@@H](O)[C@H](O2)O)[C@@H](O1)C",
    "rifampicin":     "COCCCOC1=C2NC(=O)C(=O)NC3=CC=CC(=C3OC)OC2=CC1=O",
    "phenytoin":      "O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1",
    "clopidogrel":    "COC(=O)C1(CC2=CC=CS2)CCCC2=CC=CC=C12",
}

_GENE_DRUG_PAIRS = [
    ("CYP3A4", "simvastatin"),  ("CYP3A4", "tamoxifen"),   ("CYP3A4", "rifampicin"),
    ("CYP3A4", "codeine"),      ("CYP3A4", "warfarin"),
    ("CYP2D6", "codeine"),      ("CYP2D6", "tamoxifen"),   ("CYP2D6", "methotrexate"),
    ("CYP2C9", "warfarin"),     ("CYP2C9", "ibuprofen"),   ("CYP2C9", "phenytoin"),
    ("CYP1A2", "methotrexate"), ("CYP1A2", "rifampicin"),
    ("ABCB1",  "digoxin"),      ("ABCB1",  "rifampicin"),  ("ABCB1",  "simvastatin"),
    ("UGT1A1", "ibuprofen"),    ("UGT1A1", "capecitabine"),
    ("DPYD",   "fluorouracil"), ("DPYD",   "capecitabine"),
    ("TPMT",   "azathioprine"), ("TPMT",   "mercaptopurine"),
    ("VKORC1", "warfarin"),     ("VKORC1", "rifampicin"),
    ("G6PD",   "ibuprofen"),    ("G6PD",   "rifampicin"),
]


def _synthetic_cds(seed: int, length: int = 900) -> str:
    """Synthetic but structurally realistic coding sequence (ATG…stop)."""
    rng = np.random.default_rng(seed)
    sense_codons = [
        "TTT","TTC","TTA","TTG","CTT","CTC","CTA","CTG",
        "ATT","ATC","ATA","ATG","GTT","GTC","GTA","GTG",
        "TCT","TCC","TCA","TCG","CCT","CCC","CCA","CCG",
        "ACT","ACC","ACA","ACG","GCT","GCC","GCA","GCG",
        "TAT","TAC","CAT","CAC","CAA","CAG","AAT","AAC",
        "AAA","AAG","GAT","GAC","GAA","GAG","TGT","TGC",
        "TGG","CGT","CGC","CGA","CGG","AGT","AGC","AGA",
        "AGG","GGT","GGC","GGA","GGG",
    ]
    n_codons = (length - 6) // 3
    body = "".join(rng.choice(sense_codons, n_codons))
    return "ATG" + body + "TGA"


def prepare_gene_drug():
    out_dir  = os.path.join(DATA_DIR, "gene_drug")
    out_path = os.path.join(out_dir, "interactions.csv")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return out_path

    gene_sequences = {}
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = "palm_test@example.com"
        print("  Fetching mRNA CDS sequences from NCBI Entrez...")
        for gene, acc in _GENE_ACCESSIONS.items():
            try:
                handle = Entrez.efetch(
                    db="nucleotide", id=acc, rettype="fasta", retmode="text"
                )
                record = SeqIO.read(handle, "fasta")
                handle.close()
                gene_sequences[gene] = str(record.seq)[:1200]
                print(f"    {gene} ({acc}): {len(gene_sequences[gene])} nt")
                time.sleep(0.4)   # NCBI rate limit: ≤3 req/s without API key
            except Exception as exc:
                print(f"    {gene}: NCBI failed ({exc}), using synthetic seq")
                gene_sequences[gene] = _synthetic_cds(seed=abs(hash(gene)) % 9999)
    except ImportError:
        print("  Biopython not available — using synthetic gene sequences")
        for i, gene in enumerate(_GENE_ACCESSIONS):
            gene_sequences[gene] = _synthetic_cds(seed=i * 37)

    records = [
        {
            "gene":     gene,
            "sequence": gene_sequences[gene],
            "drug":     drug,
            "smiles":   _DRUG_SMILES[drug],
        }
        for gene, drug in _GENE_DRUG_PAIRS
        if gene in gene_sequences and drug in _DRUG_SMILES
    ]

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} gene-drug pairs → {out_path}")
    return out_path


# ── Write YAML configs ─────────────────────────────────────────────────────

def write_configs(davis_path, ads_path, gene_path):
    davis_out = os.path.join(OUT_DIR, "davis")
    ads_out   = os.path.join(OUT_DIR, "adsorbate_surface")
    gene_out  = os.path.join(OUT_DIR, "gene_drug")

    _write("molecule_biomolecule", f"""\
# Davis Drug-Target Interaction — molecule (drug) + biomolecule (protein)
input_file: "{davis_path}"
output_dir: "{davis_out}"
dataset_name: "davis"

e:
  name: "drug"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets:
    - rdkit_descriptors
    - physicochemical

f:
  name: "protein"
  type: "biomolecule"
  extract:
    column: "sequence"
  feature_sets:
    - sequence_properties

splitting:
  techniques: [R, I2]
  splits: [8, 2]
  names: ["train", "test"]
  max_sec: 120
  solver: "SCIP"
""")

    _write("molecule_material", f"""\
# Adsorbate-Surface — molecule (adsorbate) + material (surface)
input_file: "{ads_path}"
output_dir: "{ads_out}"
dataset_name: "adsorbate_surface"

e:
  name: "adsorbate"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets:
    - rdkit_descriptors
    - physicochemical

f:
  name: "surface"
  type: "material"
  extract:
    column: "surface"
  feature_sets:
    - magpie_properties
    - stoichiometry
    - electronic

splitting:
  techniques: [R, I2]
  splits: [8, 2]
  names: ["train", "test"]
  max_sec: 120
  solver: "SCIP"
""")

    _write("gene_molecule", f"""\
# Gene-Drug pharmacogenomics — gene (CDS) + molecule (drug)
input_file: "{gene_path}"
output_dir: "{gene_out}"
dataset_name: "gene_drug"

e:
  name: "gene"
  type: "gene"
  extract:
    column: "sequence"
  feature_sets:
    - nucleotide_composition
    - kmer_frequencies

f:
  name: "drug"
  type: "molecule"
  extract:
    column: "smiles"
  feature_sets:
    - rdkit_descriptors
    - physicochemical

splitting:
  techniques: [R, I2]
  splits: [8, 2]
  names: ["train", "test"]
  max_sec: 120
  solver: "SCIP"
""")


def _write(name, content):
    path = os.path.join(CFG_DIR, f"{name}.yaml")
    with open(path, "w") as fh:
        fh.write(content)
    print(f"  Config written: {path}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing test datasets")
    print("=" * 60)

    print("\n[1/3] Davis Drug-Target  (molecule + biomolecule)")
    davis_path = prepare_davis()

    print("\n[2/3] Adsorbate-Surface  (molecule + material)")
    ads_path = prepare_adsorbate_surface()

    print("\n[3/3] Gene-Drug          (gene + molecule)")
    gene_path = prepare_gene_drug()

    print("\n[4/4] Writing configs")
    write_configs(davis_path, ads_path, gene_path)

    print(f"\nDone. Run: bash PALM/tests/run_tests.sh")
