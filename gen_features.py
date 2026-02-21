import json
import re
import os
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import Descriptors

# ── Load metadata ──────────────────────────────────────────────────────────
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))

# Filter out entries with empty ads_symbols
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}
system_ids = sorted(entries.keys(), key=int)
print(f"Total entries: {len(metadata)}, after filtering empty ads: {len(entries)}")

# ── Output dirs ────────────────────────────────────────────────────────────
os.makedirs("embeddings/adsorbate", exist_ok=True)
os.makedirs("embeddings/adsorbent", exist_ok=True)

# ── SMILES lookup for 9 adsorbates ─────────────────────────────────────────
ADS_SMILES = {
    "C": "[C]",
    "CO": "[C-]#[O+]",
    "H": "[H]",
    "H2O": "O",
    "HO2": "[O]O",
    "N": "[N]",
    "O": "[O]",
    "O2": "O=O",
    "OH": "[OH]",
}

# ── Formula parser ─────────────────────────────────────────────────────────
def parse_formula(formula):
    """Parse formula like 'Ta8O20' → {'Ta': 8, 'O': 20}."""
    pairs = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    comp = {}
    for elem, count in pairs:
        if elem == "":
            continue
        comp[elem] = comp.get(elem, 0) + (int(count) if count else 1)
    return comp

# ── Hardcoded elemental property table ─────────────────────────────────────
# Properties: atomic_number, atomic_mass, electronegativity (Pauling),
#             covalent_radius (pm), electron_affinity (kJ/mol), ionization_energy (kJ/mol)
ELEM_PROPS = {
    "H":  [1,   1.008,  2.20,  31,   72.8,   1312.0],
    "He": [2,   4.003,  0.00,  28,    0.0,   2372.3],
    "Li": [3,   6.941,  0.98, 128,   59.6,    520.2],
    "Be": [4,   9.012,  1.57,  96,    0.0,    899.5],
    "B":  [5,  10.81,   2.04,  84,   26.7,    800.6],
    "C":  [6,  12.011,  2.55,  76,  121.8,   1086.5],
    "N":  [7,  14.007,  3.04,  71,    0.0,   1402.3],
    "O":  [8,  15.999,  3.44,  66,  141.0,   1313.9],
    "F":  [9,  18.998,  3.98,  57,  328.0,   1681.0],
    "Ne": [10, 20.180,  0.00,  58,    0.0,   2080.7],
    "Na": [11, 22.990,  0.93, 166,   52.8,    495.8],
    "Mg": [12, 24.305,  1.31, 141,    0.0,    737.7],
    "Al": [13, 26.982,  1.61, 121,   42.5,    577.5],
    "Si": [14, 28.086,  1.90, 111,  134.1,    786.5],
    "P":  [15, 30.974,  2.19, 107,   72.0,   1011.8],
    "S":  [16, 32.06,   2.58, 105,  200.4,    999.6],
    "Cl": [17, 35.45,   3.16, 102,  349.0,   1251.2],
    "Ar": [18, 39.948,  0.00, 106,    0.0,   1520.6],
    "K":  [19, 39.098,  0.82, 203,   48.4,    418.8],
    "Ca": [20, 40.078,  1.00, 176,    2.4,    589.8],
    "Sc": [21, 44.956,  1.36, 170,   18.1,    633.1],
    "Ti": [22, 47.867,  1.54, 160,    7.6,    658.8],
    "V":  [23, 50.942,  1.63, 153,   50.9,    650.9],
    "Cr": [24, 51.996,  1.66, 139,   64.3,    652.9],
    "Mn": [25, 54.938,  1.55, 139,    0.0,    717.3],
    "Fe": [26, 55.845,  1.83, 132,   15.7,    762.5],
    "Co": [27, 58.933,  1.88, 126,   63.7,    760.4],
    "Ni": [28, 58.693,  1.91, 124,  112.0,    737.1],
    "Cu": [29, 63.546,  1.90, 132,  118.4,    745.5],
    "Zn": [30, 65.38,   1.65, 122,    0.0,    906.4],
    "Ga": [31, 69.723,  1.81, 122,   41.0,    578.8],
    "Ge": [32, 72.63,   2.01, 120,  119.0,    762.2],
    "As": [33, 74.922,  2.18, 119,   78.2,    947.0],
    "Se": [34, 78.96,   2.55, 120,  195.0,    941.0],
    "Br": [35, 79.904,  2.96, 120,  324.6,   1139.9],
    "Kr": [36, 83.798,  3.00, 116,    0.0,   1350.8],
    "Rb": [37, 85.468,  0.82, 220,   46.9,    403.0],
    "Sr": [38, 87.62,   0.95, 195,    5.0,    549.5],
    "Y":  [39, 88.906,  1.22, 190,   29.6,    600.0],
    "Zr": [40, 91.224,  1.33, 175,   41.1,    640.1],
    "Nb": [41, 92.906,  1.60, 164,   86.1,    652.1],
    "Mo": [42, 95.96,   2.16, 154,   71.9,    684.3],
    "Ru": [44, 101.07,  2.20, 146,  101.3,    710.2],
    "Rh": [45, 102.91,  2.28, 142,  109.7,    719.7],
    "Pd": [46, 106.42,  2.20, 139,   53.7,    804.4],
    "Ag": [47, 107.87,  1.93, 145,  125.6,    731.0],
    "Cd": [48, 112.41,  1.69, 144,    0.0,    867.8],
    "In": [49, 114.82,  1.78, 142,   28.9,    558.3],
    "Sn": [50, 118.71,  1.96, 139,  107.3,    708.6],
    "Sb": [51, 121.76,  2.05, 139,  103.2,    834.0],
    "Te": [52, 127.60,  2.10, 138,  190.2,    869.3],
    "I":  [53, 126.90,  2.66, 139,  295.2,   1008.4],
    "Xe": [54, 131.29,  2.60, 140,    0.0,   1170.4],
    "Cs": [55, 132.91,  0.79, 244,   45.5,    375.7],
    "Ba": [56, 137.33,  0.89, 215,   13.9,    502.9],
    "La": [57, 138.91,  1.10, 207,   48.0,    538.1],
    "Ce": [58, 140.12,  1.12, 204,   50.0,    534.4],
    "Pr": [59, 140.91,  1.13, 203,   50.0,    527.0],
    "Nd": [60, 144.24,  1.14, 201,   50.0,    533.1],
    "Sm": [62, 150.36,  1.17, 198,   50.0,    544.5],
    "Eu": [63, 151.96,  1.20, 198,   50.0,    547.1],
    "Gd": [64, 157.25,  1.20, 196,   50.0,    593.4],
    "Tb": [65, 158.93,  1.20, 194,   50.0,    565.8],
    "Dy": [66, 162.50,  1.22, 192,   50.0,    573.0],
    "Ho": [67, 164.93,  1.23, 192,   50.0,    581.0],
    "Er": [68, 167.26,  1.24, 189,   50.0,    589.3],
    "Tm": [69, 168.93,  1.25, 190,   50.0,    596.7],
    "Yb": [70, 173.05,  1.10, 187,   50.0,    603.4],
    "Lu": [71, 174.97,  1.27, 187,   50.0,    523.5],
    "Hf": [72, 178.49,  1.30, 175,    0.0,    658.5],
    "Ta": [73, 180.95,  1.50, 170,   31.0,    761.0],
    "W":  [74, 183.84,  2.36, 162,   78.6,    770.0],
    "Re": [75, 186.21,  1.90, 151,   14.5,    760.0],
    "Os": [76, 190.23,  2.20, 144,  106.1,    840.0],
    "Ir": [77, 192.22,  2.20, 141,  151.0,    880.0],
    "Pt": [78, 195.08,  2.28, 136,  205.3,    870.0],
    "Au": [79, 196.97,  2.54, 136,  222.8,    890.1],
    "Hg": [80, 200.59,  2.00, 132,    0.0,   1007.1],
    "Tl": [81, 204.38,  1.62, 145,   19.2,    589.4],
    "Pb": [82, 207.2,   2.33, 146,   35.1,    715.6],
    "Bi": [83, 208.98,  2.02, 148,   91.3,    703.0],
    "Th": [90, 232.04,  1.30, 179,    0.0,    587.0],
    "U":  [92, 238.03,  1.38, 196,    0.0,    597.6],
}
PROP_NAMES = ["atomic_number", "atomic_mass", "electronegativity",
              "covalent_radius", "electron_affinity", "ionization_energy"]


# ═══════════════════════════════════════════════════════════════════════════
# ADSORBATE FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════

def get_mol(ads_sym):
    smi = ADS_SMILES[ads_sym]
    return Chem.MolFromSmiles(smi)

# ── 1. Catalysis-relevant physicochemical properties (hardcoded) ───────────
# Replaces Morgan/MACCS fingerprints which are uninformative for these
# small radical/atomic adsorbates (extreme sparsity, collisions).
# Sources: NIST Chemistry WebBook, CRC Handbook
ADS_PHYSCHEM = {
    #              MolWt   nAtoms  nHeavy  nH  nOH  nOO  is_radical  unpaired_e  total_vale  dipole(D)  proton_aff(kJ/mol)  gas_phase_BDE(kJ/mol)  electron_aff(eV)  ioniz_E(eV)  polarizability(A^3)
    "H":          [1.008,   1,      0,      1,   0,   0,   1,          1,          1,          0.0,       0.0,                436.0,                 0.754,            13.598,      0.667],
    "C":          [12.011,  1,      1,      0,   0,   0,   1,          4,          4,          0.0,       0.0,                0.0,                   1.262,            11.260,      1.76],
    "N":          [14.007,  1,      1,      0,   0,   0,   1,          3,          5,          0.0,       0.0,                0.0,                   -0.07,            14.534,      1.10],
    "O":          [15.999,  1,      1,      0,   0,   0,   1,          2,          6,          0.0,       0.0,                0.0,                   1.461,            13.618,      0.802],
    "H2O":        [18.015,  3,      1,      2,   2,   0,   0,          0,          8,          1.85,      691.0,              497.1,                 -0.03,            12.621,      1.45],
    "OH":         [17.007,  2,      1,      1,   1,   0,   1,          1,          7,          1.66,      593.2,              428.0,                 1.828,            13.017,      1.02],
    "O2":         [31.998,  2,      2,      0,   0,   1,   1,          2,          12,         0.0,       421.0,              498.4,                 0.448,            12.070,      1.562],
    "CO":         [28.010,  2,      2,      0,   0,   0,   0,          0,          10,         0.11,      594.0,              1076.5,                1.110,            14.014,      1.95],
    "HO2":        [33.006,  3,      2,      1,   1,   1,   1,          1,          13,         2.09,      660.0,              366.0,                 1.078,            11.350,      2.20],
}
ADS_PHYSCHEM_COLS = [
    "mol_weight", "num_atoms", "num_heavy_atoms", "num_H",
    "num_OH_bonds", "num_OO_bonds", "is_radical", "unpaired_electrons",
    "total_valence_electrons", "dipole_moment", "proton_affinity",
    "gas_phase_BDE", "electron_affinity_eV", "ionization_energy_eV",
    "polarizability",
]

def compute_ads_physchem(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["ads_symbols"]
        props = ADS_PHYSCHEM[formula]
        rows.append([sid, formula] + props)
    cols = ["system_id", "ads_symbols"] + ADS_PHYSCHEM_COLS
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbate/physchem_features.csv", index=False)
    print(f"  physchem_features.csv: {df.shape}")

# ── 2. RDKit molecular descriptors (corrected for small species) ───────────
# Dropped uninformative descriptors (NumRotatableBonds=0 for all,
# FractionCSP3=0 for 8/9, NumHDonors wrong for H2O).
# Added atom-level features computed correctly via explicit H iteration.

def compute_rdkit_descriptors(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["ads_symbols"]
        mol = get_mol(formula)
        mol_h = Chem.AddHs(mol)

        # Correct O-H donor count: count O/N atoms that have at least one H neighbor
        num_hbond_donors = 0
        num_hbond_acceptors = 0
        num_lone_pairs = 0
        for atom in mol_h.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum in (7, 8):  # N, O
                h_neighbors = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
                if h_neighbors > 0:
                    num_hbond_donors += h_neighbors
                # O/N with lone pairs can accept H-bonds
                # Lone pairs: valence_electrons - bonding_electrons
                valence = 5 if anum == 7 else 6
                bond_order = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                radical = atom.GetNumRadicalElectrons()
                lone_pair_electrons = valence - bond_order - radical
                lp = max(0, int(lone_pair_electrons)) // 2
                num_lone_pairs += lp
                if lp > 0:
                    num_hbond_acceptors += 1

        row = [
            sid, formula,
            Descriptors.MolWt(mol),
            num_hbond_donors,
            num_hbond_acceptors,
            num_lone_pairs,
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
        ]
        rows.append(row)

    cols = ["system_id", "ads_symbols",
            "MolWt", "NumHBondDonors", "NumHBondAcceptors", "NumLonePairs",
            "TPSA", "MolLogP", "NumHeavyAtoms",
            "NumValenceElectrons", "NumRadicalElectrons"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbate/rdkit_descriptors_features.csv", index=False)
    print(f"  rdkit_descriptors_features.csv: {df.shape}")

# ── 3. Adsorbate composition features ─────────────────────────────────────
def compute_ads_composition(system_ids, entries):
    all_elems = set()
    for sid in system_ids:
        comp = parse_formula(entries[sid]["ads_symbols"])
        all_elems.update(comp.keys())
    all_elems = sorted(all_elems)

    rows = []
    for sid in system_ids:
        formula = entries[sid]["ads_symbols"]
        comp = parse_formula(formula)
        total = sum(comp.values())
        row = [sid, formula]
        for elem in all_elems:
            row.append(comp.get(elem, 0))
        for pidx, pname in enumerate(PROP_NAMES):
            weighted_sum = 0.0
            for elem, cnt in comp.items():
                if elem in ELEM_PROPS:
                    weighted_sum += cnt * ELEM_PROPS[elem][pidx]
            row.append(weighted_sum / total if total > 0 else 0.0)
        rows.append(row)

    elem_cols = [f"count_{e}" for e in all_elems]
    prop_cols = [f"wtd_mean_{p}" for p in PROP_NAMES]
    cols = ["system_id", "ads_symbols"] + elem_cols + prop_cols
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbate/composition_features.csv", index=False)
    print(f"  composition_features.csv: {df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# ADSORBENT FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════

def get_all_bulk_elements(system_ids, entries):
    all_elems = set()
    for sid in system_ids:
        comp = parse_formula(entries[sid]["bulk_symbols"])
        all_elems.update(comp.keys())
    return sorted(all_elems)

# 1. Element count vectors
def compute_bulk_composition(system_ids, entries, all_elems):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        row = [sid, formula] + [comp.get(e, 0) for e in all_elems]
        rows.append(row)
    cols = ["system_id", "bulk_symbols"] + [f"count_{e}" for e in all_elems]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbent/composition_features.csv", index=False)
    print(f"  composition_features.csv: {df.shape}")

# 2. Element fraction vectors
def compute_bulk_fraction(system_ids, entries, all_elems):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        total = sum(comp.values())
        fracs = [comp.get(e, 0) / total if total > 0 else 0.0 for e in all_elems]
        rows.append([sid, formula] + fracs)
    cols = ["system_id", "bulk_symbols"] + [f"frac_{e}" for e in all_elems]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbent/fraction_features.csv", index=False)
    print(f"  fraction_features.csv: {df.shape}")

# 3. Magpie-style elemental property statistics
STAT_NAMES = ["mean", "std", "min", "max", "range"]

def compute_bulk_properties(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        prop_arrays = {p: [] for p in PROP_NAMES}
        for elem, cnt in comp.items():
            if elem not in ELEM_PROPS:
                continue
            for pidx, pname in enumerate(PROP_NAMES):
                prop_arrays[pname].extend([ELEM_PROPS[elem][pidx]] * cnt)

        row = [sid, formula]
        for pname in PROP_NAMES:
            arr = np.array(prop_arrays[pname]) if prop_arrays[pname] else np.array([0.0])
            row.append(float(np.mean(arr)))
            row.append(float(np.std(arr)))
            row.append(float(np.min(arr)))
            row.append(float(np.max(arr)))
            row.append(float(np.max(arr) - np.min(arr)))
        rows.append(row)

    feat_cols = []
    for pname in PROP_NAMES:
        for stat in STAT_NAMES:
            feat_cols.append(f"{pname}_{stat}")
    cols = ["system_id", "bulk_symbols"] + feat_cols
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbent/property_features.csv", index=False)
    print(f"  property_features.csv: {df.shape}")

# 4. Stoichiometric descriptors
def compute_bulk_stoichiometry(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        counts = list(comp.values())
        total = sum(counts)
        num_elements = len(counts)
        fracs = [c / total for c in counts] if total > 0 else []
        entropy = -sum(f * math.log(f) for f in fracs if f > 0) if fracs else 0.0

        def p_norm(vec, p):
            return sum(abs(x) ** p for x in vec) ** (1.0 / p) if vec else 0.0

        row = [sid, formula, num_elements, total, entropy]
        for p in [2, 3, 5, 7, 10]:
            row.append(p_norm(fracs, p))
        rows.append(row)

    cols = ["system_id", "bulk_symbols", "num_elements", "total_atoms",
            "composition_entropy", "L2_norm", "L3_norm", "L5_norm", "L7_norm", "L10_norm"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("embeddings/adsorbent/stoichiometry_features.csv", index=False)
    print(f"  stoichiometry_features.csv: {df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_bulk_elems = get_all_bulk_elements(system_ids, entries)
    print(f"Total unique bulk elements: {len(all_bulk_elems)}")

    tasks = {
        "adsorbate/physchem": (compute_ads_physchem, (system_ids, entries)),
        "adsorbate/rdkit_desc": (compute_rdkit_descriptors, (system_ids, entries)),
        "adsorbate/composition": (compute_ads_composition, (system_ids, entries)),
        "adsorbent/composition": (compute_bulk_composition, (system_ids, entries, all_bulk_elems)),
        "adsorbent/fraction": (compute_bulk_fraction, (system_ids, entries, all_bulk_elems)),
        "adsorbent/property": (compute_bulk_properties, (system_ids, entries)),
        "adsorbent/stoichiometry": (compute_bulk_stoichiometry, (system_ids, entries)),
    }

    print(f"Running {len(tasks)} featurization tasks in parallel...")
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {}
        for name, (fn, args) in tasks.items():
            futures[executor.submit(fn, *args)] = name
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  ERROR in {name}: {e}")

    print("\nDone! All CSVs saved to embeddings/adsorbate/ and embeddings/adsorbent/")
