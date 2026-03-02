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
dataset_name = 'oc22'
metadata = json.load(open("data/oc22/is2re-total/metadata.json"))

# Filter out entries with empty ads_symbols
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}
system_ids = sorted(entries.keys(), key=int)
print(f"Total entries: {len(metadata)}, after filtering empty ads: {len(entries)}")

# ── Output dirs ────────────────────────────────────────────────────────────
os.makedirs(f"features/{dataset_name}/adsorbate", exist_ok=True)
os.makedirs(f"features/{dataset_name}/adsorbent", exist_ok=True)

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

def get_metals(comp):
    """Return {elem: count} for non-O elements in a composition."""
    return {e: c for e, c in comp.items() if e != "O"}

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

# ── Extended elemental properties for new feature sets ─────────────────────

# d-electron count (ground state configuration, 0 for s/p-block)
ELEM_D_ELECTRONS = {
    "H": 0, "He": 0, "Li": 0, "Be": 0, "B": 0, "C": 0, "N": 0, "O": 0, "F": 0, "Ne": 0,
    "Na": 0, "Mg": 0, "Al": 0, "Si": 0, "P": 0, "S": 0, "Cl": 0, "Ar": 0,
    "K": 0, "Ca": 0, "Sc": 1, "Ti": 2, "V": 3, "Cr": 5, "Mn": 5, "Fe": 6, "Co": 7, "Ni": 8,
    "Cu": 10, "Zn": 10, "Ga": 0, "Ge": 0, "As": 0, "Se": 0, "Br": 0, "Kr": 0,
    "Rb": 0, "Sr": 0, "Y": 1, "Zr": 2, "Nb": 4, "Mo": 5, "Ru": 7, "Rh": 8, "Pd": 10,
    "Ag": 10, "Cd": 10, "In": 0, "Sn": 0, "Sb": 0, "Te": 0, "I": 0, "Xe": 0,
    "Cs": 0, "Ba": 0, "La": 1, "Ce": 1, "Pr": 0, "Nd": 0, "Sm": 0, "Eu": 0, "Gd": 1,
    "Tb": 0, "Dy": 0, "Ho": 0, "Er": 0, "Tm": 0, "Yb": 0, "Lu": 1,
    "Hf": 2, "Ta": 3, "W": 4, "Re": 5, "Os": 6, "Ir": 7, "Pt": 9, "Au": 10, "Hg": 10,
    "Tl": 0, "Pb": 0, "Bi": 0, "Th": 2, "U": 1,
}

# Total valence electrons
ELEM_VALENCE_ELECTRONS = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Ru": 8, "Rh": 9, "Pd": 10,
    "Ag": 11, "Cd": 12, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
    "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Pr": 5, "Nd": 6, "Sm": 8, "Eu": 9, "Gd": 10,
    "Tb": 11, "Dy": 12, "Ho": 13, "Er": 14, "Tm": 15, "Yb": 16, "Lu": 3,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12,
    "Tl": 3, "Pb": 4, "Bi": 5, "Th": 4, "U": 6,
}

# Most common oxidation state in oxides
ELEM_OXIDATION_STATE = {
    "H": 1, "He": 0, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": -3, "O": -2, "F": -1, "Ne": 0,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": -2, "Cl": -1, "Ar": 0,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 3, "Mn": 4, "Fe": 3, "Co": 2, "Ni": 2,
    "Cu": 2, "Zn": 2, "Ga": 3, "Ge": 4, "As": 3, "Se": 4, "Br": -1, "Kr": 0,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Ru": 4, "Rh": 3, "Pd": 2,
    "Ag": 1, "Cd": 2, "In": 3, "Sn": 4, "Sb": 3, "Te": 4, "I": -1, "Xe": 0,
    "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Pr": 3, "Nd": 3, "Sm": 3, "Eu": 3, "Gd": 3,
    "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 4, "Ir": 4, "Pt": 4, "Au": 3, "Hg": 2,
    "Tl": 3, "Pb": 2, "Bi": 3, "Th": 4, "U": 6,
}

# Elemental work function (eV), 0 for non-metals/noble gases
ELEM_WORK_FUNCTION = {
    "H": 0.0, "He": 0.0, "Li": 2.93, "Be": 4.98, "B": 4.45, "C": 5.0, "N": 0.0, "O": 0.0,
    "F": 0.0, "Ne": 0.0, "Na": 2.36, "Mg": 3.66, "Al": 4.28, "Si": 4.85, "P": 0.0, "S": 0.0,
    "Cl": 0.0, "Ar": 0.0, "K": 2.29, "Ca": 2.87, "Sc": 3.50, "Ti": 4.33, "V": 4.30,
    "Cr": 4.50, "Mn": 4.10, "Fe": 4.50, "Co": 5.00, "Ni": 5.15, "Cu": 4.65, "Zn": 4.33,
    "Ga": 4.32, "Ge": 5.0, "As": 3.75, "Se": 5.9, "Br": 0.0, "Kr": 0.0,
    "Rb": 2.26, "Sr": 2.59, "Y": 3.10, "Zr": 4.05, "Nb": 4.30, "Mo": 4.60, "Ru": 4.71,
    "Rh": 4.98, "Pd": 5.12, "Ag": 4.26, "Cd": 4.22, "In": 4.12, "Sn": 4.42, "Sb": 4.55,
    "Te": 4.95, "I": 0.0, "Xe": 0.0, "Cs": 2.14, "Ba": 2.52, "La": 3.50, "Ce": 2.90,
    "Pr": 2.96, "Nd": 3.20, "Sm": 2.70, "Eu": 2.50, "Gd": 3.10, "Tb": 3.00, "Dy": 3.25,
    "Ho": 3.22, "Er": 3.25, "Tm": 3.25, "Yb": 2.60, "Lu": 3.30,
    "Hf": 3.90, "Ta": 4.25, "W": 4.55, "Re": 4.72, "Os": 4.83, "Ir": 5.27, "Pt": 5.65,
    "Au": 5.10, "Hg": 4.49, "Tl": 3.84, "Pb": 4.25, "Bi": 4.22, "Th": 3.40, "U": 3.63,
}

# Shannon ionic radius (pm) for most common oxidation state in oxides (VI coordination)
ELEM_IONIC_RADIUS = {
    "H": 0, "He": 0, "Li": 76, "Be": 45, "B": 27, "C": 16, "N": 0, "O": 140, "F": 133, "Ne": 0,
    "Na": 102, "Mg": 72, "Al": 54, "Si": 40, "P": 38, "S": 0, "Cl": 0, "Ar": 0,
    "K": 138, "Ca": 100, "Sc": 75, "Ti": 61, "V": 54, "Cr": 62, "Mn": 53, "Fe": 65,
    "Co": 75, "Ni": 69, "Cu": 73, "Zn": 74, "Ga": 62, "Ge": 53, "As": 58, "Se": 50,
    "Br": 0, "Kr": 0, "Rb": 152, "Sr": 118, "Y": 90, "Zr": 72, "Nb": 64, "Mo": 59,
    "Ru": 62, "Rh": 67, "Pd": 86, "Ag": 115, "Cd": 95, "In": 80, "Sn": 69, "Sb": 76,
    "Te": 97, "I": 0, "Xe": 0, "Cs": 167, "Ba": 135, "La": 103, "Ce": 87, "Pr": 99,
    "Nd": 98, "Sm": 96, "Eu": 95, "Gd": 94, "Tb": 92, "Dy": 91, "Ho": 90, "Er": 89,
    "Tm": 88, "Yb": 87, "Lu": 86, "Hf": 71, "Ta": 64, "W": 60, "Re": 53, "Os": 63,
    "Ir": 63, "Pt": 63, "Au": 85, "Hg": 102, "Tl": 89, "Pb": 119, "Bi": 103,
    "Th": 94, "U": 89,
}

# Atomic polarizability (Angstrom^3)
ELEM_POLARIZABILITY = {
    "H": 0.667, "He": 0.205, "Li": 24.3, "Be": 5.60, "B": 3.03, "C": 1.76, "N": 1.10,
    "O": 0.802, "F": 0.557, "Ne": 0.396, "Na": 23.6, "Mg": 10.6, "Al": 6.8, "Si": 5.38,
    "P": 3.63, "S": 2.90, "Cl": 2.18, "Ar": 1.64, "K": 43.4, "Ca": 22.8, "Sc": 17.8,
    "Ti": 14.6, "V": 12.4, "Cr": 11.6, "Mn": 9.4, "Fe": 8.4, "Co": 7.5, "Ni": 6.8,
    "Cu": 6.2, "Zn": 5.75, "Ga": 8.12, "Ge": 6.07, "As": 4.31, "Se": 3.77, "Br": 3.05,
    "Kr": 2.48, "Rb": 47.3, "Sr": 27.6, "Y": 22.7, "Zr": 17.9, "Nb": 15.7, "Mo": 12.8,
    "Ru": 9.6, "Rh": 8.6, "Pd": 4.8, "Ag": 7.2, "Cd": 7.4, "In": 10.2, "Sn": 7.7,
    "Sb": 6.6, "Te": 5.5, "I": 5.35, "Xe": 4.04, "Cs": 59.6, "Ba": 39.7, "La": 31.1,
    "Ce": 29.6, "Pr": 28.2, "Nd": 31.4, "Sm": 28.8, "Eu": 27.7, "Gd": 23.5, "Tb": 25.5,
    "Dy": 24.5, "Ho": 23.6, "Er": 22.7, "Tm": 21.8, "Yb": 21.0, "Lu": 21.9, "Hf": 16.2,
    "Ta": 13.1, "W": 11.1, "Re": 9.7, "Os": 8.5, "Ir": 7.6, "Pt": 6.5, "Au": 5.8,
    "Hg": 5.0, "Tl": 7.6, "Pb": 6.8, "Bi": 7.4, "Th": 32.1, "U": 27.4,
}

# Melting point (K)
ELEM_MELTING_POINT = {
    "H": 14, "He": 1, "Li": 454, "Be": 1560, "B": 2349, "C": 3823, "N": 63, "O": 54,
    "F": 53, "Ne": 25, "Na": 371, "Mg": 923, "Al": 933, "Si": 1687, "P": 317, "S": 388,
    "Cl": 172, "Ar": 84, "K": 337, "Ca": 1115, "Sc": 1814, "Ti": 1941, "V": 2183,
    "Cr": 2180, "Mn": 1519, "Fe": 1811, "Co": 1768, "Ni": 1728, "Cu": 1358, "Zn": 693,
    "Ga": 303, "Ge": 1211, "As": 1090, "Se": 494, "Br": 266, "Kr": 116,
    "Rb": 312, "Sr": 1050, "Y": 1799, "Zr": 2128, "Nb": 2750, "Mo": 2896, "Ru": 2607,
    "Rh": 2237, "Pd": 1828, "Ag": 1235, "Cd": 594, "In": 430, "Sn": 505, "Sb": 904,
    "Te": 723, "I": 387, "Xe": 161, "Cs": 302, "Ba": 1000, "La": 1193, "Ce": 1068,
    "Pr": 1204, "Nd": 1297, "Sm": 1345, "Eu": 1099, "Gd": 1585, "Tb": 1629, "Dy": 1680,
    "Ho": 1734, "Er": 1802, "Tm": 1818, "Yb": 1097, "Lu": 1925, "Hf": 2506, "Ta": 3290,
    "W": 3695, "Re": 3459, "Os": 3306, "Ir": 2719, "Pt": 2041, "Au": 1337, "Hg": 234,
    "Tl": 577, "Pb": 601, "Bi": 544, "Th": 2023, "U": 1405,
}

# Oxide formation enthalpy (kJ/mol per O atom) for most common oxide
# Negative = exothermic (stable oxide). 0 for noble gases/non-metals without common oxides.
ELEM_OXIDE_FORMATION_ENTHALPY = {
    "H": -143, "He": 0, "Li": -596, "Be": -609, "B": -410, "C": -394, "N": 0, "O": 0,
    "F": 0, "Ne": 0, "Na": -414, "Mg": -602, "Al": -559, "Si": -456, "P": -299, "S": -297,
    "Cl": 0, "Ar": 0, "K": -363, "Ca": -635, "Sc": -636, "Ti": -472, "V": -310,
    "Cr": -380, "Mn": -385, "Fe": -275, "Co": -238, "Ni": -240, "Cu": -157, "Zn": -351,
    "Ga": -363, "Ge": -290, "As": -310, "Se": -172, "Br": 0, "Kr": 0,
    "Rb": -339, "Sr": -592, "Y": -635, "Zr": -551, "Nb": -380, "Mo": -248, "Ru": -153,
    "Rh": -114, "Pd": -85, "Ag": -31, "Cd": -258, "In": -309, "Sn": -291, "Sb": -253,
    "Te": -161, "I": 0, "Xe": 0, "Cs": -346, "Ba": -554, "La": -598, "Ce": -545,
    "Pr": -603, "Nd": -603, "Sm": -608, "Eu": -550, "Gd": -607, "Tb": -622, "Dy": -621,
    "Ho": -627, "Er": -633, "Tm": -630, "Yb": -605, "Lu": -626, "Hf": -573, "Ta": -409,
    "W": -281, "Re": -177, "Os": -99, "Ir": -137, "Pt": -67, "Au": 27, "Hg": -91,
    "Tl": -130, "Pb": -219, "Bi": -191, "Th": -614, "U": -543,
}

# Sets for element classification
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
NOBLE_METALS = {"Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au"}
RARE_EARTHS = {
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y",
}


# ═══════════════════════════════════════════════════════════════════════════
# ADSORBATE FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════

def get_mol(ads_sym):
    smi = ADS_SMILES[ads_sym]
    return Chem.MolFromSmiles(smi)

# ── 1. Catalysis-relevant physicochemical properties (hardcoded) ───────────
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
    df.to_csv(f"features/{dataset_name}/adsorbate/physchem_features.csv", index=False)
    print(f"  physchem_features.csv: {df.shape}")

# ── 2. RDKit molecular descriptors (corrected for small species) ───────────
def compute_rdkit_descriptors(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["ads_symbols"]
        mol = get_mol(formula)
        mol_h = Chem.AddHs(mol)

        num_hbond_donors = 0
        num_hbond_acceptors = 0
        num_lone_pairs = 0
        for atom in mol_h.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum in (7, 8):  # N, O
                h_neighbors = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
                if h_neighbors > 0:
                    num_hbond_donors += h_neighbors
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
    df.to_csv(f"features/{dataset_name}/adsorbate/rdkit_descriptors_features.csv", index=False)
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
    df.to_csv(f"features/{dataset_name}/adsorbate/composition_features.csv", index=False)
    print(f"  composition_features.csv: {df.shape}")

# ── 4. Adsorbate adsorption / surface interaction features (hardcoded) ────
# Sources: NIST Chemistry WebBook, CRC Handbook, DFT literature
ADS_ADSORPTION = {
    #              HOMO_eV   LUMO_eV  hardness  softness  electrophilicity  form_enthalpy  entropy_298K  char_freq  bond_order  has_lone_pair  has_pi_bond  binding_mode
    "H":          [-13.60,    0.0,     6.42,     0.078,    3.19,             218.0,          114.7,        0,         0,          0,             0,           2],
    "C":          [-11.26,   -1.26,    5.00,     0.100,    3.94,             716.7,          158.1,        0,         0,          0,             0,           2],
    "N":          [-14.53,    0.07,    7.30,     0.068,    1.43,             472.7,          153.3,        0,         0,          1,             0,           2],
    "O":          [-13.62,   -1.46,    6.08,     0.082,    4.69,             249.2,          161.1,        0,         0,          1,             0,           2],
    "OH":         [-13.02,   -1.83,    5.59,     0.089,    4.94,             37.3,           183.7,        3570,      1,          1,             0,           0],
    "H2O":        [-12.62,    4.0,     6.33,     0.079,    0.74,            -241.8,          188.8,        3657,      1,          1,             0,           0],
    "O2":         [-12.07,   -0.45,    5.81,     0.086,    2.68,             0.0,            205.2,        1580,      2,          1,             1,           1],
    "CO":         [-14.01,   -1.11,    6.45,     0.078,    4.45,            -110.5,          197.7,        2143,      3,          1,             1,           0],
    "HO2":        [-11.35,   -1.08,    5.14,     0.097,    3.76,             12.0,           229.0,        1098,      1,          1,             0,           1],
}
ADS_ADSORPTION_COLS = [
    "HOMO_energy_eV", "LUMO_energy_eV", "chemical_hardness_eV", "chemical_softness",
    "electrophilicity_index", "formation_enthalpy_kJmol", "entropy_298K_JmolK",
    "characteristic_frequency_cm1", "bond_order", "has_lone_pair", "has_pi_bond",
    "surface_binding_mode",
]

def compute_ads_adsorption(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["ads_symbols"]
        props = ADS_ADSORPTION[formula]
        rows.append([sid, formula] + props)
    cols = ["system_id", "ads_symbols"] + ADS_ADSORPTION_COLS
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"features/{dataset_name}/adsorbate/adsorption_features.csv", index=False)
    print(f"  adsorption_features.csv: {df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# ADSORBENT FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════

# 1. Magpie-style elemental property statistics (KEPT)
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
    df.to_csv(f"features/{dataset_name}/adsorbent/property_features.csv", index=False)
    print(f"  property_features.csv: {df.shape}")

# 2. Stoichiometric descriptors (KEPT)
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
    df.to_csv(f"features/{dataset_name}/adsorbent/stoichiometry_features.csv", index=False)
    print(f"  stoichiometry_features.csv: {df.shape}")

# 3. Electronic features — d-band and valence electron statistics
def compute_bulk_electronic(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        metals = get_metals(comp)

        if not metals:
            rows.append([sid, formula] + [0.0] * 12)
            continue

        total_metal = sum(metals.values())

        # Weighted arrays for statistics
        d_elec = []
        valence = []
        oxstate = []
        wf = []
        for elem, cnt in metals.items():
            d_elec.extend([ELEM_D_ELECTRONS.get(elem, 0)] * cnt)
            valence.extend([ELEM_VALENCE_ELECTRONS.get(elem, 0)] * cnt)
            oxstate.extend([ELEM_OXIDATION_STATE.get(elem, 0)] * cnt)
            wf.extend([ELEM_WORK_FUNCTION.get(elem, 0.0)] * cnt)

        d_arr = np.array(d_elec, dtype=float)
        v_arr = np.array(valence, dtype=float)
        o_arr = np.array(oxstate, dtype=float)
        w_arr = np.array(wf, dtype=float)

        row = [
            sid, formula,
            float(np.mean(d_arr)),                    # d_electron_mean
            float(np.std(d_arr)),                     # d_electron_std
            float(np.max(d_arr) - np.min(d_arr)),     # d_electron_range
            float(np.mean(v_arr)),                    # valence_electron_mean
            float(np.std(v_arr)),                     # valence_electron_std
            float(np.mean(o_arr)),                    # oxidation_state_mean
            float(np.std(o_arr)),                     # oxidation_state_std
            float(np.mean(w_arr)),                    # work_function_mean
            float(np.std(w_arr)),                     # work_function_std
            float(np.max(d_arr)),                     # d_electron_max
            float(np.min(d_arr)),                     # d_electron_min
            float(np.max(w_arr) - np.min(w_arr)),     # work_function_range
        ]
        rows.append(row)

    cols = ["system_id", "bulk_symbols",
            "d_electron_mean", "d_electron_std", "d_electron_range",
            "valence_electron_mean", "valence_electron_std",
            "oxidation_state_mean", "oxidation_state_std",
            "work_function_mean", "work_function_std",
            "d_electron_max", "d_electron_min", "work_function_range"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"features/{dataset_name}/adsorbent/electronic_features.csv", index=False)
    print(f"  electronic_features.csv: {df.shape}")

# 4. Bonding features — metal-oxygen bonding character and crystal chemistry
def compute_bulk_bonding(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        metals = get_metals(comp)
        n_O = comp.get("O", 0)

        if not metals:
            rows.append([sid, formula] + [0.0] * 10)
            continue

        total_metal = sum(metals.values())
        en_O = ELEM_PROPS["O"][2]  # 3.44

        # Per-metal weighted arrays
        en_diffs = []
        radii = []
        polariz = []
        ionicities = []
        en_values = []

        for elem, cnt in metals.items():
            en_m = ELEM_PROPS.get(elem, [0]*6)[2]
            ir = ELEM_IONIC_RADIUS.get(elem, 0)
            pol = ELEM_POLARIZABILITY.get(elem, 0.0)

            delta_en = abs(en_O - en_m)
            ionicity = 1.0 - math.exp(-0.25 * delta_en ** 2)

            en_diffs.extend([delta_en] * cnt)
            radii.extend([ir] * cnt)
            polariz.extend([pol] * cnt)
            ionicities.extend([ionicity] * cnt)
            en_values.extend([en_m] * cnt)

        en_arr = np.array(en_diffs, dtype=float)
        r_arr = np.array(radii, dtype=float)
        p_arr = np.array(polariz, dtype=float)
        ion_arr = np.array(ionicities, dtype=float)

        # Sanderson electronegativity geometric mean (oxide basicity proxy)
        all_en = []
        for elem, cnt in comp.items():
            en = ELEM_PROPS.get(elem, [0]*6)[2]
            if en > 0:
                all_en.extend([en] * cnt)
        oxide_basicity = float(np.exp(np.mean(np.log(np.array(all_en))))) if all_en else 0.0

        # Tolerance factor for perovskite-like ABO3
        # Only compute if exactly 2 distinct metals and formula resembles ABO3
        tolerance = 0.0
        metal_elems = sorted(metals.keys())
        if len(metal_elems) == 2:
            r_A = ELEM_IONIC_RADIUS.get(metal_elems[0], 0)
            r_B = ELEM_IONIC_RADIUS.get(metal_elems[1], 0)
            r_O = 140  # O^2- ionic radius
            if r_A > 0 and r_B > 0:
                # Assign larger radius as A-site
                if r_A < r_B:
                    r_A, r_B = r_B, r_A
                tolerance = (r_A + r_O) / (math.sqrt(2) * (r_B + r_O))

        # Metal radius ratio (structural distortion indicator)
        r_max = float(np.max(r_arr)) if len(r_arr) > 0 else 0.0
        r_min = float(np.min(r_arr)) if len(r_arr) > 0 else 0.0
        radius_ratio = r_max / r_min if r_min > 0 else 0.0

        row = [
            sid, formula,
            float(np.mean(en_arr)),       # avg_metal_O_electronegativity_diff
            float(np.mean(r_arr)),         # avg_metal_ionic_radius
            radius_ratio,                  # metal_radius_ratio
            float(np.mean(ion_arr)),       # avg_bond_ionicity
            oxide_basicity,                # oxide_basicity
            total_metal / n_O if n_O > 0 else 0.0,  # metal_to_O_ratio
            float(np.mean(p_arr)),         # avg_metal_polarizability
            tolerance,                     # tolerance_factor
            float(np.std(en_arr)),         # metal_O_en_diff_std
            float(np.std(r_arr)),          # metal_ionic_radius_std
        ]
        rows.append(row)

    cols = ["system_id", "bulk_symbols",
            "avg_metal_O_electronegativity_diff", "avg_metal_ionic_radius",
            "metal_radius_ratio", "avg_bond_ionicity", "oxide_basicity",
            "metal_to_O_ratio", "avg_metal_polarizability", "tolerance_factor",
            "metal_O_en_diff_std", "metal_ionic_radius_std"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"features/{dataset_name}/adsorbent/bonding_features.csv", index=False)
    print(f"  bonding_features.csv: {df.shape}")

# 5. Thermodynamic features — stability and reactivity indicators
def compute_bulk_thermodynamic(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        metals = get_metals(comp)

        if not metals:
            rows.append([sid, formula] + [0.0] * 8)
            continue

        total_metal = sum(metals.values())
        total_atoms = sum(comp.values())

        mp_arr = []
        form_enth = []
        masses = []

        for elem, cnt in metals.items():
            mp_arr.extend([ELEM_MELTING_POINT.get(elem, 0)] * cnt)
            form_enth.extend([ELEM_OXIDE_FORMATION_ENTHALPY.get(elem, 0)] * cnt)
            masses.extend([ELEM_PROPS.get(elem, [0]*6)[1]] * cnt)

        mp = np.array(mp_arr, dtype=float)
        fe = np.array(form_enth, dtype=float)
        m = np.array(masses, dtype=float)

        # Mass-weighted mixing entropy
        mass_fracs = m / m.sum() if m.sum() > 0 else np.zeros_like(m)
        entropy_mass = -float(np.sum(mass_fracs * np.log(mass_fracs + 1e-12)))

        # Reducibility index: more negative formation enthalpy = harder to reduce
        # Use weighted mean; less negative = more reducible
        reducibility = -float(np.mean(fe))  # positive = more reducible

        row = [
            sid, formula,
            float(np.mean(mp)),                    # avg_metal_melting_point
            float(np.std(mp)),                     # metal_melting_point_std
            float(np.max(mp) - np.min(mp)),        # metal_melting_point_range
            float(np.mean(fe)),                    # avg_oxide_formation_enthalpy_per_O
            reducibility,                          # reducibility_index
            entropy_mass,                          # entropy_mixing_mass_weighted
            float(np.min(fe)),                     # min_oxide_formation_enthalpy
            float(np.max(fe)),                     # max_oxide_formation_enthalpy
        ]
        rows.append(row)

    cols = ["system_id", "bulk_symbols",
            "avg_metal_melting_point", "metal_melting_point_std", "metal_melting_point_range",
            "avg_oxide_formation_enthalpy_per_O", "reducibility_index",
            "entropy_mixing_mass_weighted",
            "min_oxide_formation_enthalpy", "max_oxide_formation_enthalpy"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"features/{dataset_name}/adsorbent/thermodynamic_features.csv", index=False)
    print(f"  thermodynamic_features.csv: {df.shape}")

# 6. Catalytic features — catalysis-specific descriptors
def compute_bulk_catalytic(system_ids, entries):
    rows = []
    for sid in system_ids:
        formula = entries[sid]["bulk_symbols"]
        comp = parse_formula(formula)
        metals = get_metals(comp)

        if not metals:
            rows.append([sid, formula] + [0.0] * 8)
            continue

        total_metal = sum(metals.values())
        total_atoms = sum(comp.values())

        # Binary flags
        has_tm = int(any(e in TRANSITION_METALS for e in metals))
        has_noble = int(any(e in NOBLE_METALS for e in metals))
        has_re = int(any(e in RARE_EARTHS for e in metals))

        # Fraction transition metal (by atom count)
        tm_count = sum(cnt for e, cnt in metals.items() if e in TRANSITION_METALS)
        frac_tm = tm_count / total_atoms if total_atoms > 0 else 0.0

        # d-band filling: avg d-electrons / 10 for d-block metals only
        d_block_d = []
        for elem, cnt in metals.items():
            if elem in TRANSITION_METALS:
                d_elec = ELEM_D_ELECTRONS.get(elem, 0)
                d_block_d.extend([d_elec / 10.0] * cnt)
        d_band_filling = float(np.mean(d_block_d)) if d_block_d else 0.0

        # Metal diversity
        metal_diversity = len(metals)

        # Electronegativity spread among metals
        en_metals = [ELEM_PROPS.get(e, [0]*6)[2] for e in metals if e in ELEM_PROPS]
        en_spread = max(en_metals) - min(en_metals) if len(en_metals) > 1 else 0.0

        # Weighted electron affinity of metals
        ea_arr = []
        for elem, cnt in metals.items():
            ea = ELEM_PROPS.get(elem, [0]*6)[4]  # electron_affinity index
            ea_arr.extend([ea] * cnt)
        wtd_ea = float(np.mean(ea_arr)) if ea_arr else 0.0

        row = [
            sid, formula,
            has_tm,                 # has_transition_metal
            has_noble,              # has_noble_metal
            has_re,                 # has_rare_earth
            frac_tm,                # fraction_transition_metal
            d_band_filling,         # d_band_filling
            metal_diversity,        # metal_diversity
            en_spread,              # metal_electronegativity_spread
            wtd_ea,                 # weighted_electron_affinity
        ]
        rows.append(row)

    cols = ["system_id", "bulk_symbols",
            "has_transition_metal", "has_noble_metal", "has_rare_earth",
            "fraction_transition_metal", "d_band_filling", "metal_diversity",
            "metal_electronegativity_spread", "weighted_electron_affinity"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"features/{dataset_name}/adsorbent/catalytic_features.csv", index=False)
    print(f"  catalytic_features.csv: {df.shape}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tasks = {
        "adsorbate/physchem": (compute_ads_physchem, (system_ids, entries)),
        "adsorbate/rdkit_desc": (compute_rdkit_descriptors, (system_ids, entries)),
        "adsorbate/composition": (compute_ads_composition, (system_ids, entries)),
        "adsorbate/adsorption": (compute_ads_adsorption, (system_ids, entries)),
        "adsorbent/property": (compute_bulk_properties, (system_ids, entries)),
        "adsorbent/stoichiometry": (compute_bulk_stoichiometry, (system_ids, entries)),
        "adsorbent/electronic": (compute_bulk_electronic, (system_ids, entries)),
        "adsorbent/bonding": (compute_bulk_bonding, (system_ids, entries)),
        "adsorbent/thermodynamic": (compute_bulk_thermodynamic, (system_ids, entries)),
        "adsorbent/catalytic": (compute_bulk_catalytic, (system_ids, entries)),
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

    print(f"\nDone! All CSVs saved to features/{dataset_name}/adsorbate/ and features/{dataset_name}/adsorbent/")
