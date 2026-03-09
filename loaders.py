"""Multi-format data loading (CSV, JSON, ASE .db, CIF, XYZ, SMILES, PDB, mmCIF, SDF, MOL, MOL2)."""

import logging
import os
import json
import re

import pandas as pd

logger = logging.getLogger(__name__)


def _detect_format(path):
    """Auto-detect file format from extension."""
    ext = os.path.splitext(path)[1].lower()
    format_map = {
        ".csv": "csv",
        ".json": "json",
        ".db": "ase_db",
        ".xyz": "xyz",
        ".smi": "smiles",
        ".smiles": "smiles",
        ".pdb": "pdb",
        ".cif": "mmcif",
        ".mmcif": "mmcif",
        ".fasta": "fasta",
        ".fa": "fasta",
        ".faa": "fasta",
        ".sdf": "sdf",
        ".mol": "mol",
        ".mol2": "mol2",
    }
    if os.path.isdir(path):
        files = os.listdir(path)
        if any(f.endswith(".pdb") for f in files):
            return "pdb_dir"
        if any(f.endswith((".cif", ".mmcif")) for f in files):
            return "mmcif_dir"
        if any(f.endswith(".xyz") for f in files):
            return "xyz_dir"
    return format_map.get(ext, "csv")


def load_csv(path):
    """Load CSV file."""
    df = pd.read_csv(path)
    df["_row_id"] = df.index.astype(str)
    return df


def load_json(path):
    """Load JSON file. Handles dict-of-dicts (OC22-style) and list-of-dicts."""
    with open(path) as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        records = []
        for key, val in data.items():
            if isinstance(val, dict):
                row = {"_row_id": str(key)}
                row.update(val)
                records.append(row)
            else:
                records.append({"_row_id": str(key), "value": val})
        df = pd.DataFrame(records)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
        df["_row_id"] = df.index.astype(str)
    else:
        raise ValueError(f"Unsupported JSON structure: {type(data)}")

    return df


def load_ase_db(path):
    """Load ASE database file (requires ase)."""
    from ase.db import connect
    db = connect(path)
    records = []
    for row in db.select():
        rec = {"_row_id": str(row.id)}
        rec["formula"] = row.formula
        rec.update(row.key_value_pairs)
        records.append(rec)
    return pd.DataFrame(records)


def load_cif_dir(path):
    """Load material CIF files from a directory (requires ase)."""
    from ase.io import read as ase_read
    records = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".cif"):
            continue
        fpath = os.path.join(path, fname)
        atoms = ase_read(fpath)
        rec = {
            "_row_id": os.path.splitext(fname)[0],
            "formula": atoms.get_chemical_formula(),
        }
        records.append(rec)
    return pd.DataFrame(records)


def load_xyz_dir(path):
    """Load XYZ files from a directory (requires ase)."""
    from ase.io import read as ase_read
    records = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".xyz"):
            continue
        fpath = os.path.join(path, fname)
        atoms = ase_read(fpath)
        rec = {
            "_row_id": os.path.splitext(fname)[0],
            "formula": atoms.get_chemical_formula(),
        }
        records.append(rec)
    return pd.DataFrame(records)


def load_smiles(path):
    """Load SMILES file (one SMILES per line, optionally with ID)."""
    records = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split()
            if len(parts) >= 2:
                records.append({"_row_id": parts[1], "smiles": parts[0]})
            elif len(parts) == 1:
                records.append({"_row_id": str(i), "smiles": parts[0]})
    return pd.DataFrame(records)


# ── PDB/mmCIF parsing helpers ────────────────────────────────────────────

# Standard amino acid 3-letter to 1-letter mapping
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard
    "MSE": "M", "SEC": "U", "PYL": "O",
}

# Common solvent/buffer molecules to exclude from ligand detection
COMMON_SOLVENTS = {
    "HOH", "WAT", "DOD",  # water
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "DMS", "ACT", "FMT",
    "CL", "NA", "MG", "ZN", "CA", "MN", "FE", "CU", "CO", "NI", "K",
    "BR", "IOD", "CIT", "TAR", "MPD", "BME", "DTT", "TRS", "MES",
    "HED", "EPE", "IPA", "ACE", "NH4", "NO3",
}


def _parse_pdb_structure(path):
    """Parse a PDB file to extract protein sequences and ligand identifiers.

    Returns:
        dict with keys:
            - chains: {chain_id: sequence}
            - ligands: list of {resname, chain, resseq} for non-protein, non-solvent residues
            - structure_id: filename stem
    """
    chains = {}
    ligands = []
    seen_residues = set()

    with open(path) as fh:
        for line in fh:
            record = line[:6].strip()

            if record in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain_id = line[21].strip() or "A"
                resseq = line[22:26].strip()
                residue_key = (chain_id, resseq, resname)

                if residue_key in seen_residues:
                    continue
                seen_residues.add(residue_key)

                if record == "ATOM" and resname in AA3TO1:
                    # Protein residue — only count once per residue
                    if chain_id not in chains:
                        chains[chain_id] = []
                    chains[chain_id].append((int(resseq), AA3TO1[resname]))

                elif record == "HETATM" and resname not in AA3TO1 and resname not in COMMON_SOLVENTS:
                    ligands.append({
                        "resname": resname,
                        "chain": chain_id,
                        "resseq": resseq,
                    })

    # Sort residues by sequence number and join
    sequences = {}
    for chain_id, residues in chains.items():
        residues.sort(key=lambda x: x[0])
        sequences[chain_id] = "".join(aa for _, aa in residues)

    # Deduplicate ligands by resname
    unique_ligands = list({lig["resname"]: lig for lig in ligands}.values())

    return {
        "chains": sequences,
        "ligands": unique_ligands,
        "structure_id": os.path.splitext(os.path.basename(path))[0],
    }


def _parse_mmcif_structure(path):
    """Parse an mmCIF file to extract protein sequences and ligand identifiers.

    Uses Biopython's MMCIFParser if available, falls back to manual parsing.

    Returns:
        dict with same structure as _parse_pdb_structure
    """
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure_id = os.path.splitext(os.path.basename(path))[0]
        structure = parser.get_structure(structure_id, path)

        chains = {}
        ligands = []
        seen_ligands = set()

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                seq = []
                for residue in chain:
                    resname = residue.get_resname().strip()
                    het_flag = residue.get_id()[0]

                    if het_flag == " " and resname in AA3TO1:
                        seq.append(AA3TO1[resname])
                    elif het_flag.startswith("H_") and resname not in COMMON_SOLVENTS:
                        if resname not in seen_ligands:
                            seen_ligands.add(resname)
                            ligands.append({
                                "resname": resname,
                                "chain": chain_id,
                                "resseq": str(residue.get_id()[1]),
                            })

                if seq:
                    chains[chain_id] = "".join(seq)
            break  # first model only

        return {
            "chains": chains,
            "ligands": ligands,
            "structure_id": structure_id,
        }

    except ImportError:
        # Fallback: parse _atom_site records manually
        return _parse_mmcif_manual(path)


def _parse_mmcif_manual(path):
    """Manual mmCIF parser for _atom_site records (no Biopython dependency)."""
    structure_id = os.path.splitext(os.path.basename(path))[0]
    chains = {}
    ligands = []
    seen_residues = set()
    seen_ligands = set()

    in_atom_site = False
    col_names = []
    col_map = {}

    with open(path) as fh:
        for line in fh:
            line = line.strip()

            if line.startswith("_atom_site."):
                in_atom_site = True
                col_name = line.split(".")[1].split()[0]
                col_names.append(col_name)
                col_map[col_name] = len(col_names) - 1
                continue

            if in_atom_site and not line.startswith("_") and line and not line.startswith("#") and not line.startswith("loop_"):
                parts = line.split()
                if len(parts) < len(col_names):
                    in_atom_site = False
                    continue

                group = parts[col_map.get("group_PDB", 0)]
                resname = parts[col_map.get("label_comp_id", 0)]
                chain_id = parts[col_map.get("label_asym_id", 0)]
                resseq = parts[col_map.get("label_seq_id", 0)]
                residue_key = (chain_id, resseq, resname)

                if residue_key in seen_residues:
                    continue
                seen_residues.add(residue_key)

                if group == "ATOM" and resname in AA3TO1:
                    if chain_id not in chains:
                        chains[chain_id] = []
                    try:
                        chains[chain_id].append((int(resseq), AA3TO1[resname]))
                    except ValueError:
                        chains[chain_id].append((0, AA3TO1[resname]))

                elif group == "HETATM" and resname not in AA3TO1 and resname not in COMMON_SOLVENTS:
                    if resname not in seen_ligands:
                        seen_ligands.add(resname)
                        ligands.append({
                            "resname": resname,
                            "chain": chain_id,
                            "resseq": resseq,
                        })

            elif in_atom_site and (line.startswith("#") or line.startswith("loop_") or line.startswith("_")):
                in_atom_site = False

    sequences = {}
    for chain_id, residues in chains.items():
        residues.sort(key=lambda x: x[0])
        sequences[chain_id] = "".join(aa for _, aa in residues)

    return {
        "chains": sequences,
        "ligands": ligands,
        "structure_id": structure_id,
    }


def _structure_to_records(parsed, file_path):
    """Convert parsed structure into DataFrame records.

    Produces one row per chain. If ligands are present, each chain gets
    paired with each ligand (protein-ligand interaction rows).
    If no ligands, produces one row per chain (protein-only).
    """
    records = []
    structure_id = parsed["structure_id"]
    chains = parsed["chains"]
    ligands = parsed["ligands"]

    if ligands:
        # Protein-ligand mode: one row per (chain, ligand) pair
        for chain_id, sequence in chains.items():
            for lig in ligands:
                records.append({
                    "_row_id": f"{structure_id}_{chain_id}_{lig['resname']}",
                    "structure_id": structure_id,
                    "chain_id": chain_id,
                    "sequence": sequence,
                    "ligand_id": lig["resname"],
                    "ligand_chain": lig["chain"],
                    "has_ligand": True,
                })
    else:
        # Protein-only mode: one row per chain
        for chain_id, sequence in chains.items():
            records.append({
                "_row_id": f"{structure_id}_{chain_id}",
                "structure_id": structure_id,
                "chain_id": chain_id,
                "sequence": sequence,
                "ligand_id": "",
                "ligand_chain": "",
                "has_ligand": False,
            })

    return records


def load_pdb(path):
    """Load a single PDB file. Returns one row per chain (or per chain-ligand pair)."""
    parsed = _parse_pdb_structure(path)
    records = _structure_to_records(parsed, path)
    return pd.DataFrame(records)


def load_mmcif(path):
    """Load a single mmCIF file. Returns one row per chain (or per chain-ligand pair)."""
    parsed = _parse_mmcif_structure(path)
    records = _structure_to_records(parsed, path)
    return pd.DataFrame(records)


def load_pdb_dir(path):
    """Load all PDB files from a directory."""
    records = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".pdb"):
            continue
        fpath = os.path.join(path, fname)
        parsed = _parse_pdb_structure(fpath)
        records.extend(_structure_to_records(parsed, fpath))
    return pd.DataFrame(records)


def load_mmcif_dir(path):
    """Load all mmCIF files from a directory."""
    records = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith((".cif", ".mmcif")):
            continue
        fpath = os.path.join(path, fname)
        parsed = _parse_mmcif_structure(fpath)
        records.extend(_structure_to_records(parsed, fpath))
    return pd.DataFrame(records)


def load_fasta(path):
    """Load FASTA file with protein sequences."""
    records = []
    current_id = None
    current_seq = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    records.append({
                        "_row_id": current_id,
                        "sequence": "".join(current_seq),
                    })
                # Parse header: >id description
                header = line[1:].split()
                current_id = header[0] if header else str(len(records))
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        records.append({
            "_row_id": current_id,
            "sequence": "".join(current_seq),
        })

    return pd.DataFrame(records)


def load_sdf(path):
    """Load an SDF file containing one or more molecules (requires rdkit).

    Extracts SMILES, molecular formula, and any SDF properties.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    suppl = Chem.SDMolSupplier(path, removeHs=False)
    records = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name").strip() else str(i)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

        rec = {
            "_row_id": mol_name,
            "smiles": smiles,
            "formula": formula,
        }
        # Extract SDF properties
        props = mol.GetPropsAsDict()
        rec.update({k: v for k, v in props.items() if k != "_Name"})

        records.append(rec)
    return pd.DataFrame(records)


def load_mol(path):
    """Load a single MOL file (requires rdkit)."""
    from rdkit import Chem

    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to parse MOL file: {path}")

    mol_name = os.path.splitext(os.path.basename(path))[0]
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

    return pd.DataFrame([{
        "_row_id": mol_name,
        "smiles": smiles,
        "formula": formula,
    }])


def load_mol2(path):
    """Load a MOL2 file (requires rdkit)."""
    from rdkit import Chem

    mol = Chem.MolFromMol2File(path, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to parse MOL2 file: {path}")

    mol_name = os.path.splitext(os.path.basename(path))[0]
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

    return pd.DataFrame([{
        "_row_id": mol_name,
        "smiles": smiles,
        "formula": formula,
    }])


LOADERS = {
    "csv": load_csv,
    "json": load_json,
    "ase_db": load_ase_db,
    "cif_dir": load_cif_dir,
    "xyz_dir": load_xyz_dir,
    "smiles": load_smiles,
    "pdb": load_pdb,
    "mmcif": load_mmcif,
    "pdb_dir": load_pdb_dir,
    "mmcif_dir": load_mmcif_dir,
    "fasta": load_fasta,
    "sdf": load_sdf,
    "mol": load_mol,
    "mol2": load_mol2,
}


def load_data(path, fmt=None, filters=None):
    """Load data from any supported format.

    Args:
        path: file or directory path
        fmt: format string, or None for auto-detection
        filters: list of FilterConfig objects

    Returns:
        DataFrame with _row_id column
    """
    if fmt is None:
        fmt = _detect_format(path)

    loader = LOADERS.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {list(LOADERS.keys())}")

    df = loader(path)

    # Apply filters
    if filters:
        for filt in filters:
            if filt.not_equal is not None:
                df = df[df[filt.column] != filt.not_equal]
            if filt.not_empty:
                df = df[df[filt.column].astype(str).str.strip() != ""]

    return df.reset_index(drop=True)


# ── Column-level entity type inference ───────────────────────────────────

# Unambiguous file format → default entity type and column
FORMAT_TYPE_DEFAULTS = {
    "smiles":    {"type": "molecule",    "column": "smiles"},
    "sdf":       {"type": "molecule",    "column": "smiles"},
    "mol":       {"type": "molecule",    "column": "smiles"},
    "mol2":      {"type": "molecule",    "column": "smiles"},
    "fasta":     {"type": "biomolecule", "column": "sequence"},
    "pdb":       {"type": "biomolecule", "column": "sequence"},
    "pdb_dir":   {"type": "biomolecule", "column": "sequence"},
    "mmcif":     {"type": "biomolecule", "column": "sequence"},
    "mmcif_dir": {"type": "biomolecule", "column": "sequence"},
    "ase_db":    {"type": "material",    "column": "formula"},
    "cif_dir":   {"type": "material",    "column": "formula"},
    "xyz_dir":   {"type": "material",    "column": "formula"},
}

TYPE_DEFAULT_NAMES = {
    "molecule":    "compound",
    "material":    "material",
    "biomolecule": "protein",
    "gene":        "gene",
}

FEAT_DEFAULTS = {
    "molecule":    ["rdkit_descriptors", "physicochemical"],
    "material":    ["magpie_properties", "stoichiometry", "electronic"],
    "biomolecule": ["sequence_properties"],
    "gene":        ["nucleotide_composition", "kmer_frequencies"],
}

# Regex for chemical formulas: e.g. Fe2O3, TiO2, CaCO3, H2O
_FORMULA_RE = re.compile(
    r"^[A-Z][a-z]?(\d+(\.\d+)?)?([A-Z][a-z]?(\d+(\.\d+)?)?)*$"
)
_AA_CHARS = set("ACDEFGHIKLMNPQRSTVWY")
_NT_CHARS = set("ATCGUN")


def _is_smiles(val):
    """Check if a string is a valid SMILES (requires RDKit)."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        if len(val) < 2:
            return False
        mol = Chem.MolFromSmiles(val)
        return mol is not None
    except Exception:
        return False


def _is_formula(val):
    """Check if a string looks like a chemical formula with at least one digit."""
    if len(val) < 2 or len(val) > 100:
        return False
    if not _FORMULA_RE.match(val):
        return False
    # Require at least one digit to distinguish from short AA sequences or element symbols
    return any(c.isdigit() for c in val)


def _is_amino_acid_sequence(val):
    """Check if a string looks like a protein sequence."""
    if len(val) < 15:
        return False
    upper = val.upper()
    aa_count = sum(1 for c in upper if c in _AA_CHARS)
    return aa_count / len(upper) >= 0.90


def _is_nucleotide_sequence(val):
    """Check if a string looks like a DNA/RNA sequence."""
    if len(val) < 15:
        return False
    upper = val.upper()
    nt_count = sum(1 for c in upper if c in _NT_CHARS)
    return nt_count / len(upper) >= 0.85


def _classify_value(val):
    """Classify a single string value as an entity type.

    Returns type string or None. Gene check runs before biomolecule
    because ATCG is a subset of amino acid codes.
    """
    s = str(val).strip()
    if not s or len(s) < 2:
        return None

    # Try numeric check — skip pure numbers
    try:
        float(s)
        return None
    except ValueError:
        pass

    # Gene before biomolecule (ATCG ⊂ amino acid alphabet)
    if _is_nucleotide_sequence(s):
        return "gene"
    if _is_amino_acid_sequence(s):
        return "biomolecule"
    if _is_formula(s):
        return "material"
    if _is_smiles(s):
        return "molecule"

    return None


def infer_column_types(df, sample_size=50):
    """Infer entity types for each column by sampling values.

    Args:
        df: DataFrame with data
        sample_size: number of rows to sample per column

    Returns:
        dict mapping column_name -> {"type": str|None, "confidence": float}
    """
    result = {}
    skip_cols = {"_row_id"}

    for col in df.columns:
        if col in skip_cols:
            continue

        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if len(values) == 0:
            result[col] = {"type": None, "confidence": 0.0}
            continue

        # Skip columns that are all short strings (likely IDs or labels)
        median_len = values.str.len().median()
        if median_len < 2:
            result[col] = {"type": None, "confidence": 0.0}
            continue

        sample = values.sample(min(sample_size, len(values)), random_state=42)
        votes = {"molecule": 0, "material": 0, "biomolecule": 0, "gene": 0}
        total = len(sample)

        for val in sample:
            t = _classify_value(val)
            if t is not None:
                votes[t] += 1

        best_type = max(votes, key=votes.get)
        best_count = votes[best_type]
        confidence = best_count / total if total > 0 else 0.0

        if confidence >= 0.5:
            result[col] = {"type": best_type, "confidence": round(confidence, 3)}
        else:
            result[col] = {"type": None, "confidence": 0.0}

    return result


def build_upload_hints(df, fmt):
    """Build auto-detection hints for the upload response.

    Args:
        df: loaded DataFrame
        fmt: detected file format string

    Returns:
        dict with column_types, suggested_e, suggested_f (or None on failure)
    """
    try:
        col_types = infer_column_types(df)
    except Exception:
        logger.debug("Column type inference failed", exc_info=True)
        col_types = {}

    format_default = FORMAT_TYPE_DEFAULTS.get(fmt)

    # Rank columns by confidence, only those with a detected type
    typed_cols = [
        (col, info["type"], info["confidence"])
        for col, info in col_types.items()
        if info["type"] is not None
    ]
    typed_cols.sort(key=lambda x: x[2], reverse=True)

    suggested_e = None
    suggested_f = None

    if format_default:
        # Unambiguous format — use the format default for the primary entity
        e_type = format_default["type"]
        e_col = format_default["column"]
        # Verify column actually exists
        if e_col not in df.columns:
            # Fallback: pick first typed column matching this type
            e_col = next((c for c, t, _ in typed_cols if t == e_type), None)

        if e_col:
            suggested_e = {
                "type": e_type,
                "column": e_col,
                "name": TYPE_DEFAULT_NAMES[e_type],
                "feature_sets": FEAT_DEFAULTS[e_type],
            }

        # For PDB/mmCIF with ligands, suggest molecule as second entity
        if fmt in ("pdb", "pdb_dir", "mmcif", "mmcif_dir") and "ligand_id" in df.columns:
            has_ligand = df.get("has_ligand", pd.Series([False]))
            if has_ligand.any():
                suggested_f = {
                    "type": "molecule",
                    "column": "ligand_id",
                    "name": "ligand",
                    "feature_sets": FEAT_DEFAULTS["molecule"],
                }

        # If no second entity yet, pick best column of a different type
        if suggested_f is None and typed_cols:
            for col, ctype, conf in typed_cols:
                if suggested_e and col == suggested_e["column"]:
                    continue
                if suggested_e and ctype != suggested_e["type"]:
                    suggested_f = {
                        "type": ctype,
                        "column": col,
                        "name": TYPE_DEFAULT_NAMES[ctype],
                        "feature_sets": FEAT_DEFAULTS[ctype],
                    }
                    break
    else:
        # Ambiguous format (CSV, JSON) — assign from inferred column types
        if len(typed_cols) >= 1:
            c1, t1, _ = typed_cols[0]
            suggested_e = {
                "type": t1,
                "column": c1,
                "name": TYPE_DEFAULT_NAMES[t1],
                "feature_sets": FEAT_DEFAULTS[t1],
            }

        if len(typed_cols) >= 2:
            # Prefer a column with a different type
            for c, t, _ in typed_cols[1:]:
                if t != typed_cols[0][1]:
                    suggested_f = {
                        "type": t,
                        "column": c,
                        "name": TYPE_DEFAULT_NAMES[t],
                        "feature_sets": FEAT_DEFAULTS[t],
                    }
                    break

            # Fallback: second highest confidence column even if same type
            if suggested_f is None:
                c2, t2, _ = typed_cols[1]
                name = TYPE_DEFAULT_NAMES[t2]
                if suggested_e and t2 == suggested_e["type"]:
                    name = name + "_2"
                suggested_f = {
                    "type": t2,
                    "column": c2,
                    "name": name,
                    "feature_sets": FEAT_DEFAULTS[t2],
                }

    return {
        "column_types": col_types,
        "format_default": format_default,
        "suggested_e": suggested_e,
        "suggested_f": suggested_f,
    }
