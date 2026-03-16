"""Integration tests for all supported data formats.

Tests format detection, loading, and format-specific split output.
"""

import os
import sys
import tempfile
import shutil

# Ensure PALM package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PALM.loaders import _detect_format, load_data, build_upload_hints
from PALM.config import EntityConfig, SplittingConfig, PipelineConfig
from PALM.pipeline import run_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def test_format_detection():
    """Test that _detect_format correctly identifies each format."""
    print("\n=== Format Detection ===")
    check("SDF file", _detect_format(os.path.join(DATA_DIR, "molecules.sdf")) == "sdf")
    check("SDF directory", _detect_format(os.path.join(DATA_DIR, "sdf_dir")) == "sdf_dir")
    check("CIF directory (material)", _detect_format(os.path.join(DATA_DIR, "cif_dir")) == "cif_dir")
    check("FASTA file", _detect_format(os.path.join(DATA_DIR, "sequences.fasta")) == "fasta")
    check("SMILES file", _detect_format(os.path.join(DATA_DIR, "molecules.smi")) == "smiles")
    check("CSV file", _detect_format(os.path.join(DATA_DIR, "davis", "interactions.csv")) == "csv")


def test_loading():
    """Test that each format loads correctly into a DataFrame."""
    print("\n=== Data Loading ===")

    # SDF
    df = load_data(os.path.join(DATA_DIR, "molecules.sdf"))
    check("SDF loads", len(df) > 0, f"got {len(df)} rows")
    check("SDF has smiles column", "smiles" in df.columns)
    check("SDF has _row_id", "_row_id" in df.columns)
    check("SDF row count", len(df) == 20, f"expected 20, got {len(df)}")

    # SDF directory
    df = load_data(os.path.join(DATA_DIR, "sdf_dir"))
    check("SDF dir loads", len(df) > 0, f"got {len(df)} rows")
    check("SDF dir has smiles column", "smiles" in df.columns)
    check("SDF dir has _source_file", "_source_file" in df.columns)
    check("SDF dir row count", len(df) == 20, f"expected 20, got {len(df)}")

    # CIF directory
    df = load_data(os.path.join(DATA_DIR, "cif_dir"))
    check("CIF dir loads", len(df) > 0, f"got {len(df)} rows")
    check("CIF dir has formula column", "formula" in df.columns)
    check("CIF dir row count", len(df) == 20, f"expected 20, got {len(df)}")

    # FASTA
    df = load_data(os.path.join(DATA_DIR, "sequences.fasta"))
    check("FASTA loads", len(df) > 0, f"got {len(df)} rows")
    check("FASTA has sequence column", "sequence" in df.columns)
    check("FASTA row count", len(df) == 20, f"expected 20, got {len(df)}")

    # SMILES
    df = load_data(os.path.join(DATA_DIR, "molecules.smi"))
    check("SMILES loads", len(df) > 0, f"got {len(df)} rows")
    check("SMILES has smiles column", "smiles" in df.columns)
    check("SMILES row count", len(df) == 20, f"expected 20, got {len(df)}")


def test_upload_hints():
    """Test that upload hints are generated correctly."""
    print("\n=== Upload Hints ===")

    # SDF
    df = load_data(os.path.join(DATA_DIR, "molecules.sdf"))
    hints = build_upload_hints(df, "sdf")
    check("SDF hints: type=molecule", hints["suggested_e"]["type"] == "molecule")
    check("SDF hints: column=smiles", hints["suggested_e"]["column"] == "smiles")

    # CIF dir
    df = load_data(os.path.join(DATA_DIR, "cif_dir"))
    hints = build_upload_hints(df, "cif_dir")
    check("CIF dir hints: type=material", hints["suggested_e"]["type"] == "material")

    # FASTA
    df = load_data(os.path.join(DATA_DIR, "sequences.fasta"))
    hints = build_upload_hints(df, "fasta")
    check("FASTA hints: type=biomolecule", hints["suggested_e"]["type"] == "biomolecule")

    # SMILES
    df = load_data(os.path.join(DATA_DIR, "molecules.smi"))
    hints = build_upload_hints(df, "smiles")
    check("SMILES hints: type=molecule", hints["suggested_e"]["type"] == "molecule")

    # SDF dir
    df = load_data(os.path.join(DATA_DIR, "sdf_dir"))
    hints = build_upload_hints(df, "sdf_dir")
    check("SDF dir hints: type=molecule", hints["suggested_e"]["type"] == "molecule")


def _run_pipeline_test(name, input_file, entity_type, entity_col, feature_sets):
    """Helper to run a 1D pipeline test and check outputs."""
    out_dir = tempfile.mkdtemp(prefix=f"palm_test_{name}_")
    try:
        config = PipelineConfig(
            input_file=input_file,
            output_dir=out_dir,
            dataset_name=name,
            e1=EntityConfig(
                name="entity",
                type=entity_type,
                extract_column=entity_col,
                feature_sets=feature_sets,
            ),
            e2=None,
            splitting=SplittingConfig(
                techniques=["R"],
                splits=[8, 2],
                names=["train", "test"],
            ),
        )
        run_pipeline(config)

        # Check that split results exist
        split_dir = os.path.join(out_dir, "split_result")
        check(f"{name}: split_result dir exists", os.path.isdir(split_dir))

        # Check for technique-specific output directory
        technique_dir = os.path.join(split_dir, f"R_{name}")
        check(f"{name}: technique dir exists", os.path.isdir(technique_dir))

        # List output files
        if os.path.isdir(technique_dir):
            files = []
            for root, dirs, fnames in os.walk(technique_dir):
                for f in fnames:
                    files.append(os.path.relpath(os.path.join(root, f), technique_dir))
            check(f"{name}: has output files", len(files) > 0, f"files: {files}")
            return files
        return []
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_pipeline_sdf():
    """Test SDF pipeline produces .sdf output files."""
    print("\n=== Pipeline: SDF ===")
    files = _run_pipeline_test(
        "sdf_test",
        os.path.join(DATA_DIR, "molecules.sdf"),
        "molecule", "smiles",
        ["rdkit_descriptors"],
    )
    check("SDF output: train.sdf exists", "train.sdf" in files, f"got: {files}")
    check("SDF output: test.sdf exists", "test.sdf" in files, f"got: {files}")


def test_pipeline_sdf_dir():
    """Test SDF directory pipeline produces train/test folders."""
    print("\n=== Pipeline: SDF Directory ===")
    files = _run_pipeline_test(
        "sdf_dir_test",
        os.path.join(DATA_DIR, "sdf_dir"),
        "molecule", "smiles",
        ["rdkit_descriptors"],
    )
    train_files = [f for f in files if f.startswith("train/")]
    test_files = [f for f in files if f.startswith("test/")]
    check("SDF dir output: has train/ files", len(train_files) > 0, f"got: {train_files}")
    check("SDF dir output: has test/ files", len(test_files) > 0, f"got: {test_files}")


def test_pipeline_cif_dir():
    """Test CIF directory pipeline produces train/test folders."""
    print("\n=== Pipeline: CIF Directory ===")
    files = _run_pipeline_test(
        "cif_dir_test",
        os.path.join(DATA_DIR, "cif_dir"),
        "material", "formula",
        ["magpie_properties"],
    )
    train_files = [f for f in files if f.startswith("train/")]
    test_files = [f for f in files if f.startswith("test/")]
    check("CIF dir output: has train/ files", len(train_files) > 0, f"got: {train_files}")
    check("CIF dir output: has test/ files", len(test_files) > 0, f"got: {test_files}")


def test_pipeline_fasta():
    """Test FASTA pipeline produces .fasta output files."""
    print("\n=== Pipeline: FASTA ===")
    files = _run_pipeline_test(
        "fasta_test",
        os.path.join(DATA_DIR, "sequences.fasta"),
        "biomolecule", "sequence",
        ["sequence_properties"],
    )
    check("FASTA output: train.fasta exists", "train.fasta" in files, f"got: {files}")
    check("FASTA output: test.fasta exists", "test.fasta" in files, f"got: {files}")


def test_pipeline_smiles():
    """Test SMILES pipeline produces .smi output files."""
    print("\n=== Pipeline: SMILES ===")
    files = _run_pipeline_test(
        "smi_test",
        os.path.join(DATA_DIR, "molecules.smi"),
        "molecule", "smiles",
        ["rdkit_descriptors"],
    )
    check("SMILES output: train.smi exists", "train.smi" in files, f"got: {files}")
    check("SMILES output: test.smi exists", "test.smi" in files, f"got: {files}")


if __name__ == "__main__":
    test_format_detection()
    test_loading()
    test_upload_hints()
    test_pipeline_sdf()
    test_pipeline_sdf_dir()
    test_pipeline_cif_dir()
    test_pipeline_fasta()
    test_pipeline_smiles()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)
