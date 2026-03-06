#!/usr/bin/env bash
# Run the full PALM test suite using the palm conda environment.
#
# Usage:
#   bash PALM/tests/run_tests.sh           # run all tests
#   bash PALM/tests/run_tests.sh --data    # only prepare data

set -e

PYBIN=/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/palm/bin/python
TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"        # PALM/tests/
PALM_DIR="$(cd "$TESTS_DIR/.." && pwd)"            # PALM root (one level up)
PARENT_DIR="$(cd "$PALM_DIR/.." && pwd)"           # Parent of PALM (for -m invocation)

echo "======================================================"
echo "  Data Splitting Agent — Integration Test Suite"
echo "======================================================"
echo "  PALM dir : $PALM_DIR"
echo "  Python   : $PYBIN"
echo ""

# ── Step 1: prepare datasets and configs ───────────────────────────────────
echo "[1/2] Preparing datasets and configs..."
$PYBIN "$TESTS_DIR/prepare_data.py"

if [[ "$1" == "--data" ]]; then
    echo "Data preparation complete. Exiting (--data flag set)."
    exit 0
fi

# ── Step 2: run the pipeline for each config ──────────────────────────────
echo ""
echo "[2/2] Running pipelines..."

PASS=0
FAIL=0
FAILED_CASES=()

run_case() {
    local label="$1"
    local config="$2"
    echo ""
    echo "------------------------------------------------------"
    echo "  Testing: $label"
    echo "------------------------------------------------------"
    if $PYBIN -m PALM "$config"; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("$label")
    fi
}

cd "$PARENT_DIR"
run_case "molecule + biomolecule (Davis DTI)"          "$TESTS_DIR/configs/molecule_biomolecule.yaml"
run_case "molecule + material (adsorbate-surface)"     "$TESTS_DIR/configs/molecule_material.yaml"
run_case "gene + molecule (gene-drug)"                 "$TESTS_DIR/configs/gene_molecule.yaml"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Results: $PASS passed, $FAIL failed"
if [[ ${#FAILED_CASES[@]} -gt 0 ]]; then
    echo "  Failed cases:"
    for c in "${FAILED_CASES[@]}"; do
        echo "    - $c"
    done
    echo "======================================================"
    exit 1
else
    echo "  All tests passed!"
    echo "======================================================"
fi
