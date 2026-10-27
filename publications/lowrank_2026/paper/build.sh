#!/usr/bin/env bash
set -euo pipefail
paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mlst_python="${MLST_PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
report_flags=()
if [[ "${1:-}" == "--allow-partial" ]]; then
  report_flags=(--allow-incomplete)
fi
"$mlst_python" "$paper_dir/build_tools/systematic_report.py" "${report_flags[@]}"
cd -- "$paper_dir"
mkdir -p logs
for document in manuscript supplement; do
  pdflatex -interaction=nonstopmode -halt-on-error "$document.tex" > "logs/${document}_pass1.log" 2>&1
  bibtex "$document" > "logs/${document}_bibtex.log" 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error "$document.tex" > "logs/${document}_pass2.log" 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error "$document.tex" > "logs/${document}_pass3.log" 2>&1
done
"$mlst_python" "$paper_dir/build_tools/validate_systematic.py" "${report_flags[@]}"
printf "%s\n" "Built and checked $paper_dir/manuscript.pdf and supplement.pdf"
