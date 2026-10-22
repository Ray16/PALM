#!/usr/bin/env bash
set -euo pipefail
paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mlst_python="${MLST_PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
report_flags=()
if [[ "${1:-}" == "--allow-partial" ]]; then
  report_flags=(--allow-partial)
fi
"$mlst_python" "$paper_dir/build_tools/revision_numbers.py"
"$mlst_python" "$paper_dir/build_tools/dataset_report.py"
"$mlst_python" "$paper_dir/build_tools/metric_support_report.py"
"$mlst_python" "$paper_dir/build_tools/robustness_report.py"
"$mlst_python" "$paper_dir/build_tools/scaling_report.py"
"$mlst_python" "$paper_dir/build_tools/local_baselines_report.py"
"$mlst_python" "$paper_dir/build_tools/sensitivity_report.py"
"$mlst_python" "$paper_dir/build_tools/calibration_report.py"
"$mlst_python" "$paper_dir/build_tools/ablation_report.py"
"$mlst_python" "$paper_dir/build_tools/datasail_report.py"
"$mlst_python" "$paper_dir/build_tools/benchmark_report.py" "${report_flags[@]}"
"$mlst_python" "$paper_dir/build_tools/runtime_report.py" "${report_flags[@]}"
"$mlst_python" "$paper_dir/build_tools/revision_figures.py"
cd -- "$paper_dir"
mkdir -p logs
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > logs/latex_pass1.log 2>&1
bibtex manuscript > logs/bibtex.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > logs/latex_pass2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > logs/latex_pass3.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > logs/supplement_pass1.log 2>&1
bibtex supplement > logs/supplement_bibtex.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > logs/supplement_pass2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > logs/supplement_pass3.log 2>&1
if [[ "${1:-}" != "--allow-partial" ]]; then
  "$mlst_python" "$paper_dir/build_tools/validate_revision.py"
fi
printf '%s\n' "Built $paper_dir/manuscript.pdf"
