#!/usr/bin/env bash
# Regenerate the entire master benchmark on one GPU (one-job-per-GPU rule).
#
#   benchmarks/master/run_all.sh <gpu_id> [--limit N] [--seeds "0 1 2 3 4"]
#
# Produces:
#   results/master_benchmark.csv     registry: split quality + generalization gap
#   results/chemixhub_quality.csv    mixture suite: split quality (from committed report)
#   results/figures/*.png            summary figures
#   master/INSIGHTS.md               written interpretation
set -euo pipefail

GPU="${1:?usage: run_all.sh <gpu_id> [extra args to run_benchmark]}"; shift || true
PY=/homes/rzhu/miniforge3/envs/palm/bin/python
export LD_LIBRARY_PATH=/homes/rzhu/miniforge3/envs/palm/lib
cd "$(dirname "$0")/../.."/..              # -> PALM parent (so `python -m PALM...` resolves)

echo "== [1/3] registry sweep (GPU $GPU) =="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m PALM.benchmarks.master.run_benchmark \
    --seeds 0 1 2 --limit 10000 "$@"

echo "== [2/3] fold in CheMixHub split-quality =="
"$PY" -m PALM.benchmarks.master.chemixhub_ingest

echo "== [3/3] figures + INSIGHTS.md =="
"$PY" -m PALM.benchmarks.master.analyze

echo "== done: see PALM/benchmarks/results/ and PALM/benchmarks/master/INSIGHTS.md =="
