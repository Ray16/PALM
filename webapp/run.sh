#!/usr/bin/env bash
# Launch the Data Splitting web app.
#
# Usage: bash PALM/webapp/run.sh [--gpu] [--dev] [port]      (works from any directory)
#   (default)  CPU only: GPUs are hidden from the server, so it never touches one.
#   --gpu      reserve ONE free GPU on this host through a `gpu_reserve` gate
#              (GPU_RESERVE=/path/to/gpu_reserve if it is not on PATH); it is
#              released when the server exits. Speeds up ESM2 / ChemBERTa / NT
#              embeddings and the k-NN steps.
#   --dev      auto-reload when files under PALM/webapp/ change.
#
# Environment:
#   PALM_CONDA_ENV   conda env to run in (default: palm). Set PALM_CONDA_ENV=""
#                    to use the currently active Python environment instead.
#   WEBAPP_BIND_HOST bind address (default 127.0.0.1, loopback only).

set -euo pipefail

PORT=8080; USE_GPU=0; RELOAD=()
for arg in "$@"; do
  case "$arg" in
    --gpu) USE_GPU=1 ;;
    --dev) RELOAD=(--reload --reload-dir PALM/webapp) ;;
    [0-9]*) PORT="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PALM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(dirname "$PALM_DIR")"            # directory containing the PALM checkout
if [ "$(basename "$PALM_DIR")" != "PALM" ]; then
  echo "The repository folder must be named PALM (it is imported as the PALM package);" >&2
  echo "found: $PALM_DIR. Clone with: git clone https://github.com/Ray16/PALM.git" >&2
  exit 2
fi
cd "$ROOT_DIR"

# Shared nodes: cap math-library threads (t-SNE, sklearn, torch CPU).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

if [ "$USE_GPU" = 1 ]; then
  GPU_RESERVE="${GPU_RESERVE:-gpu_reserve}"
  if ! command -v "$GPU_RESERVE" >/dev/null 2>&1; then
    echo "--gpu needs the gpu_reserve gate (not found: $GPU_RESERVE). Set GPU_RESERVE or run without --gpu." >&2
    exit 1
  fi
  host="$(hostname -s)"; GPU_IDX=""
  for idx in $("$GPU_RESERVE" list --hosts "$host" | awk -v h="$host" '$1==h{print $2}'); do
    if "$GPU_RESERVE" acquire "$idx" --label palm-webapp --pid $$ >/dev/null 2>&1; then
      GPU_IDX="$idx"; break
    fi
  done
  if [ -z "$GPU_IDX" ]; then
    echo "No free GPU on $host (see: $GPU_RESERVE list). Run without --gpu for CPU mode." >&2
    exit 1
  fi
  trap '"$GPU_RESERVE" release "$GPU_IDX" >/dev/null 2>&1' EXIT INT TERM
  export CUDA_VISIBLE_DEVICES="$GPU_IDX"      # reserved above through the gate
  echo "Reserved GPU $GPU_IDX on $host"
else
  export CUDA_VISIBLE_DEVICES=""              # hide all GPUs: CPU-only server
fi

CONDA_ENV="${PALM_CONDA_ENV-palm}"
RUNNER=()
if [ -n "$CONDA_ENV" ]; then
  RUNNER=(conda run --no-capture-output -n "$CONDA_ENV")
fi

echo "Starting Data Splitting web app on http://${WEBAPP_BIND_HOST:-127.0.0.1}:${PORT}"
${RUNNER[@]+"${RUNNER[@]}"} uvicorn PALM.webapp.app:app \
    --host "${WEBAPP_BIND_HOST:-127.0.0.1}" --port "$PORT" ${RELOAD[@]+"${RELOAD[@]}"}
