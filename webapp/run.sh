#!/usr/bin/env bash
# Launch the Data Splitting Agent web server using the palm conda environment.
# Usage: bash PALM/webapp/run.sh [port]

set -e

PORT=${1:-8080}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PALM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(cd "$PALM_DIR/.." && pwd)"

cd "$PARENT_DIR"

echo "Starting Data Splitting Agent on http://localhost:${PORT}"
exec conda run -n palm uvicorn PALM.webapp.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --reload
