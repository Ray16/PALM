#!/usr/bin/env bash
# Download + extract the OMol25 4M-train / val / test splits from the public FAIR
# CDN (no HuggingFace gating needed) into the PALM 1D dataset area.
# Resumable: re-running continues partial downloads (wget -c) and skips existing
# extracted dirs. Usage:  bash download_omol25.sh
set -uo pipefail

DEST=/nfs/lambda_stor_01/homes/rzhu/PALM/data/DataSAIL_data/1D/omol25
BASE=https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514
SPLITS=(train_4M val test)

mkdir -p "$DEST"
cd "$DEST"

echo "[$(date +%T)] downloading ${SPLITS[*]} in parallel ..."
pids=()
for f in "${SPLITS[@]}"; do
  wget -c -q "$BASE/${f}.tar.gz" -O "${f}.tar.gz" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then echo "[$(date +%T)] ERROR: a download failed"; exit 1; fi
echo "[$(date +%T)] downloads complete:"; ls -la ./*.tar.gz

for f in "${SPLITS[@]}"; do
  if [ -d "$f" ]; then echo "[$(date +%T)] $f/ already extracted, skipping"; continue; fi
  echo "[$(date +%T)] extracting ${f}.tar.gz ..."
  tar xzf "${f}.tar.gz" || { echo "extract failed: $f"; exit 1; }
done

echo "[$(date +%T)] DONE. contents:"
ls -la "$DEST"
echo "--- *.aselmdb counts ---"
for f in "${SPLITS[@]}"; do
  n=$(find "$DEST" -path "*${f}*" -name '*.aselmdb' 2>/dev/null | wc -l)
  echo "$f: $n aselmdb files"
done
