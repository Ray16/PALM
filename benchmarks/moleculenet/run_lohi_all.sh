#!/bin/bash
# Lo-Hi Hi-split across MoleculeNet, 4 at a time, 25 min cap each.
# A timeout is a RESULT (the MILP does not scale) -- recorded, not hidden.
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/../results"
cd "$HERE/../../.."                          # directory containing PALM/
PY="${PYTHON:-python}"
for d in freesolv esol sider clintox bace bbbp lipophilicity tox21 qm8 hiv muv; do
  ( timeout 1500 $PY -m PALM.benchmarks.moleculenet.benchmark_lohi --dataset $d \
      > $RES/lohi_$d.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "{\"dataset\":\"$d\",\"status\":\"timeout_or_fail\",\"rc\":$rc}" \
        > $RES/lohi_$d.json
      echo "[$d] FAILED rc=$rc"
    else
      echo "[$d] ok"
    fi ) &
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo "ALL LOHI DONE"
