#!/usr/bin/env bash
# Run every version back-to-back, unattended (use the branch v6-paper, which contains the code of all versions):
#   cd /data/ProvSeg && git fetch && git checkout v6-paper && git pull
#   tmux new -s provseg   ->   bash scripts/run_all_versions.sh      (Ctrl+B D to detach)
# Each version waits for a free GPU, logs to ../logs/, verifies itself and writes ../results/STATUS_<v>.md.
# A failed go/no-go GATE (verify exit 1) is recorded but does NOT stop the chain: with ~1 month of GPU we
# collect all experiments first and decide afterwards. A CRASH (exit >= 2) stops the chain so it can be fixed.
# Already finished runs are skipped automatically, so re-running this script resumes where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
VERSIONS=${VERSIONS:-"v1 v2 v3 v4 v5 v6"}
mkdir -p ../results
SUMMARY=../results/STATUS_ALL.md
echo "# STATUS_ALL (started $(date +%F' '%T), commit $(git rev-parse --short HEAD))" > $SUMMARY
for v in $VERSIONS; do
  bash scripts/run_unattended.sh $v
  rc=$?
  line="- $v: $([ $rc -eq 0 ] && echo PASS || ([ $rc -eq 1 ] && echo 'GATE FAIL (continuing)' || echo "CRASH exit $rc"))"
  echo "$line" | tee -a $SUMMARY
  if [ $rc -ge 2 ]; then
    echo "- stopped at $v: see ../results/STATUS_$v.md and the log in ../logs/" | tee -a $SUMMARY
    exit $rc
  fi
done
echo "- all done $(date +%F' '%T)" | tee -a $SUMMARY
