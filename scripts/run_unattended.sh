#!/usr/bin/env bash
# Run a version end-to-end without supervision ("chạy treo"):
#   1) wait until the GPU has >= NEED MiB free for STABLE consecutive minutes
#   2) run versions/<v>/run.sh, logging everything to ../logs/<v>_<timestamp>.log
#   3) the version's run.sh ends with scripts/verify_run.py; the PASS/FAIL line is written to
#      ../results/STATUS_<v>.md so you can check it later with one `cat`.
# Usage (inside tmux or with nohup):
#   bash scripts/run_unattended.sh v2
#   NEED=16000 STABLE=10 nohup bash scripts/run_unattended.sh v3 > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
V=${1:?usage: run_unattended.sh <version, e.g. v2>}
NEED=${NEED:-12000}; STABLE=${STABLE:-5}
[ -f versions/$V/run.sh ] || { echo "versions/$V/run.sh not found"; exit 2; }
mkdir -p ../logs ../results
LOG=../logs/${V}_$(date +%Y%m%d_%H%M).log
echo "[$(date +%F' '%T)] $V: waiting for >= ${NEED} MiB free GPU memory, ${STABLE} min stable" | tee -a $LOG
ok=0
while [ $ok -lt $STABLE ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free=${free:-0}
  if [ "$free" -ge "$NEED" ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -lt $STABLE ] && sleep 60
done
echo "[$(date +%F' '%T)] $V: GPU free (${free} MiB) -> start (commit $(git rev-parse --short HEAD), branch $(git rev-parse --abbrev-ref HEAD))" | tee -a $LOG
LOGFILE=$LOG bash versions/$V/run.sh >> $LOG 2>&1
rc=$?
status=$([ $rc -eq 0 ] && echo PASS || echo "FAIL (exit $rc)")
{ echo "# STATUS $V: $status"; echo "- finished: $(date +%F' '%T)"; echo "- commit: $(git rev-parse --short HEAD)";
  echo "- log: $LOG"; [ -f ../results/VERIFY_$V.md ] && cat ../results/VERIFY_$V.md; } > ../results/STATUS_$V.md
echo "[$(date +%F' '%T)] $V: $status -> ../results/STATUS_$V.md" | tee -a $LOG
exit $rc
