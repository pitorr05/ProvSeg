# shared helpers for versions/*/run.sh (sourced)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /data/venv/bin/activate 2>/dev/null || true
need() {  # need <file> <task id in PROMPT.md>: fail fast if a piece of code the version needs is missing
  [ -e "$1" ] || { echo "MISSING $1 -> implement task $2 of versions/$V/PROMPT.md first"; exit 3; }
}
step() { echo; echo "=== [$(date +%T)] $V: $*"; }
# output roots (LIMIT>0 = quick test dirs, same convention as scripts/_env.sh)
if [ "${LIMIT:-0}" -gt 0 ]; then R=../runs_test; RES=../results_test; FD=../data/feats_small_test; VQ=--no_auroc_check; LIMARG="--limit $LIMIT"
else R=../runs; RES=../results; FD=../data/feats_small; VQ=""; LIMARG=""; fi
mkdir -p "$RES"
