#!/usr/bin/env bash
# v2-core: Evidence Reader (S2) + Boundary Curator (S2b) + Auditor (S4: evidence dropout + Shapley), failure diagnosis.
V=v2; source "$(dirname "$0")/../_lib.sh"
need $R/seqxgpt "v1 (run v1 first)"
# runs made before the fp32 fix (2026-11-16) are invalid (NaN): their meta.json has no "boundary_head"
for d in $R/provseg/S2_* $R/provseg/S4_*; do
  [ -d "$d" ] && ! grep -q '"boundary_head"' "$d/meta.json" 2>/dev/null && { echo "removing stale $d"; rm -rf "$d"; }
done
step "S2 S2b S4";      STEPS="feats S2 S2b S4 eval" bash run_provseg.sh || exit 10
step "failure modes";  python scripts/diagnose.py --runs "$R/seqxgpt/S0_*" "$R/provseg/*" \
                           --data ../data/processed --out $RES/diagnose || exit 11
step "verify"
python scripts/verify_run.py $VQ --tag $V --runs "$R/provseg/S2_*" "$R/provseg/S2b_*" "$R/provseg/S4_*" \
    --data ../data/processed --log "${LOGFILE:-}" --ladder $RES/compare/ladder.csv \
    --gate "S2b>S0:+2.0:2" --out $RES
