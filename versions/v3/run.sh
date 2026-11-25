#!/usr/bin/env bash
# v3-sota: baselines 2024-2026 on the same test documents (seed 13).
V=v3; source "$(dirname "$0")/../_lib.sh"
need scripts/baseline_markov.py T1
need scripts/baseline_wcp.py T2
step "supervised + zero-shot baselines"; bash run_baselines.sh || exit 10
SETS=${SETS:-seqxgpt,semeval8c,beemo}
for ds in ${SETS//,/ }; do
  step "Markov-informed calibration (ICLR'26) on $ds"
  python scripts/baseline_markov.py --ds $ds --score fdg --t 10 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_markov.py --ds $ds --score bino --t 10 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_markov.py --ds $ds --score fdg --t 0 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_markov.py --ds $ds --score bino --t 0 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  step "WCP change-point (2026) on $ds"
  python scripts/baseline_wcp.py --ds $ds --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 12
done
step "compare"
python scripts/eval_ladder.py --runs "$R/seqxgpt/*" "$R/provseg/*" "$R/baselines/*" \
    --data ../data/processed --out $RES/compare || exit 13
step "verify"
python scripts/verify_run.py $VQ --tag $V --runs "$R/baselines/*" --data ../data/processed --log "${LOGFILE:-}" \
    --ladder $RES/compare/ladder.csv --gate "S4>BEST:+0:2" --out $RES
