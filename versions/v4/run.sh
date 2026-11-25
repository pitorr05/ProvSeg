#!/usr/bin/env bash
# v4-main: 3 seeds for every method, 5 benchmarks, paired bootstrap significance -> main table.
V=v4; source "$(dirname "$0")/../_lib.sh"
need scripts/baseline_markov.py "v3/T1"
need scripts/baseline_wcp.py "v3/T2"
need provseg/data/loaders/coauthor.py T1
export SEEDS="13 21 42"
export SETS=${SETS:-seqxgpt,semeval8c,beemo,pasted,coauthor}
step "build new benchmarks"
python scripts/build_data.py --hf ../data/hf --refs ../refs --out ../data/processed --only pasted,coauthor || exit 9
step "SeqXGPT x3 seeds";    STEPS="feats train eval" bash run_seqxgpt.sh || exit 10
step "ProvSeg x3 seeds";    STEPS="feats S1 S2 S2b S4 eval" bash run_provseg.sh || exit 11
step "baselines x3 seeds";  bash run_baselines.sh || exit 12
for ds in ${SETS//,/ }; do
  python scripts/baseline_markov.py --ds $ds --score fdg --t 10 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_markov.py --ds $ds --score bino --t 10 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_markov.py --ds $ds --score fdg --t 0 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_markov.py --ds $ds --score bino --t 0 --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 11
  python scripts/baseline_wcp.py --ds $ds --data ../data/processed --feats $FD --out $R/baselines $LIMARG || exit 14
done
step "compare + significance"
python scripts/eval_ladder.py --runs "$R/seqxgpt/*" "$R/provseg/*" "$R/baselines/*" \
    --data ../data/processed --out $RES/compare || exit 15
python scripts/significance.py --runs "$R/seqxgpt/*" "$R/provseg/*" "$R/baselines/*" \
    --data ../data/processed --target S4 --out $RES/significance || exit 16
step "verify"
python scripts/verify_run.py $VQ --tag $V --runs "$R/provseg/*_s21" "$R/provseg/*_s42" --data ../data/processed \
    --log "${LOGFILE:-}" --ladder $RES/compare/ladder.csv --gate "S4>BEST:+0:3" --out $RES
