# Shared settings for run_seqxgpt.sh / run_provseg.sh / run_baselines.sh (sourced, not run).
SETS=${SETS:-seqxgpt,semeval8c,beemo}          # 3 benchmarks, trained and tested separately
SEEDS=${SEEDS:-"13"}
LIMIT=${LIMIT:-0}                              # LIMIT=50 -> quick test: 50 docs/split, 1-2 epochs, *_test dirs
DATA=../data/processed
FEATS=../data/feats; RUNS_ROOT=../runs; RES_ROOT=../results
LIM=""; EP_SMALL="--epochs 20"; EP_ENC="--epochs 3"; EP_RB="--epochs 4"
if [ "$LIMIT" -gt 0 ]; then
  FEATS=../data/feats_test; RUNS_ROOT=../runs_test; RES_ROOT=../results_test
  LIM="--limit $LIMIT"; EP_SMALL="--epochs 2"; EP_ENC="--epochs 1"; EP_RB="--epochs 1"
fi
mkdir -p $FEATS $RUNS_ROOT $RES_ROOT
has() { [[ " $STEPS " == *" $1 "* ]]; }
