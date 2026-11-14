#!/usr/bin/env bash
# LATER - other baselines, each with its official model / scoring code:
#   Seq-RoBERTa, Sent-RoBERTa (SeqXGPT repo), Fast-DetectGPT [ICLR'24], Binoculars [ICML'24] (sentence level)
#   outputs : ../runs/baselines/<run>/   ../results/baselines/   and an updated ../results/compare/
set -euo pipefail
cd "$(dirname "$0")"
source scripts/_env.sh
STEPS=${STEPS:-"supervised zeroshot eval"}
OBS=${OBS:-Qwen/Qwen2.5-0.5B}
PERF=${PERF:-Qwen/Qwen2.5-0.5B-Instruct}
RB=${RB:-roberta-base}
RUNS=$RUNS_ROOT/baselines; RES=$RES_ROOT/baselines; mkdir -p $RUNS $RES
for s in $SEEDS; do
  for ds in ${SETS//,/ }; do
    T="--train $ds --seed $s --data $DATA --feats $FEATS --out $RUNS $LIM"
    if has supervised; then
      python scripts/train_seq.py --name seqroberta_${ds}_s$s --step baseline --arch seqroberta --encoder $RB --lr 1e-5 --enc_lr 1e-5 $EP_RB $T
      python scripts/train_seq.py --name sentroberta_${ds}_s$s --step baseline --arch sentroberta --encoder $RB --lr 1e-5 --enc_lr 1e-5 $EP_RB $T
    fi
  done
done
if has zeroshot; then
  for ds in ${SETS//,/ }; do
    for m in fastdetect binoculars; do
      python scripts/zeroshot_sent.py --method $m --ds $ds --data $DATA --out $RUNS --observer $OBS --performer $PERF $LIM
    done
  done
fi
if has eval; then
  python scripts/eval_ladder.py --runs "$RUNS/*" --data $DATA --out $RES
  python scripts/eval_ladder.py --runs "$RUNS_ROOT/seqxgpt/*" "$RUNS_ROOT/provseg/*" "$RUNS/*" --data $DATA --out $RES_ROOT/compare
fi
