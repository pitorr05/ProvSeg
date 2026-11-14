#!/usr/bin/env bash
# RUN 2 - our improvements on top of SeqXGPT (same backbone, one change per step):
#   S1  + contrastive per-word evidence (Fast-DetectGPT z, Binoculars x-entropy, entropy, rank; Qwen2.5-1.5B pair)
#   S2  + context encoder (DeBERTa-v3-large)
#   S3  + certified Unknown (Learn-then-Test) -> Cov@0.05 column of the table
#   S4  + evidence dropout + exact per-word Shapley explanations  = ProvSeg
#   outputs : ../runs/provseg/<run>/   ../results/provseg/{table_ladder.md, explain_*.csv, cases_*.html}
#   compare : if RUN 1 exists, ../results/compare/table_ladder.md = SeqXGPT vs S1..S4 on the same test docs
# Usage:  bash run_provseg.sh        (STEPS="feats S1 S2 S4 eval", LIMIT=50 for a quick test)
set -euo pipefail
cd "$(dirname "$0")"
source scripts/_env.sh
STEPS=${STEPS:-"feats S1 S2 S4 eval"}
OBS=${OBS:-Qwen/Qwen2.5-1.5B}
PERF=${PERF:-Qwen/Qwen2.5-1.5B-Instruct}
ENC=${ENC:-microsoft/deberta-v3-large}
RUNS=$RUNS_ROOT/provseg; RES=$RES_ROOT/provseg; mkdir -p $RUNS $RES

if has feats; then
  for ds in ${SETS//,/ }; do
    [ -f $FEATS/$ds.npz ] || python -m provseg.features.token_features --records $DATA/$ds.jsonl \
        --out $FEATS/$ds.npz --observer $OBS --performer $PERF $LIM
  done
fi
for s in $SEEDS; do
  for ds in ${SETS//,/ }; do
    T="--train $ds --seed $s --data $DATA --feats $FEATS --out $RUNS $LIM"
    has S1 && python scripts/train_seq.py --name S1_${ds}_s$s --step S1 --arch seqx --feat_kind contrast $EP_SMALL $T
    has S2 && python scripts/train_seq.py --name S2_${ds}_s$s --step S2 --arch seqx --feat_kind contrast \
        --context $ENC --bs 16 --lr 1e-4 --enc_lr 2e-5 $EP_ENC $T
    has S4 && python scripts/train_seq.py --name S4_${ds}_s$s --step S4 --arch seqx --feat_kind contrast \
        --context $ENC --bs 16 --lr 1e-4 --enc_lr 2e-5 --evidence_dropout 0.15 --shapley $EP_ENC $T
  done
done
if has eval; then
  python scripts/eval_ladder.py --runs "$RUNS/*" --data $DATA --out $RES
  for d in $RUNS/S4_*; do [ -d "$d" ] && python scripts/explain_s4.py --run "$d" --data $DATA --out $RES; done
  if ls $RUNS_ROOT/seqxgpt/S0_* >/dev/null 2>&1; then
    python scripts/eval_ladder.py --runs "$RUNS_ROOT/seqxgpt/*" "$RUNS/*" --data $DATA --out $RES_ROOT/compare
    echo "==> SeqXGPT vs ProvSeg: $RES_ROOT/compare/table_ladder.md"
  fi
  echo "==> ProvSeg table: $RES/table_ladder.md"
fi
