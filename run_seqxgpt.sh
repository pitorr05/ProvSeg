#!/usr/bin/env bash
# RUN 1 - baseline SeqXGPT (Wang et al., EMNLP 2023) with its OFFICIAL model code.
#   features : per-word log-prob from 4 white-box LMs. The paper used GPT-2-xl, GPT-Neo-2.7B, GPT-J-6B, LLaMA-7B
#              (the generators of SeqXGPT-Bench); to save VRAM/disk we use 4 SMALL LMs, the same budget as
#              ProvSeg (set S0_LMS=... to override, e.g. the paper's LMs for an exact reproduction).
#   model    : baselines/seqxgpt/official/seqxgpt_model.py (unchanged), 20 epochs, lr 5e-5
#   outputs  : ../runs/seqxgpt/<run>/   ../results/seqxgpt/table_ladder.md
#   repro    : 6-way SeqXGPT-Bench protocol (matches the paper's numbers only with the paper's LMs)
# Usage:  bash run_seqxgpt.sh        (STEPS="feats train repro eval", LIMIT=50 for a quick test)
set -euo pipefail
cd "$(dirname "$0")"
source scripts/_env.sh
STEPS=${STEPS:-"feats train repro eval"}
S0_LMS=${S0_LMS:-gpt2,EleutherAI/gpt-neo-125m,EleutherAI/pythia-410m,Qwen/Qwen2.5-0.5B}
RUNS=$RUNS_ROOT/seqxgpt; RES=$RES_ROOT/seqxgpt; mkdir -p $RUNS $RES

if has feats; then
  for ds in ${SETS//,/ }; do
    [ -f $FEATS/${ds}_lp4.npz ] || python -m provseg.features.llm_logprob --records $DATA/$ds.jsonl \
        --out $FEATS/${ds}_lp4.npz --lms $S0_LMS $LIM
  done
fi
for s in $SEEDS; do
  for ds in ${SETS//,/ }; do
    has train && python scripts/train_seq.py --name S0_${ds}_s$s --step S0 --arch seqxgpt --feat_kind lp4 \
        --train $ds --seed $s --data $DATA --feats $FEATS --out $RUNS $LIM $EP_SMALL
  done
  if has repro && [[ ",$SETS," == *",seqxgpt,"* ]]; then
    python scripts/train_seq.py --name repro_seqxgpt6_s$s --step repro --arch seqxgpt --feat_kind lp4 --multiclass \
        --train seqxgpt --seed $s --data $DATA --feats $FEATS --out $RUNS $LIM $EP_SMALL
  fi
done
if has eval; then
  python scripts/eval_ladder.py --runs "$RUNS/*" --data $DATA --out $RES
  for f in $RUNS/repro_*/metrics_seqxgpt.json; do [ -f "$f" ] && cp "$f" $RES/repro_6way_$(basename $(dirname $f)).json \
      && echo "== SeqXGPT-Bench 6-way reproduction ($f)" && grep multiclass "$f"; done
  echo "==> SeqXGPT table: $RES/table_ladder.md"
fi
