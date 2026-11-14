#!/usr/bin/env bash
# ProvSeg ladder on the GPU server. Run from ProvSeg/:   bash scripts/run_ladder.sh
#
# Base paper: SeqXGPT (EMNLP 2023), official code in baselines/seqxgpt/official. Each step adds ONE thing:
#   S0  SeqXGPT, official model + its 4-LM log-prob features      (reproduction / main baseline)
#   S1  + contrastive per-word evidence (Fast-DetectGPT z, Binoculars x-ent, entropy, rank; Qwen2.5-1.5B pair)
#   S2  + context encoder (DeBERTa-v3-large word states)
#   S3  + certified Unknown (Learn-then-Test), applied in eval to every row -> Cov@0.05 column
#   S4  + evidence dropout + exact per-word Shapley explanation = ProvSeg
# Baselines: Seq-RoBERTa, Sent-RoBERTa (official SeqXGPT repo), Fast-DetectGPT, Binoculars (official scoring code).
# Benchmarks (in-domain train/test per benchmark): SeqXGPT-Bench, SemEval-24 8C, Beemo.
#
#   STEPS="feats S0 S1 S2 S4 baselines repro eval"   (default: all)
#   LIMIT=50  -> quick test (50 docs per split, 1-2 epochs, separate *_test dirs)
set -euo pipefail
cd "$(dirname "$0")/.."
SETS=${SETS:-seqxgpt,semeval8c,beemo}
STEPS=${STEPS:-"feats S0 S1 S2 S4 baselines repro eval"}
SEEDS=${SEEDS:-"13"}
S0_LMS=${S0_LMS:-gpt2-xl,EleutherAI/gpt-neo-2.7B,EleutherAI/gpt-j-6b,huggyllama/llama-7b}
OBS=${OBS:-Qwen/Qwen2.5-1.5B}
PERF=${PERF:-Qwen/Qwen2.5-1.5B-Instruct}
ENC=${ENC:-microsoft/deberta-v3-large}
RB=${RB:-roberta-base}
LIMIT=${LIMIT:-0}
DATA=../data/processed; FEATS=../data/feats; RUNS=../runs_ladder; RES=../results_ladder
LIM=""; EP_SMALL="--epochs 20"; EP_ENC="--epochs 3"; EP_RB="--epochs 4"
if [ "$LIMIT" -gt 0 ]; then
  FEATS=../data/feats_ladder_test; RUNS=../runs_ladder_test; RES=../results_ladder_test
  LIM="--limit $LIMIT"; EP_SMALL="--epochs 2"; EP_ENC="--epochs 1"; EP_RB="--epochs 1"
fi
mkdir -p $FEATS $RUNS $RES
has() { [[ " $STEPS " == *" $1 "* ]]; }
COMMON="--data $DATA --feats $FEATS --out $RUNS $LIM"

if has feats; then
  for ds in ${SETS//,/ }; do
    [ -f $FEATS/${ds}_lp4.npz ] || python -m provseg.features.llm_logprob --records $DATA/$ds.jsonl \
        --out $FEATS/${ds}_lp4.npz --lms $S0_LMS $LIM
    [ -f $FEATS/$ds.npz ] || python -m provseg.features.token_features --records $DATA/$ds.jsonl \
        --out $FEATS/$ds.npz --observer $OBS --performer $PERF $LIM
  done
fi

for s in $SEEDS; do
  for ds in ${SETS//,/ }; do
    T="--train $ds --seed $s $COMMON"
    has S0 && python scripts/train_seq.py --name S0_${ds}_s$s --step S0 --arch seqxgpt --feat_kind lp4 $EP_SMALL $T
    has S1 && python scripts/train_seq.py --name S1_${ds}_s$s --step S1 --arch seqx --feat_kind contrast $EP_SMALL $T
    has S2 && python scripts/train_seq.py --name S2_${ds}_s$s --step S2 --arch seqx --feat_kind contrast \
        --context $ENC --bs 16 --lr 1e-4 --enc_lr 2e-5 $EP_ENC $T
    has S4 && python scripts/train_seq.py --name S4_${ds}_s$s --step S4 --arch seqx --feat_kind contrast \
        --context $ENC --bs 16 --lr 1e-4 --enc_lr 2e-5 --evidence_dropout 0.15 --shapley $EP_ENC $T
    if has baselines; then
      python scripts/train_seq.py --name seqroberta_${ds}_s$s --step baseline --arch seqroberta --encoder $RB \
          --lr 1e-5 --enc_lr 1e-5 $EP_RB $T
      python scripts/train_seq.py --name sentroberta_${ds}_s$s --step baseline --arch sentroberta --encoder $RB \
          --lr 1e-5 --enc_lr 1e-5 $EP_RB $T
      if [ "$s" = "$(echo $SEEDS | cut -d' ' -f1)" ]; then   # zero-shot: deterministic, one run
        python scripts/zeroshot_sent.py --method fastdetect --ds $ds --data $DATA --out $RUNS --observer $OBS --performer $PERF $LIM
        python scripts/zeroshot_sent.py --method binoculars --ds $ds --data $DATA --out $RUNS --observer $OBS --performer $PERF $LIM
      fi
    fi
  done
  # SeqXGPT-Bench 6-way reproduction (compare multiclass_sent_f1 with the SeqXGPT paper)
  if has repro && [[ ",$SETS," == *",seqxgpt,"* ]]; then
    python scripts/train_seq.py --name repro_seqxgpt6_s$s --step repro --arch seqxgpt --feat_kind lp4 --multiclass \
        --train seqxgpt --seed $s $COMMON $EP_SMALL
  fi
done

if has eval; then
  python scripts/eval_ladder.py --runs "$RUNS/*" --data $DATA --out $RES
  for d in $RUNS/S4_*; do [ -d "$d" ] && python scripts/explain_s4.py --run "$d" --data $DATA --out $RES; done
  for f in $RUNS/repro_*/metrics_seqxgpt.json; do [ -f "$f" ] && echo "== reproduction $f" && cat "$f"; done
  echo "==> table: $RES/table_ladder.md"
fi
