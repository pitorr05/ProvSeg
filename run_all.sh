#!/usr/bin/env bash
# ProvSeg pipeline (GPU server). Run from ProvSeg/:  bash run_all.sh
#
# PROFILE=small (default)  first round: 3 benchmarks x 5 baselines + ProvSeg (+records)
#   benchmarks : SemEval-24 8C (H->A boundary), PASTED (AI paraphrase of human text), Beemo (human edits of AI text)
#   baselines  : Fast-DetectGPT, Binoculars, RepreGuard (zero-shot, adapted to words), SeqXGPT, PASTED (trained)
# PROFILE=full             all 7 datasets, ablations, extra baselines (second round)
# STAGES="data syntax feats train eval" to choose stages (data/syntax are already done locally).
set -euo pipefail
PROFILE=${PROFILE:-small}
OBS=${OBS:-Qwen/Qwen2.5-7B}
PERF=${PERF:-Qwen/Qwen2.5-7B-Instruct}
ENC=${ENC:-microsoft/deberta-v3-large}
SEEDS=${SEEDS:-"13"}
STAGES=${STAGES:-"feats train eval"}
LIMIT=${LIMIT:-0}                         # LIMIT=50 -> quick environment test (50 docs/split, 1 epoch)
DATA=../data/processed; FEATS=../data/feats; RUNS=../runs_$PROFILE; RES=../results_$PROFILE
LIM=""; EP=""
if [ "$LIMIT" -gt 0 ]; then
  FEATS=../data/feats_test; RUNS=../runs_test; RES=../results_test; LIM="--limit $LIMIT"; EP="--epochs 1"
  mkdir -p $FEATS && cp -n ../data/feats/*_dep.npz $FEATS/ 2>/dev/null || true
fi
if [ -z "${REP_TRAIN:-}" ]; then
  if [ -f assets/detectrl_train_Google-PaLM.json ]; then REP_TRAIN=assets/detectrl_train_Google-PaLM.json
  else REP_TRAIN=$(ls ../RepreGuard/datasets/detectrl_dataset/original_dataset/detectrl_train_dataset_llm_type_*.json | paste -sd, -); fi
fi
has() { [[ " $STAGES " == *" $1 "* ]]; }

if [ "$PROFILE" = small ]; then
  TRAIN=${TRAIN:-semeval8c,pasted,beemo}
  TEST=${TEST:-semeval8c,pasted,beemo}
  FEAT_SETS=$TEST
else
  TRAIN=${TRAIN:-seqxgpt,semeval8c,beemo,pasted,editlens}
  TEST=${TEST:-seqxgpt,semeval8c,beemo,pasted,pasted_ood,apt_eval,editlens}
  FEAT_SETS=$TEST
fi

has data   && python scripts/build_data.py --hf ../data/hf --refs ../refs --out $DATA
has syntax && python -m provseg.features.syntax_graph --data $DATA --out $FEATS

# per-word LLM evidence (GPU, dominant cost). Train/dev are capped with the same uid-hash
# subset that train.py uses, so every model sees exactly the same documents.
if has feats; then
  for ds in ${FEAT_SETS//,/ }; do
    [ -f $FEATS/$ds.npz ] && continue
    SPL=""; [[ ",$TRAIN," == *",$ds,"* ]] || SPL="--splits cal,test"
    python -m provseg.features.token_features --records $DATA/$ds.jsonl --out $FEATS/$ds.npz $SPL $LIM \
        --observer $OBS --performer $PERF --rep_train "$REP_TRAIN" --rep_n 512 --rep_dir_cache $FEATS/rep_dirs.pt
  done
fi

if has train; then
  for s in $SEEDS; do
    T="--train $TRAIN --eval $TEST --seed $s --out $RUNS --data $DATA --feats $FEATS $LIM $EP"
    python scripts/train.py --name provseg       --encoder $ENC --model efp $T            # ours (+ provseg_s*_records)
    python scripts/train.py --name seqxgpt_feats --encoder none --model blackbox $T       # SeqXGPT (EMNLP'23)
    python scripts/train.py --name pasted_cls    --encoder roberta-large --no_feats --no_crf --model blackbox $T  # PASTED (ACL-F'24)
    if [ "$PROFILE" = full ]; then
      python scripts/train.py --name seqroberta     --encoder roberta-large --no_feats --model blackbox $T
      python scripts/train.py --name abl_blackbox   --encoder $ENC --model blackbox $T
      python scripts/train.py --name abl_additive   --encoder $ENC --model additive $T
      python scripts/train.py --name abl_noedit_sup --encoder $ENC --model efp --no_edit $T
      python scripts/train.py --name abl_nosyntax   --encoder $ENC --model efp --no_syntax $T
      python scripts/train.py --name abl_nofeat     --encoder $ENC --model efp --no_feats $T
    fi
  done
fi

if has eval; then
  python scripts/evaluate.py --runs $RUNS/*_s* --data $DATA --feats $FEATS --train_sets $TRAIN \
      --test_sets $TEST --zs fastdetect,binoculars,repreguard --out $RES
  python -m provseg.eval.explain --run $RUNS/provseg_s* --data $DATA --out $RES
  if [ "$PROFILE" = full ]; then
    python scripts/motivation.py --data $DATA --out $RES --figs paper/figs
    python scripts/plots.py --res $RES --out paper/figs
  fi
  echo "==> main table: $RES/table_main.md"
fi
