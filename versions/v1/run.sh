#!/usr/bin/env bash
# v1-baseline: official SeqXGPT (S0, small feature LMs) + S1 contrastive evidence, 3 benchmarks, seed 13.
V=v1; source "$(dirname "$0")/../_lib.sh"
step "run 1: SeqXGPT";  bash run_seqxgpt.sh || exit 10
step "run 2: S1";       STEPS="feats S1 eval" bash run_provseg.sh || exit 11
step "verify"
python scripts/verify_run.py $VQ --tag $V --runs "$R/seqxgpt/S0_*" "$R/provseg/S1_*" \
    --data ../data/processed --log "${LOGFILE:-}" --ladder $RES/compare/ladder.csv --out $RES
