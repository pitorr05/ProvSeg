#!/usr/bin/env bash
# v5-robust: cross-domain, paraphrase OOD, explanation faithfulness, risk-coverage.
V=v5; source "$(dirname "$0")/../_lib.sh"
need scripts/explain_deletion.py T2
need scripts/plot_risk_coverage.py T3
SETS=${SETS:-seqxgpt,semeval8c,beemo}
ENC=${ENC:-microsoft/deberta-v3-base}
S4CFG="--step S4 --arch seqx --feat_kind contrast --context $ENC --bf16 --boundary_head --evidence_dropout 0.15 --bs 16 --lr 1e-4 --enc_lr 2e-5 --epochs 3"
D="--data ../data/processed --feats $FD --out $R/robust"
step "cross-domain (train A -> test B), seed 13"
for A in ${SETS//,/ }; do for B in ${SETS//,/ }; do
  [ "$A" = "$B" ] && continue
  python scripts/train_seq.py --name xd_${A}_to_${B} $S4CFG --train $A --eval $B $D || exit 10
done; done
step "paraphrase OOD (PASTED-OOD)"
[ -f $FD/pasted_ood.npz ] || python -m provseg.features.token_features \
    --records ../data/processed/pasted_ood.jsonl --out $FD/pasted_ood.npz \
    --observer Qwen/Qwen2.5-0.5B --performer Qwen/Qwen2.5-0.5B-Instruct || exit 11
python scripts/train_seq.py --name ood_pasted $S4CFG --train pasted --eval pasted,pasted_ood $D || exit 12
step "explanation faithfulness"
python scripts/explain_deletion.py --runs "$R/provseg/S4_*" --data ../data/processed --out $RES/robust || exit 13
step "risk-coverage"
python scripts/plot_risk_coverage.py --runs "$R/seqxgpt/S0_*" "$R/provseg/*" "$R/baselines/*" \
    --data ../data/processed --out $RES/robust || exit 14
step "robust table"
python scripts/eval_ladder.py --runs "$R/robust/*" --data ../data/processed --out $RES/robust || exit 15
step "verify"
python scripts/verify_run.py $VQ --tag $V --runs "$R/robust/*" --data ../data/processed --log "${LOGFILE:-}" --out $RES
