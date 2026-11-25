#!/usr/bin/env bash
# v6-paper: regenerate every table and figure of the paper from saved predictions (CPU is enough).
V=v6; source "$(dirname "$0")/../_lib.sh"
need scripts/make_paper_assets.py T1
step "motivation M1/M2";  python scripts/motivation.py --data ../data/processed --out $RES --figs paper/figs || exit 10
step "assets";            python scripts/make_paper_assets.py --results $RES --runs $R --out paper || exit 11
step "verify";            python scripts/verify_run.py $VQ --tag $V --runs "$R/provseg/S4_*" --data ../data/processed --out $RES
