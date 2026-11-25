"""Compute cost table from run logs: peak VRAM, training minutes, inference time per document.

Parses the lines printed by train_seq.py / feature extractors inside ../logs/*.log:
  "[<run>] arch=... train=N ..."              (run start)
  "epoch k: ... (<minutes> min)"              (cumulative minutes at the end of each epoch)
  "[<run>] peak VRAM <x> GB"
Writes cost.csv (one row per run) to --out.

  python scripts/parse_cost.py --logs "../logs/*.log" --out ../results/robust
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd


def parse(text):
    rows, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\[(\S+)\] arch=(\S+) .*train=(\d+)", line)
        if m:
            cur = m.group(1)
            rows[cur] = {"run": cur, "arch": m.group(2), "train_docs": int(m.group(3))}
            continue
        m = re.search(r"epoch (\d+): .*\(([\d.]+) min\)", line)
        if m and cur:
            rows[cur]["epochs"] = int(m.group(1))
            rows[cur]["train_minutes"] = float(m.group(2))
        m = re.match(r"\[(\S+)\] peak VRAM ([\d.]+) GB", line)
        if m and m.group(1) in rows:
            rows[m.group(1)]["peak_vram_gb"] = float(m.group(2))
        m = re.match(r"(\S+): peak VRAM ([\d.]+) GB", line)           # feature extractors
        if m and not line.startswith("["):
            rows[f"feat:{m.group(1)}"] = {"run": f"features:{m.group(1)}", "arch": "feature-LM",
                                          "peak_vram_gb": float(m.group(2))}
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="../logs/*.log")
    ap.add_argument("--out", default="../results/robust")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(a.logs)):
        rows += [{**r, "log": os.path.basename(f)} for r in parse(open(f, encoding="utf8", errors="ignore").read())]
    if not rows:
        raise SystemExit(f"no parsable logs in {a.logs}")
    df = pd.DataFrame(rows).drop_duplicates("run", keep="last")
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "cost.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
