"""Automatic verification of a version run: sanity checks + go/no-go gate. Exit code 0 = PASS.

Checks (every run directory under --runs):
  1. meta.json and at least one pred_<ds>.pkl exist
  2. probabilities are finite and not constant (a dead/NaN model gives constant 0.5)
  3. word AUROC on test is not ~0.5 (|AUROC-0.5| > 0.02) for trained models
  4. the run log (if --log is given) has no "aborting"/"Traceback" and skipped batches < 1%
Gate (optional, from ladder.csv written by eval_ladder.py):
  --gate "S2b>S0:+2.0:2"   method S2b must beat S0 by >= 2.0 word-F1 points on >= 2 benchmarks
  --gate "S4>BEST:+0:2"    S4 must be >= the strongest baseline (step == baseline) on >= 2 benchmarks
Several --gate options may be given; all must hold. Writes <out>/VERIFY_<tag>.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.data.torch_data import load_records  # noqa: E402
from provseg.eval.metrics import word_auroc  # noqa: E402


def check_run(d, data, auroc_check=True):
    issues = []
    preds = glob.glob(os.path.join(d, "pred_*.pkl"))
    if not preds:
        return issues + ["no pred_*.pkl"]
    for p in preds:
        ds = os.path.basename(p)[5:-4]
        dump = pickle.load(open(p, "rb"))
        meta = dump.pop("__meta__", None)
        if meta is None:
            issues.append(f"{ds}: prediction dump has no __meta__")
            continue
        probs = np.concatenate([v["prob"][:, 1] for v in dump.values()])
        if not np.isfinite(probs).all():
            issues.append(f"{ds}: non-finite probabilities")
        if np.nanstd(probs) < 1e-4:
            issues.append(f"{ds}: constant probabilities (dead model)")
        if auroc_check and meta.get("step") not in ("repro",) and meta.get("arch") not in ("fastdetect", "binoculars"):
            recs = {r["meta"]["uid"]: r for r in load_records([os.path.join(data, f"{ds}.jsonl")], {"test"})}
            us = [u for u, v in dump.items() if v["split"] == "test" and u in recs]
            if us:
                auc = word_auroc([np.asarray(recs[u]["labels"]) for u in us], [dump[u]["prob"] for u in us])
                if np.isfinite(auc) and abs(auc - 0.5) <= 0.02:
                    issues.append(f"{ds}: test AUROC {auc:.3f} ~ chance")
    return issues


def check_log(path):
    if not path or not os.path.exists(path):
        return []
    txt = open(path, encoding="utf8", errors="ignore").read()
    issues = []
    for pat in ("aborting", "Traceback", "out of memory"):
        if pat in txt:
            issues.append(f"log contains '{pat}'")
    sk = [(int(a), int(b)) for a, b in re.findall(r"skipped_batches (\d+)/(\d+)", txt)]
    if sk and sum(a for a, _ in sk) > 0.01 * max(sum(b for _, b in sk), 1):
        issues.append(f"skipped batches {sum(a for a, _ in sk)}/{sum(b for _, b in sk)} > 1%")
    return issues


def check_gate(gate, ladder):
    m = re.fullmatch(r"(\w+)>(\w+):\+?([\d.]+):(\d+)", gate)
    if not m:
        return [f"bad gate spec '{gate}'"]
    a, b, gain, k = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
    df = pd.read_csv(ladder)
    agg = df[df.step != "baseline"].groupby(["dataset", "step"])["F1w"].mean().unstack()
    base = df[df.step == "baseline"].groupby(["dataset", "method"])["F1w"].mean()
    if not base.empty:
        agg["BEST"] = base.groupby(level=0).max()       # strongest baseline per benchmark
    if a not in agg or b not in agg:
        return [f"gate {gate}: method missing in {ladder}"]
    d = 100 * (agg[a] - agg[b])
    k = min(k, len(d))                        # fewer benchmarks (quick tests): require all of them
    ok = int((d >= gain).sum())
    detail = ", ".join(f"{ds}: {v:+.2f}" for ds, v in d.items())
    return [] if ok >= k else [f"gate {gate} FAILED ({ok}/{len(d)} benchmarks; {detail})"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--log", default="")
    ap.add_argument("--ladder", default="")
    ap.add_argument("--gate", action="append", default=[])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no_auroc_check", action="store_true", help="quick tests: tiny models are near chance")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    report, failed = [f"# VERIFY {a.tag}", ""], False
    dirs = sorted(d for pat in a.runs for d in glob.glob(pat) if os.path.isdir(d))
    if not dirs:
        report.append("- FAIL: no run directories found")
        failed = True
    for d in dirs:
        iss = check_run(d, a.data, not a.no_auroc_check)
        failed |= bool(iss)
        report.append(f"- {'FAIL' if iss else 'ok  '} {os.path.basename(d)}" + (f": {'; '.join(iss)}" if iss else ""))
    for iss in check_log(a.log):
        failed = True
        report.append(f"- FAIL log: {iss}")
    for g in a.gate:
        iss = check_gate(g, a.ladder) if a.ladder and os.path.exists(a.ladder) else [f"gate {g}: no ladder.csv"]
        failed |= bool(iss)
        report.append(f"- {'FAIL' if iss else 'PASS'} gate {g}" + (f": {iss[0]}" if iss else ""))
    report += ["", f"**RESULT: {'FAIL' if failed else 'PASS'}**"]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"VERIFY_{a.tag}.md"), "w", encoding="utf8").write("\n".join(report) + "\n")
    print("\n".join(report))
    json.dump({"tag": a.tag, "pass": not failed}, open(os.path.join(a.out, f"VERIFY_{a.tag}.json"), "w"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
