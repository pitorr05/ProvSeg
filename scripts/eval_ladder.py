"""Results table for the ladder (S0..S4) and the baselines, on the same test documents.

For every benchmark, all runs are scored on the test documents they *all* predicted.
Columns:
  F1w     word-level macro-F1 (H vs AI)
  AUROC   word-level AUROC of P(AI)
  F1s     sentence-level macro-F1 (SeqXGPT protocol: majority label per sentence)
  MAE     boundary MAE in words (SemEval-24 8C official metric; 8C only)
  Cov@a   S3: share of words answered when "Unknown" is calibrated on the cal split so that the
          word error rate among answered words is <= alpha with prob >= 1-delta (Learn-then-Test)
  Risk@a  realised error rate on the answered test words (should be <= alpha)

  python scripts/eval_ladder.py --runs ../runs_ladder/* --data ../data/processed --out ../results_ladder
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.conformal.selective import SelectiveCalibrator  # noqa: E402
from provseg.data.torch_data import load_records  # noqa: E402
from provseg.eval.metrics import (boundary_mae, sentence_metrics, span_f1, window_diff,  # noqa: E402
                                  word_auroc, word_metrics)

ORDER = ["S0", "S1", "S2", "S2b", "S4", "baseline"]
PRETTY = {
    "S0": "S0  SeqXGPT [EMNLP'23] (official model, small feature LMs)",
    "S1": "S1  + contrastive evidence (Fast-DetectGPT/Binoculars per word)",
    "S2": "S2  + context encoder (Evidence Reader)",
    "S2b": "S2b + Boundary Curator",
    "S4": "S4  + evidence dropout = **ProvSeg (ours)**",
    "seqroberta": "Seq-RoBERTa [SeqXGPT repo]",
    "sentroberta": "Sent-RoBERTa [SeqXGPT repo]",
    "fastdetect": "Fast-DetectGPT [ICLR'24] (sentence)",
    "binoculars": "Binoculars [ICML'24] (sentence)",
}


def load_runs(patterns):
    runs = {}
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            for p in glob.glob(os.path.join(d, "pred_*.pkl")):
                ds = os.path.basename(p)[5:-4]
                dump = pickle.load(open(p, "rb"))
                meta = dump.pop("__meta__")
                if meta.get("step") == "repro":      # 6-way SeqXGPT reproduction: see its metrics_*.json
                    continue
                runs.setdefault(ds, []).append((meta, dump))
    return runs


def label(meta):
    if meta.get("step") in PRETTY and meta["step"] != "baseline":
        return PRETTY[meta["step"]]
    return PRETTY.get(meta.get("arch"), meta.get("name"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results_ladder")
    ap.add_argument("--alphas", default="0.05,0.1", help="certified error levels for the Unknown (S3) columns")
    ap.add_argument("--delta", type=float, default=0.1)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for ds, runs in sorted(load_runs(a.runs).items()):
        recs = {r["meta"]["uid"]: r for sp in ("cal", "test") for r in load_records([os.path.join(a.data, f"{ds}.jsonl")], {sp})}
        test = set.intersection(*[{u for u, v in d.items() if v["split"] == "test"} for _, d in runs])
        test = sorted(u for u in test if u in recs)
        print(f"{ds}: {len(runs)} runs, {len(test)} common test docs")
        for meta, d in runs:
            g = [np.asarray(recs[u]["labels"]) for u in test]
            p = [np.asarray(d[u]["pred"]) for u in test]
            pr = [np.asarray(d[u]["prob"]) for u in test]
            wm = word_metrics(g, p)
            row = {"dataset": ds, "step": meta.get("step") or "baseline", "method": label(meta), "run": meta["name"],
                   "n_docs": len(test), "F1w": wm["macro_f1"], "AUROC": word_auroc(g, pr),
                   "F1s": sentence_metrics([recs[u]["words"] for u in test], g, p)["sent_f1"]}
            row["SpanF1"] = span_f1(g, p)
            row["WinDiff"] = window_diff(g, p)
            if ds == "semeval8c":
                row["MAE"] = boundary_mae(g, p)
            cal = [u for u, v in d.items() if v["split"] == "cal" and u in recs]
            if len(cal) >= 50:
                for al in alphas:
                    c = SelectiveCalibrator(alpha=al, delta=a.delta).fit(
                        [np.asarray(d[u]["pred"]) for u in cal], [np.asarray(d[u]["prob"]) for u in cal],
                        [np.asarray(recs[u]["labels"]) for u in cal])
                    sel = word_metrics(g, [c.apply(x, y) for x, y in zip(p, pr)])
                    row[f"Cov@{al:g}"] = sel["coverage"]
                    row[f"Risk@{al:g}"] = sel.get("selective_risk", float("nan"))
            rows.append(row)
    if not rows:
        print("no runs found for", a.runs)
        return
    df = pd.DataFrame(rows)
    df["_o"] = df["step"].map(lambda s: ORDER.index(s) if s in ORDER else len(ORDER))
    df = df.sort_values(["dataset", "_o", "method"]).drop(columns="_o")
    df.to_csv(os.path.join(a.out, "ladder.csv"), index=False)
    lines = ["# ProvSeg ladder (test split; same documents for every row of a benchmark)", ""]
    for ds, g in df.groupby("dataset", sort=False):
        sel_cols = [c for al in alphas for c in (f"Cov@{al:g}", f"Risk@{al:g}")]
        cols = [c for c in ["F1w", "AUROC", "F1s", "SpanF1", "WinDiff", "MAE"] + sel_cols if c in g and g[c].notna().any()]
        lines += [f"## {ds}  (n={int(g['n_docs'].iloc[0])} docs)", "",
                  "| method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for _, r in g.iterrows():
            vals = [("%.2f" % r[c]) if c in ("MAE", "WinDiff") else ("%.1f" % (100 * r[c])) if pd.notna(r[c]) else "-" for c in cols]
            lines.append(f"| {r['method']} | " + " | ".join(vals) + " |")
        lines.append("")
    lines += [f"Cov@a: % words answered with certified error <= a (prob >= 1-{a.delta}); Risk@a: realised error on "
              "answered words. This is step S3 (+ certified Unknown), applied on top of each row's probabilities.", ""]
    open(os.path.join(a.out, "table_ladder.md"), "w", encoding="utf8").write("\n".join(lines))
    print("\n".join(lines))
    json.dump({"alphas": alphas, "delta": a.delta}, open(os.path.join(a.out, "eval_config.json"), "w"))


if __name__ == "__main__":
    main()
