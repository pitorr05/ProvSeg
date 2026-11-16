"""Failure-mode diagnosis of word-level detectors (motivation from our own results).

Measured on saved predictions (pred_<ds>.pkl, test split), for each run:
  boundary blur      word error rate as a function of the distance to the nearest gold
                     human<->AI boundary (0, 1-2, 3-5, 6-10, 11-20, >20 words)
  forced commitment  word error rate per confidence decile, and the share of all errors that
                     sit in the 20% least-confident words (what a calibrated "Unknown" can remove)
  fragmentation      predicted AI segments per document / gold AI segments per document
                     (>1: the tagger shatters segments, a symptom of per-word score noise)

  python scripts/diagnose.py --runs "../runs/seqxgpt/S0_*" "../runs/provseg/S1_*" \
      --data ../data/processed --out ../results/diagnose
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.data.torch_data import load_records  # noqa: E402
from provseg.eval.metrics import spans  # noqa: E402

BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
BIN_NAMES = ["0", "1-2", "3-5", "6-10", "11-20", ">20"]


def boundary_distance(g):
    """Distance (in words) of every word to the nearest position where the gold label changes."""
    g = np.asarray(g)
    ch = np.where((g[1:] != g[:-1]) & (g[1:] >= 0) & (g[:-1] >= 0))[0] + 1
    if len(ch) == 0:
        return np.full(len(g), 10 ** 9)
    idx = np.arange(len(g))
    # a change at position c sits between words c-1 and c: both are at distance 0
    d = np.minimum(np.abs(idx[:, None] - ch[None, :]), np.abs(idx[:, None] - (ch[None, :] - 1))).min(1)
    return d


def diagnose(golds, preds, probs):
    err, dist, conf = [], [], []
    frag = []
    for g, p, pr in zip(golds, preds, probs):
        g, p, pr = np.asarray(g), np.asarray(p), np.asarray(pr)
        m = g >= 0
        err.append((p != g)[m])
        dist.append(boundary_distance(g)[m])
        conf.append(pr.max(-1)[: len(g)][m])
        ng = len(spans(np.where(g == 1, 1, 0)))
        if ng:
            frag.append(len(spans(np.where(p == 1, 1, 0))) / ng)
    err, dist, conf = np.concatenate(err), np.concatenate(dist), np.concatenate(conf)
    out = {"word_error": float(err.mean()), "fragmentation": float(np.mean(frag)) if frag else float("nan")}
    for (lo, hi), name in zip(BINS, BIN_NAMES):
        s = (dist >= lo) & (dist <= hi)
        out[f"err_dist_{name}"] = float(err[s].mean()) if s.any() else float("nan")
        out[f"share_words_dist_{name}"] = float(s.mean())
    q = np.quantile(conf, np.linspace(0, 1, 11))
    for k in range(10):
        s = (conf >= q[k]) & (conf <= q[k + 1])
        out[f"err_conf_dec{k + 1}"] = float(err[s].mean()) if s.any() else float("nan")
    low = conf <= np.quantile(conf, 0.2)
    out["share_errors_in_low20_conf"] = float(err[low].sum() / max(err.sum(), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results/diagnose")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    cache = {}
    for pat in a.runs:
        for d in sorted(glob.glob(pat)):
            for p in glob.glob(os.path.join(d, "pred_*.pkl")):
                ds = os.path.basename(p)[5:-4]
                dump = pickle.load(open(p, "rb"))
                meta = dump.pop("__meta__")
                if meta.get("step") == "repro":
                    continue
                if ds not in cache:
                    cache[ds] = {r["meta"]["uid"]: r for r in load_records([os.path.join(a.data, f"{ds}.jsonl")], {"test"})}
                recs = cache[ds]
                uids = [u for u, v in dump.items() if v["split"] == "test" and u in recs]
                res = diagnose([recs[u]["labels"] for u in uids], [dump[u]["pred"] for u in uids],
                               [dump[u]["prob"] for u in uids])
                rows.append({"dataset": ds, "step": meta.get("step"), "run": meta["name"], "n_docs": len(uids), **res})
    if not rows:
        raise SystemExit(f"no predictions found in {a.runs}")
    df = pd.DataFrame(rows).sort_values(["dataset", "step"])
    df.to_csv(os.path.join(a.out, "failure_modes.csv"), index=False)
    show = ["dataset", "step", "word_error", "fragmentation", "share_errors_in_low20_conf"] + \
           [f"err_dist_{n}" for n in BIN_NAMES]
    print(df[show].round(3).to_string(index=False))

    sets = list(df["dataset"].unique())
    fig, axes = plt.subplots(2, len(sets), figsize=(3.2 * len(sets), 5.0), squeeze=False)
    for j, ds in enumerate(sets):
        for _, r in df[df.dataset == ds].iterrows():
            axes[0, j].plot(range(len(BIN_NAMES)), [r[f"err_dist_{n}"] for n in BIN_NAMES], "o-", label=r["step"])
            axes[1, j].plot(range(1, 11), [r[f"err_conf_dec{k}"] for k in range(1, 11)], "o-", label=r["step"])
        axes[0, j].set_xticks(range(len(BIN_NAMES)), BIN_NAMES, fontsize=7)
        axes[0, j].set_title(ds, fontsize=9)
        axes[0, j].set_xlabel("distance to gold boundary (words)", fontsize=7)
        axes[1, j].set_xlabel("confidence decile (1 = least confident)", fontsize=7)
    axes[0, 0].set_ylabel("word error rate")
    axes[1, 0].set_ylabel("word error rate")
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figF1_failure_modes.pdf"))
    print(f"==> {a.out}/failure_modes.csv, figF1_failure_modes.pdf")


if __name__ == "__main__":
    main()
