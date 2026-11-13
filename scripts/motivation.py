"""Motivation experiments (CPU, existing data only).

M1  "A detector score is not a contribution measure".
    EditLens ships document scores of Binoculars, Fast-DetectGPT, EditLens and
    Pangram. We compare them with the *true* fraction of AI-written words
    (retained provenance via diff) on human / AI-edited / AI-generated docs:
      - Spearman rho(score, AI ratio) overall and within AI-edited docs
      - for documents in the same score decile, the spread (IQR, 10-90 range)
        of their true AI ratio  -> one score maps to many contributions.
M2  "Contribution is undefined without a stated policy".
    On version-paired data (APT-Eval by polish degree, Beemo by edit ratio)
    we compare the retained-word policy with an equally defensible
    sentence policy (a sentence the editor touched belongs to the editor) and
    report word-label disagreement and the gap in document AI ratio.
    (A first attempt, context-limited Bayes error from recurring +-3-word
    windows, gave eps~0.002 everywhere: contexts almost never recur across
    documents, so that estimator is uninformative and was dropped.)

Usage: python scripts/motivation.py --data ../data/processed --out ../results --figs paper/figs
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCORES = {"binoculars_score": "Binoculars", "fastdetectgpt_score": "Fast-DetectGPT",
          "editlens_roberta-large_score": "EditLens (RoBERTa)", "pangram_v3.2_score": "Pangram"}
APT_ORDER = ["none", "extreme_minor", "minor", "slight_major", "major"]
BEEMO_ORDER = ["<= 0.10", "<= 0.25", "<= 0.50", "<= 0.75", "> 0.75"]


def read(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f]


def ai_ratio(r):
    lab = [x for x in r["labels"] if x >= 0]
    return sum(lab) / max(len(lab), 1)


# ---------------------------------------------------------------- M1
def m1(data_dir, out, figs):
    recs = read(os.path.join(data_dir, "editlens.jsonl"))
    rows = []
    for r in recs:
        row = {"type": r["edit_level"], "ai_ratio": ai_ratio(r)}
        row.update(r["meta"].get("doc_scores", {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Binoculars: lower = more AI -> flip so that higher = more AI for every score
    if "binoculars_score" in df:
        df["binoculars_score"] = -df["binoculars_score"]
    res = []
    for c, name in SCORES.items():
        if c not in df:
            continue
        d = df.dropna(subset=[c])
        ed = d[d["type"] == "ai_edited"]
        rho_all = spearmanr(d[c], d["ai_ratio"]).statistic
        rho_ed = spearmanr(ed[c], ed["ai_ratio"]).statistic
        # spread of true ratio inside score deciles (AI-edited docs)
        q = pd.qcut(ed[c].rank(method="first"), 10, labels=False)
        spreads = ed.groupby(q)["ai_ratio"].agg(lambda s: np.subtract(*np.percentile(s, [90, 10])))
        res.append({"score": name, "n": len(d), "spearman_all": rho_all, "spearman_ai_edited": rho_ed,
                    "mean_p10_p90_ratio_range_within_decile": float(spreads.mean())})
    pd.DataFrame(res).to_csv(os.path.join(out, "motivation_m1.csv"), index=False)
    print(pd.DataFrame(res).round(3).to_string(index=False))

    cols = [c for c in SCORES if c in df]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.6 * len(cols), 2.5), squeeze=False)
    colors = {"human_written": "#1f77b4", "ai_edited": "#ff7f0e", "ai_generated": "#d62728"}
    for ax, c in zip(axes[0], cols):
        for t, g in df.dropna(subset=[c]).groupby("type"):
            ax.scatter(g[c], g["ai_ratio"], s=2, alpha=0.25, color=colors.get(t, "k"), label=t.replace("_", " "))
        ax.set_title(SCORES[c], fontsize=9)
        ax.set_xlabel("detector score" + (" (negated)" if c == "binoculars_score" else ""))
    axes[0, 0].set_ylabel("true AI-word ratio")
    axes[0, -1].legend(fontsize=6, markerscale=4, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "figM1_score_vs_ratio.pdf"))
    plt.close(fig)
    return res


# ---------------------------------------------------------------- M2
def sentence_policy(words, labels, editor):
    """Alternative, equally defensible policy: a sentence touched by the editor belongs to the editor."""
    out, s0, n = list(labels), 0, len(words)
    for i in range(n):
        if words[i].endswith((".", "!", "?")) or i == n - 1:
            if any(l == editor for l in labels[s0:i + 1]):
                out[s0:i + 1] = [editor] * (i + 1 - s0)
            s0 = i + 1
    return out


def m2(data_dir, out, figs, alpha=0.05, k=3):
    """How much does 'AI contribution' depend on the labelling policy (i.e. on history semantics)?"""
    rows = []
    for ds, order, editor in (("apt_eval", APT_ORDER, 1), ("beemo", BEEMO_ORDER, 0)):
        p = os.path.join(data_dir, f"{ds}.jsonl")
        if not os.path.exists(p):
            continue
        recs = [r for r in read(p) if r["versions"]]
        for lv in order:
            pool = [r for r in recs if r["edit_level"] == lv]
            if not pool:
                continue
            dis, gap = [], []
            for r in pool:
                alt = sentence_policy(r["words"], r["labels"], editor)
                lab = np.array(r["labels"]); alt = np.array(alt); m = lab >= 0
                dis.append(float((lab[m] != alt[m]).mean()))
                gap.append(float(abs(lab[m].mean() - alt[m].mean())))
            rows.append({"dataset": ds, "level": lv, "n_docs": len(pool),
                         "ai_ratio_word_policy": float(np.mean([ai_ratio(r) for r in pool])),
                         "word_disagreement": float(np.mean(dis)),
                         "ai_ratio_gap_mean": float(np.mean(gap)),
                         "ai_ratio_gap_p90": float(np.percentile(gap, 90))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "motivation_m2.csv"), index=False)
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, (ds, order) in zip(axes, (("apt_eval", APT_ORDER), ("beemo", BEEMO_ORDER))):
        d = df[df.dataset == ds].set_index("level").reindex(order).dropna(subset=["n_docs"])
        x = list(range(len(d)))
        ax.plot(x, d["word_disagreement"], "o-", label="words whose label flips")
        ax.plot(x, d["ai_ratio_gap_mean"], "s--", label="|Δ AI ratio| (mean)")
        ax.plot(x, d["ai_ratio_gap_p90"], "^:", label="|Δ AI ratio| (p90)")
        ax.set_xticks(x, [s.replace("_", " ") for s in d.index], rotation=30, ha="right", fontsize=7)
        ax.set_title({"apt_eval": "APT-Eval (polish degree)", "beemo": "Beemo (human edit ratio)"}[ds], fontsize=9)
    axes[0].set_ylabel("word vs. sentence policy")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "figM2_policy.pdf"))
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--figs", default="paper/figs")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.figs, exist_ok=True)
    print("== M1: detector score vs. true AI contribution (EditLens)")
    m1(a.data, a.out, a.figs)
    print("\n== M2: AI ratio under word vs. sentence attribution policy")
    m2(a.data, a.out, a.figs, a.alpha, a.k)


if __name__ == "__main__":
    main()
