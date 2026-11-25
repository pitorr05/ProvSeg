"""Baseline: weighted change-point segmentation (WCP) of human-LLM co-authored text
(Li, Zhu, Li, Shi, arXiv 2605.03723, STAI-X 2026). Official code: github.com/Mamba413/DetectLLMSegmentation
(commit 60fe32e; scripts/detector_value_cp.py + scripts/utils_cp.py).

Pipeline (the repository's recommended configuration: --cp_method DP --cp_selection aic --weight_type invar):
  1. sentence statistic = mean of the per-word Fast-DetectGPT z over the sentence (our S1 evidence, same
     Qwen2.5-0.5B pair; the official code scores each sentence with its own detector call);
  2. weights w_i = (1 / v_i) * n_i^power, normalised to sum to the number of sentences (official invar
     weights; v_i = variance of the sentence statistic, estimated as the within-sentence variance of the
     word z divided by n_i, floored);
  3. weighted L2 dynamic-programming segmentation, number of change points chosen by AIC with the official
     _get_information_criterion (copied verbatim below). The official code calls a weighted fork of
     `ruptures.Dynp(model="l2", min_size=1, jump=2)`; that fork needs a C build, so the same exact DP is
     implemented here in numpy with jump=1 (every sentence boundary is a candidate);
  4. segment score = standardised sum of the word z in the segment (official NFT segment score), every
     sentence takes its segment's score, every word its sentence's score;
  5. label: the official code thresholds either by cross-validation on labelled documents or per document
     by 1-D 2-means clustering; we choose ONE global threshold on the dev split (labelled, like every other
     supervised row of our table) maximising word macro-F1, and map scores to P(AI) by a logistic fitted on dev.

  python scripts/baseline_wcp.py --ds semeval8c --data ../data/processed --feats ../data/feats_small
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _baseline_io import dump, gold, load, word_macro_f1  # noqa: E402
from provseg.eval.metrics import sentence_spans  # noqa: E402


def _get_information_criterion(rss, n_samples, n_bkps, criterion, r, eps=1e-12):
    """Compute an information criterion from the weighted residual sum of squares. (verbatim, utils_cp.py)"""
    criterion = criterion.lower()
    n_segments = n_bkps + 1
    p = 2 * n_segments
    rss_safe = max(float(rss) / max(int(n_samples), 1), eps)
    base = n_samples * np.log(rss_safe)
    if criterion == "aic":
        return base + r * 2 * p
    if criterion == "bic":
        return base + r * p * np.log(n_samples)
    if criterion == "aicc":
        aic = base + r * 2 * p
        denom = n_samples - p - 1
        if denom <= 0:
            return np.inf
        return aic + (2 * p * (p + 1)) / denom
    raise ValueError(f"Unsupported information criterion: {criterion}")


def weighted_dp(y, w, max_bkps):
    """Exact weighted-L2 segmentation for every k <= max_bkps. Returns {k: (rss, breakpoints)}."""
    n = len(y)
    W = np.r_[0, np.cumsum(w)]
    WY = np.r_[0, np.cumsum(w * y)]
    WY2 = np.r_[0, np.cumsum(w * y * y)]

    def cost(i, j):                       # segment y[i:j]
        sw = W[j] - W[i]
        return 0.0 if sw <= 0 else (WY2[j] - WY2[i]) - (WY[j] - WY[i]) ** 2 / sw

    C = np.array([[cost(i, j) if j > i else np.inf for j in range(n + 1)] for i in range(n + 1)])
    D = np.full((max_bkps + 1, n + 1), np.inf)
    arg = np.zeros((max_bkps + 1, n + 1), dtype=int)
    D[0, 1:] = C[0, 1:]
    for k in range(1, max_bkps + 1):
        for j in range(k + 1, n + 1):
            cand = D[k - 1, k:j] + C[k:j, j]
            b = int(np.argmin(cand))
            D[k, j], arg[k, j] = cand[b], b + k
    out = {}
    for k in range(max_bkps + 1):
        if not np.isfinite(D[k, n]):
            continue
        bk, j = [], n
        for kk in range(k, 0, -1):
            j = arg[kk, j]
            bk.append(j)
        out[k] = (float(D[k, n]), sorted(bk))
    return out


def segment_doc(z, words, power, criterion="aic", r=1.0):
    """Word z-scores of one document -> per-word WCP score."""
    sents = sentence_spans(words)
    stat = np.array([z[a:b].mean() for a, b in sents])
    n_i = np.array([b - a for a, b in sents], dtype=float)
    v = np.array([max(np.var(z[a:b]), 0.05) / (b - a) for a, b in sents])
    w = (1.0 / v) * n_i ** power
    w = len(w) * w / w.sum()
    ns = len(sents)
    if ns <= 2:
        bk = [1] if ns == 2 else []
    else:
        res = weighted_dp(stat, w, max(1, ns // 2))
        bk = min(res.items(), key=lambda kv: (_get_information_criterion(kv[1][0], ns, kv[0], criterion, r), kv[0]))[1][1]
    out = np.zeros(len(z))
    bounds = [0] + bk + [ns]
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        a, b = sents[s0][0], sents[s1 - 1][1]
        out[a:b] = z[a:b].sum() / np.sqrt(b - a)        # standardised segment statistic
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--feats", default="../data/feats_small")
    ap.add_argument("--out", default="../runs/baselines")
    ap.add_argument("--power", type=float, default=2.0, help="official README recommends 2.0 with invar weights")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    name = f"wcp_{a.ds}"
    if os.path.exists(os.path.join(a.out, name, f"pred_{a.ds}.pkl")):
        print(f"[{name}] already done -> skip")
        return
    S = load(a.ds, a.data, a.feats, "fdg", a.limit)
    tr = np.concatenate([s for _, s in S["train"]])
    mu, sd = tr.mean(), tr.std() + 1e-8                    # z-scale with train statistics
    score = {sp: [segment_doc((s - mu) / sd, r["words"], a.power) for r, s in S[sp]] for sp in ("dev", "cal", "test")}
    dg = [gold(r) for r, _ in S["dev"]]
    allv = np.concatenate(score["dev"])
    grid = np.unique(np.percentile(allv, np.linspace(1, 99, 99)))
    f1s = [word_macro_f1(dg, [(s > t).astype(int) for s in score["dev"]]) for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    g = np.concatenate(dg)
    m = g >= 0
    lr = LogisticRegression().fit((allv[m] - thr)[:, None], g[m]) if len(set(g[m].tolist())) == 2 else None
    items = S["dev"] + S["cal"] + S["test"]
    sc = score["dev"] + score["cal"] + score["test"]
    probs = [lr.predict_proba((s - thr)[:, None])[:, 1] if lr is not None else (s > thr).astype(float) for s in sc]
    rd = dump(a.out, name, a.ds, "wcp", items, probs, [(s > thr).astype(int) for s in sc],
              {"power": a.power, "threshold": thr, "best_dev_f1": float(max(f1s))})
    print(f"[{name}] dev F1 {max(f1s):.4f} (thr {thr:.3f}) -> {rd}")


if __name__ == "__main__":
    main()
