"""Evaluation metrics for word-level provenance.

All functions take per-document integer arrays: gold in {0,1,-1(ignore)},
pred in {0,1} or {0,1,2(Unknown)}.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _flat(golds, preds):
    g = np.concatenate([np.asarray(x) for x in golds])
    p = np.concatenate([np.asarray(x) for x in preds])
    m = g >= 0
    return g[m], p[m]


def word_metrics(golds, preds) -> dict:
    g, p = _flat(golds, preds)
    ans = p != 2
    out = {"coverage": float(ans.mean())}
    if ans.any():
        out["acc"] = float((p[ans] == g[ans]).mean())
        out["macro_f1"] = float(f1_score(g[ans], p[ans], average="macro", labels=[0, 1]))
        out["ai_f1"] = float(f1_score(g[ans], p[ans], pos_label=1, average="binary"))
        out["selective_risk"] = 1 - out["acc"]
    return out


def spans(lab):
    """Maximal runs of label 1 as (start, end) half-open."""
    out, i, lab = [], 0, list(lab)
    while i < len(lab):
        if lab[i] == 1:
            j = i
            while j < len(lab) and lab[j] == 1:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def span_f1(golds, preds, iou: float = 0.5) -> float:
    """AI-span F1 with IoU-matching (greedy, one-to-one)."""
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        gs, ps = spans(np.where(np.asarray(g) == 1, 1, 0)), spans(np.where(np.asarray(p) == 1, 1, 0))
        used = set()
        for a in ps:
            hit = None
            for k, b in enumerate(gs):
                if k in used:
                    continue
                inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
                union = max(a[1], b[1]) - min(a[0], b[0])
                if union and inter / union >= iou:
                    hit = k
                    break
            if hit is None:
                fp += 1
            else:
                used.add(hit)
                tp += 1
        fn += len(gs) - len(used)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def boundary_mae(golds, preds) -> float:
    """SemEval-24 8C metric: |first AI word index (gold) - (pred)| for H->A documents."""
    errs = []
    for g, p in zip(golds, preds):
        g, p = np.asarray(g), np.asarray(p)
        if not (g == 1).any() or not (g == 0).any():
            continue
        gb = int(np.argmax(g == 1))
        pb = int(np.argmax(p == 1)) if (p == 1).any() else len(p)
        errs.append(abs(gb - pb))
    return float(np.mean(errs)) if errs else float("nan")


def ratio_mae(golds, preds) -> float:
    """|gold AI-word ratio - predicted ratio| per document (Unknown counted 0.5)."""
    errs = []
    for g, p in zip(golds, preds):
        g, p = np.asarray(g), np.asarray(p)
        m = g >= 0
        if not m.any():
            continue
        pv = np.where(p[m] == 2, 0.5, p[m])
        errs.append(abs(g[m].mean() - pv.mean()))
    return float(np.mean(errs)) if errs else float("nan")


def risk_coverage(golds, preds, confs, n: int = 50):
    """Curve of (coverage, selective risk) when thresholding word confidence."""
    g, p = _flat(golds, preds)
    c = np.concatenate([np.asarray(x)[: len(y)] for x, y in zip(confs, golds)])
    c = c[np.concatenate([np.asarray(x) >= 0 for x in golds])]
    order = np.argsort(-c)
    err = (p[order] != g[order]).astype(float)
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    idx = np.linspace(0, len(err) - 1, n).astype(int)
    cov = (idx + 1) / len(err)
    aurc = float(cum.mean())
    return cov, cum[idx], aurc


def ece(golds, confs_pos, bins: int = 15) -> float:
    """Expected calibration error of P(AI) at word level."""
    g = np.concatenate([np.asarray(x) for x in golds])
    c = np.concatenate([np.asarray(x)[: len(y)] for x, y in zip(confs_pos, golds)])
    m = g >= 0
    g, c = g[m], c[m]
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (c >= lo) & (c < hi if hi < 1 else c <= hi)
        if s.any():
            tot += s.mean() * abs(c[s].mean() - g[s].mean())
    return float(tot)


def word_auroc(golds, probs) -> float:
    """Word-level AUROC of P(AI) (Binoculars / Fast-DetectGPT style, at word granularity)."""
    from sklearn.metrics import roc_auc_score
    g = np.concatenate([np.asarray(x) for x in golds])
    s = np.concatenate([np.asarray(pr)[: len(x), 1] for x, pr in zip(golds, probs)])
    m = (g >= 0) & np.isfinite(s)
    return float(roc_auc_score(g[m], s[m])) if len(set(g[m].tolist())) == 2 else float("nan")


def all_metrics(golds, preds, probs=None) -> dict:
    m = word_metrics(golds, preds)
    m["span_f1"] = span_f1(golds, preds)
    m["boundary_mae"] = boundary_mae(golds, preds)
    m["ratio_mae"] = ratio_mae(golds, preds)
    if probs is not None:
        hard = [np.where(np.asarray(p) == 2, np.asarray(pr)[: len(p)].argmax(-1), p) for p, pr in zip(preds, probs)]
        _, _, m["aurc"] = risk_coverage(golds, hard, [np.asarray(pr)[: len(p)].max(-1) for p, pr in zip(preds, probs)])
        m["ece"] = ece(golds, [np.asarray(pr)[:, 1] for pr in probs])
        m["auroc"] = word_auroc(golds, probs)
    return m


_END = (".", "!", "?")


def sentence_spans(words):
    """Rule-based sentence split over whitespace words: a sentence ends at a word whose text,
    stripped of closing quotes/brackets, ends with . ! or ?  Returns [(start, end)] half-open."""
    out, s = [], 0
    for i, w in enumerate(words):
        if w.rstrip("\"')]}”’").endswith(_END):
            out.append((s, i + 1))
            s = i + 1
    if s < len(words):
        out.append((s, len(words)))
    return out


def _majority(x):
    v, c = np.unique(x, return_counts=True)
    return int(v[np.argmax(c)])


def sentence_metrics(words_list, golds, preds) -> dict:
    """SeqXGPT's sentence-level protocol: each sentence takes the majority word label (gold and
    predicted separately); report accuracy and macro-F1 over sentences."""
    g_all, p_all = [], []
    for ws, g, p in zip(words_list, golds, preds):
        g, p = np.asarray(g), np.asarray(p)
        for a, b in sentence_spans(ws[: len(g)]):
            m = g[a:b] >= 0
            if not m.any():
                continue
            g_all.append(_majority(g[a:b][m]))
            p_all.append(_majority(p[a:b][m]))
    if not g_all:
        return {"sent_acc": float("nan"), "sent_f1": float("nan")}
    g_all, p_all = np.array(g_all), np.array(p_all)
    return {"sent_acc": float((g_all == p_all).mean()),
            "sent_f1": float(f1_score(g_all, p_all, average="macro", labels=sorted(set(g_all.tolist()))))}


def window_diff(golds, preds, k=None) -> float:
    """WindowDiff (Pevzner & Hearst, 2002) on word labels, averaged over documents (lower is better).
    A boundary is a position where the label changes. k defaults to half the mean gold segment length
    of each document (the convention of the 2026 change-point segmentation work)."""
    vals = []
    for g, p in zip(golds, preds):
        g, p = np.asarray(g), np.asarray(p)
        m = g >= 0
        g, p = g[m], np.where(p[m] == 2, -1, p[m])
        n = len(g)
        if n < 3:
            continue
        bg = np.r_[0, (g[1:] != g[:-1]).astype(int)]
        bp = np.r_[0, (p[1:] != p[:-1]).astype(int)]
        kk = k or max(2, int(round(n / (bg.sum() + 1) / 2)))
        if kk >= n:
            continue
        cg, cp = np.cumsum(bg), np.cumsum(bp)
        dg = cg[kk:] - cg[:-kk]
        dp = cp[kk:] - cp[:-kk]
        vals.append(float((dg != dp).mean()))
    return float(np.mean(vals)) if vals else float("nan")
