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
    m = g >= 0
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
