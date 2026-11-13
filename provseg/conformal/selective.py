"""Selective provenance with a finite-sample guarantee (Learn-then-Test).

Each word gets a confidence c = max_y p(y | text). For a threshold lam the
tagger answers only words with c >= lam and abstains ("Unknown") otherwise.
Selective risk  R(lam) = P(wrong | answered).

We pick the smallest lam (max coverage) such that R(lam) <= alpha is certified
with prob >= 1 - delta, via Learn-then-Test (Angelopoulos et al., 2021) with a
fixed-sequence binomial test over a decreasing-risk grid. Calibration units are
*documents* (words inside a document are dependent), so the test runs on
document-level sums: we use the Hoeffding-Bentkus p-value on per-document
error fractions weighted by answered words, conservative but valid for
exchangeable documents.

Doc-level span abstention: optionally a span-level variant abstains on whole
maximal runs whose mean confidence < lam (``span_mode=True``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


def _hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: R > alpha given empirical mean rhat over n bounded units."""
    if rhat >= alpha:
        return 1.0
    def h(a, b):
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b)) if 0 < a < 1 else (
            -math.log(1 - b) if a == 0 else -math.log(b))
    hoeff = math.exp(-n * h(max(rhat, 1e-12), alpha))
    bentkus = math.e * binom.cdf(math.ceil(n * rhat), n, alpha)
    return min(hoeff, bentkus, 1.0)


def span_confidence(pred: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Replace each word's confidence by the mean confidence of its predicted run."""
    out = conf.copy()
    i = 0
    while i < len(pred):
        j = i
        while j < len(pred) and pred[j] == pred[i]:
            j += 1
        out[i:j] = conf[i:j].mean()
        i = j
    return out


@dataclass
class SelectiveCalibrator:
    alpha: float = 0.05
    delta: float = 0.1
    span_mode: bool = True
    lam: float = 1.01   # abstain on everything until calibrated

    def _doc_stats(self, preds, confs, golds, lam):
        risks, answered = [], 0
        for p, c, g in zip(preds, confs, golds):
            keep = (c >= lam) & (g >= 0)
            k = keep.sum()
            if k == 0:
                continue
            answered += k
            risks.append(((p[keep] != g[keep]).sum(), k))
        return risks, answered

    def fit(self, preds, probs, golds, grid=None, ambig=None):
        """preds/golds: list of int arrays per doc; probs: list of (T,2) arrays;
        ambig: optional list of per-word ambiguity u >= 0 (confidence becomes p_max / (1 + u))."""
        ambig = ambig if ambig is not None else [None] * len(preds)
        confs = [self.confidence(p, pr, u) for p, pr, u in zip(preds, probs, ambig)]
        grid = np.linspace(0.999, 0.0, 400) if grid is None else grid  # high -> low lam (u can push c below .5)
        best = None
        for lam in grid:   # fixed-sequence testing: stop at first failure
            risks, _ = self._doc_stats(preds, confs, golds, lam)
            if len(risks) < 10:
                continue
            # per-document error fraction is in [0,1]: valid bounded unit for HB
            fr = np.array([e / k for e, k in risks])
            if _hb_pvalue(float(fr.mean()), len(fr), self.alpha) <= self.delta:
                best = lam
            else:
                break
        self.lam = 1.01 if best is None else float(best)
        return self

    def confidence(self, pred, prob, u=None):
        c = np.asarray(prob).max(-1)[: len(pred)]
        if u is not None:
            c = c / (1.0 + np.asarray(u)[: len(pred)])
        return span_confidence(np.asarray(pred), c) if self.span_mode else c

    def apply(self, pred, prob, u=None):
        """Returns labels in {0:H, 1:A, 2:Unknown}."""
        c = self.confidence(pred, prob, u)
        out = np.asarray(pred).copy()
        out[c < self.lam] = 2
        return out
