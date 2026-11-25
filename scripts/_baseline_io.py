"""Shared I/O for score-based baselines (Markov'26, WCP'26): per-word detector scores + prediction dumps.

Per-word scores come from the S1 evidence file data/feats_small/<ds>.npz (token_features.FEAT_NAMES), i.e.
exactly the per-token Fast-DetectGPT / Binoculars statistics our own model sees (same LLM pair, same budget):
  fdg   Fast-DetectGPT analytic per-token z (higher = more machine-like)
  bino  per-token Binoculars ratio  -log p_perf(x_t) / x-entropy_t, negated so that higher = more machine-like
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.data.torch_data import load_records  # noqa: E402
from provseg.features.token_features import FEAT_NAMES  # noqa: E402

SPLITS = ("train", "dev", "cal", "test")


def word_scores(arr, kind):
    a = np.asarray(arr, dtype=np.float64)
    if kind == "fdg":
        s = a[:, FEAT_NAMES.index("fdg_z")]
    elif kind == "bino":
        ll_perf, xent = a[:, FEAT_NAMES.index("ll_perf")], a[:, FEAT_NAMES.index("xent")]
        s = -((-ll_perf) / np.maximum(xent, 1e-6))
    else:
        raise ValueError(kind)
    return np.where(np.isfinite(s), s, np.nan)


def load(ds, data, feats, kind, limit=0, cap=25000, dev_cap=2500):
    """{split: [(record, scores)]} with NaN scores (no left context) imputed by the train median."""
    z = np.load(os.path.join(feats, f"{ds}.npz"))
    caps = {"train": cap, "dev": dev_cap, "cal": 0, "test": 0}
    out = {}
    for sp in SPLITS:
        recs = load_records([os.path.join(data, f"{ds}.jsonl")], {sp}, limit, caps[sp])
        out[sp] = [(r, word_scores(z[r["meta"]["uid"]], kind)) for r in recs if r["meta"]["uid"] in z.files
                   and len(z[r["meta"]["uid"]]) == len(r["words"])]
    med = np.nanmedian(np.concatenate([s for _, s in out["train"]])) if out["train"] else 0.0
    for sp in SPLITS:
        out[sp] = [(r, np.where(np.isnan(s), med, s)) for r, s in out[sp]]
    return out


def dump(out_dir, name, ds, arch, items, probs, preds, extra=None):
    """items: [(record, ...)] over dev/cal/test; probs/preds aligned lists of per-word arrays."""
    d = {"__meta__": {"name": name, "step": "baseline", "arch": arch, "train": ds, "channels": [], **(extra or {})}}
    for (r, *_), p, y in zip(items, probs, preds):
        p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
        d[r["meta"]["uid"]] = {"pred": np.asarray(y, dtype=int), "prob": np.stack([1 - p, p], 1), "split": r["split"]}
    rd = os.path.join(out_dir, name)
    os.makedirs(rd, exist_ok=True)
    pickle.dump(d, open(os.path.join(rd, f"pred_{ds}.pkl"), "wb"))
    return rd


def gold(r):
    return np.asarray(r["labels"])


def word_macro_f1(golds, preds):
    from sklearn.metrics import f1_score
    g = np.concatenate(golds)
    p = np.concatenate(preds)
    m = g >= 0
    return float(f1_score(g[m], p[m], average="macro", labels=[0, 1]))
