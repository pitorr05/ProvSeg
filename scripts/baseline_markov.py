"""Baseline: Markov-informed score calibration (Wu et al., ICLR 2026, "Beyond Raw Detection Scores").

Official code: github.com/tmlr-group/MRF_Calibration (commit a21add1, improve_mrf.py, class GCN).
What is taken verbatim from the official implementation:
  * token scores min-max normalised to [0, 1] and turned into a 2-state unary H = [1 - s, s]
  * pairwise matrix W_in = [[-|w1|, |w'1|], [|w'11|, -|w11|]] (4 learned scalars, init N(0, 1e-3))
  * neighbour adjacency with positional weights sigmoid(i -/+ 1 - filter_pos) and self loops
  * mean-field update repeated t times:  Q <- softmax(log(Q + 1e-10) - (A Q) W_in)
  * hyper-parameters of their long-text setting: t = 10, filter_pos = 30; Adam lr 0.05, 10 epochs,
    best epoch on validation
Adaptation to word-level provenance (stated in the paper):
  * the official model aggregates Q over tokens into ONE document score; our task labels every word, so
    the final logistic layer sigmoid(w_final * Q_t + b_final) is applied per word and trained with
    word-level BCE (documents mix human and AI words, so a document label does not exist);
  * per-word scores are the S1 evidence (Fast-DetectGPT z or Binoculars ratio from the same Qwen2.5-0.5B
    pair our model uses), min-max range from train words (1st-99th percentile, then clipped);
  * model selection on dev word macro-F1 (threshold 0.5) instead of document AUROC.
Variants: --t 10 (Markov calibration) and --t 0 (raw per-word score + logistic: the uncalibrated detector).

  python scripts/baseline_markov.py --ds semeval8c --score fdg --data ../data/processed --feats ../data/feats_small
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))
from _baseline_io import dump, gold, load, word_macro_f1  # noqa: E402


def adjacency(L, filter_pos):
    """Official my_dataset.adj_in (both loops), for a sequence of length L."""
    A = np.zeros((L, L), dtype=np.float32)
    if filter_pos > 0:
        for i in range(1, L):
            A[i, i - 1] = 1 / (1 + np.exp(-(i - 1 - filter_pos)))
            A[i, i] = 1
        for i in range(0, L - 1):
            A[i, i + 1] = 1 / (1 + np.exp(-(i + 1 - filter_pos)))
            A[i, i] = 1
    else:
        for i in range(1, L):
            A[i, i - 1] = 1
            A[i, i] = 1
        for i in range(0, L - 1):
            A[i, i + 1] = 1
            A[i, i] = 1
    return torch.tensor(A)


class MRFCalibration(nn.Module):
    """Official GCN mean-field layer, with the document aggregation replaced by a per-word logistic output."""

    def __init__(self, t=10, filter_pos=30):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1) * 0.001)
        self.w11 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_prime1 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_prime11 = nn.Parameter(torch.randn(1) * 0.001)
        self.w_final = nn.Parameter(torch.randn(1) * 0.0001)
        self.b_final = nn.Parameter(torch.randn(1) * 0.)
        self.t, self.filter_pos = t, filter_pos

    def calibrated(self, x, A, mask):
        """x: (B, L, 2) unary; A: (L, L); mask (B, L) -> Q_t (B, L) machine probability after mean field."""
        diag = torch.diagflat(torch.stack([-torch.abs(self.w1), -torch.abs(self.w11)]).squeeze(-1))
        off = torch.fliplr(torch.diagflat(torch.stack([torch.abs(self.w_prime1), torch.abs(self.w_prime11)]).squeeze(-1)))
        w_in = diag + off
        x = x * mask[..., None]
        for _ in range(self.t):
            x2 = torch.matmul(torch.einsum("ij,bjk->bik", A, x), w_in)
            x = torch.softmax(torch.log(x + 1e-10) - x2, dim=-1) * mask[..., None]
        return x[..., 1]

    def forward(self, x, A, mask):
        return torch.sigmoid(self.calibrated(x, A, mask) * self.w_final + self.b_final)


def batches(items, lo, hi, bs, shuffle, rng):
    idx = np.arange(len(items))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [items[j] for j in idx[i:i + bs]]
        L = max(len(s) for _, s in chunk)
        x = np.zeros((len(chunk), L), np.float32)
        m = np.zeros((len(chunk), L), np.float32)
        y = np.full((len(chunk), L), -1, np.int64)
        for k, (r, s) in enumerate(chunk):
            n = len(s)
            x[k, :n] = np.clip((s - lo) / (hi - lo + 1e-10), 0, 1)
            m[k, :n] = 1
            y[k, :n] = gold(r)[:n]
        x = torch.tensor(x)
        yield chunk, torch.stack([1 - x, x], -1), torch.tensor(m), torch.tensor(y)


def predict(model, items, lo, hi, bs, A_cache, filter_pos):
    probs = []
    model.eval()
    with torch.no_grad():
        for chunk, x, m, _ in batches(items, lo, hi, bs, False, None):
            L = x.shape[1]
            if L not in A_cache:
                A_cache[L] = adjacency(L, filter_pos)
            p = model(x, A_cache[L], m).numpy()
            probs += [p[k, :len(s)] for k, (_, s) in enumerate(chunk)]
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--score", choices=["fdg", "bino"], default="fdg")
    ap.add_argument("--t", type=int, default=10, help="mean-field steps (official long-text setting: 10; 0 = raw)")
    ap.add_argument("--filter_pos", type=int, default=30)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--feats", default="../data/feats_small")
    ap.add_argument("--out", default="../runs/baselines")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    name = f"markov{'' if a.t > 0 else '_raw'}_{a.score}_{a.ds}"
    if os.path.exists(os.path.join(a.out, name, f"pred_{a.ds}.pkl")):
        print(f"[{name}] already done -> skip")
        return
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    S = load(a.ds, a.data, a.feats, a.score, a.limit)
    tr = np.concatenate([s for _, s in S["train"]])
    lo, hi = np.percentile(tr, [1, 99])
    filter_pos = a.filter_pos if a.t > 0 else 0
    model = MRFCalibration(a.t, filter_pos)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    A_cache, best, best_state = {}, -1.0, None
    for ep in range(a.epochs):
        model.train()
        for _, x, m, y in batches(S["train"], lo, hi, a.bs, True, rng):
            L = x.shape[1]
            if L not in A_cache:
                A_cache[L] = adjacency(L, filter_pos)
            p = model(x, A_cache[L], m)
            ok = y >= 0
            loss = nn.functional.binary_cross_entropy(p[ok].clamp(1e-6, 1 - 1e-6), y[ok].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        dp = predict(model, S["dev"], lo, hi, a.bs, A_cache, filter_pos)
        f1 = word_macro_f1([gold(r) for r, _ in S["dev"]], [(p >= 0.5).astype(int) for p in dp])
        print(f"[{name}] epoch {ep + 1}: dev word macro-F1 {f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    items = S["dev"] + S["cal"] + S["test"]
    probs = predict(model, items, lo, hi, a.bs, A_cache, filter_pos)
    rd = dump(a.out, name, a.ds, f"markov{'' if a.t > 0 else '_raw'}_{a.score}", items, probs,
              [(p >= 0.5).astype(int) for p in probs], {"t": a.t, "filter_pos": filter_pos, "best_dev_f1": best})
    print(f"[{name}] best dev F1 {best:.4f} -> {rd}")


if __name__ == "__main__":
    main()
