"""Faithfulness of ProvSeg's evidence explanations (deletion test), computed offline from shap_<ds>.pkl.

S4 runs store, for every test word: phi (exact Shapley value of each evidence group for the margin
log P(AI) - log P(human)), m_full (margin with all evidence) and m_loo[k] (margin with group k removed).
For each word we compare the margin change when removing
    top  = the group with the largest |phi|,
    rand = a uniformly random group (seeded),
    min  = the group with the smallest |phi|.
A faithful explanation gives |d_top| > |d_rand| > |d_min| and flips more decisions when `top` is removed.
Reported with a document-level bootstrap 95% CI of mean(|d_top| - |d_rand|), plus Spearman(|phi|, |d|).

  python scripts/explain_deletion.py --runs "../runs/provseg/S4_*" --data ../data/processed --out ../results/robust
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def analyse(sh, rng, B=1000):
    players = sh.pop("__players__")
    docs = [v for v in sh.values() if "m_loo" in v]
    if not docs:
        return None
    per_doc, rows_phi, rows_d = [], [], []
    flips = {"top": 0, "rand": 0, "min": 0}
    n_words = 0
    for v in docs:
        phi, mf, loo = np.asarray(v["phi"]), np.asarray(v["m_full"]), np.asarray(v["m_loo"])
        K = phi.shape[1]
        a = np.abs(phi)
        top, low = a.argmax(1), a.argmin(1)
        rnd = rng.integers(0, K, len(mf))
        idx = np.arange(len(mf))
        d = {"top": mf - loo[idx, top], "rand": mf - loo[idx, rnd], "min": mf - loo[idx, low]}
        for k in flips:
            flips[k] += int((np.sign(mf) != np.sign(mf - d[k])).sum())
        n_words += len(mf)
        per_doc.append((np.abs(d["top"]).mean() - np.abs(d["rand"]).mean(), {k: np.abs(x).mean() for k, x in d.items()}))
        rows_phi.append(a.ravel())
        rows_d.append(np.abs(mf[:, None] - loo).ravel())
    diff = np.array([x for x, _ in per_doc])
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rho = spearmanr(np.concatenate(rows_phi), np.concatenate(rows_d)).statistic
    return {"players": len(players), "docs": len(docs), "words": n_words,
            **{f"mean_abs_d_{k}": float(np.mean([m[k] for _, m in per_doc])) for k in flips},
            **{f"flip_rate_{k}": flips[k] / max(n_words, 1) for k in flips},
            "top_minus_rand": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "faithful": bool(lo > 0), "spearman_absphi_absdelta": float(rho)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results/robust")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for pat in a.runs:
        for d in sorted(glob.glob(pat)):
            for p in sorted(glob.glob(os.path.join(d, "shap_*.pkl"))):
                res = analyse(pickle.load(open(p, "rb")), rng)
                if res is None:
                    print(f"[skip] {p}: no m_loo (run made before leave-one-out margins were stored)")
                    continue
                rows.append({"run": os.path.basename(d), "dataset": os.path.basename(p)[5:-4], **res})
    if not rows:
        raise SystemExit("no shap_*.pkl with m_loo found")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "explain_faithfulness.csv"), index=False)
    print(df[["run", "dataset", "mean_abs_d_top", "mean_abs_d_rand", "mean_abs_d_min", "flip_rate_top",
              "flip_rate_rand", "ci_lo", "ci_hi", "faithful", "spearman_absphi_absdelta"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
