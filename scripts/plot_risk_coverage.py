"""Risk-coverage curves and the certified Unknown (S3) vs a naive confidence threshold.

For every run and benchmark (test split, words ordered by confidence, span-mode as in the calibrator):
  * the full risk-coverage curve;
  * the Learn-then-Test (Bonferroni) point at alpha = 0.05 and 0.10, calibrated on the cal split;
  * a NAIVE threshold (no guarantee): the smallest confidence threshold whose *empirical* cal error is
    <= alpha. We report how often its test risk exceeds alpha, the failure mode the guarantee prevents.
Outputs: risk_coverage_<ds>.pdf and selective_summary.csv in --out.

  python scripts/plot_risk_coverage.py --runs "../runs/seqxgpt/S0_*" "../runs/provseg/*" "../runs/baselines/*" \
      --data ../data/processed --out ../results/robust
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.conformal.selective import SelectiveCalibrator  # noqa: E402
from provseg.data.torch_data import load_records  # noqa: E402
from provseg.eval.metrics import word_metrics  # noqa: E402
from scripts.eval_ladder import label, load_runs  # noqa: E402


def flat_conf(cal_obj, preds, probs, golds):
    c, e = [], []
    for p, pr, g in zip(preds, probs, golds):
        cc = cal_obj.confidence(p, pr)
        m = g >= 0
        c.append(cc[m])
        e.append((p != g)[m])
    return np.concatenate(c), np.concatenate(e).astype(float)


def naive_threshold(conf, err, alpha):
    order = np.argsort(-conf)
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    ok = np.where(risk <= alpha)[0]
    return float(conf[order][ok.max()]) if len(ok) else 1.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results/robust")
    ap.add_argument("--alphas", default="0.05,0.1")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--min_cal", type=int, default=50, help="skip runs with fewer calibration documents")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for ds, runs in sorted(load_runs(a.runs).items()):
        recs = {r["meta"]["uid"]: r for sp in ("cal", "test") for r in load_records([os.path.join(a.data, f"{ds}.jsonl")], {sp})}
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for meta, d in runs:
            cal = [u for u, v in d.items() if v["split"] == "cal" and u in recs]
            te = [u for u, v in d.items() if v["split"] == "test" and u in recs]
            if len(cal) < a.min_cal or not te:
                continue
            G = lambda us: [np.asarray(recs[u]["labels"]) for u in us]  # noqa: E731
            P = lambda us: [np.asarray(d[u]["pred"]) for u in us]       # noqa: E731
            PR = lambda us: [np.asarray(d[u]["prob"]) for u in us]      # noqa: E731
            base = SelectiveCalibrator(alpha=0.1, delta=a.delta)
            conf, err = flat_conf(base, P(te), PR(te), G(te))
            order = np.argsort(-conf)
            cov = np.arange(1, len(err) + 1) / len(err)
            risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
            line, = ax.plot(cov, risk, lw=1, label=label(meta))
            row = {"dataset": ds, "method": label(meta), "run": meta["name"], "test_error": float(err.mean())}
            ccal, ecal = flat_conf(base, P(cal), PR(cal), G(cal))
            for al in alphas:
                c = SelectiveCalibrator(alpha=al, delta=a.delta).fit(P(cal), PR(cal), G(cal))
                sel = word_metrics(G(te), [c.apply(x, y) for x, y in zip(P(te), PR(te))])
                thr = naive_threshold(ccal, ecal, al)
                ans = conf >= thr
                row.update({f"ltt_cov@{al:g}": sel["coverage"], f"ltt_risk@{al:g}": sel.get("selective_risk", np.nan),
                            f"naive_cov@{al:g}": float(ans.mean()),
                            f"naive_risk@{al:g}": float(err[ans].mean()) if ans.any() else np.nan})
                if sel["coverage"] > 0:
                    ax.scatter([sel["coverage"]], [sel.get("selective_risk", np.nan)], color=line.get_color(), s=14, zorder=3)
            rows.append(row)
        for al in alphas:
            ax.axhline(al, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("coverage (fraction of words answered)")
        ax.set_ylabel("selective risk (word error)")
        ax.set_title(ds, fontsize=9)
        ax.legend(fontsize=5, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"risk_coverage_{ds}.pdf"))
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "selective_summary.csv"), index=False)
    if len(df):
        for al in alphas:
            viol_l = (df[f"ltt_risk@{al:g}"] > al).mean()
            viol_n = (df[f"naive_risk@{al:g}"] > al).mean()
            print(f"alpha={al:g}: test risk > alpha in {100 * viol_l:.0f}% of (run, benchmark) with LTT vs "
                  f"{100 * viol_n:.0f}% with the naive threshold")
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
