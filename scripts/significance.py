"""Seed aggregation + paired bootstrap significance (the evidence behind every "better than" claim).

For each benchmark:
  * groups runs by method (step for our ladder, arch for baselines) across seeds -> mean +- std
    of word macro-F1 on the common test documents;
  * paired bootstrap over test DOCUMENTS (words inside a document are dependent): resample documents
    with replacement B times, recompute word macro-F1 of the target method and of every other method on
    the same resample; report the 95% CI of the difference (seed-averaged predictions are compared
    seed-by-seed and the CI is taken over seeds x resamples).
A claim "target > other" is allowed only when the whole CI is above 0.

  python scripts/significance.py --runs "../runs/seqxgpt/*" "../runs/provseg/*" "../runs/baselines/*" \
      --data ../data/processed --target S4 --out ../results/significance
Outputs: significance.csv, table_main.md (mean+-std, * = CI of (target - row) excludes 0 in target's favour).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.data.torch_data import load_records  # noqa: E402
from scripts.eval_ladder import label, load_runs  # noqa: E402


def method_key(meta):
    st = meta.get("step") or "baseline"
    return st if st != "baseline" else meta.get("arch", meta.get("name"))


def doc_counts(g, p):
    """Per-document confusion counts (tp, fp, fn for both classes) so macro-F1 of any resample is cheap."""
    m = g >= 0
    g, p = g[m], p[m]
    return np.array([((p == 1) & (g == 1)).sum(), ((p == 1) & (g == 0)).sum(), ((p == 0) & (g == 1)).sum(),
                     ((p == 0) & (g == 0)).sum()], dtype=np.float64)


def macro_f1_from_counts(c):
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    f_ai = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f_h = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    return (f_ai + f_h) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--target", default="S4", help="method whose superiority is tested (step name or arch)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default="../results/significance")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, lines = [], ["# Main table: word macro-F1 (mean ± std over seeds); * = 95% bootstrap CI of "
                       f"({a.target} − method) > 0", ""]
    for ds, runs in sorted(load_runs(a.runs).items()):
        recs = {r["meta"]["uid"]: r for r in load_records([os.path.join(a.data, f"{ds}.jsonl")], {"test"})}
        test = sorted(set.intersection(*[{u for u, v in d.items() if v["split"] == "test"} for _, d in runs]) & set(recs))
        gold = [np.asarray(recs[u]["labels"]) for u in test]
        by = defaultdict(list)          # method -> list of (meta, per-doc counts (D, 4))
        for meta, d in runs:
            C = np.stack([doc_counts(g, np.asarray(d[u]["pred"])) for g, u in zip(gold, test)])
            by[method_key(meta)].append((meta, C))
        idx = rng.integers(0, len(test), size=(a.B, len(test)))
        boot = {k: np.stack([macro_f1_from_counts(C[idx].sum(1)) for _, C in v]) for k, v in by.items()}  # (S, B)
        lines += [f"## {ds} (n={len(test)} test docs)", "", "| method | seeds | F1w mean ± std | Δ vs target [95% CI] |",
                  "|---|---:|---:|---:|"]
        tgt = boot.get(a.target)
        for k, v in sorted(by.items(), key=lambda kv: -np.mean([macro_f1_from_counts(C.sum(0)) for _, C in kv[1]])):
            f1s = np.array([macro_f1_from_counts(C.sum(0)) for _, C in v])
            row = {"dataset": ds, "method": k, "label": label(v[0][0]), "seeds": len(v),
                   "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(ddof=1)) if len(v) > 1 else 0.0}
            cell = "—"
            if tgt is not None and k != a.target:
                n = min(len(tgt), len(boot[k]))
                diff = (tgt[:n] - boot[k][:n]).ravel()
                lo, hi = np.percentile(diff, [2.5, 97.5])
                row.update(delta_lo=float(lo), delta_hi=float(hi), target_better=bool(lo > 0))
                cell = f"{100 * diff.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]" + (" *" if lo > 0 else "")
            rows.append(row)
            lines.append(f"| {row['label']} | {len(v)} | {100 * row['f1_mean']:.2f} ± {100 * row['f1_std']:.2f} | {cell} |")
        lines.append("")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "significance.csv"), index=False)
    open(os.path.join(a.out, "table_main.md"), "w", encoding="utf8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
