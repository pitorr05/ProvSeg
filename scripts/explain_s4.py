"""Evidence explanations of ProvSeg (S4): exact channel Shapley values per word.

For every word, phi_k is the Shapley value of evidence group k (each per-word LLM signal, and the
context encoder) for the margin  log P(AI) - log P(human). Efficiency holds exactly:
sum_k phi_k = margin(all evidence) - margin(no evidence). S4 is trained with evidence dropout, so
"evidence absent" is a state the model has seen.

Outputs in --out:
  explain_<ds>.csv   per evidence group: mean phi on gold-AI and gold-human words, share of |phi|,
                     and how often it is the top contributor of words predicted AI
  cases_<ds>.html    a few test documents, words coloured by P(AI), hover = per-group phi
"""
from __future__ import annotations

import argparse
import glob
import html
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.data.torch_data import load_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="S4 run dir(s), glob allowed")
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results_ladder")
    ap.add_argument("--cases", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.run, "shap_*.pkl"))):
        ds = os.path.basename(p)[5:-4]
        sh = pickle.load(open(p, "rb"))
        players = sh.pop("__players__")
        pred = pickle.load(open(os.path.join(os.path.dirname(p), f"pred_{ds}.pkl"), "rb"))
        recs = {r["meta"]["uid"]: r for r in load_records([os.path.join(a.data, f"{ds}.jsonl")], {"test"})}
        uids = [u for u in sh if u in recs]
        phi = np.concatenate([sh[u]["phi"] for u in uids])
        gap = np.concatenate([sh[u]["m_full"] - sh[u]["m_empty"] for u in uids])
        gold = np.concatenate([np.asarray(recs[u]["labels"]) for u in uids])
        pa = np.concatenate([pred[u]["pred"] for u in uids])
        eff = float(np.abs(phi.sum(1) - gap).max())
        top = phi.argmax(1)
        rows = []
        for k, name in enumerate(players):
            rows.append({"evidence": name,
                         "mean_phi_goldAI": float(phi[gold == 1, k].mean()),
                         "mean_phi_goldH": float(phi[gold == 0, k].mean()),
                         "share_abs_phi": float(np.abs(phi[:, k]).sum() / np.abs(phi).sum()),
                         "top_on_predAI": float((top[pa == 1] == k).mean()) if (pa == 1).any() else float("nan")})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, f"explain_{ds}.csv"), index=False)
        print(f"== {ds}: {len(uids)} docs, max efficiency error {eff:.2e}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        # cases: documents with both human and AI words
        mixed = [u for u in uids if len(set(np.asarray(recs[u]["labels"]).tolist()) & {0, 1}) == 2][: a.cases]
        parts = [f"<h2>{html.escape(ds)}: per-word evidence (hover a word)</h2>"]
        for u in mixed:
            ws, pr, ph = recs[u]["words"], pred[u]["prob"][:, 1], sh[u]["phi"]
            g = np.asarray(recs[u]["labels"])
            spans = []
            for i, w in enumerate(ws):
                tip = " | ".join(f"{n}: {ph[i, k]:+.2f}" for k, n in enumerate(players))
                bg = f"rgba(220,60,60,{pr[i]:.2f})"
                ul = "text-decoration:underline" if g[i] == 1 else ""
                spans.append(f'<span title="P(AI)={pr[i]:.2f} | {html.escape(tip)}" style="background:{bg};{ul}">{html.escape(w)}</span>')
            parts.append(f"<p><b>{html.escape(u)}</b> (underline = gold AI)<br>" + " ".join(spans) + "</p>")
        open(os.path.join(a.out, f"cases_{ds}.html"), "w", encoding="utf8").write(
            "<!doctype html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px'>" + "\n".join(parts))


if __name__ == "__main__":
    main()
