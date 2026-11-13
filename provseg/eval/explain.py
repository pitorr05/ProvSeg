"""Evaluation of the evidence-bottleneck explanations.

For every run with ``contrib`` in its prediction dumps:

E1 exactness        max |z - b - sum_k contrib_k| (should be ~0: the explanation IS the decision)
E2 deletion         per predicted span: remove the top channel (comprehensiveness: does the
                    span decision flip?) / keep only the top channel (sufficiency: does it stay?)
                    -- computed on the additive logit, CRF transitions excluded.
E3 channel share    mean |contrib_k| share per dataset, per gold edit status and per gold label
E4 edit-trace       AUROC of predicted P(edited) vs gold ``edited`` on held-out data
E5 ambiguity        AUROC of u for predicting errors; error rate per u-quintile
cases               JSON + HTML of example spans with their evidence (case-study material)

Usage: python -m provseg.eval.explain --run ../runs/provseg_s13 --data ../data/processed --out ../results
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score



PRETTY = {"lexical": "LLM-likelihood", "representation": "hidden representation", "context": "context",
          "syntax": "syntax", "edit_trace": "edit trace"}


def runs_of(pred):
    """Maximal runs of identical predicted labels -> (start, end, label)."""
    out, i = [], 0
    while i < len(pred):
        j = i
        while j < len(pred) and pred[j] == pred[i]:
            j += 1
        out.append((i, j, int(pred[i])))
        i = j
    return out


def _test_recs(data_dir, ds, P):
    recs = {}
    with open(os.path.join(data_dir, f"{ds}.jsonl"), encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "test" and r["meta"]["uid"] in P:
                recs[r["meta"]["uid"]] = r
    return recs


def explain_efp(run, data_dir, out_dir, n_cases=30):
    """History explanations for Edit-Factorized Provenance.

    H1 factor faithfulness  : per span and factor, delete the top channel of the factor logit
                              (comprehensiveness) / keep only it (sufficiency); exact by additivity
    H2 edit detection       : AUROC of pi vs gold edited (privileged label, held-out data)
    H3 Unknown sources      : share of total variance p(1-p) from edit / editor / base uncertainty,
                              overall and on errors; AUROC of p(1-p) for errors
    H4 Prop. B check        : error rate vs |rho - beta| for words with uncertain edits (0.2<pi<0.8)
    H5 channel shares       : mean |contrib| share per factor and channel
    cases                   : spans with pi/rho/beta and the dominant channel of each factor
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(run.rstrip("/\\"))
    R = {k: [] for k in ("faith", "edit", "unknown", "propb", "share")}
    cases = []
    for pf in sorted(glob.glob(os.path.join(run, "pred_*.pkl"))):
        ds = os.path.basename(pf)[5:-4]
        P = pickle.load(open(pf, "rb"))
        if P.get("__model__") != "efp":
            continue
        ch = P["__channels__"]
        recs = _test_recs(data_dir, ds, P)
        faith = {f: [0, 0, 0] for f in ("pi", "rho", "beta")}   # flips, keeps, n
        e_true, e_prob, err, var, src, gap = [], [], [], [], [], []
        share = {f: [] for f in ("pi", "rho", "beta")}
        for uid, r in recs.items():
            o = P[uid]
            g = np.array(r["labels"])
            for f in ("pi", "rho", "beta"):
                C = o[f"contrib_{f}"]
                share[f].append((np.abs(C) / (np.abs(C).sum(-1, keepdims=True) + 1e-9)).mean(0))
                lg = np.log(np.clip(o[f], 1e-6, 1 - 1e-6) / np.clip(1 - o[f], 1e-6, 1))
                b0 = float(np.median(lg - C.sum(-1)))
                for s, e, _ in runs_of(o["pred"]):
                    cs = C[s:e].sum(0)
                    zs = b0 * (e - s) + cs.sum()
                    sign = 1 if zs >= 0 else -1
                    top = int(np.argmax(sign * cs))
                    faith[f][0] += int(np.sign(zs - cs[top]) != sign)
                    faith[f][1] += int(np.sign(b0 * (e - s) + cs[top]) == sign)
                    faith[f][2] += 1
            ed = np.array(r.get("edited") or [-1] * len(g))
            m = ed >= 0
            e_true += ed[m].tolist()
            e_prob += o["pi"][m].tolist()
            m = g >= 0
            wrong = (o["pred"][m] != g[m]).astype(int)
            U = np.stack([o["u_edit"], o["u_editor"], o["u_base"]], -1)[m]
            err += wrong.tolist()
            var += U.sum(-1).tolist()
            src += U.argmax(-1).tolist()
            unc = (o["pi"][m] > 0.2) & (o["pi"][m] < 0.8)
            gap += [(float(abs(a - b)), int(w)) for a, b, w, k in
                    zip(o["rho"][m], o["beta"][m], wrong, unc) if k]
            if len(cases) < n_cases and 0 < g[g >= 0].mean() < 1:
                sp = []
                for s, e, lab in runs_of(o["pred"]):
                    dom = {f: ch[int(np.argmax(np.abs(o[f"contrib_{f}"][s:e].sum(0))))] for f in ("pi", "rho", "beta")}
                    sp.append({"start": s, "end": e, "pred": "AI" if lab == 1 else "Human",
                               "gold_ai_frac": float(g[s:e][g[s:e] >= 0].mean()) if (g[s:e] >= 0).any() else None,
                               "pi": float(o["pi"][s:e].mean()), "rho": float(o["rho"][s:e].mean()),
                               "beta": float(o["beta"][s:e].mean()), "dominant": dom,
                               "unknown_source": ["edit", "editor", "base"][int(np.argmax(
                                   [o["u_edit"][s:e].mean(), o["u_editor"][s:e].mean(), o["u_base"][s:e].mean()]))],
                               "text": " ".join(r["words"][s:e])})
                cases.append({"dataset": ds, "uid": uid, "spans": sp})
        for f, (fl, kp, n) in faith.items():
            R["faith"].append({"dataset": ds, "factor": f, "n_spans": n,
                               "comprehensiveness_flip_rate": fl / max(n, 1), "sufficiency_keep_rate": kp / max(n, 1)})
        if len(set(e_true)) == 2:
            R["edit"].append({"dataset": ds, "n_words": len(e_true), "pi_edit_auroc": roc_auc_score(e_true, e_prob)})
        if len(set(err)) == 2:
            src_a, err_a = np.array(src), np.array(err)
            R["unknown"].append({"dataset": ds, "var_error_auroc": roc_auc_score(err, var),
                                 **{f"share_{n}_all": float((src_a == k).mean()) for k, n in enumerate(["edit", "editor", "base"])},
                                 **{f"share_{n}_errors": float((src_a[err_a == 1] == k).mean()) for k, n in enumerate(["edit", "editor", "base"])}})
        if len(gap) > 50:
            gdf = pd.DataFrame(gap, columns=["gap", "err"])
            gdf["bin"] = pd.cut(gdf.gap, [0, 0.1, 0.3, 0.6, 1.0], include_lowest=True)
            for b, gg in gdf.groupby("bin", observed=True):
                R["propb"].append({"dataset": ds, "rho_beta_gap": str(b), "n": len(gg), "error_rate": gg.err.mean()})
        for f, v in share.items():
            if v:
                R["share"].append({"dataset": ds, "factor": f, **dict(zip(ch, np.mean(v, 0).tolist()))})
    for k, rows in R.items():
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"explain_efp_{k}_{name}.csv"), index=False)
        if rows:
            print(f"-- {k}\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    json.dump(cases, open(os.path.join(out_dir, f"explain_efp_cases_{name}.json"), "w"), indent=1)
    write_html_efp(cases, os.path.join(out_dir, f"explain_efp_cases_{name}.html"))


def write_html_efp(cases, path):
    css = ("body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.6}"
           ".A{background:#fde2e1}.H{background:#e1effd}.span{border-radius:3px;padding:1px 2px}"
           ".ev{font-size:12px;color:#555;margin:2px 0 8px 0}")
    out = [f"<!doctype html><meta charset=utf-8><title>ProvSeg history explanations</title><style>{css}</style>",
           "<h1>ProvSeg (EFP): which words survived from whom, and why</h1>"]
    for c in cases:
        out.append(f"<h3>{html.escape(c['dataset'])} · {html.escape(c['uid'])}</h3><p>")
        for s in c["spans"]:
            cls = "A" if s["pred"] == "AI" else "H"
            out.append(f"<span class='span {cls}'>{html.escape(s['text'])}</span> ")
        out.append("</p>")
        for s in c["spans"]:
            d = s["dominant"]
            out.append(
                f"<div class=ev><b>{s['pred']}</b> ({s['end'] - s['start']} words): "
                f"edited in last step π={s['pi']:.2f} [{PRETTY.get(d['pi'], d['pi'])}], "
                f"editor AI ρ={s['rho']:.2f} [{PRETTY.get(d['rho'], d['rho'])}], "
                f"base AI β={s['beta']:.2f} [{PRETTY.get(d['beta'], d['beta'])}]; "
                f"main uncertainty: {s['unknown_source']}</div>")
    open(path, "w", encoding="utf8").write("\n".join(out))


def explain_run(run, data_dir, out_dir, n_cases=30):
    os.makedirs(out_dir, exist_ok=True)
    if any(pickle.load(open(p, "rb")).get("__model__") == "efp" for p in glob.glob(os.path.join(run, "pred_*.pkl"))[:1]):
        return explain_efp(run, data_dir, out_dir, n_cases)
    rows_share, rows_del, rows_edit, rows_amb, cases = [], [], [], [], []
    exact = 0.0
    for pf in sorted(glob.glob(os.path.join(run, "pred_*.pkl"))):
        ds = os.path.basename(pf)[5:-4]
        P = pickle.load(open(pf, "rb"))
        ch = P.get("__channels__", [])
        if not ch:
            continue
        recs = {}
        with open(os.path.join(data_dir, f"{ds}.jsonl"), encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == "test" and r["meta"]["uid"] in P:
                    recs[r["meta"]["uid"]] = r
        share = {c: [] for c in ch}
        share_by = {}
        flips = keeps = nsp = 0
        e_true, e_prob, err, amb = [], [], [], []
        for uid, r in recs.items():
            o = P[uid]
            if "contrib" not in o:
                continue
            C, z = o["contrib"], o.get("z")
            g = np.array(r["labels"])
            if z is not None:
                b = z - C.sum(-1)
                exact = max(exact, float(np.abs(b - np.median(b)).max()))   # b is a constant
                b0 = float(np.median(b))
            else:
                b0 = 0.0
            mag = np.abs(C)
            tot = mag.sum(-1, keepdims=True) + 1e-9
            sh = mag / tot
            for k, c in enumerate(ch):
                share[c].append(sh[:, k].mean())
            ed = np.array(r.get("edited") or [-1] * len(g))
            for key, m in (("gold=A", g == 1), ("gold=H", g == 0), ("edited", ed == 1), ("copied", ed == 0)):
                if m.any():
                    share_by.setdefault(key, []).append(sh[m].mean(0))
            # E2 deletion on predicted runs
            for s, e, lab in runs_of(o["pred"]):
                span_c = C[s:e].sum(0)                       # summed evidence for the run
                zs = b0 * (e - s) + span_c.sum()
                sign = 1 if zs >= 0 else -1
                top = int(np.argmax(sign * span_c))           # channel pushing hardest towards the decision
                zs_del = zs - span_c[top]
                zs_only = b0 * (e - s) + span_c[top]
                flips += int(np.sign(zs_del) != sign)
                keeps += int(np.sign(zs_only) == sign)
                nsp += 1
            if "edit_prob" in o:
                m = ed >= 0
                e_true += ed[m].tolist()
                e_prob += o["edit_prob"][m].tolist()
            if "u" in o:
                m = g >= 0
                err += (o["pred"][m] != g[m]).astype(int).tolist()
                amb += o["u"][m].tolist()
            if len(cases) < n_cases and 0 < g.mean() < 1:
                sp = []
                for s, e, lab in runs_of(o["pred"]):
                    cs = C[s:e].sum(0)
                    sp.append({"start": s, "end": e, "pred": "AI" if lab == 1 else "Human",
                               "gold_ai_frac": float(g[s:e][g[s:e] >= 0].mean()) if (g[s:e] >= 0).any() else None,
                               "evidence": {c: round(float(v), 3) for c, v in zip(ch, cs)},
                               "ambiguity": float(o["u"][s:e].mean()) if "u" in o else None,
                               "text": " ".join(r["words"][s:e])})
                cases.append({"dataset": ds, "uid": uid, "spans": sp})
        rows_share.append({"dataset": ds, **{c: float(np.mean(v)) for c, v in share.items() if v}})
        for key, v in share_by.items():
            rows_share.append({"dataset": f"{ds}|{key}", **dict(zip(ch, np.mean(v, 0).tolist()))})
        rows_del.append({"dataset": ds, "n_spans": nsp, "comprehensiveness_flip_rate": flips / max(nsp, 1),
                         "sufficiency_keep_rate": keeps / max(nsp, 1)})
        if len(set(e_true)) == 2:
            rows_edit.append({"dataset": ds, "n_words": len(e_true), "edit_auroc": roc_auc_score(e_true, e_prob)})
        if len(set(err)) == 2:
            q = pd.qcut(pd.Series(amb).rank(method="first"), 5, labels=False)
            rates = pd.Series(err).groupby(q).mean().tolist()
            rows_amb.append({"dataset": ds, "u_error_auroc": roc_auc_score(err, amb),
                             **{f"err_rate_u_q{i + 1}": v for i, v in enumerate(rates)}})
    name = os.path.basename(run.rstrip("/\\"))
    for fn, rows in (("share", rows_share), ("deletion", rows_del), ("edit", rows_edit), ("ambiguity", rows_amb)):
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"explain_{fn}_{name}.csv"), index=False)
    json.dump(cases, open(os.path.join(out_dir, f"explain_cases_{name}.json"), "w"), indent=1)
    write_html(cases, os.path.join(out_dir, f"explain_cases_{name}.html"))
    print(f"[explain] {name}: max |additivity error| = {exact:.2e}")
    for t, rows in (("deletion", rows_del), ("edit", rows_edit), ("ambiguity", rows_amb)):
        if rows:
            print(f"-- {t}\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    return exact


def write_html(cases, path):
    css = ("body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.6}"
           ".A{background:#fde2e1}.H{background:#e1effd}.span{border-radius:3px;padding:1px 2px}"
           ".ev{font-size:12px;color:#555;margin:2px 0 10px 0}")
    out = [f"<!doctype html><meta charset=utf-8><title>ProvSeg evidence</title><style>{css}</style>",
           "<h1>ProvSeg: span decisions with their evidence</h1>"]
    for c in cases:
        out.append(f"<h3>{html.escape(c['dataset'])} · {html.escape(c['uid'])}</h3><p>")
        for s in c["spans"]:
            cls = "A" if s["pred"] == "AI" else "H"
            ev = ", ".join(f"{PRETTY.get(k, k)} {v:+.2f}" for k, v in
                           sorted(s["evidence"].items(), key=lambda kv: -abs(kv[1])))
            tip = html.escape(f"{s['pred']} | {ev} | ambiguity {s['ambiguity']}")
            out.append(f"<span class='span {cls}' title='{tip}'>{html.escape(s['text'])}</span> ")
        out.append("</p>")
        for s in c["spans"]:
            if s["pred"] == "AI":
                ev = ", ".join(f"<b>{PRETTY.get(k, k)}</b> {v:+.2f}" for k, v in
                               sorted(s["evidence"].items(), key=lambda kv: -abs(kv[1]))[:3])
                out.append(f"<div class=ev>AI span ({s['end'] - s['start']} words, gold AI={s['gold_ai_frac']}): {ev}</div>")
    open(path, "w", encoding="utf8").write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    for r in a.run:
        explain_run(r, a.data, a.out)


if __name__ == "__main__":
    main()
