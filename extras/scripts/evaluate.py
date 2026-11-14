"""Evaluate trained runs + zero-shot baselines, with and without selective abstention.

Zero-shot baselines are document detectors adapted to words by a sliding window
(moving average of per-word scores, window --zs_win words), thresholded at the
dev-optimal macro-F1 point and Platt-calibrated on dev (favourable to them).
  binoculars : -(ll_obs / xent)   (Hans et al., ICML 2024; low ppl/x-ppl => AI)
  fastdetect : fdg_z              (Bao et al., ICLR 2024)
  loglik     : ll_obs             (likelihood baseline)
  repreguard : rep                (RepreGuard-style projection, Chen et al., TACL 2025)

Outputs (in --out): main.csv, selective.csv, by_level.csv, curves.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from provseg.conformal.selective import SelectiveCalibrator  # noqa: E402
from provseg.data.torch_data import load_records  # noqa: E402
from provseg.eval.metrics import all_metrics, risk_coverage, word_metrics  # noqa: E402
from provseg.features.token_features import FEAT_NAMES  # noqa: E402

ZS = {
    "binoculars": lambda F: -(F[:, 0] / np.maximum(F[:, 3], 1e-3)),
    "fastdetect": lambda F: F[:, 4],
    "loglik": lambda F: F[:, 0],
    "repreguard": lambda F: F[:, 6],
}


def load_feats(feat_dir):
    arr = {}
    for p in glob.glob(os.path.join(feat_dir, "*.npz")):
        if p.endswith("_dep.npz"):      # dependency graphs share uids: never mix them with features
            continue
        z = np.load(p)
        arr.update({k: z[k] for k in z.files})
    return arr


class ZeroShot:
    def __init__(self, name, win):
        self.name, self.win, self.f = name, win, ZS[name]

    def score(self, F):
        s = self.f(np.nan_to_num(F, nan=np.nanmean(F, 0)))
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        return uniform_filter1d(s, size=self.win, mode="nearest")

    def fit(self, recs, feats):
        X = np.concatenate([self.score(feats[r["meta"]["uid"]]) for r in recs])
        y = np.concatenate([np.array(r["labels"]) for r in recs])
        m = y >= 0
        self.lr = LogisticRegression().fit(X[m, None], y[m])  # Platt; also sets the 0.5 threshold
        return self

    def predict(self, r, feats):
        p1 = self.lr.predict_proba(self.score(feats[r["meta"]["uid"]])[:, None])[:, 1]
        prob = np.stack([1 - p1, p1], -1)
        return prob.argmax(-1), prob, None


def collect(recs, getter):
    G, P, PR, U, meta = [], [], [], [], []
    for r in recs:
        out = getter(r)
        if out is None:
            continue
        p, pr, u = out
        G.append(np.array(r["labels"]))
        P.append(np.asarray(p))
        PR.append(np.asarray(pr))
        U.append(None if u is None else np.asarray(u))
        meta.append(r)
    return G, P, PR, (U if any(x is not None for x in U) else None), meta


PRETTY = {
    "zs_fastdetect": "Fast-DetectGPT (ICLR'24)*", "zs_binoculars": "Binoculars (ICML'24)*",
    "zs_repreguard": "RepreGuard (TACL'25)*", "zs_loglik": "Log-likelihood*",
    "seqxgpt_feats": "SeqXGPT (EMNLP'23)", "pasted_cls": "PASTED (ACL-F'24)", "seqroberta": "Seq-RoBERTa",
    "provseg": "**ProvSeg (ours)**", "provseg_records": "**ProvSeg + records (ours)**",
}
ORDER = list(PRETTY)


def _base(name):
    import re
    return re.sub(r"_s\d+", "", name)


def main_table(df, sel, test_sets, alpha):
    """Markdown table: rows = systems, per dataset Macro-F1 / AUROC / (MAE for semeval8c) / Cov@alpha.
    Best per column in bold (lowest for MAE). Seeds are averaged."""
    df = df.copy()
    df["sys"] = df.system.map(_base)
    sel = sel[np.isclose(sel.alpha, alpha)].copy() if len(sel) else sel
    if len(sel):
        sel["sys"] = sel.system.map(_base)
    cols, vals = [], {}
    for ds in test_sets:
        metrics = [("macro_f1", "F1", 1), ("auroc", "AUROC", 1)]
        if ds.startswith("semeval8c"):
            metrics.append(("boundary_mae", "MAE↓", -1))
        for key, lab, sgn in metrics:
            col = f"{ds} {lab}"
            cols.append((col, sgn))
            vals[col] = df[df.dataset == ds].groupby("sys")[key].mean()
        if len(sel):
            col = f"{ds} Cov@{alpha:g}"
            cols.append((col, 1))
            vals[col] = sel[sel.dataset == ds].groupby("sys")["coverage"].mean()
    systems = sorted(set(df.sys), key=lambda s: ORDER.index(s) if s in ORDER else len(ORDER))
    lines = ["| System | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for s in systems:
        cells = []
        for c, sgn in cols:
            v = vals[c].get(s, np.nan)
            col = vals[c].dropna()
            best = (col.max() if sgn > 0 else col.min()) if len(col) else np.nan
            txt = "–" if pd.isna(v) else (f"{v:.1f}" if "MAE" in c else f"{100 * v:.1f}")
            cells.append(f"**{txt}**" if not pd.isna(v) and np.isclose(v, best) else txt)
        lines.append(f"| {PRETTY.get(s, s)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"F1 = word-level macro-F1 (%), AUROC = word-level (%), MAE = SemEval-24 8C boundary error (words), "
                 f"Cov@{alpha:g} = % words answered with certified selective risk <= {alpha:g} (Learn-then-Test, delta=0.1). "
                 "* document detector adapted to words with a sliding window + dev-tuned Platt calibration.")
    return chr(10).join(lines) + chr(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs with pred_<ds>.pkl")
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--feats", default="../data/feats")
    ap.add_argument("--train_sets", default="seqxgpt,semeval8c,beemo")
    ap.add_argument("--test_sets", default="seqxgpt,semeval8c,beemo,pasted,pasted_ood,apt_eval,editlens")
    ap.add_argument("--alphas", default="0.02,0.05,0.1,0.15,0.2")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--zs_win", type=int, default=15)
    ap.add_argument("--no_zs", action="store_true")
    ap.add_argument("--zs", default="fastdetect,binoculars,repreguard", help="zero-shot baselines to run")
    ap.add_argument("--table_alpha", type=float, default=0.05)
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    alphas = [float(x) for x in a.alphas.split(",")]
    path = lambda ds: os.path.join(a.data, f"{ds}.jsonl")  # noqa: E731
    train_sets = a.train_sets.split(",")
    test_sets = [d for d in a.test_sets.split(",") if os.path.exists(path(d))]

    # ---------- systems: name -> getter(record) ----------
    systems = {}
    for rd in a.runs:
        name = os.path.basename(rd.rstrip("/\\"))
        preds = {}
        for p in glob.glob(os.path.join(rd, "pred_*.pkl")):
            preds.update(pickle.load(open(p, "rb")))
        systems[name] = (lambda P: (lambda r: (P[r["meta"]["uid"]]["pred"], P[r["meta"]["uid"]]["prob"],
                                               P[r["meta"]["uid"]].get("u"))
                                    if r["meta"]["uid"] in P else None))(preds)
    feats = {} if a.no_zs else load_feats(a.feats)
    if feats:
        dev = [r for r in load_records([path(d) for d in train_sets], splits={"dev"}) if r["meta"]["uid"] in feats]
        for zn in [z for z in a.zs.split(",") if z in ZS]:
            z = ZeroShot(zn, a.zs_win).fit(dev, feats)
            systems[f"zs_{zn}"] = (lambda Z: (lambda r: Z.predict(r, feats) if r["meta"]["uid"] in feats else None))(z)

    # calibration pool: cal split of the training datasets (in-distribution guarantee)
    cal_recs = load_records([path(d) for d in train_sets], splits={"cal"})
    rows_main, rows_sel, rows_lvl, curves = [], [], [], {}
    for sname, get in systems.items():
        Gc, Pc, PRc, Uc, _ = collect(cal_recs, get)
        pooled = {al: SelectiveCalibrator(alpha=al, delta=a.delta).fit(Pc, PRc, Gc, ambig=Uc) for al in alphas}
        for ds in test_sets:
            # calibrate on the cal split of the same benchmark when it has one (in-distribution guarantee)
            G1, P1, PR1, U1, _ = collect(load_records([path(ds)], splits={"cal"}), get)
            cals = ({al: SelectiveCalibrator(alpha=al, delta=a.delta).fit(P1, PR1, G1, ambig=U1) for al in alphas}
                    if len(G1) >= 50 else pooled)
            recs = load_records([path(ds)], splits={"test"})
            G, P, PR, U, M = collect(recs, get)
            Ui = U if U is not None else [None] * len(P)
            if not G:
                continue
            m = all_metrics(G, P, PR)
            rows_main.append({"system": sname, "dataset": ds, "n_docs": len(G), **m})
            cov, risk, _ = risk_coverage(G, P, [pr.max(-1) for pr in PR])
            curves.setdefault(sname, {})[ds] = {"coverage": cov.tolist(), "risk": risk.tolist()}
            for al, cal in cals.items():
                Ps = [cal.apply(p, pr, u) for p, pr, u in zip(P, PR, Ui)]
                wm = word_metrics(G, Ps)
                # per-document selective risk = the quantity LTT certifies
                dr = []
                for g, ps in zip(G, Ps):
                    k = (ps != 2) & (g >= 0)
                    if k.any():
                        dr.append(float((ps[k] != g[k]).mean()))
                rows_sel.append({"system": sname, "dataset": ds, "alpha": al, "lambda": cal.lam,
                                 "doc_risk": float(np.mean(dr)) if dr else float("nan"),
                                 "valid": (np.mean(dr) <= al) if dr else True, **wm})
            # by edit level (polish degree / edit-ratio bucket)
            levels = sorted({r["edit_level"] for r in M})
            if len(levels) > 1:
                cal = cals[0.1] if 0.1 in cals else list(cals.values())[0]
                for lv in levels:
                    idx = [i for i, r in enumerate(M) if r["edit_level"] == lv]
                    g = [G[i] for i in idx]
                    full = word_metrics(g, [P[i] for i in idx])
                    sel = word_metrics(g, [cal.apply(P[i], PR[i], Ui[i]) for i in idx])
                    rows_lvl.append({"system": sname, "dataset": ds, "level": lv, "n": len(idx),
                                     "macro_f1": full.get("macro_f1"), "acc": full.get("acc"),
                                     "sel_acc": sel.get("acc"), "sel_coverage": sel["coverage"]})
        print(f"[done] {sname}")

    pd.DataFrame(rows_main).to_csv(os.path.join(a.out, "main.csv"), index=False)
    pd.DataFrame(rows_sel).to_csv(os.path.join(a.out, "selective.csv"), index=False)
    pd.DataFrame(rows_lvl).to_csv(os.path.join(a.out, "by_level.csv"), index=False)
    json.dump(curves, open(os.path.join(a.out, "curves.json"), "w"))
    df = pd.DataFrame(rows_main)
    if len(df):
        print(df.pivot_table(index="system", columns="dataset", values="macro_f1").round(3).to_string())
        table = main_table(df, pd.DataFrame(rows_sel), test_sets, a.table_alpha)
        open(os.path.join(a.out, "table_main.md"), "w", encoding="utf8").write(table)
        print(chr(10) + table)


if __name__ == "__main__":
    main()
