"""Paper figures from results/*.csv|json -> paper/figs/*.pdf

fig1_teaser      word accuracy vs. polish degree (APT-Eval) / edit ratio (Beemo):
                 baselines degrade; ProvSeg (full coverage, dashed) and ProvSeg+abstain
                 (selective accuracy, solid) with coverage as a secondary panel.
fig2_riskcov     risk-coverage curves per test set.
fig3_validity    realised per-document selective risk vs. target alpha (diagonal = guarantee).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

APT_ORDER = ["none", "extreme_minor", "minor", "slight_major", "major"]
APT_PCT = ["1.0%", "5.0%", "10.0%", "20.0%", "35.0%", "50.0%", "75.0%"]
BEEMO_ORDER = ["<= 0.10", "<= 0.25", "<= 0.50", "<= 0.75", "> 0.75"]
PRETTY = {"zs_binoculars": "Binoculars*", "zs_fastdetect": "Fast-DetectGPT*", "zs_loglik": "Log-likelihood*",
          "zs_repreguard": "RepreGuard*"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def pretty(s):
    if s in PRETTY:
        return PRETTY[s]
    return s.split("_s")[0].replace("provseg", "ProvSeg").replace("seqroberta", "Seq-RoBERTa")


def fig_teaser(lvl, out, ours):
    panels = [("apt_eval", APT_ORDER, "APT-Eval polish degree"), ("apt_eval", APT_PCT, "APT-Eval polish %"),
              ("beemo", BEEMO_ORDER, "Beemo human edit ratio")]
    panels = [p for p in panels if ((lvl.dataset == p[0]) & lvl.level.isin(p[1])).any()]
    if not panels:
        return
    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 4.2), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for j, (ds, order, title) in enumerate(panels):
        d = lvl[(lvl.dataset == ds) & lvl.level.isin(order)]
        for s, g in d.groupby("system"):
            g = g.set_index("level").reindex(order).dropna(subset=["acc"])
            x = [order.index(i) for i in g.index]
            is_ours = s.startswith(ours)
            axes[0, j].plot(x, g["acc"], ls="--" if is_ours else "-", marker="o", ms=3,
                            lw=1.8 if is_ours else 1, label=pretty(s) + (" (full)" if is_ours else ""))
            if is_ours:
                axes[0, j].plot(x, g["sel_acc"], "-", marker="s", ms=3, lw=2.2, label=pretty(s) + " + abstain")
                axes[1, j].bar(x, g["sel_coverage"], color="0.6")
        axes[0, j].set_title(title)
        axes[1, j].set_xticks(range(len(order)), [o.replace("_", " ") for o in order], rotation=30, ha="right")
        axes[1, j].set_ylim(0, 1)
    axes[0, 0].set_ylabel("word accuracy")
    axes[1, 0].set_ylabel("coverage")
    axes[0, -1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.savefig(os.path.join(out, "fig1_teaser.pdf"))
    plt.close(fig)


def fig_riskcov(curves, out):
    ds_all = sorted({d for v in curves.values() for d in v})
    fig, axes = plt.subplots(1, len(ds_all), figsize=(2.6 * len(ds_all), 2.4), squeeze=False)
    for j, ds in enumerate(ds_all):
        for s, v in curves.items():
            if ds in v:
                axes[0, j].plot(v[ds]["coverage"], v[ds]["risk"], label=pretty(s), lw=1.2)
        axes[0, j].set_title(ds)
        axes[0, j].set_xlabel("coverage")
    axes[0, 0].set_ylabel("selective risk")
    axes[0, -1].legend(fontsize=6, frameon=False)
    fig.savefig(os.path.join(out, "fig2_riskcov.pdf"))
    plt.close(fig)


def fig_validity(sel, out, ours):
    d = sel[sel.system.str.startswith(ours)]
    if d.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.6))
    for ds, g in d.groupby("dataset"):
        ax[0].plot(g.alpha, g.doc_risk, marker="o", ms=3, label=ds)
        ax[1].plot(g.alpha, g.coverage, marker="o", ms=3, label=ds)
    lim = d.alpha.max() * 1.1
    ax[0].plot([0, lim], [0, lim], "k:", lw=1)
    ax[0].set_xlabel(r"target $\alpha$")
    ax[0].set_ylabel("realised risk")
    ax[1].set_xlabel(r"target $\alpha$")
    ax[1].set_ylabel("coverage")
    ax[1].legend(fontsize=6, frameon=False)
    fig.savefig(os.path.join(out, "fig3_validity.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="../results")
    ap.add_argument("--out", default="paper/figs")
    ap.add_argument("--ours", default="provseg")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lvl = pd.read_csv(os.path.join(a.res, "by_level.csv"))
    sel = pd.read_csv(os.path.join(a.res, "selective.csv"))
    curves = json.load(open(os.path.join(a.res, "curves.json")))
    if len(lvl):
        fig_teaser(lvl, a.out, a.ours)
    fig_riskcov(curves, a.out)
    fig_validity(sel, a.out, a.ours)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
