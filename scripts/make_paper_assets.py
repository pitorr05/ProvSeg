"""Regenerate every paper table/figure from saved results, plus RESULTS_INDEX.md (provenance of each asset).

Inputs (all produced by versions v2-v5):
  results/significance/significance.csv      main table (mean +- std over seeds, bootstrap CI vs ProvSeg)
  results/compare/ladder.csv                 ablation ladder S0 -> S4 and baselines (single seed rows)
  results/robust/ladder.csv                  cross-domain / OOD runs (names xd_A_to_B, ood_pasted)
  results/robust/selective_summary.csv       certified Unknown (LTT) vs naive threshold
  results/robust/explain_faithfulness.csv    deletion test
  results/robust/cost.csv                    GPU cost
  results/diagnose/figF1_failure_modes.pdf, results/robust/risk_coverage_*.pdf, paper/figs/figM*.pdf
Outputs: paper/tables/*.tex (booktabs), paper/figs/*.pdf, paper/RESULTS_INDEX.md.
--check_tex paper/main.tex: every decimal number in the .tex must appear in some generated table
(guards against hand-typed or stale numbers).

  python scripts/make_paper_assets.py --results ../results --runs ../runs --out paper [--check_tex paper/main.tex]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess

import pandas as pd

BENCH = {"seqxgpt": "SeqXGPT-Bench", "semeval8c": "SemEval-8C", "beemo": "Beemo", "pasted": "PASTED",
         "coauthor": "CoAuthor", "pasted_ood": "PASTED-OOD"}


def esc(s):
    return str(s).replace("**", "").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write(path, text, index, source):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf8").write(text)
    index.append((os.path.relpath(path), source))


def table_main(sig, out, index):
    df = pd.read_csv(sig)
    sets = [d for d in BENCH if d in set(df.dataset)]
    methods = list(dict.fromkeys(df.sort_values("f1_mean", ascending=False)["label"]))
    best = df.loc[df.groupby("dataset")["f1_mean"].idxmax()].set_index("dataset")["label"].to_dict()
    lines = [r"\begin{tabular}{l" + "c" * len(sets) + "}", r"\toprule",
             "Method & " + " & ".join(BENCH[s] for s in sets) + r" \\", r"\midrule"]
    for m in methods:
        cells = []
        for s in sets:
            r = df[(df.dataset == s) & (df.label == m)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            c = f"{100 * r.f1_mean:.1f}" + (f"$\\pm${100 * r.f1_std:.1f}" if r.seeds > 1 else "")
            if best.get(s) == m:
                c = r"\textbf{" + c + "}"
            if r.get("target_better") is True or r.get("target_better") == 1.0:
                c += "$^*$"
            cells.append(c)
        lines.append(esc(m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tables", "main.tex"), "\n".join(lines) + "\n", index,
          f"{sig}: word macro-F1 mean±std over seeds; * = 95% bootstrap CI of (ProvSeg - method) > 0")


def table_ladder(ladder, out, index):
    df = pd.read_csv(ladder)
    df = df[df.step.isin(["S0", "S1", "S2", "S2b", "S4"])]
    agg = df.groupby(["step", "method", "dataset"])[["F1w", "AUROC"]].mean().reset_index()
    sets = [d for d in BENCH if d in set(agg.dataset)]
    lines = [r"\begin{tabular}{l" + "cc" * len(sets) + "}", r"\toprule",
             "Step & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{BENCH[s]}}}" for s in sets) + r" \\",
             " & " + " & ".join(["F1 & AUROC"] * len(sets)) + r" \\", r"\midrule"]
    for st in ["S0", "S1", "S2", "S2b", "S4"]:
        sub = agg[agg.step == st]
        if sub.empty:
            continue
        cells = []
        for s in sets:
            r = sub[sub.dataset == s]
            cells += ["--", "--"] if r.empty else [f"{100 * r.F1w.iloc[0]:.1f}", f"{100 * r.AUROC.iloc[0]:.1f}"]
        lines.append(esc(sub.method.iloc[0]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tables", "ladder.tex"), "\n".join(lines) + "\n", index, f"{ladder}: rows S0..S4 (seed-mean)")


def table_selective(sel, out, index):
    df = pd.read_csv(sel)
    cols = [c for c in df.columns if c.startswith(("ltt_", "naive_"))]
    lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             "Benchmark & Method & " + " & ".join(esc(c.replace("@", " @")) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{BENCH.get(r.dataset, r.dataset)} & {esc(r.method)} & " +
                     " & ".join("--" if pd.isna(r[c]) else f"{100 * r[c]:.1f}" for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tables", "selective.tex"), "\n".join(lines) + "\n", index,
          f"{sel}: coverage/risk (%) of certified LTT vs naive threshold")


def table_robust(robust_ladder, out, index):
    df = pd.read_csv(robust_ladder)
    rows = []
    for _, r in df.iterrows():
        m = re.match(r"xd_(\w+?)_to_(\w+)", str(r.run))
        if m and m.group(2) == r.dataset:
            rows.append((BENCH.get(m.group(1), m.group(1)), BENCH.get(r.dataset, r.dataset), r.F1w, r.AUROC))
        elif str(r.run).startswith("ood_"):
            rows.append(("PASTED", BENCH.get(r.dataset, r.dataset), r.F1w, r.AUROC))
    lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Train & Test & F1 & AUROC \\", r"\midrule"]
    lines += [f"{a} & {b} & {100 * f:.1f} & {100 * u:.1f}" + r" \\" for a, b, f, u in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tables", "robust.tex"), "\n".join(lines) + "\n", index, f"{robust_ladder}: cross-domain + OOD")


def table_csv(csv, out, index, name, cols, note):
    df = pd.read_csv(csv)
    cols = [c for c in cols if c in df.columns]
    lines = [r"\begin{tabular}{" + "l" * 2 + "c" * (len(cols) - 2) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else esc(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(out, "tables", f"{name}.tex"), "\n".join(lines) + "\n", index, f"{csv}: {note}")


def check_tex(tex, out):
    nums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", open(tex, encoding="utf8").read()))
    pool = "".join(open(p, encoding="utf8").read() for p in glob.glob(os.path.join(out, "tables", "*.tex")))
    missing = sorted(n for n in nums if n not in pool)
    print(f"check_tex: {len(nums)} decimal numbers in {tex}; {len(missing)} not found in generated tables")
    for n in missing[:40]:
        print("   not in tables:", n)
    return not missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--runs", default="../runs")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--check_tex", default="")
    a = ap.parse_args()
    R = a.results
    index = []
    jobs = [(os.path.join(R, "significance", "significance.csv"), table_main),
            (os.path.join(R, "compare", "ladder.csv"), table_ladder),
            (os.path.join(R, "robust", "selective_summary.csv"), table_selective),
            (os.path.join(R, "robust", "ladder.csv"), table_robust)]
    for path, fn in jobs:
        if os.path.exists(path):
            fn(path, a.out, index)
        else:
            print(f"[missing] {path} (run the version that produces it)")
    extra = [("explain_faithfulness.csv", "faithfulness", ["run", "dataset", "mean_abs_d_top", "mean_abs_d_rand",
              "mean_abs_d_min", "flip_rate_top", "flip_rate_rand", "spearman_absphi_absdelta"], "deletion test"),
             ("cost.csv", "cost", ["run", "arch", "train_docs", "epochs", "train_minutes", "peak_vram_gb"], "GPU cost")]
    for f, name, cols, note in extra:
        p = os.path.join(R, "robust", f)
        if os.path.exists(p):
            table_csv(p, a.out, index, name, cols, note)
    figs = glob.glob(os.path.join(R, "diagnose", "*.pdf")) + glob.glob(os.path.join(R, "robust", "*.pdf"))
    os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
    for f in figs:
        dst = os.path.join(a.out, "figs", os.path.basename(f))
        shutil.copy(f, dst)
        index.append((os.path.relpath(dst), f))
    for f in glob.glob(os.path.join(a.out, "figs", "figM*.pdf")):
        index.append((os.path.relpath(f), "scripts/motivation.py (EditLens released detector scores; APT-Eval, Beemo)"))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "?"
    lines = ["# RESULTS_INDEX: every paper asset and where it comes from", "", f"Generated at commit `{commit}` "
             "by `python scripts/make_paper_assets.py`.", "", "| asset | source |", "|---|---|"]
    lines += [f"| `{p}` | {s} |" for p, s in sorted(set(index))]
    open(os.path.join(a.out, "RESULTS_INDEX.md"), "w", encoding="utf8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.check_tex and not check_tex(a.check_tex, a.out):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
