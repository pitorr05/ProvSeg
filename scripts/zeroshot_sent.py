"""Zero-shot baselines at sentence level, scored with the *official* scoring functions:

  fastdetect  Fast-DetectGPT (Bao et al., ICLR 2024): get_sampling_discrepancy_analytic
              (baselines/zeroshot/fast_detect_gpt.py, verbatim), reference = performer, scoring = observer
  binoculars  Binoculars (Hans et al., ICML 2024): perplexity / entropy from binoculars/metrics.py
              (verbatim); score = ppl(performer) / x-ppl(observer, performer), lower = AI

As in SeqXGPT's sentence-level comparison, every sentence is scored on its own. A logistic map
(Platt) from score to P(AI) is fitted on train-split sentences; each word takes its sentence's
prediction. Observer/performer default to the same small Qwen2.5 pair used for our evidence.

  python scripts/zeroshot_sent.py --method fastdetect --ds semeval8c --out ../runs_ladder
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baselines.zeroshot.binoculars_metrics import entropy, perplexity  # noqa: E402
from baselines.zeroshot.fast_detect_gpt import get_sampling_discrepancy_analytic  # noqa: E402
from provseg.data.torch_data import cap_subset, load_records  # noqa: E402
from provseg.eval.metrics import sentence_spans  # noqa: E402


class Scorer:
    def __init__(self, observer, performer, max_len=256):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.bfloat16 if self.dev == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(observer)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.obs = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=dt).to(self.dev).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=dt).to(self.dev).eval()
        self.max_len = max_len

    @torch.no_grad()
    def score(self, texts, method, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.dev)
            lo = self.obs(**enc).logits.float()
            lp = self.perf(**enc).logits.float()
            if method == "binoculars":
                ppl = perplexity(enc, lp)
                xppl = entropy(lo, lp, enc, self.tok.pad_token_id)
                out += list(-(ppl / np.maximum(xppl, 1e-6)))           # higher = more AI-like
            else:
                for k in range(lo.shape[0]):
                    n = int(enc["attention_mask"][k].sum())
                    if n < 2:
                        out.append(0.0)
                        continue
                    lab = enc["input_ids"][k:k + 1, 1:n]
                    out.append(get_sampling_discrepancy_analytic(lp[k:k + 1, :n - 1], lo[k:k + 1, :n - 1], lab))
        return np.nan_to_num(np.asarray(out, dtype=np.float64))


def sentences(recs):
    items = []
    for ri, r in enumerate(recs):
        g = np.asarray(r["labels"])
        for s, e in sentence_spans(r["words"]):
            m = g[s:e] >= 0
            items.append((ri, s, e, " ".join(r["words"][s:e]), int(np.bincount(g[s:e][m]).argmax()) if m.any() else -1))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fastdetect", "binoculars"], required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--data", default="../data/processed")
    ap.add_argument("--out", default="../runs_ladder")
    ap.add_argument("--name", default="")
    ap.add_argument("--observer", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--performer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--fit_docs", type=int, default=500, help="train docs used to fit the score->P(AI) map")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev_cap", type=int, default=2500)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, a.name or f"{a.method}_{a.ds}", f"pred_{a.ds}.pkl")):
        print(f"[{a.method}_{a.ds}] already done -> skip")
        return
    path = os.path.join(a.data, f"{a.ds}.jsonl")
    sc = Scorer(a.observer, a.performer)

    fit = sentences(cap_subset(load_records([path], {"train"}, a.limit), a.fit_docs))
    fit = [x for x in fit if x[4] >= 0]
    s_fit = sc.score([x[3] for x in fit], a.method)
    y_fit = np.array([x[4] for x in fit])
    clf = LogisticRegression().fit(s_fit[:, None], y_fit) if len(set(y_fit)) == 2 else None

    recs = [r for sp in ("dev", "cal", "test") for r in load_records([path], {sp}, a.limit, a.dev_cap if sp == "dev" else 0)]
    items = sentences(recs)
    s = sc.score([x[3] for x in tqdm(items, desc=f"{a.method}:{a.ds}", mininterval=60)], a.method)
    p_ai = clf.predict_proba(s[:, None])[:, 1] if clf is not None else np.full(len(s), 0.5)
    name = a.name or f"{a.method}_{a.ds}"
    dump = {"__meta__": {"name": name, "step": "baseline", "arch": a.method, "train": a.ds, "channels": []}}
    for r in recs:
        n = len(r["words"])
        dump[r["meta"]["uid"]] = {"pred": np.zeros(n, int), "prob": np.tile([1.0, 0.0], (n, 1)), "split": r["split"]}
    for (ri, st, en, _, _), p in zip(items, p_ai):
        d = dump[recs[ri]["meta"]["uid"]]
        d["prob"][st:en] = [1 - p, p]
        d["pred"][st:en] = int(p >= 0.5)
    rd = os.path.join(a.out, name)
    os.makedirs(rd, exist_ok=True)
    pickle.dump(dump, open(os.path.join(rd, f"pred_{a.ds}.pkl"), "wb"))
    print(f"[{name}] wrote {len(recs)} docs / {len(items)} sentences -> {rd}")


if __name__ == "__main__":
    main()
