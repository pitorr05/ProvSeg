"""S0 evidence = the original SeqXGPT features: per-word negative log-likelihood under several
white-box LMs (paper: GPT-2-xl, GPT-Neo-2.7B, GPT-J-6B, LLaMA-7B).

SeqXGPT spreads every token's NLL over its bytes and averages the bytes of each word
(``calc_token_ppl`` in backend_utils.py); the first token has no left context and gets 0.
We do the same at character level (identical for ASCII text) on ``" ".join(words)``, calling
the models directly instead of through SeqXGPT's HTTP inference servers.

  python -m provseg.features.llm_logprob --records ../data/processed/semeval8c.jsonl \
      --out ../data/feats/semeval8c_lp4.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LMS = "gpt2-xl,EleutherAI/gpt-neo-2.7B,EleutherAI/gpt-j-6b,huggyllama/llama-7b"


def word_char_spans(words):
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    return spans


def tokens_to_words(offsets, values, words):
    """Character-weighted mean of token values over each word's characters (SeqXGPT's
    byte-averaging). offsets: [(a, b)] char span per token; values: per-token floats."""
    n_chars = sum(len(w) for w in words) + max(len(words) - 1, 0)
    char_val = np.full(n_chars, np.nan)
    for (a, b), v in zip(offsets, values):
        if b > a:
            char_val[a:b] = v
    out = np.zeros(len(words), dtype=np.float32)
    for i, (a, b) in enumerate(word_char_spans(words)):
        seg = char_val[a:b]
        seg = seg[np.isfinite(seg)]
        out[i] = seg.mean() if len(seg) else 0.0
    return out


class LogProbLM:
    def __init__(self, name, max_len=1024, dtype=torch.bfloat16):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype if self.dev == "cuda" else torch.float32).to(self.dev).eval()
        self.max_len = min(max_len, getattr(self.model.config, "max_position_embeddings", max_len) or max_len)

    @torch.no_grad()
    def nll_windows(self, windows):
        """windows: list of token-id lists -> list of per-token NLL arrays (first token = 0)."""
        L = max(len(w) for w in windows)
        ids = torch.full((len(windows), L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for k, w in enumerate(windows):
            ids[k, :len(w)] = torch.tensor(w)
            att[k, :len(w)] = 1
        logits = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:].to(self.dev), reduction="none")
        out = []
        for k, w in enumerate(windows):
            out.append(np.concatenate([[0.0], nll[k, :len(w) - 1].cpu().numpy()]))
        return out

    def doc_features(self, docs, bs=8):
        """docs: list of word lists -> list of (n_words,) arrays."""
        encs = [self.tok(" ".join(ws), return_offsets_mapping=True, add_special_tokens=False) for ws in docs]
        jobs = []   # (doc, start)
        for d, e in enumerate(encs):
            for s in range(0, max(len(e["input_ids"]), 1), self.max_len):
                if len(e["input_ids"][s:s + self.max_len]):
                    jobs.append((d, s))
        tok_vals = [np.zeros(len(e["input_ids"])) for e in encs]
        jobs.sort(key=lambda j: -len(encs[j[0]]["input_ids"][j[1]:j[1] + self.max_len]))
        for i in range(0, len(jobs), bs):
            chunk = jobs[i:i + bs]
            res = self.nll_windows([encs[d]["input_ids"][s:s + self.max_len] for d, s in chunk])
            for (d, s), v in zip(chunk, res):
                tok_vals[d][s:s + len(v)] = v
        return [tokens_to_words(e["offset_mapping"], v, ws) for e, v, ws in zip(encs, tok_vals, docs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lms", default=DEFAULT_LMS, help="comma-separated HF causal LMs, one channel each")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="")
    ap.add_argument("--cap", default="train:25000,dev:2500")
    a = ap.parse_args()

    from provseg.data.torch_data import select_records
    todo = select_records(a.records, a.splits, a.limit, a.cap)
    docs = [r["words"] for r in todo]
    cols = []
    for name in a.lms.split(","):
        lm = LogProbLM(name, a.max_len)
        vals = []
        for i in tqdm(range(0, len(docs), 256), desc=name, mininterval=30):
            vals += lm.doc_features(docs[i:i + 256], a.bs)
        cols.append(vals)
        del lm
        torch.cuda.empty_cache()
    feats = {r["meta"]["uid"]: np.stack([c[k] for c in cols], 1).astype(np.float32) for k, r in enumerate(todo)}
    np.savez_compressed(a.out, **feats)
    print(f"wrote {len(feats)} docs x {len(cols)} channels -> {a.out}")


if __name__ == "__main__":
    main()
