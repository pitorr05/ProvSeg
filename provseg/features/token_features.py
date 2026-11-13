"""Per-word evidence features from an observer/performer LLM pair (one forward pass each).

Features (per word = mean over its sub-tokens; first token of a window has no
left context and is skipped):
  0 ll_obs      observer log p(x_t | x_<t)                 (likelihood, DetectGPT family)
  1 ent_obs     observer predictive entropy                (Binoculars numerator/denominator)
  2 ll_perf     performer log p(x_t | x_<t)
  3 xent        cross-entropy of performer dist under observer logits (Binoculars x-ppl term)
  4 fdg_z       Fast-DetectGPT per-token analytic z-score:
                (ll_obs - E_{q}[log p_obs]) / sqrt(Var_q[log p_obs]),  q = performer
  5 log_rank    log(1 + rank of x_t under observer)          (GLTR)
  6 rep         RepreGuard-style representation score: projection of the observer's
                hidden states on an AI-vs-human direction fitted on DetectRL pairs,
                averaged over layers (reimplemented; see fit_rep_direction).

Run on GPU; observer and performer must share a tokenizer (e.g. Qwen2.5-7B /
Qwen2.5-7B-Instruct, or Falcon-7B / Falcon-7B-Instruct as in Binoculars).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

FEAT_NAMES = ["ll_obs", "ent_obs", "ll_perf", "xent", "fdg_z", "log_rank", "rep"]


def _word_char_spans(words):
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    return " ".join(words), spans


class FeatureExtractor:
    def __init__(self, observer: str, performer: str, max_len: int = 1024,
                 rep_layers: int = 8, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(observer, use_fast=True)
        if not torch.cuda.is_available():
            dtype = torch.float32
        self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        dev2 = "cuda:1" if torch.cuda.device_count() > 1 else self.dev
        self.obs = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=dtype, device_map={"": self.dev}).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=dtype, device_map={"": dev2}).eval()
        self.max_len = max_len
        n = self.obs.config.num_hidden_layers
        self.rep_layers = list(range(n - rep_layers, n + 1))  # last k hidden states
        self.rep_dirs = None  # (L, D) unit vectors, sign-corrected so AI > human

    # ---------- RepreGuard-style direction ----------
    @torch.no_grad()
    def _mean_hidden(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len).to(self.dev)
        hs = self.obs(**enc, output_hidden_states=True).hidden_states
        return torch.stack([hs[l][0].float().mean(0) for l in self.rep_layers]).cpu()  # L,D

    def fit_rep_direction(self, ai_texts, human_texts):
        """First principal component of paired (AI - human) mean hidden-state differences per layer."""
        diffs = torch.stack([self._mean_hidden(a) - self._mean_hidden(h)
                             for a, h in tqdm(list(zip(ai_texts, human_texts)), desc="rep-dir")])  # N,L,D
        dirs = []
        for l in range(diffs.size(1)):
            X = diffs[:, l]
            mu = X.mean(0)
            _, _, V = torch.pca_lowrank(X - mu, q=1, center=False)
            d = V[:, 0]
            if (X @ d).mean() < 0:
                d = -d
            dirs.append(d / d.norm())
        self.rep_dirs = torch.stack(dirs)
        return self.rep_dirs

    # ---------- per-token features ----------
    @torch.no_grad()
    def _window(self, ids):
        ids = ids.unsqueeze(0)
        o = self.obs(input_ids=ids.to(self.dev), output_hidden_states=self.rep_dirs is not None)
        lo = o.logits[0, :-1].float()
        lp = self.perf(input_ids=ids.to(self.perf.device)).logits[0, :-1].float().to(lo.device)
        tgt = ids[0, 1:].to(lo.device)
        logp_o = lo.log_softmax(-1)
        logp_p = lp.log_softmax(-1)
        p_o, p_p = logp_o.exp(), logp_p.exp()
        ll_obs = logp_o.gather(1, tgt[:, None])[:, 0]
        ent = -(p_o * logp_o).sum(-1)
        ll_perf = logp_p.gather(1, tgt[:, None])[:, 0]
        xent = -(p_p * logp_o).sum(-1)
        mean_ref = (p_p * logp_o).sum(-1)
        var_ref = (p_p * logp_o.square()).sum(-1) - mean_ref.square()
        fdg = (ll_obs - mean_ref) / var_ref.clamp(min=1e-6).sqrt()
        rank = (logp_o > ll_obs[:, None]).sum(-1).float()
        feats = [ll_obs, ent, ll_perf, xent, fdg, torch.log1p(rank)]
        if self.rep_dirs is not None:
            hs = torch.stack([o.hidden_states[l][0, 1:].float() for l in self.rep_layers])  # L,T-1,D
            rep = torch.einsum("ltd,ld->lt", hs, self.rep_dirs.to(hs.device)).mean(0)
        else:
            rep = torch.zeros_like(ll_obs)
        feats.append(rep)
        return torch.stack(feats, -1).cpu().numpy()  # (T-1, F) for tokens 1..T-1

    def word_features(self, words):
        text, wspans = _word_char_spans(words)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = torch.tensor(enc["input_ids"])
        offs = enc["offset_mapping"]
        F = len(FEAT_NAMES)
        tok_feat = np.full((len(ids), F), np.nan, dtype=np.float32)
        for s in range(0, len(ids), self.max_len):
            chunk = ids[s:s + self.max_len]
            if len(chunk) < 2:
                continue
            tok_feat[s + 1:s + len(chunk)] = self._window(chunk)
        # token -> word by char offsets
        out = np.zeros((len(words), F), dtype=np.float32)
        cnt = np.zeros(len(words))
        wi = 0
        for t, (a, b) in enumerate(offs):
            if b <= a or np.isnan(tok_feat[t, 0]):
                continue
            while wi < len(wspans) and wspans[wi][1] <= a:
                wi += 1
            if wi >= len(wspans):
                break
            out[wi] += tok_feat[t]
            cnt[wi] += 1
        out[cnt > 0] /= cnt[cnt > 0, None]
        out[cnt == 0] = np.nan  # no left context / unmapped -> imputed to the train mean later
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="jsonl of provseg Records")
    ap.add_argument("--out", required=True, help=".npz output: one (n_words, F) array per record")
    ap.add_argument("--observer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--performer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--rep_train", default="", help="DetectRL train json for the rep direction")
    ap.add_argument("--rep_n", type=int, default=512)
    ap.add_argument("--rep_dir_cache", default="")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", default="", help="only these splits (comma); default all")
    ap.add_argument("--cap", default="train:25000,dev:2500",
                    help="per-split max docs, same uid-hash subset as train.py (empty = no cap)")
    a = ap.parse_args()

    fx = FeatureExtractor(a.observer, a.performer, a.max_len)
    if a.rep_dir_cache and os.path.exists(a.rep_dir_cache):
        fx.rep_dirs = torch.load(a.rep_dir_cache)
    elif a.rep_train:
        data = []
        for p in a.rep_train.split(","):
            data += json.load(open(p, encoding="utf8"))
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(data))[: a.rep_n]
        fx.fit_rep_direction([data[i]["direct_prompt"] for i in idx], [data[i]["human_text"] for i in idx])
        if a.rep_dir_cache:
            torch.save(fx.rep_dirs, a.rep_dir_cache)

    from provseg.data.torch_data import cap_subset
    keep = set(a.splits.split(",")) if a.splits else None
    caps = dict((k, int(v)) for k, v in (x.split(":") for x in a.cap.split(",") if x)) if a.cap else {}
    by_split = {}
    with open(a.records, encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if keep and r["split"] not in keep:
                continue
            by_split.setdefault(r["split"], []).append(r)
    # --limit is per split and in file order, exactly like train.py's load_records(limit=...)
    todo = [r for s, rs in by_split.items() for r in (rs[: a.limit] if a.limit else cap_subset(rs, caps.get(s, 0)))]
    feats = {}
    for r in tqdm(todo, desc="features", mininterval=60):
        feats[r["meta"]["uid"]] = fx.word_features(r["words"])
    np.savez_compressed(a.out, **feats)


if __name__ == "__main__":
    main()
