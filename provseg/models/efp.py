"""Edit-Factorized Provenance (EFP) — the ProvSeg main model.

Retained provenance is recursive: a word's author is the last editor if the word was
(re)written in the last edit step, otherwise it is inherited from the previous version:

    y_i = a_K            if e_i = 1
    y_i = y_i^(K-1)      if e_i = 0

EFP models exactly this structure instead of a direct classifier:

    P(y_i = A | x) = pi_i * rho_i + (1 - pi_i) * beta_i
      pi_i   = P(e_i = 1 | x)            was the word written in the last edit?
      rho_i  = P(a_K = A | x, e_i = 1)   who made that edit?
      beta_i = P(y^(K-1) = A | x, e_i=0) who wrote the retained base text?

Each factor is an *additive evidence bottleneck* over named channels
(LLM-likelihood, hidden representation, context, dependency syntax), so every
factor decision decomposes exactly into per-channel contributions.

By the law of total variance the predictive variance splits exactly into
    p(1-p) = pi(1-pi)(rho-beta)^2   [edit-uncertain]
           + pi rho(1-rho)          [editor-uncertain]
           + (1-pi) beta(1-beta)    [base-uncertain]
which tells *why* a word is Unknown.

Training: CRF NLL on log-odds(p) for every word; with privileged version
supervision where available: BCE(pi, e), BCE(rho, y | e=1), BCE(beta, y | e=0).
Datasets without versions train pi as a latent variable through the CRF term.
Records track: substitute the observed e for pi (exact plug-in, Prop. "records cannot hurt").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .provseg import LEXICAL, DepGCN, LinearChainCRF, LocalConv

FACTORS = ["pi", "rho", "beta"]
EPS = 1e-5


class EFProvSeg(nn.Module):
    def __init__(self, encoder_name: str, feat_names: list[str] | None = None, dropout: float = 0.1,
                 use_crf: bool = True, use_syntax: bool = True, use_edit_sup: bool = True,
                 channel_dropout: float = 0.1, feat_dim: int = 64, ch_dim: int = 128):
        super().__init__()
        self.feat_names = list(feat_names or [])
        self.use_edit_sup = use_edit_sup
        self.channel_dropout = channel_dropout
        self.enc = None if encoder_name == "none" else AutoModel.from_pretrained(encoder_name)
        hid = self.enc.config.hidden_size if self.enc is not None else 0
        self.drop = nn.Dropout(dropout)
        self.lex_idx = [self.feat_names.index(f) for f in LEXICAL if f in self.feat_names]
        self.rep_idx = [self.feat_names.index("rep")] if "rep" in self.feat_names else []

        # ---- channel trunks: each produces a hidden vector per word -------------------------
        self.channels, dims = [], []
        if self.lex_idx:
            self.lex = nn.Sequential(nn.Linear(len(self.lex_idx), feat_dim), nn.GELU(), LocalConv(feat_dim))
            self.channels.append("lexical"); dims.append(feat_dim)
        if self.rep_idx:
            self.rep = nn.Sequential(nn.Linear(1, 16), nn.GELU(), LocalConv(16))
            self.channels.append("representation"); dims.append(16)
        if hid:
            self.ctx = nn.Sequential(nn.Linear(hid, ch_dim), nn.GELU())
            self.channels.append("context"); dims.append(ch_dim)
        if use_syntax and hid:
            self.syn_in = nn.Linear(hid + (feat_dim if self.lex_idx else 0), ch_dim)
            self.gcn = DepGCN(ch_dim, ch_dim, 2, dropout)
            self.channels.append("syntax"); dims.append(ch_dim)
        assert self.channels, "no evidence channel"
        K = len(self.channels)
        # ---- one linear scorer per (factor, channel): additive bottleneck per factor -------
        self.scorers = nn.ModuleDict({f: nn.ModuleList([nn.Linear(d, 1) for d in dims]) for f in FACTORS})
        self.bias = nn.ParameterDict({f: nn.Parameter(torch.zeros(1)) for f in FACTORS})
        self.crf = LinearChainCRF(2) if use_crf else None
        self.K = K

    def word_states(self, input_ids, attention_mask, word_index, W):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, S, D = h.shape
        idx = word_index.clamp(min=0)
        valid = (word_index >= 0).unsqueeze(-1).to(h.dtype)
        pooled = h.new_zeros(B, W, D).scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), h * valid)
        cnt = h.new_zeros(B, W, 1).scatter_add_(1, idx.unsqueeze(-1), valid)
        return pooled / cnt.clamp(min=1)

    def channel_states(self, batch):
        W = batch["word_mask"].size(1)
        x = self.word_states(batch["input_ids"], batch["attention_mask"], batch["word_index"], W) \
            if self.enc is not None else None
        feats = batch.get("feats")
        H, lex_h = [], None
        if "lexical" in self.channels:
            lex_h = self.lex(feats[..., self.lex_idx])
            H.append(lex_h)
        if "representation" in self.channels:
            H.append(self.rep(feats[..., self.rep_idx]))
        xd = self.drop(x) if x is not None else None
        if "context" in self.channels:
            H.append(self.ctx(xd))
        if "syntax" in self.channels:
            side = torch.cat([xd, lex_h], -1) if lex_h is not None else xd
            H.append(self.gcn(self.syn_in(side), batch.get("edges")))
        return H

    def factors(self, batch):
        """Returns logits/probabilities of pi, rho, beta and contrib[f] (B,W,K)."""
        H = self.channel_states(batch)
        B, W = batch["word_mask"].shape
        keep = None
        if self.training and self.channel_dropout > 0:
            keep = (torch.rand(B, 1, self.K, device=H[0].device) > self.channel_dropout).float() / (1 - self.channel_dropout)
        out = {}
        for f in FACTORS:
            c = torch.cat([sc(h) for sc, h in zip(self.scorers[f], H)], -1).float()   # B,W,K
            if keep is not None:
                c = c * keep
            out[f"contrib_{f}"] = c
            out[f"logit_{f}"] = self.bias[f] + c.sum(-1)
            out[f] = torch.sigmoid(out[f"logit_{f}"])
        return out

    @staticmethod
    def mix(pi, rho, beta):
        p = pi * rho + (1 - pi) * beta
        u_edit = pi * (1 - pi) * (rho - beta) ** 2
        u_editor = pi * rho * (1 - rho)
        u_base = (1 - pi) * beta * (1 - beta)
        return p, u_edit, u_editor, u_base

    @staticmethod
    def emissions_from_p(p):
        p = p.clamp(EPS, 1 - EPS)
        lo = torch.log(p) - torch.log1p(-p)
        return torch.stack([-lo / 2, lo / 2], -1)

    def forward(self, batch, lam_edit: float = 0.5, lam_fac: float = 0.5, **_):
        fo = self.factors(batch)
        p, *_u = self.mix(fo["pi"], fo["rho"], fo["beta"])
        em = self.emissions_from_p(p)
        mask = batch["word_mask"].bool()
        y = batch["labels"]
        if self.crf is not None:
            tags = torch.where(y >= 0, y, em.argmax(-1))
            loss = self.crf.nll(em, tags, mask).sum() / mask.sum().clamp(min=1)
        else:
            m = mask & (y >= 0)
            pm, ym = p[m].clamp(EPS, 1 - EPS), y[m].float()     # manual BCE: autocast-safe
            loss = -(ym * pm.log() + (1 - ym) * (1 - pm).log()).mean()
        logs = {"crf": loss.detach()}
        e = batch.get("edited")
        if self.use_edit_sup and e is not None:
            me = mask & (e >= 0)
            if me.any():
                l_pi = F.binary_cross_entropy_with_logits(fo["logit_pi"][me], e[me].float())
                loss = loss + lam_edit * l_pi
                logs["pi"] = l_pi.detach()
            for f, cond in (("rho", 1), ("beta", 0)):
                mf = me & (e == cond) & (y >= 0)
                if mf.any():
                    lf = F.binary_cross_entropy_with_logits(fo[f"logit_{f}"][mf], y[mf].float())
                    loss = loss + lam_fac * lf
                    logs[f] = lf.detach()
        return {"loss": loss, "logs": logs}

    @torch.no_grad()
    def predict(self, batch, use_records: bool = False):
        fo = self.factors(batch)
        pi = fo["pi"]
        if use_records and batch.get("edited") is not None:
            e = batch["edited"]
            pi = torch.where(e >= 0, e.float(), pi)        # plug in the observed edit indicator
        p, u_edit, u_editor, u_base = self.mix(pi, fo["rho"], fo["beta"])
        em = self.emissions_from_p(p)
        mask = batch["word_mask"].bool()
        if self.crf is not None:
            prob, path = self.crf.marginals(em, mask), self.crf.viterbi(em, mask)
        else:
            prob, path = torch.stack([1 - p, p], -1) * mask.unsqueeze(-1), (p > 0.5).long()
        out = {"pred": path.masked_fill(~mask, -1), "prob": prob, "p": p, "pi": pi, "rho": fo["rho"],
               "beta": fo["beta"], "u_edit": u_edit, "u_editor": u_editor, "u_base": u_base}
        for f in FACTORS:
            out[f"contrib_{f}"] = fo[f"contrib_{f}"]
        return out
