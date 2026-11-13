"""ProvSeg v2: Evidence-Bottleneck provenance tagger.

Every word decision is an *additive* sum of named evidence channels:

    z_i = b + sum_k  w_k * s_k(i)          (logit A vs H)

  lexical         per-word LLM statistics (ll, entropy, x-ppl, Fast-DetectGPT z, rank)
  representation  RepreGuard-style hidden-state projection
  context         contextual encoder state (DeBERTa)
  syntax          2-layer GCN over the spaCy dependency graph (PRDetect graph, word level)
  edit_trace      sigma(edit logit) * editor direction; the edit logit is trained with
                  *privileged* version supervision (which words were changed in the last
                  edit step) that is only available at training time.

contrib[i, k] = w_k s_k(i) is the explanation and is exact by construction
(z_i - b = sum_k contrib[i, k]). An ambiguity head u_i >= 0, fed by the channel
scores and their disagreement, is trained with a heteroscedastic loss and lowers
confidence where evidence conflicts. A linear-chain CRF over [-z/2, z/2] smooths
span boundaries. ``blackbox=True`` reproduces the v1 concat+linear model (ablation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

LEXICAL = ["ll_obs", "ent_obs", "ll_perf", "xent", "fdg_z", "log_rank"]
CHANNELS = ["lexical", "representation", "context", "syntax", "edit_trace"]


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))

    def _forward_alg(self, em, mask):  # em: B,T,K  mask: B,T (bool, left-aligned)
        alpha = self.start + em[:, 0]
        alphas = [alpha]
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t:t + 1], nxt, alpha)
            alphas.append(alpha)
        return torch.stack(alphas, 1), torch.logsumexp(alpha + self.end, dim=1)

    def _backward_alg(self, em, mask):
        B, T, K = em.shape
        beta = self.end.expand(B, K)
        betas = [beta]
        for t in range(T - 2, -1, -1):
            nxt = torch.logsumexp(self.trans + (em[:, t + 1] + beta).unsqueeze(1), dim=2)
            beta = torch.where(mask[:, t + 1:t + 2], nxt, beta)
            betas.append(beta)
        return torch.stack(betas[::-1], 1)

    def nll(self, em, tags, mask):
        _, logZ = self._forward_alg(em, mask)
        tags = tags.clamp(min=0)
        score = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        last = tags[:, 0]
        for t in range(1, em.size(1)):
            s = self.trans[last, tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s * mask[:, t]
            last = torch.where(mask[:, t], tags[:, t], last)
        score = score + self.end[last]
        return logZ - score

    def marginals(self, em, mask):
        alphas, logZ = self._forward_alg(em, mask)
        betas = self._backward_alg(em, mask)
        return torch.softmax(alphas + betas - logZ[:, None, None], dim=-1) * mask.unsqueeze(-1)

    def viterbi(self, em, mask):
        B, T, K = em.shape
        score = self.start + em[:, 0]
        bps = []
        for t in range(1, T):
            best, bp = (score.unsqueeze(2) + self.trans).max(1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t:t + 1], nxt, score)
            bps.append(torch.where(mask[:, t:t + 1], bp, torch.arange(K, device=em.device).expand(B, K)))
        score = score + self.end
        path = [score.argmax(1)]
        for bp in reversed(bps):
            path.append(bp.gather(1, path[-1].unsqueeze(1)).squeeze(1))
        return torch.stack(path[::-1], 1)


class DepGCN(nn.Module):
    """GCN with symmetric normalisation D^-1/2 (A+I) D^-1/2 over per-word dependency edges.

    edges: LongTensor (E, 3) of [batch, i, j] (undirected; both directions are added here).
    Same propagation rule as PRDetect's GCNConv layers, without document pooling.
    """

    def __init__(self, dim_in: int, dim: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dim_in if l == 0 else dim, dim) for l in range(layers)])
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def propagate(x, edges):
        if edges is None or edges.numel() == 0:
            return x
        B, W, D = x.shape
        # always propagate in float32: under bf16 autocast, rsqrt is promoted to fp32 and
        # index_add would otherwise mix dtypes (and degree sums are more stable in fp32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.float().reshape(B * W, D)
            b, i, j = edges[:, 0], edges[:, 1], edges[:, 2]
            src = torch.cat([b * W + i, b * W + j])
            dst = torch.cat([b * W + j, b * W + i])
            deg = torch.ones(B * W, device=x.device).index_add_(0, dst, torch.ones(len(dst), device=x.device))
            norm = (deg[src] * deg[dst]).rsqrt().unsqueeze(-1)
            out = flat / deg.unsqueeze(-1)                  # self loop term
            out = out.index_add(0, dst, flat[src] * norm)
        return out.reshape(B, W, D)

    def forward(self, x, edges):
        for k, lin in enumerate(self.lins):
            x = self.propagate(lin(x), edges)
            if k < len(self.lins) - 1:
                x = self.drop(F.gelu(x))
        return x


def _scorer(d_in, hidden=64):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1))


class LocalConv(nn.Module):
    """Small 1-D conv so a channel can use +-2 neighbouring words of its own signal."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 5, padding=2)

    def forward(self, x):
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


class ProvSeg(nn.Module):
    def __init__(self, encoder_name: str, feat_names: list[str] | None = None, num_tags: int = 2,
                 dropout: float = 0.1, use_crf: bool = True, use_syntax: bool = True,
                 use_edit: bool = True, use_ambiguity: bool = True, blackbox: bool = False,
                 channel_dropout: float = 0.1, feat_dim: int = 64):
        super().__init__()
        self.feat_names = list(feat_names or [])
        self.blackbox = blackbox
        self.use_ambiguity = use_ambiguity and not blackbox
        self.channel_dropout = channel_dropout
        self.enc = None if encoder_name == "none" else AutoModel.from_pretrained(encoder_name)
        hid = self.enc.config.hidden_size if self.enc is not None else 0
        self.drop = nn.Dropout(dropout)
        self.lex_idx = [self.feat_names.index(f) for f in LEXICAL if f in self.feat_names]
        self.rep_idx = [self.feat_names.index("rep")] if "rep" in self.feat_names else []

        if blackbox:  # v1: concat(encoder, projected features) -> linear emissions
            nf = len(self.feat_names)
            self.feat_proj = nn.Sequential(nn.Linear(nf, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim)) if nf else None
            self.feat_conv = LocalConv(feat_dim) if nf else None
            self.head = nn.Linear(hid + (feat_dim if nf else 0), num_tags)
            self.channels = []
            # SeqXGPT-style features-only tagger (conv + transformer) when there is no PLM encoder
            self.feat_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                feat_dim, 4, 4 * feat_dim, dropout, batch_first=True), 2) if (nf and not hid) else None
        else:
            self.channels = []
            if self.lex_idx:
                self.lex = nn.Sequential(nn.Linear(len(self.lex_idx), feat_dim), nn.GELU())
                self.lex_conv, self.lex_out = LocalConv(feat_dim), _scorer(feat_dim)
                self.channels.append("lexical")
            if self.rep_idx:
                self.rep = nn.Sequential(nn.Linear(1, 16), nn.GELU())
                self.rep_conv, self.rep_out = LocalConv(16), _scorer(16, 16)
                self.channels.append("representation")
            if hid:
                self.ctx_out = _scorer(hid, 128)
                self.channels.append("context")
            if use_syntax and hid:
                self.syn_in = nn.Linear(hid + (feat_dim if self.lex_idx else 0), 128)
                self.gcn = DepGCN(128, 128, 2, dropout)
                self.syn_out = _scorer(128)
                self.channels.append("syntax")
            if use_edit and hid:
                self.edit_head = _scorer(hid + (feat_dim if self.lex_idx else 0), 128)   # P(edited)
                self.editor_dir = _scorer(hid, 64)                                     # who edited (A>0)
                self.channels.append("edit_trace")
            K = len(self.channels)
            assert K > 0, "no evidence channel available"
            self.w = nn.Parameter(torch.ones(K))
            self.b = nn.Parameter(torch.zeros(1))
            if self.use_ambiguity:
                self.amb = nn.Sequential(nn.Linear(2 * K, 32), nn.GELU(), nn.Linear(32, 1))
        self.crf = LinearChainCRF(num_tags) if use_crf else None

    # ------------------------------------------------------------------ encoder
    def word_states(self, input_ids, attention_mask, word_index, W):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, S, D = h.shape
        idx = word_index.clamp(min=0)
        valid = (word_index >= 0).unsqueeze(-1).to(h.dtype)
        pooled = h.new_zeros(B, W, D).scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), h * valid)
        cnt = h.new_zeros(B, W, 1).scatter_add_(1, idx.unsqueeze(-1), valid)
        return pooled / cnt.clamp(min=1)

    # ------------------------------------------------------------------ evidence
    def evidence(self, batch):
        """Returns dict with z (B,W), contrib (B,W,K), scores (B,W,K), edit_logit (B,W) or None, u."""
        W = batch["word_mask"].size(1)
        x = self.word_states(batch["input_ids"], batch["attention_mask"], batch["word_index"], W) \
            if self.enc is not None else None
        feats = batch.get("feats")
        if self.blackbox:
            parts = [x] if x is not None else []
            if self.feat_proj is not None:
                f = self.feat_conv(self.feat_proj(feats))
                if self.feat_tf is not None:
                    f = self.feat_tf(f, src_key_padding_mask=~batch["word_mask"].bool())
                parts.append(f)
            em = self.head(self.drop(torch.cat(parts, -1)))
            return {"emissions": em, "z": em[..., 1] - em[..., 0], "contrib": None, "u": None, "edit_logit": None}

        scores, out = [], {"edit_logit": None}
        lex_h = None
        if "lexical" in self.channels:
            lex_h = self.lex_conv(self.lex(feats[..., self.lex_idx]))
            scores.append(self.lex_out(lex_h))
        if "representation" in self.channels:
            scores.append(self.rep_out(self.rep_conv(self.rep(feats[..., self.rep_idx]))))
        xd = self.drop(x) if x is not None else None
        if "context" in self.channels:
            scores.append(self.ctx_out(xd))
        side = torch.cat([xd, lex_h], -1) if lex_h is not None and xd is not None else xd
        if "syntax" in self.channels:
            scores.append(self.syn_out(self.gcn(self.syn_in(side), batch.get("edges"))))
        if "edit_trace" in self.channels:
            e = self.edit_head(side).squeeze(-1)
            out["edit_logit"] = e
            scores.append(torch.sigmoid(e).unsqueeze(-1) * self.editor_dir(xd))
        s = torch.cat(scores, -1).float()                                   # B,W,K
        if self.training and self.channel_dropout > 0:                      # each channel must stand alone
            keep = (torch.rand(s.shape[0], 1, s.shape[2], device=s.device) > self.channel_dropout).float()
            s = s * keep / (1 - self.channel_dropout)
        contrib = s * self.w
        z = self.b + contrib.sum(-1)
        out.update(scores=s, contrib=contrib, z=z)
        if self.use_ambiguity:
            dis = (contrib - contrib.mean(-1, keepdim=True)).abs()
            out["u"] = F.softplus(self.amb(torch.cat([contrib, dis], -1))).squeeze(-1)
        else:
            out["u"] = None
        out["emissions"] = torch.stack([-z / 2, z / 2], -1)
        return out

    # ------------------------------------------------------------------ training
    def forward(self, batch, lam_edit: float = 0.5, lam_amb: float = 0.2, n_mc: int = 8, **_):
        ev = self.evidence(batch)
        em = ev["emissions"].float()
        mask = batch["word_mask"].bool()
        labels = batch["labels"]
        lab_mask = mask & (labels >= 0)
        if self.crf is not None:
            tags = torch.where(labels >= 0, labels, em.argmax(-1))
            loss = self.crf.nll(em, tags, mask).sum() / mask.sum().clamp(min=1)   # per-word NLL
        else:
            loss = F.cross_entropy(em[lab_mask], labels[lab_mask])
        logs = {"crf": loss.detach()}
        ed = batch.get("edited")
        if ev.get("edit_logit") is not None and ed is not None and lam_edit > 0:
            m = mask & (ed >= 0)
            if m.any():
                le = F.binary_cross_entropy_with_logits(ev["edit_logit"][m].float(), ed[m].float())
                loss = loss + lam_edit * le
                logs["edit"] = le.detach()
        if ev.get("u") is not None and lam_amb > 0 and lab_mask.any():
            z, u = ev["z"][lab_mask], ev["u"][lab_mask]
            y = labels[lab_mask].float() * 2 - 1
            eps = torch.randn(n_mc, *z.shape, device=z.device)
            zz = z.unsqueeze(0) + u.sqrt().unsqueeze(0) * eps                 # heteroscedastic logits
            la = -(torch.logsumexp(F.logsigmoid(y * zz), 0) - torch.log(torch.tensor(float(n_mc)))).mean()
            loss = loss + lam_amb * la
            logs["amb"] = la.detach()
        return {"loss": loss, "logs": logs}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, batch):
        ev = self.evidence(batch)
        em = ev["emissions"].float()
        mask = batch["word_mask"].bool()
        if self.crf is not None:
            probs, path = self.crf.marginals(em, mask), self.crf.viterbi(em, mask)
        else:
            probs, path = em.softmax(-1) * mask.unsqueeze(-1), em.argmax(-1)
        out = {"pred": path.masked_fill(~mask, -1), "prob": probs}
        for k in ("contrib", "u", "edit_logit", "z"):
            if ev.get(k) is not None:
                out[k] = ev[k].float()
        return out
