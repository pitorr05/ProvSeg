"""Minimal stand-in for the three fastNLP symbols the official SeqXGPT code imports
(``MLP``, ``ConditionalRandomField``, ``allowed_transitions``).

It is only put on ``sys.path`` when the real ``fastNLP`` package cannot be imported
(see ``baselines/seqxgpt/__init__.py``). Behaviour matches fastNLP 1.0 for the way SeqXGPT
uses it: the CRF transition matrix is zeroed and never trained, so decoding is a Viterbi
search whose only effect is to forbid invalid BMES transitions
(B->M/E of the same label, M->M/E of the same label, E/S->B/S of any label;
start/end constraints are not used because ``include_start_end_trans=False``).
"""
from __future__ import annotations

import torch
from torch import nn


def _split(tag: str):
    tag = tag.lower()
    if "-" in tag:
        p, lab = tag.split("-", 1)
        return p, lab
    return tag, ""


def _allowed(from_tag: str, to_tag: str) -> bool:
    fp, fl = _split(from_tag)
    tp, tl = _split(to_tag)
    if fp in ("b", "m"):
        return tp in ("m", "e") and fl == tl
    if fp in ("e", "s"):
        return tp in ("b", "s")
    return False


def allowed_transitions(tag_vocab, encoding_type=None, include_start_end=False):
    """List of allowed (from_id, to_id) pairs for a BMES tag set given as {id: tag}."""
    id2tag = dict(tag_vocab)
    return [(i, j) for i, a in id2tag.items() for j, b in id2tag.items() if _allowed(a, b)]


class MLP(nn.Module):  # imported but unused by SeqXGPT
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("fastNLP MLP is not provided by the compatibility shim")


class ConditionalRandomField(nn.Module):
    def __init__(self, num_tags: int, include_start_end_trans: bool = False, allowed_transitions=None):
        super().__init__()
        self.num_tags = num_tags
        self.include_start_end_trans = include_start_end_trans
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        constrain = torch.zeros(num_tags, num_tags)
        if allowed_transitions is not None:
            constrain.fill_(-10000.0)
            for i, j in allowed_transitions:
                constrain[i, j] = 0.0
        self.register_buffer("_constrain", constrain, persistent=False)

    @torch.no_grad()
    def viterbi_decode(self, logits: torch.Tensor, mask: torch.Tensor, unpad: bool = False):
        """logits (B, L, T); mask (B, L) left-contiguous. Returns (paths (B, L) long, best scores (B,))."""
        B, L, T = logits.shape
        logits = logits.float()
        trans = (self._constrain + self.trans_m.data).float().to(logits.device)
        mask = mask.bool()
        score = logits[:, 0]                                     # (B, T)
        back = torch.zeros(B, L, T, dtype=torch.long, device=logits.device)
        for t in range(1, L):
            cand = score.unsqueeze(2) + trans.unsqueeze(0)       # (B, from, to)
            best, idx = cand.max(1)
            new = best + logits[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, new, score)
            back[:, t] = torch.where(m, idx, torch.arange(T, device=logits.device).expand(B, T))
        best_score, last = score.max(1)
        paths = torch.zeros(B, L, dtype=torch.long, device=logits.device)
        cur = last
        for t in range(L - 1, -1, -1):
            paths[:, t] = cur
            if t > 0:
                cur = back[:, t].gather(1, cur.unsqueeze(1)).squeeze(1)
        return paths, best_score
