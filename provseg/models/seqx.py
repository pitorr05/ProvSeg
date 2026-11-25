"""ProvSeg ladder model, built *on top of* the official SeqXGPT architecture.

SeqXGPT (EMNLP 2023): every evidence channel (one per-word score sequence) goes through the
same 1-D CNN, the channel outputs are concatenated, a sinusoidal position code is added,
a 2-layer Transformer encoder mixes the sequence and a linear layer emits BMES tag logits.
Training is token cross-entropy; decoding is a BMES-constrained Viterbi (CRF with zero
transitions). ``SeqXNet`` reproduces that computation exactly (``tests`` checks it against
the vendored official class with shared weights) and adds, one at a time:

  S1  evidence: any number of channels (contrastive Fast-DetectGPT / Binoculars evidence)
  S2  context:  word states of a pretrained encoder (DeBERTa-v3), projected and concatenated
  S2b Boundary Curator: a per-word "a new segment starts here" head (trained with BCE) whose
      log-probabilities are added to the B/S vs M/E tag scores before decoding, so evidence about
      authorship *changes* is modelled explicitly instead of only through per-word labels.
  S4  evidence dropout: whole channels are zeroed at random during training, so "channel
      absent" is in-distribution and exact channel-level Shapley values are meaningful.
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from baselines.seqxgpt import ConvFeatureExtractionModel
from baselines.seqxgpt.official.seqxgpt_model import ConditionalRandomField, allowed_transitions

BMES = ("B-", "M-", "E-", "S-")
CONV_LAYERS = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]   # SeqXGPT's feature_enc_layers


# ---------------------------------------------------------------- tags
def bmes_id2label(classes):
    """Same construction as SeqXGPT's construct_bmes_labels: class-major, B/M/E/S minor."""
    return {k * 4 + j: p + c for k, c in enumerate(classes) for j, p in enumerate(BMES)}


def segment_starts(tags):
    """BMES tag ids (-1 = pad) -> 1 where a segment starts (B-* or S-*), 0 elsewhere, -1 on padding."""
    return torch.where(tags < 0, -1, ((tags % 4 == 0) | (tags % 4 == 3)).long())


def to_bmes(cls_ids, n_classes=None):
    """Per-word class ids (-1 = ignore) -> BMES tag ids (class*4 + {0:B,1:M,2:E,3:S})."""
    out, i, n = [-1] * len(cls_ids), 0, len(cls_ids)
    while i < n:
        c = cls_ids[i]
        if c < 0:
            i += 1
            continue
        j = i
        while j < n and cls_ids[j] == c:
            j += 1
        if j - i == 1:
            out[i] = c * 4 + 3
        else:
            out[i] = c * 4
            for k in range(i + 1, j - 1):
                out[k] = c * 4 + 1
            out[j - 1] = c * 4 + 2
        i = j
    return out


def class_probs(logits, n_classes):
    """Tag logits (..., 4C) -> class probabilities (..., C) (sum over B/M/E/S)."""
    p = logits.float().softmax(-1)
    return p.view(*p.shape[:-1], n_classes, 4).sum(-1)


def class_margin(logits, pos: int = 1, n_classes: int = 2):
    """log P(class pos) - log P(rest): the per-word evidence score explained by Shapley values."""
    lp = logits.float().log_softmax(-1).view(*logits.shape[:-1], n_classes, 4).logsumexp(-1)
    rest = torch.cat([lp[..., :pos], lp[..., pos + 1:]], -1).logsumexp(-1)
    return lp[..., pos] - rest


def sinusoid(seq_len: int, dim: int) -> torch.Tensor:
    """Exactly SeqXGPT's position code (including its exponent convention), vectorised."""
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, dtype=torch.float32)[None]
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos / 10000 ** ((2 * i) / dim))
    pe[:, 1::2] = torch.cos(pos / 10000 ** ((2 * (i + 1)) / dim))
    return pe


def pool_words(h, word_index, W):
    """Mean of sub-token states per word. Padding states are *overwritten* (masked_fill), never
    multiplied by 0, because some encoders (DeBERTa-v3 under bf16) emit NaN at padding and
    NaN*0 = NaN would leak into word 0 through the clamped index."""
    B, S, D = h.shape
    valid = word_index >= 0
    h = torch.nan_to_num(h.float()).masked_fill(~valid.unsqueeze(-1), 0.0)
    idx = word_index.clamp(min=0)
    pooled = h.new_zeros(B, W, D).scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, D), h)
    cnt = h.new_zeros(B, W, 1).scatter_add_(1, idx.unsqueeze(-1), valid.unsqueeze(-1).float())
    return pooled / cnt.clamp(min=1)


# ---------------------------------------------------------------- model
class SeqXNet(nn.Module):
    """SeqXGPT backbone generalised to C evidence channels + an optional context encoder.

    With n_channels=4 and no context it has the same parameters/names and computes the same
    logits as ``ModelWiseTransformerClassifier`` from the official SeqXGPT code.
    """

    def __init__(self, id2labels, seq_len, n_channels=4, context=None, ctx_dim=256,
                 intermediate_size=512, num_layers=2, nhead=16, dropout_rate=0.1, boundary=False):
        super().__init__()
        self.n_channels, self.seq_len = n_channels, seq_len
        self.last_boundary = None
        self.conv = ConvFeatureExtractionModel(conv_layers=CONV_LAYERS, conv_dropout=0.0, conv_bias=False)
        self.enc = None
        emb = n_channels * 64
        if context:
            from transformers import AutoModel
            # fp32 master weights: transformers>=5 loads checkpoints in their stored dtype (fp16 for
            # DeBERTa-v3), and AdamW on fp16 weights (eps=1e-8 underflows) turns them into NaN
            self.enc = AutoModel.from_pretrained(context, torch_dtype=torch.float32)
            self.ctx_proj = nn.Linear(self.enc.config.hidden_size, ctx_dim)
            emb += ctx_dim
        self.embedding_size = emb
        self.encoder_layer = TransformerEncoderLayer(d_model=emb, nhead=nhead, dim_feedforward=intermediate_size,
                                                     dropout=dropout_rate, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.register_buffer("position_encoding", sinusoid(seq_len, emb), persistent=False)
        self.norm = nn.LayerNorm(emb)
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(emb, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0
        self.boundary = boundary
        if boundary:
            self.bnd = nn.Linear(emb, 1)
            start = torch.tensor([1.0 if j % 4 in (0, 3) else 0.0 for j in range(self.label_num)])  # B-*, S-*
            self.register_buffer("start_tags", start, persistent=False)

    @property
    def n_players(self):
        """Evidence groups that can be switched off: each channel, plus the context encoder."""
        return self.n_channels + (1 if self.enc is not None else 0)

    # -- context
    def context_states(self, input_ids, attention_mask, word_index, W):
        # the encoder always runs in fp32: DeBERTa-v3 under bf16 produces NaN at padding positions,
        # which poisons the *backward* pass even when the forward output is masked
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = pool_words(h, word_index, W)
        return self.ctx_proj(pooled)

    # -- head (everything after the expensive encoder; cheap enough to run 2^K times for Shapley)
    def head(self, x, mask, ctx=None, keep=None):
        """x (B, L, C) evidence; mask (B, L) bool valid words; ctx (B, L, ctx_dim) or None;
        keep (B, K) {0,1}: which evidence groups are present (None = all)."""
        B, L, C = x.shape
        if keep is not None:
            x = x * keep[:, None, :C].to(x.dtype)
            if ctx is not None:
                ctx = ctx * keep[:, None, C:C + 1].to(ctx.dtype)
        x = x.transpose(1, 2)
        outs = [self.conv(x[:, c:c + 1, :]).transpose(1, 2) for c in range(C)]
        if ctx is not None:
            outs.append(ctx)
        out = torch.cat(outs, dim=2)
        out = out + self.position_encoding[:L].to(out.dtype)
        out = self.norm(out)
        out = self.dropout(self.encoder(out, src_key_padding_mask=~mask))
        logits = self.classifier(out)
        if self.boundary:
            b = self.bnd(out).squeeze(-1).float()                  # logit of "segment starts here"
            self.last_boundary = b
            s = self.start_tags.to(logits.device)
            logits = logits.float() + F.logsigmoid(b)[..., None] * s + F.logsigmoid(-b)[..., None] * (1 - s)
        return logits

    def forward(self, x, mask, ctx=None, keep=None):
        return self.head(x, mask, ctx, keep)

    def decode(self, logits, mask):
        paths, _ = self.crf.viterbi_decode(logits=logits, mask=mask)
        return paths.masked_fill(~mask, -1)


def shapley_weights(K: int):
    """Shapley coefficient |S|!(K-|S|-1)!/K! for every coalition size."""
    return [math.factorial(s) * math.factorial(K - s - 1) / math.factorial(K) for s in range(K)]


@torch.no_grad()
def channel_shapley(model: SeqXNet, x, mask, ctx=None, pos: int = 1, n_classes: int = 2, return_loo: bool = False):
    """Exact Shapley values of the K evidence groups for the per-word margin
    log P(AI) - log P(human). Returns (phi (B, L, K), m_full (B, L), m_empty (B, L));
    efficiency holds exactly: phi.sum(-1) == m_full - m_empty."""
    K = model.n_players
    B = x.shape[0]
    val = {}
    for bits in itertools.product((0, 1), repeat=K):
        keep = torch.tensor(bits, device=x.device, dtype=x.dtype).expand(B, K)
        val[bits] = class_margin(model.head(x, mask, ctx, keep), pos, n_classes)
    w = shapley_weights(K)
    phi = torch.zeros(*val[(1,) * K].shape, K, device=x.device)
    for bits, v in val.items():
        s = sum(bits)
        for k in range(K):
            if bits[k] == 0:
                with_k = bits[:k] + (1,) + bits[k + 1:]
                phi[..., k] += w[s] * (val[with_k] - v)
    if return_loo:   # margin with evidence group k removed (all others kept): offline deletion tests
        loo = torch.stack([val[tuple(0 if j == k else 1 for j in range(K))] for k in range(K)], -1)
        return phi, val[(1,) * K], val[(0,) * K], loo
    return phi, val[(1,) * K], val[(0,) * K]
