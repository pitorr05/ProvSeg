import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baselines.seqxgpt import ModelWiseTransformerClassifier  # noqa: E402
from baselines.seqxgpt.official.seqxgpt_model import ConditionalRandomField, allowed_transitions  # noqa: E402
from provseg.eval.metrics import sentence_metrics, sentence_spans  # noqa: E402
from provseg.features.llm_logprob import tokens_to_words  # noqa: E402
from provseg.models.seqx import (SeqXNet, bmes_id2label, channel_shapley, class_margin,  # noqa: E402
                                 class_probs, pool_words, to_bmes)

ID2 = bmes_id2label(["human", "AI"])


def test_bmes_tags():
    assert ID2 == {0: "B-human", 1: "M-human", 2: "E-human", 3: "S-human",
                   4: "B-AI", 5: "M-AI", 6: "E-AI", 7: "S-AI"}
    assert to_bmes([0, 0, 0, 1, 1, 0, -1, 1]) == [0, 1, 2, 4, 6, 3, -1, 7]


def test_constrained_viterbi_matches_brute_force():
    torch.manual_seed(0)
    crf = ConditionalRandomField(num_tags=8, allowed_transitions=allowed_transitions(ID2))
    crf.trans_m.data *= 0
    ok = set(allowed_transitions(ID2))
    L = 5
    logits = torch.randn(2, L, 8)
    mask = torch.tensor([[1] * L, [1, 1, 1, 0, 0]]).bool()
    paths, _ = crf.viterbi_decode(logits, mask)
    for b in range(2):
        n = int(mask[b].sum())
        best, arg = -1e9, None
        for seq in itertools.product(range(8), repeat=n):
            if all((seq[i], seq[i + 1]) in ok for i in range(n - 1)):
                s = sum(float(logits[b, i, t]) for i, t in enumerate(seq))
                if s > best:
                    best, arg = s, seq
        assert tuple(paths[b, :n].tolist()) == arg


def test_seqxnet_equals_official_seqxgpt():
    torch.manual_seed(0)
    L = 32
    off = ModelWiseTransformerClassifier(id2labels=ID2, seq_len=L).eval()
    ours = SeqXNet(ID2, seq_len=L, n_channels=4).eval()
    ours.load_state_dict(off.state_dict())
    x = torch.randn(3, L, 4)
    lab = torch.zeros(3, L, dtype=torch.long)
    lab[1, 20:] = -1
    o = off(x, lab)
    lo = ours(x, lab > -1)
    assert torch.allclose(o["logits"], lo, atol=1e-5)
    assert torch.equal(o["preds"], ours.decode(lo, lab > -1))


def test_pool_words_ignores_nan_padding():
    h = torch.randn(2, 6, 4)
    h[0, 4:] = float("nan")                     # padding states of an encoder in bf16
    widx = torch.tensor([[0, 0, 1, 2, -1, -1], [0, 1, 1, 2, 3, 3]])
    out = pool_words(h, widx, 4)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 0], h[0, :2].mean(0), atol=1e-6)
    assert torch.allclose(out[1, 3], h[1, 4:].mean(0), atol=1e-6)


def test_tokens_to_words_char_weighted():
    words = ["ab", "cde"]                         # text "ab cde"
    offs = [(0, 1), (1, 2), (2, 6)]              # 'a', 'b', ' cde'
    v = tokens_to_words(offs, [0.0, 2.0, 3.0], words)
    assert np.allclose(v, [1.0, 3.0])


def test_channel_shapley_efficiency_and_dropout_keep():
    torch.manual_seed(0)
    m = SeqXNet(ID2, seq_len=16, n_channels=3).eval()
    x = torch.randn(2, 16, 3)
    mask = torch.ones(2, 16, dtype=torch.bool)
    phi, mf, me = channel_shapley(m, x, mask)
    assert torch.allclose(phi.sum(-1), mf - me, atol=1e-4)
    assert torch.allclose(mf, class_margin(m(x, mask)), atol=1e-5)
    assert torch.allclose(class_probs(m(x, mask), 2).sum(-1), torch.ones(2, 16), atol=1e-5)


def test_sentence_protocol():
    ws = ["Hello", "world.", "This", "is", "AI!", "tail"]
    assert sentence_spans(ws) == [(0, 2), (2, 5), (5, 6)]
    g = [np.array([0, 0, 1, 1, 1, 1])]
    p = [np.array([0, 1, 1, 1, 0, 0])]
    m = sentence_metrics([ws], g, p)
    assert abs(m["sent_acc"] - 2 / 3) < 1e-9
