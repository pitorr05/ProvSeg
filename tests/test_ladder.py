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


def test_segment_starts_and_boundary_fusion():
    from provseg.models.seqx import segment_starts
    tags = torch.tensor([[0, 1, 2, 4, 6, 3, -1]])          # B-h M-h E-h B-AI E-AI S-h pad
    assert segment_starts(tags).tolist() == [[1, 0, 0, 1, 0, 1, -1]]
    torch.manual_seed(0)
    m = SeqXNet(ID2, seq_len=16, n_channels=3, boundary=True).eval()
    x, mask = torch.randn(2, 16, 3), torch.ones(2, 16, dtype=torch.bool)
    lo = m(x, mask)
    assert torch.isfinite(lo).all() and m.last_boundary.shape == (2, 16)
    phi, mf, me = channel_shapley(m, x, mask)               # explanation still exact with the curator
    assert torch.allclose(phi.sum(-1), mf - me, atol=1e-4)


def test_context_encoder_runs_outside_autocast():
    m = SeqXNet(ID2, seq_len=8, n_channels=2).eval()
    seen = {}

    class FakeEnc(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            seen["autocast"] = torch.is_autocast_enabled("cpu")
            h = torch.randn(*input_ids.shape, 4)
            h[attention_mask == 0] = float("nan")           # what DeBERTa-v3 emits at padding in bf16
            return type("O", (), {"last_hidden_state": h})()

    m.enc, m.ctx_proj = FakeEnc(), torch.nn.Linear(4, 3)
    ids = torch.ones(1, 5, dtype=torch.long)
    att = torch.tensor([[1, 1, 1, 0, 0]])
    widx = torch.tensor([[0, 1, 1, -1, -1]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m.context_states(ids, att, widx, 2)
    assert seen["autocast"] is False and torch.isfinite(out).all()


def test_bonferroni_calibrator_certifies_partial_coverage():
    from provseg.conformal.selective import SelectiveCalibrator
    rng = np.random.default_rng(0)

    def docs(n):
        P, Pr, G = [], [], []
        for _ in range(n):
            conf = rng.uniform(0.5, 1.0, 80)
            gold = rng.integers(0, 2, 80)
            wrong = rng.random(80) > conf            # calibrated: error prob = 1 - conf (~25% overall)
            pred = np.where(wrong, 1 - gold, gold)
            prob = np.stack([np.where(pred == 0, conf, 1 - conf), np.where(pred == 1, conf, 1 - conf)], 1)
            P.append(pred); Pr.append(prob); G.append(gold)
        return P, Pr, G
    c = SelectiveCalibrator(alpha=0.1, delta=0.1, span_mode=False).fit(*docs(300))
    P, Pr, G = docs(300)
    out = np.concatenate([c.apply(p, pr) for p, pr in zip(P, Pr)])
    g = np.concatenate(G)
    ans = out != 2
    assert 0.05 < ans.mean() < 1.0
    assert (out[ans] != g[ans]).mean() <= 0.1


def test_shapley_leave_one_out():
    torch.manual_seed(1)
    m = SeqXNet(ID2, seq_len=8, n_channels=2).eval()
    x, mask = torch.randn(1, 8, 2), torch.ones(1, 8, dtype=torch.bool)
    phi, mf, me, loo = channel_shapley(m, x, mask, return_loo=True)
    keep = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(loo[..., 0], class_margin(m(x, mask, None, keep)), atol=1e-5)


def test_window_diff_and_bootstrap_counts():
    from provseg.eval.metrics import window_diff
    from scripts.significance import doc_counts, macro_f1_from_counts
    g = [np.array([0] * 10 + [1] * 10)]
    assert window_diff(g, g) == 0.0
    assert window_diff(g, [np.array([0] * 5 + [1] * 15)]) > 0
    p = np.array([0] * 12 + [1] * 8)
    from sklearn.metrics import f1_score
    assert abs(macro_f1_from_counts(doc_counts(g[0], p)) - f1_score(g[0], p, average="macro")) < 1e-9


def test_wcp_finds_single_change_point():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from baseline_wcp import segment_doc, weighted_dp
    y = np.r_[np.zeros(6), np.ones(6) * 3] + np.random.default_rng(0).normal(0, 0.1, 12)
    res = weighted_dp(y, np.ones(12), 3)
    assert res[1][1] == [6]
    words = (["w"] * 4 + ["end."]) * 6
    z = np.r_[np.zeros(15), np.ones(15) * 4] + np.random.default_rng(1).normal(0, 0.3, 30)
    s = segment_doc(z, words, power=2.0)
    assert s[:15].max() < s[15:].min()


def test_markov_layer_smooths_isolated_flips():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from baseline_markov import MRFCalibration, adjacency
    m = MRFCalibration(t=10, filter_pos=0)
    with torch.no_grad():
        m.w1.fill_(1.0); m.w11.fill_(1.0); m.w_prime1.fill_(0.0); m.w_prime11.fill_(0.0)
    s = torch.tensor([[0.9] * 5 + [0.2] + [0.9] * 5])          # one noisy human-looking word inside AI text
    x = torch.stack([1 - s, s], -1)
    q = m.calibrated(x, adjacency(11, 0), torch.ones(1, 11))
    assert q[0, 5] > 0.5 > 0.2                                  # neighbours pull it back to "machine"
