import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from provseg.conformal.selective import SelectiveCalibrator  # noqa: E402
from provseg.data.diff_provenance import chain, edit_ratio  # noqa: E402
from provseg.data.schema import char_labels_to_words  # noqa: E402
from provseg.eval.metrics import boundary_mae, span_f1  # noqa: E402
from provseg.models.provseg import LinearChainCRF  # noqa: E402


def test_chain_h_a_h():
    v0 = "the cat sat on the mat today"
    v1 = "the cat quietly sat on the mat today"          # AI inserts "quietly"
    v2 = "the cat quietly sat on a mat today"            # human replaces "the"->"a"
    w, l = chain([v0, v1, v2], [0, 1, 0])
    assert w == v2.split()
    assert l == [0, 0, 1, 0, 0, 0, 0, 0]
    assert abs(edit_ratio(v0, v1) - 1 / 8) < 1e-9


def test_char_labels():
    w, l = char_labels_to_words("ab cd  ef", [0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert w == ["ab", "cd", "ef"] and l == [0, 1, 1]


def test_crf_marginals_and_nll_bruteforce():
    torch.manual_seed(0)
    crf = LinearChainCRF(2)
    with torch.no_grad():
        crf.trans.normal_()
        crf.start.normal_()
        crf.end.normal_()
    T = 5
    em = torch.randn(1, T, 2)
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    L = 4
    scores = {}
    for y in itertools.product([0, 1], repeat=L):
        s = crf.start[y[0]] + em[0, 0, y[0]]
        for t in range(1, L):
            s = s + crf.trans[y[t - 1], y[t]] + em[0, t, y[t]]
        scores[y] = s + crf.end[y[-1]]
    logZ = torch.logsumexp(torch.stack(list(scores.values())), 0)
    y = (1, 0, 0, 1)
    tags = torch.tensor([list(y) + [0]])
    assert torch.allclose(crf.nll(em, tags, mask)[0], logZ - scores[y], atol=1e-4)
    marg = crf.marginals(em, mask)[0]
    for t in range(L):
        p1 = sum(torch.exp(s - logZ) for k, s in scores.items() if k[t] == 1)
        assert torch.allclose(marg[t, 1], p1, atol=1e-4)
    best = max(scores, key=lambda k: scores[k].item())
    assert tuple(crf.viterbi(em, mask)[0, :L].tolist()) == best


def test_calibrator_controls_risk():
    rng = np.random.default_rng(0)
    def make(n):
        P, PR, G = [], [], []
        for _ in range(n):
            T = 50
            conf = rng.uniform(0.5, 1.0, T)
            g = rng.integers(0, 2, T)
            correct = rng.uniform(size=T) < conf          # calibrated confidences
            p = np.where(correct, g, 1 - g)
            pr = np.zeros((T, 2))
            pr[np.arange(T), p] = conf
            pr[np.arange(T), 1 - p] = 1 - conf
            P.append(p); PR.append(pr); G.append(g)
        return P, PR, G
    cal = SelectiveCalibrator(alpha=0.1, delta=0.1, span_mode=False).fit(*make(400))
    P, PR, G = make(2000)
    risks = []
    for p, pr, g in zip(P, PR, G):
        out = cal.apply(p, pr)
        k = out != 2
        if k.any():
            risks.append((out[k] != g[k]).mean())
    assert cal.lam < 1.0 and np.mean(risks) <= 0.1


def test_metrics():
    g = [np.array([0, 0, 1, 1, 1])]
    assert span_f1(g, [np.array([0, 0, 1, 1, 1])]) == 1.0
    assert boundary_mae(g, [np.array([0, 1, 1, 1, 1])]) == 1.0


def test_word_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from provseg.eval.metrics import word_auroc
    rng = np.random.default_rng(0)
    G = [rng.integers(0, 2, 20) for _ in range(5)]
    G[0][:2] = -1
    PR = [np.stack([1 - p, p], -1) for p in (rng.uniform(size=20) for _ in range(5))]
    g = np.concatenate(G); s = np.concatenate([pr[:, 1] for pr in PR]); m = g >= 0
    assert abs(word_auroc(G, PR) - roc_auc_score(g[m], s[m])) < 1e-12


def test_cap_subset_deterministic_and_nested():
    from provseg.data.torch_data import cap_subset
    recs = [{"meta": {"uid": f"d:{i}"}} for i in range(100)]
    a = cap_subset(recs, 30)
    b = cap_subset(list(reversed(recs)), 30)
    assert len(a) == 30 and {r["meta"]["uid"] for r in a} == {r["meta"]["uid"] for r in b}
    assert {r["meta"]["uid"] for r in cap_subset(recs, 10)} <= {r["meta"]["uid"] for r in a}


def test_featurestore_handles_inf_and_nan(tmp_path):
    from provseg.data.torch_data import FeatureStore
    a = np.array([[1.0, np.inf], [np.nan, 2.0], [3.0, -np.inf], [1e30, 4.0]], dtype=np.float32)
    np.savez(tmp_path / "f.npz", **{"d:0": a})
    fs = FeatureStore([str(tmp_path / "f.npz")]).fit_norm(["d:0"])
    x = fs.get("d:0", 4)
    assert np.isfinite(x).all() and np.abs(x).max() <= 10
