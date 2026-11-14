import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from provseg.data.diff_provenance import chain_edits  # noqa: E402
from extras.models.provseg import DepGCN, ProvSeg  # noqa: E402


def test_chain_edits():
    v0 = "the cat sat on the mat"
    v1 = "the cat quietly sat on the mat"            # AI inserts
    v2 = "the cat quietly sat on a mat"              # human replaces the->a
    w, lab, ed = chain_edits([v0, v1, v2], [0, 1, 0])
    assert lab == [0, 0, 1, 0, 0, 0, 0]
    assert ed == [0, 0, 0, 0, 0, 1, 0]               # only the last step's change is "edited"


def test_depgcn_matches_dense():
    torch.manual_seed(0)
    B, W, D = 2, 5, 3
    x = torch.randn(B, W, D)
    edges = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 3, 4]])
    out = DepGCN.propagate(x, edges)
    for b in range(B):
        A = torch.eye(W)
        for e in edges[edges[:, 0] == b]:
            A[e[1], e[2]] = A[e[2], e[1]] = 1
        d = A.sum(1).rsqrt()
        ref = (d[:, None] * A * d[None, :]) @ x[b]
        assert torch.allclose(out[b], ref, atol=1e-5)


def _toy_model():
    torch.manual_seed(0)
    names = ["ll_obs", "ent_obs", "ll_perf", "xent", "fdg_z", "log_rank", "rep"]
    return ProvSeg("none", names, use_syntax=False, use_edit=False).eval(), names


def test_additive_explanation_is_exact():
    m, names = _toy_model()
    B, W = 2, 7
    batch = {"feats": torch.randn(B, W, len(names)), "word_mask": torch.ones(B, W, dtype=torch.bool),
             "labels": torch.randint(0, 2, (B, W))}
    ev = m.evidence(batch)
    assert m.channels == ["lexical", "representation"]
    assert torch.allclose(ev["z"], m.b + ev["contrib"].sum(-1), atol=1e-5)
    # deleting channel k shifts z by exactly contrib_k
    with torch.no_grad():
        w = m.w.clone()
        m.w[0] = 0
        z_del = m.evidence(batch)["z"]
        m.w.copy_(w)
    assert torch.allclose(ev["z"] - z_del, ev["contrib"][..., 0], atol=1e-5)


def test_forward_losses_and_predict():
    m, names = _toy_model()
    m.train()
    B, W = 3, 9
    batch = {"feats": torch.randn(B, W, len(names)), "word_mask": torch.ones(B, W, dtype=torch.bool),
             "labels": torch.randint(0, 2, (B, W)), "edited": torch.randint(0, 2, (B, W))}
    o = m(batch)
    o["loss"].backward()
    assert torch.isfinite(o["loss"]) and "amb" in o["logs"]
    m.eval()
    p = m.predict(batch)
    assert p["pred"].shape == (B, W) and p["u"].min() >= 0
    assert np.allclose(p["prob"].sum(-1).numpy(), 1, atol=1e-4)
