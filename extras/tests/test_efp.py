import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extras.models.efp import EFProvSeg  # noqa: E402

NAMES = ["ll_obs", "ent_obs", "ll_perf", "xent", "fdg_z", "log_rank", "rep"]


def _batch(B=3, W=9, edited=True):
    torch.manual_seed(1)
    b = {"feats": torch.randn(B, W, len(NAMES)), "word_mask": torch.ones(B, W, dtype=torch.bool),
         "labels": torch.randint(0, 2, (B, W))}
    if edited:
        b["edited"] = torch.randint(0, 2, (B, W))
    return b


def test_mixture_limits_and_variance_decomposition():
    pi, rho, beta = torch.rand(100), torch.rand(100), torch.rand(100)
    p, ue, ued, ub = EFProvSeg.mix(pi, rho, beta)
    assert torch.allclose(p, pi * rho + (1 - pi) * beta)
    # law of total variance: Bernoulli variance splits exactly into the three sources
    assert torch.allclose(p * (1 - p), ue + ued + ub, atol=1e-6)
    one, zero = torch.ones(100), torch.zeros(100)
    assert torch.allclose(EFProvSeg.mix(one, rho, beta)[0], rho)
    assert torch.allclose(EFProvSeg.mix(zero, rho, beta)[0], beta)


def test_factor_explanations_are_exact():
    m = EFProvSeg("none", NAMES, use_syntax=False).eval()
    fo = m.factors(_batch())
    for f in ("pi", "rho", "beta"):
        assert torch.allclose(fo[f"logit_{f}"], m.bias[f] + fo[f"contrib_{f}"].sum(-1), atol=1e-5)


def test_training_step_and_records_plugin():
    m = EFProvSeg("none", NAMES, use_syntax=False)
    b = _batch()
    o = m(b)
    o["loss"].backward()
    assert torch.isfinite(o["loss"]) and {"crf", "pi", "rho", "beta"} <= set(o["logs"])
    m.eval()
    txt, rec = m.predict(b), m.predict(b, use_records=True)
    # records track uses the observed edit indicator exactly
    assert torch.equal(rec["pi"], b["edited"].float())
    exp = b["edited"].float() * rec["rho"] + (1 - b["edited"].float()) * rec["beta"]
    assert torch.allclose(rec["p"], exp, atol=1e-6)
    assert not torch.equal(txt["pi"], rec["pi"])


def test_latent_pi_without_version_data():
    m = EFProvSeg("none", NAMES, use_syntax=False)
    o = m(_batch(edited=False))
    o["loss"].backward()          # pi is learned only through the CRF term
    assert set(o["logs"]) == {"crf"}
    assert m.scorers["pi"][0].weight.grad is not None
