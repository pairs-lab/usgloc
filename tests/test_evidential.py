import torch

from usgloc.evidential import (
    dempster_fuse,
    edl_loss,
    evidence_to_subjective_opinion,
    fuse_opinions,
    opinion_to_probability,
    yager_fuse,
)


def _opinion(evidence):
    e = torch.tensor(evidence, dtype=torch.float64)
    _, b, u = evidence_to_subjective_opinion(e[None])
    return b[0], u[0]


def test_opinion_normalization():
    torch.manual_seed(0)
    e = torch.rand(50, 7, dtype=torch.float64) * 10
    p, b, u = evidence_to_subjective_opinion(e)
    assert torch.allclose(b.sum(-1) + u, torch.ones(50, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(p.sum(-1), torch.ones(50, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(opinion_to_probability(b, u), p, atol=1e-12)


def test_vacuity_decreases_with_evidence():
    _, _, u_weak = evidence_to_subjective_opinion(torch.tensor([[0.5, 0.2, 0.1]]))
    _, _, u_strong = evidence_to_subjective_opinion(torch.tensor([[20.0, 0.2, 0.1]]))
    assert u_strong < u_weak


def test_dempster_agreement_concentrates():
    b1, u1 = _opinion([8.0, 0.1, 0.1])
    b2, u2 = _opinion([9.0, 0.2, 0.1])
    b, u, eta = dempster_fuse(b1, u1, b2, u2)
    assert u < min(u1, u2)          # agreement reduces vacuity
    assert b[0] > max(b1[0], b2[0])  # and concentrates belief
    assert b.sum() + u <= 1.0 + 1e-9


def test_dempster_vs_yager_on_conflict():
    # Two confident, contradictory opinions: Dempster renormalizes the
    # conflict away (low fused vacuity); Yager routes it into vacuity.
    b1, u1 = _opinion([15.0, 0.1, 0.1])
    b2, u2 = _opinion([0.1, 15.0, 0.1])
    _, u_d, eta_d = dempster_fuse(b1, u1, b2, u2)
    _, u_y, eta_y = yager_fuse(b1, u1, b2, u2)
    assert eta_d > 0.5 and torch.allclose(eta_d, eta_y)
    assert u_y > 0.5          # conflict shows up as vacuity
    assert u_d < 0.1          # Dempster suppresses it
    assert u_y > u_d


def test_fuse_opinions_accumulates_conflict():
    b = torch.stack([_opinion([15.0, 0.1, 0.1])[0], _opinion([0.1, 15.0, 0.1])[0]])
    u = torch.stack([_opinion([15.0, 0.1, 0.1])[1], _opinion([0.1, 15.0, 0.1])[1]])
    _, _, eta = fuse_opinions(b, u, rule="dempster")
    assert eta > 0.5


def test_edl_loss_prefers_correct_confident_evidence():
    targets = torch.tensor([0])
    e_good = torch.tensor([[10.0, 0.0, 0.0]])
    e_bad = torch.tensor([[0.0, 10.0, 0.0]])
    e_vacuous = torch.tensor([[0.0, 0.0, 0.0]])
    l_good = edl_loss(e_good, targets)
    l_bad = edl_loss(e_bad, targets)
    l_vac = edl_loss(e_vacuous, targets)
    assert l_good < l_vac < l_bad


def test_edl_loss_gradient_flows():
    e = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
    loss = edl_loss(e, torch.tensor([1]))
    loss.backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()
