"""Evidential Deep Learning (EDL) and Dempster--Shafer (DST) operations.

Implements Eqs. (3)--(5) and the EDL loss Eq. (7) of the draft:

* ``evidence_to_subjective_opinion`` -- Dirichlet evidence ``e`` to expected
  probability ``p``, belief mass ``b`` and vacuity ``u`` (Eq. 3), with
  ``u + sum_c b_c = 1``.
* ``dempster_fuse`` -- Dempster's combination rule for two opinions (Eq. 5).
* ``yager_fuse`` -- Yager's variant that routes the conflict mass into vacuity
  instead of renormalizing.  Dempster's rule *suppresses* conflict (two
  confident, contradictory opinions fuse to a confident compromise with low
  vacuity), which is the opposite of what the localization weighting needs for
  dynamic objects; Yager's rule makes persistent semantic conflict show up as
  vacuity.  Both are provided; the map builder records the accumulated
  conflict either way.
* ``edl_loss`` -- Bayes risk of the Brier score under the Dirichlet plus the
  KL regularizer toward the uniform Dirichlet (Eq. 7).
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

FuseRule = Literal["dempster", "yager"]


def evidence_to_subjective_opinion(
    evidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map non-negative Dirichlet evidence to (p_hat, belief, vacuity).

    Args:
        evidence: (..., C) non-negative evidence ``e_c``.

    Returns:
        p_hat:  (..., C) expected class probability ``alpha_c / S``.
        belief: (..., C) belief mass ``(alpha_c - 1) / S = e_c / S``.
        vacuity: (...,) ``u = C / S``.
    """
    if (evidence < 0).any():
        raise ValueError("evidence must be non-negative")
    C = evidence.shape[-1]
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    p_hat = alpha / S
    belief = evidence / S
    vacuity = (C / S).squeeze(-1)
    return p_hat, belief, vacuity


def opinion_to_probability(belief: torch.Tensor, vacuity: torch.Tensor) -> torch.Tensor:
    """``p_c = b_c + u / C`` -- the pignistic probability of an opinion."""
    C = belief.shape[-1]
    return belief + (vacuity / C)[..., None]


def _conflict(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """``eta = sum_{x != y} b1^x b2^y`` for (..., C) belief masses."""
    total = b1.sum(dim=-1) * b2.sum(dim=-1)
    agree = (b1 * b2).sum(dim=-1)
    return total - agree


def dempster_fuse(
    b1: torch.Tensor,
    u1: torch.Tensor,
    b2: torch.Tensor,
    u2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dempster's rule (Eq. 5).  Returns (belief, vacuity, conflict eta)."""
    eta = _conflict(b1, b2)
    denom = (1.0 - eta).clamp_min(1e-8)
    b = (b1 * b2 + b1 * u2[..., None] + b2 * u1[..., None]) / denom[..., None]
    u = (u1 * u2) / denom
    return b, u, eta


def yager_fuse(
    b1: torch.Tensor,
    u1: torch.Tensor,
    b2: torch.Tensor,
    u2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Yager's rule: same intersection masses, conflict goes to vacuity."""
    eta = _conflict(b1, b2)
    b = b1 * b2 + b1 * u2[..., None] + b2 * u1[..., None]
    u = u1 * u2 + eta
    return b, u, eta


def fuse_opinions(
    beliefs: torch.Tensor,
    vacuities: torch.Tensor,
    rule: FuseRule = "dempster",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold a set of opinions into one, accumulating mean pairwise conflict.

    Args:
        beliefs: (N, C) belief masses of the member observations.
        vacuities: (N,) vacuity masses.
        rule: "dempster" (Eq. 5 of the draft) or "yager".

    Returns:
        belief (C,), vacuity (scalar), mean conflict eta over the N-1 folds.
    """
    if beliefs.ndim != 2:
        raise ValueError("beliefs must be (N, C)")
    fuse = dempster_fuse if rule == "dempster" else yager_fuse
    b, u = beliefs[0], vacuities[0]
    etas = []
    for i in range(1, beliefs.shape[0]):
        b, u, eta = fuse(b, u, beliefs[i], vacuities[i])
        etas.append(eta)
    mean_eta = torch.stack(etas).mean() if etas else torch.zeros_like(u)
    return b, u, mean_eta


def softmax_opinions(
    logits: torch.Tensor, obs_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The non-evidential twin of Eq. 3 for the softmax ablation.

    ``p = softmax(logits)``; vacuity is the normalized entropy
    ``u = H(p) / ln C`` (deflated toward 1 when ``obs_scale`` < 1, mirroring
    the evidential observation scaling); belief ``b = (1 - u) p`` keeps the
    subjective-logic identity ``u + sum b = 1`` so every downstream consumer
    works unchanged.
    """
    p = torch.softmax(logits, dim=-1)
    C = p.shape[-1]
    H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=-1)
    u = H / torch.log(torch.tensor(float(C), device=p.device, dtype=p.dtype))
    if obs_scale is not None:
        u = 1.0 - (1.0 - u) * obs_scale.clamp(0.0, 1.0)
    return (1.0 - u)[..., None] * p, u


def edl_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """EDL loss (Eq. 7): Dirichlet Brier risk + KL(misleading evidence || uniform).

    Args:
        evidence: (B, C) non-negative evidence outputs of the head.
        targets: (B,) integer labels or (B, C) one-hot.
        kl_weight: annealing coefficient ``lambda_t`` (ramp 0 -> 1 in training).
    """
    C = evidence.shape[-1]
    if targets.ndim == 1:
        y = torch.nn.functional.one_hot(targets, C).to(evidence.dtype)
    else:
        y = targets.to(evidence.dtype)
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S
    risk = ((y - p) ** 2 + p * (1.0 - p) / (S + 1.0)).sum(dim=-1)

    # KL between Dirichlet of the misleading evidence and the uniform Dirichlet.
    alpha_tilde = y + (1.0 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    ones = torch.ones_like(alpha_tilde)
    kl = (
        torch.lgamma(S_tilde.squeeze(-1))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + torch.lgamma(ones).sum(dim=-1)
        - torch.lgamma(ones.sum(dim=-1))
        + (
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
        ).sum(dim=-1)
    )
    return (risk + kl_weight * kl).mean()


class EvidentialHead(torch.nn.Module):
    """Small MLP head mapping features to non-negative Dirichlet evidence.

    Both the map-building observations and the query instances are passed
    through the *same* head so that map and query vacuities are comparable
    (Sec. IV-B of the draft).
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns evidence (..., C) via softplus to keep it non-negative."""
        return torch.nn.functional.softplus(self.net(x))
