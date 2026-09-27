"""Gaussian-mixture pose posterior on SE(3) and its condensation (Eq. 11--12).

Each component is a tangent-space Gaussian centred at its refined mode
``T_k``: a pose ``T`` has component coordinates ``xi = log(T T_k^{-1})`` (left
perturbation) and density ``N(xi; 0, Sigma_k)``.  The mixture is

    p(T | I_q, M) = sum_k pi_k N(xi_k(T); 0, Sigma_k)

For a downstream EKF/UKF the mixture is condensed to a single Gaussian by
moment matching in a common chart (Eq. after 12):

    mu = sum_k pi_k xi_k,   Sigma = sum_k pi_k (Sigma_k + (xi_k - mu)(xi_k - mu)^T)

Chart curvature is neglected when transporting component covariances to the
common chart, which is accurate when the modes are within a few degrees /
centimetres of each other -- the regime after refinement and fidelity pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from . import se3

_LOG_2PI = 1.8378770664093453


@dataclass
class PoseGaussian:
    """Tangent-space Gaussian on SE(3): xi = log(T T_mode^{-1}) ~ N(0, Sigma)."""

    T_mode: torch.Tensor   # (4, 4)
    Sigma: torch.Tensor    # (6, 6)

    def log_prob(self, T: torch.Tensor) -> torch.Tensor:
        xi = se3.relative_tangent(self.T_mode, T)
        L = torch.linalg.cholesky(self.Sigma)
        alpha = torch.cholesky_solve(xi[:, None], L)[:, 0]
        maha = xi @ alpha
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        return -0.5 * (maha + logdet + 6 * _LOG_2PI)

    def mahalanobis_sq(self, T: torch.Tensor) -> torch.Tensor:
        xi = se3.relative_tangent(self.T_mode, T)
        return xi @ torch.linalg.solve(self.Sigma, xi)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        L = torch.linalg.cholesky(self.Sigma)
        eps = torch.randn(n, 6, dtype=self.Sigma.dtype, device=self.Sigma.device,
                          generator=generator)
        xi = eps @ L.T
        return se3.se3_exp(xi) @ self.T_mode

    @property
    def translation_sigma(self) -> torch.Tensor:
        """Marginal translation std (m), a scalar summary used for AUSE."""
        return torch.sqrt(torch.diagonal(self.Sigma)[:3].sum())

    @property
    def rotation_sigma(self) -> torch.Tensor:
        """Marginal rotation std (rad)."""
        return torch.sqrt(torch.diagonal(self.Sigma)[3:].sum())


def _project_pd(Sigma: torch.Tensor, rel_floor: float = 1e-6) -> torch.Tensor:
    """Nearest symmetric positive-definite matrix (eigenvalue floor).

    A moment-matched covariance is PSD in exact arithmetic, but a component
    whose Laplace Hessian was nearly singular can leave it indefinite in
    float32, which then yields a negative Mahalanobis distance and a NaN
    log-determinant (observed on about one query in 700). Flooring the
    spectrum at rel_floor * max eigenvalue restores a valid Gaussian
    without changing any well-conditioned covariance. The floor must sit
    above float32 resolution (about 1e-7 of the largest eigenvalue), or the
    reconstruction rounds a clamped eigenvalue negative again.
    """
    S = 0.5 * (Sigma + Sigma.T)
    evals, evecs = torch.linalg.eigh(S.double())
    floor = rel_floor * evals.max().clamp_min(1e-30)
    evals = evals.clamp_min(floor)
    return ((evecs * evals) @ evecs.T).to(Sigma.dtype)


@dataclass
class PoseMixture:
    """Weighted mixture of tangent-space Gaussians (Eq. 12)."""

    components: List[PoseGaussian]
    weights: torch.Tensor  # (K,), sums to 1

    def __post_init__(self):
        self.weights = self.weights / self.weights.sum()

    def log_prob(self, T: torch.Tensor) -> torch.Tensor:
        lp = torch.stack([c.log_prob(T) for c in self.components])
        return torch.logsumexp(torch.log(self.weights) + lp, dim=0)

    def nll(self, T_gt: torch.Tensor) -> float:
        return float(-self.log_prob(T_gt))

    @property
    def mode(self) -> PoseGaussian:
        """Highest-weight component."""
        return self.components[int(torch.argmax(self.weights))]

    def condense(self, at_mode: bool = False) -> PoseGaussian:
        """Moment-matched single Gaussian for a downstream Bayesian filter.

        ``at_mode`` keeps the moment-matched covariance -- which includes the
        between-component spread, so it grows when the modes disagree -- but
        centres it on the highest-weight component instead of on the mixture
        mean. For a multi-modal posterior the mean lies between the modes and
        is therefore a pose the model considers unlikely; on Stairs, where the
        modes sit one stair tread apart, reporting the mean rather than the
        mode roughly doubles the median error. The mean remains the correct
        choice for a filter that will itself re-linearize, so both are
        available.
        """
        if len(self.components) == 1:
            c0 = self.components[0]
            return PoseGaussian(T_mode=c0.T_mode, Sigma=_project_pd(c0.Sigma))
        T_ref = self.mode.T_mode
        xis = torch.stack(
            [se3.relative_tangent(T_ref, c.T_mode) for c in self.components]
        )  # (K, 6)
        w = self.weights[:, None]
        mu_xi = (w * xis).sum(dim=0)
        T_mean = se3.apply_tangent(T_ref, mu_xi)
        # Re-center component modes in the chart at the matched mean.
        d = torch.stack(
            [se3.relative_tangent(T_mean, c.T_mode) for c in self.components]
        )
        Sigma = torch.zeros(6, 6, dtype=xis.dtype, device=xis.device)
        for k, c in enumerate(self.components):
            Sigma = Sigma + self.weights[k] * (c.Sigma + torch.outer(d[k], d[k]))
        Sigma = _project_pd(0.5 * (Sigma + Sigma.T))
        return PoseGaussian(T_mode=self.mode.T_mode if at_mode else T_mean,
                            Sigma=Sigma)
