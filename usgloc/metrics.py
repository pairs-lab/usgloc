"""Accuracy and calibration metrics (Sec. V of the draft).

Accuracy: median translation/rotation error and success rate.
Calibration (the paper's focus):

* ``coverage`` -- empirical frequency with which the Mahalanobis error
  ``d^2 = e^T Sigma^{-1} e`` falls inside the chi-square quantile of each
  nominal credibility level (68.3 / 95.4 / 99.7), as in KSE-Net.
* ``ause`` -- Area Under the Sparsification Error curve: rank queries by
  *predicted* uncertainty, remove the worst fraction, and compare the error
  decay against the oracle ranking by true error.
* ``coverage_ece`` -- expected calibration error over a grid of credibility
  levels (the regression analogue of classification ECE).
* ``nll`` is provided by :meth:`usgloc.posterior.PoseMixture.nll`.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from scipy.stats import chi2

from . import se3


def pose_errors(T_est: torch.Tensor, T_gt: torch.Tensor) -> Tuple[float, float]:
    """Translation error (m) and rotation error (deg) between two poses."""
    dT = T_est @ se3.se3_inverse(T_gt)
    t_err = float(torch.linalg.norm(dT[:3, 3]))
    r_err = float(se3.rotation_angle_deg(dT[:3, :3]))
    return t_err, r_err


def success_rate(
    t_errs: Sequence[float],
    r_errs: Sequence[float],
    t_thresh: float = 0.05,
    r_thresh: float = 5.0,
) -> float:
    """Fraction of queries within (t_thresh [m], r_thresh [deg])."""
    t = np.asarray(t_errs)
    r = np.asarray(r_errs)
    return float(np.mean((t <= t_thresh) & (r <= r_thresh)))


def mahalanobis_sq(
    T_est: torch.Tensor, Sigma: torch.Tensor, T_gt: torch.Tensor
) -> float:
    """d^2 = e^T Sigma^{-1} e with e = log(T_gt T_est^{-1}) in the tangent space."""
    e = se3.relative_tangent(T_est, T_gt)
    return float(e @ torch.linalg.solve(Sigma, e))


def coverage(
    d2: Sequence[float],
    dof: int = 6,
    levels: Sequence[float] = (0.683, 0.954, 0.997),
) -> dict:
    """Empirical coverage at each nominal credibility level.

    A calibrated covariance gives ``coverage[level] ~= level``; an
    overconfident one undershoots, an underconfident one overshoots.
    """
    d2 = np.asarray(d2)
    return {lv: float(np.mean(d2 <= chi2.ppf(lv, dof))) for lv in levels}


def coverage_ece(
    d2: Sequence[float], dof: int = 6, n_levels: int = 99
) -> float:
    """Mean |empirical coverage - nominal level| over a uniform level grid."""
    levels = np.linspace(0.01, 0.99, n_levels)
    cov = coverage(d2, dof=dof, levels=levels)
    return float(np.mean([abs(cov[lv] - lv) for lv in levels]))


def ause(
    uncertainties: Sequence[float],
    errors: Sequence[float],
    n_steps: int = 20,
) -> float:
    """Area Under the Sparsification Error curve (lower is better).

    At each fraction f, remove the f highest-uncertainty (resp. highest-error,
    for the oracle) queries and record the mean error of the rest.  Both
    curves are normalized by the full-set mean error; AUSE is the mean gap.
    Zero means the predicted uncertainty ranks queries exactly like the true
    error does.
    """
    u = np.asarray(uncertainties, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if u.shape != e.shape or u.ndim != 1:
        raise ValueError("uncertainties and errors must be 1-D and same length")
    n = len(e)
    base = e.mean()
    if base <= 0:
        return 0.0
    by_u = np.argsort(-u)   # most uncertain first
    by_e = np.argsort(-e)   # largest error first (oracle)
    gaps = []
    for f in np.linspace(0.0, 0.95, n_steps):
        # Keep at least one element so small query sets stay well-defined.
        k = min(int(round(f * n)), n - 1)
        rest_u = e[by_u[k:]].mean() / base
        rest_e = e[by_e[k:]].mean() / base
        gaps.append(rest_u - rest_e)
    return float(np.mean(gaps))
