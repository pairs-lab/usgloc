"""Scalar temperature calibration of pose covariances (Sec. IV-D of the draft).

The Laplace covariance ``Sigma_xi`` produced by photometric refinement
(:mod:`usgloc.refinement`) assumes i.i.d. Gaussian residuals.  Real
photometric residuals are heavy-tailed and spatially correlated, so the raw
covariance tends to be optimistic.  As promised in Sec. IV-D of the RA-L
draft, we therefore "apply a single scalar inflation factor calibrated on a
held-out split (a temperature on ``Sigma_xi``)": the calibrated posterior
uses ``tau * Sigma_xi`` with a single scalar ``tau > 0`` shared across all
queries and components.

The statistical model behind the fit: if the pose error
``e = log(T_gt T_est^{-1})`` is distributed as ``N(0, tau * Sigma)``, then
the *raw* Mahalanobis-squared statistic ``d^2 = e^T Sigma^{-1} e`` follows a
scaled chi-square law ``tau * chi2(dof)`` with ``dof = 6`` on SE(3).  The
temperature is hence identifiable from held-out ``d^2`` values alone:

* ``median``: ``tau = median(d^2) / chi2.ppf(0.5, dof)`` -- robust to the
  heavy right tail contributed by wrong-basin queries (the default).
* ``mean`` / ``mle``: ``tau = mean(d^2) / dof``.  This *is* the Gaussian
  maximum-likelihood estimate of a shared scalar scale: maximising
  ``sum_i log N(e_i; 0, tau Sigma_i)`` over ``tau`` gives
  ``d/dtau [-1/2 sum_i (d_i^2 / tau + dof log tau)] = 0`` and thus
  ``tau = (1/n) sum_i d_i^2 / dof``.

Dividing the held-out ``d^2`` by ``tau`` (equivalently, inflating every
``Sigma`` by ``tau``) restores the nominal chi-square coverage levels
reported by :func:`usgloc.metrics.coverage`; the raw vs. tempered ablation
of Sec. V is driven by :func:`calibrate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from scipy.stats import chi2

from . import metrics
from .posterior import PoseGaussian, PoseMixture

#: Lower clamp on the fitted temperature, guarding against degenerate
#: calibration splits (e.g. all-zero d^2) producing a singular covariance.
TAU_FLOOR = 1e-12

_METHODS = ("median", "mean", "mle")


def _as_d2_array(d2: Sequence[float]) -> np.ndarray:
    """Validate and convert Mahalanobis-squared values to a 1-D float array."""
    arr = np.asarray(d2, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("calibration requires at least one d^2 value")
    if not np.all(np.isfinite(arr)):
        raise ValueError("d^2 values must be finite")
    if np.any(arr < 0.0):
        raise ValueError("d^2 values must be non-negative")
    return arr


def fit_temperature(
    d2: Sequence[float],
    dof: int = 6,
    method: str = "median",
) -> float:
    """Fit the scalar covariance temperature ``tau`` from held-out ``d^2``.

    Args:
        d2: Mahalanobis-squared values ``e^T Sigma^{-1} e`` computed with the
            *raw* (untempered) covariances on a held-out calibration split.
        dof: Degrees of freedom of the error; 6 for full SE(3) poses.
        method: ``'median'`` (robust, default), ``'mean'``, or ``'mle'``.
            ``'mle'`` is an alias of ``'mean'`` because the sample mean over
            ``dof`` is exactly the Gaussian MLE of a shared scalar scale
            (see module docstring).

    Returns:
        The temperature ``tau`` such that ``tau * Sigma`` is calibrated,
        clamped from below at :data:`TAU_FLOOR`.

    Raises:
        ValueError: If ``d2`` is empty, contains non-finite or negative
            values, or ``method`` is unknown.
    """
    arr = _as_d2_array(d2)
    if method == "median":
        tau = float(np.median(arr)) / float(chi2.ppf(0.5, dof))
    elif method in ("mean", "mle"):
        tau = float(np.mean(arr)) / float(dof)
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")
    return max(tau, TAU_FLOOR)


def fit_temperature_from_poses(
    T_estimates: Sequence[torch.Tensor],
    Sigmas: Sequence[torch.Tensor],
    T_gts: Sequence[torch.Tensor],
    dof: int = 6,
    method: str = "median",
) -> float:
    """Fit ``tau`` directly from (estimate, covariance, ground truth) triplets.

    For each held-out query the raw statistic
    ``d^2 = e^T Sigma^{-1} e`` with ``e = log(T_gt T_est^{-1})`` is computed
    via :func:`usgloc.metrics.mahalanobis_sq`, then passed to
    :func:`fit_temperature`.

    Args:
        T_estimates: Estimated camera-from-world poses, each ``(4, 4)``.
        Sigmas: Raw tangent-space covariances, each ``(6, 6)``.
        T_gts: Ground-truth camera-from-world poses, each ``(4, 4)``.
        dof: Degrees of freedom (6 for SE(3)).
        method: Estimator passed through to :func:`fit_temperature`.

    Returns:
        The fitted temperature ``tau``.

    Raises:
        ValueError: If the three sequences have mismatched lengths.
    """
    if not (len(T_estimates) == len(Sigmas) == len(T_gts)):
        raise ValueError(
            "T_estimates, Sigmas and T_gts must have the same length, got "
            f"{len(T_estimates)}, {len(Sigmas)}, {len(T_gts)}"
        )
    d2 = [
        metrics.mahalanobis_sq(T_est, Sigma, T_gt)
        for T_est, Sigma, T_gt in zip(T_estimates, Sigmas, T_gts)
    ]
    return fit_temperature(d2, dof=dof, method=method)


def temper(g: PoseGaussian, tau: float) -> PoseGaussian:
    """Return a new :class:`PoseGaussian` with covariance ``tau * Sigma``.

    The mode is unchanged (a temperature only rescales the spread); the
    input ``g`` is not mutated.
    """
    return PoseGaussian(T_mode=g.T_mode, Sigma=g.Sigma * tau)


def temper_mixture(m: PoseMixture, tau: float) -> PoseMixture:
    """Return a new :class:`PoseMixture` with every component tempered.

    Weights are preserved (a shared scalar temperature does not change the
    relative responsibility of mixture components); the input ``m`` is not
    mutated.
    """
    return PoseMixture(
        components=[temper(c, tau) for c in m.components],
        weights=m.weights.clone(),
    )


@dataclass
class CalibrationReport:
    """Outcome of the held-out temperature calibration (Sec. IV-D / Sec. V).

    Attributes:
        tau: Fitted scalar temperature on ``Sigma_xi``.
        method: Estimator used (``'median'``, ``'mean'`` or ``'mle'``).
        dof: Degrees of freedom of the chi-square reference (6 on SE(3)).
        n_calibration: Number of held-out queries used for the fit.
        coverage_before: Empirical coverage of the raw ``d^2`` at the nominal
            credibility levels (:func:`usgloc.metrics.coverage`).
        coverage_after: Coverage of ``d^2 / tau``, i.e. with ``tau * Sigma``.
    """

    tau: float
    method: str
    dof: int
    n_calibration: int
    coverage_before: Dict[float, float]
    coverage_after: Dict[float, float]

    def __str__(self) -> str:
        before = ", ".join(
            f"{lv:.3f}:{c:.3f}" for lv, c in self.coverage_before.items()
        )
        after = ", ".join(
            f"{lv:.3f}:{c:.3f}" for lv, c in self.coverage_after.items()
        )
        return (
            f"CalibrationReport(tau={self.tau:.4g}, method={self.method}, "
            f"dof={self.dof}, n={self.n_calibration}, "
            f"coverage {before} -> {after})"
        )


def calibrate(
    d2: Sequence[float],
    dof: int = 6,
    method: str = "median",
) -> CalibrationReport:
    """Fit ``tau`` and report before/after coverage on the calibration split.

    Args:
        d2: Raw held-out Mahalanobis-squared values (untempered covariances).
        dof: Degrees of freedom (6 for SE(3)).
        method: Estimator passed to :func:`fit_temperature`.

    Returns:
        A :class:`CalibrationReport` whose ``coverage_after`` is evaluated on
        ``d2 / tau`` -- the statistic obtained when every covariance is
        inflated to ``tau * Sigma``.
    """
    arr = _as_d2_array(d2)
    tau = fit_temperature(arr, dof=dof, method=method)
    return CalibrationReport(
        tau=tau,
        method=method,
        dof=dof,
        n_calibration=int(arr.size),
        coverage_before=metrics.coverage(arr, dof=dof),
        coverage_after=metrics.coverage(arr / tau, dof=dof),
    )
