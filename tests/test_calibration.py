import numpy as np
import pytest
import torch

from usgloc import se3
from usgloc.calibration import (
    TAU_FLOOR,
    CalibrationReport,
    calibrate,
    fit_temperature,
    fit_temperature_from_poses,
    temper,
    temper_mixture,
)
from usgloc.metrics import coverage, coverage_ece
from usgloc.posterior import PoseGaussian, PoseMixture


def _gaussian(seed=0, scale=0.05):
    torch.manual_seed(seed)
    T = se3.se3_exp(torch.randn(6) * 0.5)
    A = torch.randn(6, 6) * scale
    Sigma = A @ A.T + 1e-4 * torch.eye(6)
    return PoseGaussian(T_mode=T, Sigma=Sigma)


# ---------------------------------------------------------------- (a) fitting


@pytest.mark.parametrize("method", ["median", "mean", "mle"])
@pytest.mark.parametrize("tau_true", [0.4, 3.0])
def test_fit_temperature_recovers_scale(method, tau_true):
    rng = np.random.default_rng(0)
    d2 = tau_true * rng.chisquare(6, size=500)
    tau = fit_temperature(d2, dof=6, method=method)
    assert abs(tau - tau_true) / tau_true < 0.15, (method, tau)


def test_mle_is_alias_of_mean():
    rng = np.random.default_rng(1)
    d2 = 2.0 * rng.chisquare(6, size=100)
    assert fit_temperature(d2, method="mle") == fit_temperature(d2, method="mean")


def test_fit_temperature_from_poses_recovers_scale():
    tau_true = 4.0
    g = _gaussian(seed=2)
    # Ground truths drawn around the estimate with the *inflated* covariance:
    # raw d^2 then follows tau_true * chi2(6).
    truth = PoseGaussian(T_mode=g.T_mode, Sigma=g.Sigma * tau_true)
    gen = torch.Generator().manual_seed(3)
    T_gts = list(truth.sample(400, generator=gen))
    T_estimates = [g.T_mode] * len(T_gts)
    Sigmas = [g.Sigma] * len(T_gts)
    tau = fit_temperature_from_poses(T_estimates, Sigmas, T_gts, method="median")
    assert abs(tau - tau_true) / tau_true < 0.15, tau


# ------------------------------------------------------------- (b) tempering


def test_temper_scales_sigma_and_keeps_mode():
    g = _gaussian(seed=4)
    Sigma_orig = g.Sigma.clone()
    tau = 2.5
    g_t = temper(g, tau)
    assert torch.allclose(g_t.Sigma, Sigma_orig * tau)
    assert torch.equal(g_t.T_mode, g.T_mode)
    # Input is not mutated.
    assert torch.allclose(g.Sigma, Sigma_orig)


def test_tempered_coverage_improves_toward_nominal():
    tau_true = 5.0
    g = _gaussian(seed=5)  # overconfident: truth is tau_true x wider
    truth = PoseGaussian(T_mode=g.T_mode, Sigma=g.Sigma * tau_true)
    gen = torch.Generator().manual_seed(6)
    samples = truth.sample(1500, generator=gen)

    d2_raw = [float(g.mahalanobis_sq(T)) for T in samples]
    g_t = temper(g, fit_temperature(d2_raw, method="median"))
    d2_tmp = [float(g_t.mahalanobis_sq(T)) for T in samples]

    ece_raw = coverage_ece(d2_raw, dof=6)
    ece_tmp = coverage_ece(d2_tmp, dof=6)
    assert ece_tmp < ece_raw
    cov_tmp = coverage(d2_tmp, dof=6)
    for level, c in cov_tmp.items():
        assert abs(c - level) < 0.05, (level, c)


# --------------------------------------------------------------- (c) mixture


def test_temper_mixture_preserves_weights_and_components():
    g1 = _gaussian(seed=7)
    g2 = PoseGaussian(
        T_mode=se3.apply_tangent(g1.T_mode, torch.tensor([0.1, 0, 0, 0, 0, 0.0])),
        Sigma=g1.Sigma * 0.5,
    )
    mix = PoseMixture(components=[g1, g2], weights=torch.tensor([0.7, 0.3]))
    tau = 3.0
    mixed = temper_mixture(mix, tau)
    assert len(mixed.components) == len(mix.components)
    assert torch.allclose(mixed.weights, mix.weights)
    for c_new, c_old in zip(mixed.components, mix.components):
        assert torch.allclose(c_new.Sigma, c_old.Sigma * tau)
        assert torch.equal(c_new.T_mode, c_old.T_mode)
    # Original mixture untouched.
    assert torch.allclose(mix.components[0].Sigma, g1.Sigma)


# ---------------------------------------------------------------- (d) report


def test_calibration_report_improves_coverage():
    rng = np.random.default_rng(8)
    tau_true = 6.0  # grossly overconfident raw covariances
    d2 = tau_true * rng.chisquare(6, size=800)
    report = calibrate(d2, dof=6, method="median")

    assert isinstance(report, CalibrationReport)
    assert report.n_calibration == 800
    assert report.method == "median"
    assert report.dof == 6
    for level in report.coverage_before:
        before_gap = abs(report.coverage_before[level] - level)
        after_gap = abs(report.coverage_after[level] - level)
        assert after_gap < before_gap, (level, report)
        assert after_gap < 0.05
    line = str(report)
    assert "\n" not in line
    assert "tau=" in line and "median" in line


# ------------------------------------------------------------ (e) edge cases


def test_empty_d2_raises():
    with pytest.raises(ValueError):
        fit_temperature([])
    with pytest.raises(ValueError):
        calibrate([])


def test_tau_floor_clamp():
    assert fit_temperature([0.0, 0.0, 0.0]) == TAU_FLOOR
    assert TAU_FLOOR == 1e-12


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        fit_temperature([1.0, 2.0], method="mode")
    with pytest.raises(ValueError):
        fit_temperature([1.0, -2.0])
    with pytest.raises(ValueError):
        fit_temperature([1.0, np.nan])
    with pytest.raises(ValueError):
        fit_temperature_from_poses([torch.eye(4)], [], [torch.eye(4)])
