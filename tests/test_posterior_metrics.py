import numpy as np
import torch

from usgloc import se3
from usgloc.metrics import ause, coverage, coverage_ece, mahalanobis_sq, pose_errors, success_rate
from usgloc.posterior import PoseGaussian, PoseMixture


def _gaussian(seed=0, scale=0.05):
    torch.manual_seed(seed)
    T = se3.se3_exp(torch.randn(6) * 0.5)
    A = torch.randn(6, 6) * scale
    Sigma = A @ A.T + 1e-4 * torch.eye(6)
    return PoseGaussian(T_mode=T, Sigma=Sigma)


def test_sampling_coverage_consistency():
    """Samples of a PoseGaussian must hit the chi2 credible regions at the
    nominal rates -- this jointly validates sample(), mahalanobis and the
    coverage metric."""
    g = _gaussian()
    gen = torch.Generator().manual_seed(42)
    samples = g.sample(4000, generator=gen)
    d2 = [float(g.mahalanobis_sq(T)) for T in samples]
    cov = coverage(d2, dof=6)
    for level, c in cov.items():
        assert abs(c - level) < 0.03, (level, c)
    assert coverage_ece(d2, dof=6) < 0.03


def test_log_prob_peaks_at_mode():
    g = _gaussian(1)
    lp_mode = g.log_prob(g.T_mode)
    far = se3.apply_tangent(g.T_mode, torch.ones(6) * 0.5)
    assert lp_mode > g.log_prob(far)


def test_mixture_nll_and_condense():
    g1 = _gaussian(2)
    xi = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    g2 = PoseGaussian(T_mode=se3.apply_tangent(g1.T_mode, xi), Sigma=g1.Sigma.clone())
    mix = PoseMixture(components=[g1, g2], weights=torch.tensor([0.5, 0.5]))

    assert np.isfinite(mix.nll(g1.T_mode))
    cond = mix.condense()
    # Condensed mean lies midway between the two modes.
    d1 = se3.relative_tangent(g1.T_mode, cond.T_mode)
    assert torch.allclose(d1, xi / 2, atol=1e-5)
    # Between-mode spread inflates the condensed covariance.
    assert torch.trace(cond.Sigma) > torch.trace(g1.Sigma)
    # Single-component mixtures condense to the component itself.
    single = PoseMixture(components=[g1], weights=torch.tensor([1.0]))
    assert torch.allclose(single.condense().Sigma, g1.Sigma)


def test_mahalanobis_metric_matches_component():
    g = _gaussian(3)
    T = se3.apply_tangent(g.T_mode, torch.randn(6) * 0.05)
    assert abs(mahalanobis_sq(g.T_mode, g.Sigma, T) - float(g.mahalanobis_sq(T))) < 1e-5


def test_pose_errors_and_success_rate():
    T = torch.eye(4)
    T2 = se3.se3_exp(torch.tensor([0.03, 0.0, 0.0, 0.0, 0.0, np.deg2rad(2.0)], dtype=torch.float32))
    t_err, r_err = pose_errors(T2, T)
    assert abs(t_err - 0.03) < 1e-3
    assert abs(r_err - 2.0) < 1e-3
    assert success_rate([0.03], [2.0]) == 1.0
    assert success_rate([0.10], [2.0]) == 0.0


def test_ause_ranking():
    rng = np.random.default_rng(0)
    errors = rng.exponential(1.0, 200)
    perfect = ause(errors, errors)               # predicted == true error
    noisy = ause(errors + rng.normal(0, 1.0, 200), errors)
    anti = ause(-errors, errors)                 # inverted ranking
    assert perfect < 1e-9
    assert perfect <= noisy <= anti
    assert anti > 0.1


def test_overconfident_covariance_fails_coverage():
    g = _gaussian(4)
    gen = torch.Generator().manual_seed(7)
    samples = g.sample(1000, generator=gen)
    shrunk = PoseGaussian(T_mode=g.T_mode, Sigma=g.Sigma * 0.1)
    d2 = [float(shrunk.mahalanobis_sq(T)) for T in samples]
    cov = coverage(d2, dof=6)
    assert cov[0.683] < 0.30  # gross undercoverage is detected
