"""Tests for scripts/analysis/pose_unc_np.py (numpy pose-uncertainty helpers).

The module must reproduce usgloc.se3's conventions exactly (left perturbation
of T_cw, translation first), because uncertainty_fair.py recomputes the
runner's d2 with it and compares ACE's PnP covariance against ours.
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))
import pose_unc_np as pu  # noqa: E402


def _rand_pose(rng, rot=0.5, trans=1.0):
    return pu.se3_exp(np.concatenate([rng.normal(0, trans, 3), rng.normal(0, rot, 3)]))


def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        xi = np.concatenate([rng.normal(0, 1, 3), rng.normal(0, 0.8, 3)])
        assert np.allclose(pu.se3_log(pu.se3_exp(xi)), xi, atol=1e-7)
        T = pu.se3_exp(xi)
        assert np.allclose(pu.se3_inverse(T) @ T, np.eye(4), atol=1e-10)


def test_relative_tangent_is_left_perturbation():
    rng = np.random.default_rng(1)
    Tr = _rand_pose(rng)
    xi = rng.normal(0, 0.1, 6)
    T = pu.se3_exp(xi) @ Tr
    assert np.allclose(pu.relative_tangent(Tr, T), xi, atol=1e-8)


def test_matches_torch_se3():
    torch = pytest.importorskip("torch")
    from usgloc import se3
    from usgloc.metrics import pose_errors
    from usgloc.posterior import _project_pd

    rng = np.random.default_rng(2)
    for _ in range(20):
        Ta, Tb = _rand_pose(rng), _rand_pose(rng)
        ref = se3.relative_tangent(torch.tensor(Ta), torch.tensor(Tb)).numpy()
        assert np.allclose(pu.relative_tangent(Ta, Tb), ref, atol=1e-6)
        t_err, _ = pose_errors(torch.tensor(Ta), torch.tensor(Tb))
        assert abs(pu.translation_error(Ta, Tb) - t_err) < 1e-9
    A = rng.normal(size=(6, 6))
    S = A @ A.T
    S[0, 0] += 1e-9
    assert np.allclose(pu.project_pd(S), _project_pd(torch.tensor(S)).numpy(), atol=1e-10)


def _scene(rng, n=200):
    T = _rand_pose(rng, rot=0.3, trans=0.5)
    # points in front of the camera, expressed in the world frame
    pc = np.column_stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1, 1, n),
                          rng.uniform(1.0, 4.0, n)])
    Xw = (pc - T[:3, 3]) @ T[:3, :3]
    K = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1]])
    return T, Xw, K


def test_pnp_jacobian_finite_difference():
    rng = np.random.default_rng(3)
    T, Xw, K = _scene(rng, 30)
    uv = np.zeros((30, 2))
    J = pu.pnp_jacobian(T, Xw, K)
    h = 1e-6
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        rp, _ = pu.pnp_residuals(pu.se3_exp(d) @ T, Xw, uv, K)
        rm, _ = pu.pnp_residuals(pu.se3_exp(-d) @ T, Xw, uv, K)
        assert np.allclose((rp - rm) / (2 * h), J[:, :, k], rtol=1e-4, atol=1e-3)


def _gauss_newton(T, Xw, uv, K, iters=10):
    for _ in range(iters):
        r, _ = pu.pnp_residuals(T, Xw, uv, K)
        J = pu.pnp_jacobian(T, Xw, K).reshape(-1, 6)
        T = pu.se3_exp(-np.linalg.solve(J.T @ J, J.T @ r.reshape(-1))) @ T
    return T


def test_pnp_covariance_is_calibrated_under_iid_noise():
    """With i.i.d. Gaussian pixel noise the Gauss-Newton covariance at the
    least-squares pose must give chi2_6-distributed d2 for the true pose."""
    rng = np.random.default_rng(4)
    d2 = []
    for _ in range(300):
        T, Xw, K = _scene(rng, 120)
        r0, _ = pu.pnp_residuals(T, Xw, np.zeros((120, 2)), K)
        uv = r0 + rng.normal(0, 1.5, r0.shape)          # projections + noise
        T_hat = _gauss_newton(pu.se3_exp(rng.normal(0, 1e-3, 6)) @ T, Xw, uv, K)
        out = pu.pnp_covariance(T_hat, Xw, uv, K)
        assert out["n_inliers"] == 120
        d2.append(pu.mahalanobis_sq(pu.relative_tangent(T_hat, T), out["Sigma"]))
    d2 = np.array(d2)
    assert abs(d2.mean() - 6.0) < 0.9
    assert abs(np.mean(d2 <= chi2.ppf(0.954, 6)) - 0.954) < 0.04


def test_pnp_covariance_inlier_threshold_and_degenerate():
    rng = np.random.default_rng(5)
    T, Xw, K = _scene(rng, 50)
    r0, _ = pu.pnp_residuals(T, Xw, np.zeros((50, 2)), K)
    uv = r0.copy()
    uv[:10] += 40.0                                          # 10 gross outliers
    out = pu.pnp_covariance(T, Xw, uv + rng.normal(0, 0.5, uv.shape), K, inlier_px=10.0)
    assert out["n_inliers"] == 40 and out["rms_px"] < 2.0
    assert pu.pnp_covariance(T, Xw[:3], uv[:3], K)["Sigma"] is None


def test_gaussian_logpdf_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(6)
    A = rng.normal(size=(6, 6))
    S = A @ A.T + np.eye(6)
    e = rng.normal(size=6)
    tau = 2.5
    ref = multivariate_normal(np.zeros(6), tau * S).logpdf(e)
    got = pu.gaussian_logpdf(pu.mahalanobis_sq(e, S), np.linalg.slogdet(S)[1], tau)
    assert abs(float(got) - ref) < 1e-9
