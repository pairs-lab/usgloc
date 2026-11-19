import numpy as np
import torch

from usgloc import se3
from usgloc.pipeline import PipelineConfig, _pnp_covariance, _pnp_jacobian
from usgloc.rendering import Camera


def _scene(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cam = Camera(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    T = se3.apply_tangent(torch.eye(4, dtype=torch.float64),
                          torch.tensor([0.10, -0.05, 0.20, 0.03, -0.02, 0.05], dtype=torch.float64))
    Xc = np.stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1.0, 1.0, n), rng.uniform(1.5, 4.0, n)], 1)
    Tn = T.numpy()
    obj = (Xc - Tn[:3, 3]) @ Tn[:3, :3]                  # world points seen at T (camera-from-world)
    img = np.stack([cam.fx * Xc[:, 0] / Xc[:, 2] + cam.cx, cam.fy * Xc[:, 1] / Xc[:, 2] + cam.cy], 1)
    return cam, T, obj, img, rng


def test_pnp_jacobian_matches_finite_differences_in_the_left_chart():
    cam, T, obj, img, _ = _scene(50)
    r0, J = _pnp_jacobian(obj, img, T, cam, thresh_px=1e9)
    eps = 1e-6
    for k in range(6):
        d = torch.zeros(6, dtype=torch.float64); d[k] = eps
        rp, _ = _pnp_jacobian(obj, img, se3.apply_tangent(T, d), cam, thresh_px=1e9)
        rm, _ = _pnp_jacobian(obj, img, se3.apply_tangent(T, -d), cam, thresh_px=1e9)
        assert np.allclose((rp - rm) / (2 * eps), J[:, :, k], rtol=1e-4, atol=1e-3), k


def test_pnp_covariance_scale_and_shape():
    cam, T, obj, img, rng = _scene(400)
    sigma_px = 0.5
    noisy = img + rng.normal(0.0, sigma_px, img.shape)
    S = _pnp_covariance(obj, noisy, T, cam, thresh_px=10.0)
    assert S is not None and S.shape == (6, 6)
    assert torch.allclose(S, S.T) and bool((torch.linalg.eigvalsh(S) > 0).all())
    r, J = _pnp_jacobian(obj, noisy, T, cam, thresh_px=10.0)
    s2 = float((r ** 2).sum() / (2 * len(r) - 6))
    assert abs(s2 / sigma_px ** 2 - 1.0) < 0.15          # the residual scale estimates the pixel noise
    H = np.einsum("nij,nik->jk", J, J)
    assert np.allclose(S.numpy(), s2 * np.linalg.inv(H), rtol=1e-6)


def test_pnp_covariance_needs_enough_inliers():
    cam, T, obj, img, _ = _scene(8)
    assert _pnp_covariance(obj, img, T, cam, thresh_px=2.0) is None
    assert _pnp_covariance(None, None, T, cam, thresh_px=2.0) is None
    far = img + 50.0                                      # every pair beyond the threshold
    cam2, T2, obj2, img2, _ = _scene(100)
    assert _pnp_covariance(obj2, img2 + 50.0, T2, cam2, thresh_px=2.0) is None


def test_default_covariance_source_is_photometric():
    assert PipelineConfig().cov_source == "photometric"
