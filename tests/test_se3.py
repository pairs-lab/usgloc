import torch

from usgloc import se3


def test_exp_log_roundtrip():
    torch.manual_seed(0)
    xi = torch.randn(64, 6, dtype=torch.float64) * 0.7
    T = se3.se3_exp(xi)
    xi_back = se3.se3_log(T)
    assert torch.allclose(xi, xi_back, atol=1e-8)


def test_exp_small_angle_stable():
    xi = torch.tensor([1e-9, -2e-9, 3e-9, 1e-10, 0.0, -1e-10], dtype=torch.float64)
    T = se3.se3_exp(xi)
    assert torch.allclose(T, torch.eye(4, dtype=torch.float64), atol=1e-8)
    assert torch.isfinite(se3.se3_log(T)).all()


def test_inverse():
    torch.manual_seed(1)
    T = se3.se3_exp(torch.randn(10, 6, dtype=torch.float64))
    eye = T @ se3.se3_inverse(T)
    assert torch.allclose(eye, torch.eye(4, dtype=torch.float64).expand(10, 4, 4), atol=1e-10)


def test_relative_tangent_roundtrip():
    torch.manual_seed(2)
    T_ref = se3.se3_exp(torch.randn(6, dtype=torch.float64))
    xi = torch.randn(6, dtype=torch.float64) * 0.3
    T = se3.apply_tangent(T_ref, xi)
    assert torch.allclose(se3.relative_tangent(T_ref, T), xi, atol=1e-9)


def test_rotation_matrices_orthonormal():
    torch.manual_seed(3)
    R = se3.so3_exp(torch.randn(32, 3, dtype=torch.float64))
    eye = R @ R.transpose(-1, -2)
    assert torch.allclose(eye, torch.eye(3, dtype=torch.float64).expand(32, 3, 3), atol=1e-10)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32, dtype=torch.float64), atol=1e-10)


def test_rotation_angle_deg():
    phi = torch.tensor([0.0, 0.0, torch.pi / 6], dtype=torch.float64)
    R = se3.so3_exp(phi)
    assert abs(float(se3.rotation_angle_deg(R)) - 30.0) < 1e-6
