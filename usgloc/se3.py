"""SE(3) / SO(3) Lie-group utilities in PyTorch.

Conventions
-----------
* A pose ``T`` is a 4x4 homogeneous matrix mapping world points to camera
  points (or vice versa -- the module is agnostic; the pipeline documents its
  own convention).
* A tangent vector is ``xi = (rho, phi)`` with translation part ``rho`` first
  and rotation part ``phi`` (axis-angle) last, so ``xi`` has shape ``(6,)`` and
  ``exp(xi) = [[exp(phi^), V(phi) rho], [0, 1]]``.
* All functions are differentiable and batched: leading dimensions are
  preserved.
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def hat(phi: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix of (..., 3) vectors -> (..., 3, 3)."""
    zero = torch.zeros_like(phi[..., 0])
    rows = (
        torch.stack([zero, -phi[..., 2], phi[..., 1]], dim=-1),
        torch.stack([phi[..., 2], zero, -phi[..., 0]], dim=-1),
        torch.stack([-phi[..., 1], phi[..., 0], zero], dim=-1),
    )
    return torch.stack(rows, dim=-2)


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula, (..., 3) -> (..., 3, 3); stable near zero."""
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(_EPS)
    theta_ = theta[..., None]
    K = hat(phi / theta)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    R = eye + torch.sin(theta_) * K + (1.0 - torch.cos(theta_)) * (K @ K)
    # For very small angles fall back to first order to keep gradients clean.
    small = (theta_ < 1e-6)
    R_taylor = eye + hat(phi)
    return torch.where(small, R_taylor, R)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Inverse of so3_exp, (..., 3, 3) -> (..., 3)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos)
    w = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta)).clamp_min(_EPS)
    small = theta[..., None] < 1e-6
    return torch.where(small, 0.5 * w, scale[..., None] * w)


def _left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    """Left Jacobian V(phi) of SO(3), used in the SE(3) exponential."""
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(_EPS)
    theta_ = theta[..., None]
    K = hat(phi / theta)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    V = (
        eye
        + ((1.0 - torch.cos(theta_)) / theta_) * K
        + ((theta_ - torch.sin(theta_)) / theta_) * (K @ K)
    )
    small = theta_ < 1e-6
    V_taylor = eye + 0.5 * hat(phi)
    return torch.where(small, V_taylor, V)


def _left_jacobian_inv(phi: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp_min(_EPS)
    theta_ = theta[..., None]
    K = hat(phi / theta)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    half = theta_ * 0.5
    cot = half / torch.tan(half).clamp_min(_EPS)
    Vinv = eye - 0.5 * hat(phi) + (1.0 - cot) * (K @ K)
    small = theta_ < 1e-6
    Vinv_taylor = eye - 0.5 * hat(phi)
    return torch.where(small, Vinv_taylor, Vinv)


def _assemble(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build (..., 4, 4) from rotation and translation without in-place ops
    (keeps the map compatible with forward-mode AD)."""
    top = torch.cat([R, t[..., None]], dim=-1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom = torch.cat(
        [bottom[..., :3], torch.ones_like(bottom[..., :1])], dim=-1
    )
    return torch.cat([top, bottom], dim=-2)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map, (..., 6) -> (..., 4, 4). xi = (rho, phi)."""
    rho, phi = xi[..., :3], xi[..., 3:]
    R = so3_exp(phi)
    t = (_left_jacobian(phi) @ rho[..., None])[..., 0]
    return _assemble(R, t)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Logarithm map, (..., 4, 4) -> (..., 6)."""
    phi = so3_log(T[..., :3, :3])
    rho = (_left_jacobian_inv(phi) @ T[..., :3, 3, None])[..., 0]
    return torch.cat([rho, phi], dim=-1)


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    RT = R.transpose(-1, -2)
    return _assemble(RT, -(RT @ t[..., None])[..., 0])


def relative_tangent(T_ref: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Tangent-space coordinates of ``T`` in the chart centred at ``T_ref``,
    i.e. ``xi`` such that ``T = exp(xi^) T_ref`` (left perturbation)."""
    return se3_log(T @ se3_inverse(T_ref))


def apply_tangent(T_ref: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`relative_tangent`: ``exp(xi^) T_ref``."""
    return se3_exp(xi) @ T_ref


def rotation_angle_deg(R_err: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation angle of (..., 3, 3) rotation matrices, in degrees."""
    phi = so3_log(R_err)
    return torch.linalg.norm(phi, dim=-1) * (180.0 / torch.pi)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion(s) in (w, x, y, z) order, (..., 4) -> (..., 3, 3).

    The input is normalized internally, so raw values parsed from pose files
    (e.g. COLMAP / visloc pseudo-GT lines) can be passed directly.
    """
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))
