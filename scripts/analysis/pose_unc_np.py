"""NumPy-only pose-uncertainty helpers for the fair-baseline study.

Used by ``uncertainty_fair.py`` (CPU analysis of saved run JSONs) and by
``run_ace_baseline.py`` (the ACE PnP covariance).  Kept free of torch so the
analysis runs on any machine with numpy + scipy.

Conventions are those of :mod:`usgloc.se3` and :mod:`usgloc.posterior`, and
``tests/test_pose_unc_np.py`` checks them against the torch implementation:

* a pose ``T`` is a 4x4 camera-from-world matrix;
* a tangent vector is ``xi = (rho, phi)``, translation first;
* perturbations are on the LEFT: ``T = exp(xi^) T_ref``, so
  ``relative_tangent(T_ref, T) = log(T T_ref^{-1})`` -- the error ``e`` whose
  Mahalanobis norm under the stored ``Sigma`` is the runner's ``d2``.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8
LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------
# SE(3)
# --------------------------------------------------------------------------
def hat(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(phi))
    if th < 1e-6:
        return np.eye(3) + hat(phi)
    K = hat(phi / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    return th / (2.0 * np.sin(th)) * w


def _left_jacobian(phi: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(phi))
    if th < 1e-6:
        return np.eye(3) + 0.5 * hat(phi)
    K = hat(phi / th)
    return (np.eye(3) + (1.0 - np.cos(th)) / th * K
            + (th - np.sin(th)) / th * (K @ K))


def _left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(phi))
    if th < 1e-6:
        return np.eye(3) - 0.5 * hat(phi)
    K = hat(phi / th)
    half = 0.5 * th
    cot = half / np.tan(half)
    return np.eye(3) - 0.5 * hat(phi) + (1.0 - cot) * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3, 3] = _left_jacobian(xi[3:]) @ xi[:3]
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    phi = so3_log(T[:3, :3])
    rho = _left_jacobian_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def relative_tangent(T_ref: np.ndarray, T: np.ndarray) -> np.ndarray:
    """``xi`` with ``T = exp(xi^) T_ref`` (left perturbation)."""
    return se3_log(np.asarray(T, np.float64) @ se3_inverse(np.asarray(T_ref, np.float64)))


def translation_error(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """Same quantity as :func:`usgloc.metrics.pose_errors` (metres)."""
    return float(np.linalg.norm((np.asarray(T_est) @ se3_inverse(np.asarray(T_gt)))[:3, 3]))


# --------------------------------------------------------------------------
# Gaussians in the tangent space
# --------------------------------------------------------------------------
def project_pd(Sigma: np.ndarray, rel_floor: float = 1e-6) -> np.ndarray:
    """Mirror of :func:`usgloc.posterior._project_pd` (in float64)."""
    S = 0.5 * (np.asarray(Sigma, np.float64) + np.asarray(Sigma, np.float64).T)
    ev, U = np.linalg.eigh(S)
    ev = np.maximum(ev, rel_floor * max(ev.max(), 1e-30))
    return (U * ev) @ U.T


def mahalanobis_sq(e: np.ndarray, Sigma: np.ndarray) -> float:
    return float(e @ np.linalg.solve(Sigma, e))


def gaussian_logpdf(d2, logdet, scale=1.0, dof: int = 6):
    """``log N(e; 0, scale * Sigma)`` from the RAW ``d2 = e^T Sigma^-1 e`` and
    ``logdet = log|Sigma|``; vectorised over arrays."""
    d2, logdet, scale = np.asarray(d2), np.asarray(logdet), np.asarray(scale)
    return -0.5 * (d2 / scale + logdet + dof * np.log(scale) + dof * LOG_2PI)


# --------------------------------------------------------------------------
# PnP covariance (the fair external baseline)
# --------------------------------------------------------------------------
def pnp_residuals(T_cw: np.ndarray, X_w: np.ndarray, uv: np.ndarray,
                  K: np.ndarray):
    """Reprojection residuals ``proj(T X) - uv`` (n, 2) and depths (n,)."""
    p = X_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = p[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    u = K[0, 0] * p[:, 0] / zs + K[0, 2]
    v = K[1, 1] * p[:, 1] / zs + K[1, 2]
    return np.stack([u - uv[:, 0], v - uv[:, 1]], axis=1), z


def pnp_jacobian(T_cw: np.ndarray, X_w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """d residual / d xi for the left perturbation ``exp(xi^) T_cw``: (n, 2, 6).

    ``exp(xi^) p = p + rho + phi x p`` to first order, so
    ``dp/dxi = [I, -hat(p)]``; chained with the pinhole Jacobian.
    """
    p = X_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    n = len(p)
    Jp = np.zeros((n, 2, 3))
    Jp[:, 0, 0] = fx / z
    Jp[:, 0, 2] = -fx * x / z ** 2
    Jp[:, 1, 1] = fy / z
    Jp[:, 1, 2] = -fy * y / z ** 2
    Dxi = np.zeros((n, 3, 6))
    Dxi[:, :, :3] = np.eye(3)
    # -hat(p) for every point
    Dxi[:, 0, 4], Dxi[:, 0, 5] = z, -y
    Dxi[:, 1, 3], Dxi[:, 1, 5] = -z, x
    Dxi[:, 2, 3], Dxi[:, 2, 4] = y, -x
    return Jp @ Dxi


def pnp_covariance(T_cw: np.ndarray, X_w: np.ndarray, uv: np.ndarray,
                   K: np.ndarray, inlier_px: float | None = None) -> dict:
    """Gauss-Newton (Laplace) covariance of a PnP pose: ``s^2 (J^T J)^-1``.

    ``s^2 = sum r^2 / (2n - 6)`` is the residual variance estimated on the
    inliers (points in front of the camera with reprojection error below
    ``inlier_px`` at ``T_cw``; all such points when ``inlier_px`` is None).
    This is the textbook covariance any PnP solver could report; it ignores
    the spatial correlation of the 2D-3D errors, so like our Laplace
    covariance it needs a temperature.

    Returns ``{"Sigma", "n_inliers", "rms_px", "sigma2"}``; ``Sigma`` is None
    when fewer than 4 inliers remain.
    """
    T_cw = np.asarray(T_cw, np.float64)
    X_w = np.asarray(X_w, np.float64)
    uv = np.asarray(uv, np.float64)
    r, z = pnp_residuals(T_cw, X_w, uv, K)
    m = z > 1e-6
    if inlier_px is not None:
        m &= np.linalg.norm(r, axis=1) < inlier_px
    n = int(m.sum())
    if n < 4:
        return {"Sigma": None, "n_inliers": n, "rms_px": float("nan"),
                "sigma2": float("nan")}
    r = r[m]
    J = pnp_jacobian(T_cw, X_w[m], K).reshape(-1, 6)
    sigma2 = float((r ** 2).sum() / (2 * n - 6))
    H = J.T @ J
    Sigma = sigma2 * np.linalg.inv(H)
    return {"Sigma": 0.5 * (Sigma + Sigma.T), "n_inliers": n,
            "rms_px": float(np.sqrt((r ** 2).sum(axis=1).mean())),
            "sigma2": sigma2}
