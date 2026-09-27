"""Uncertainty-aware rendering refinement and Laplace pose covariance
(Sec. IV-D of the draft).

Each coarse hypothesis ``T0`` is refined in the tangent space,
``T(xi) = exp(xi^) T0``, by minimizing Eq. 9:

    E(xi) = sum_x w(x) || I_q(x) - I_r(x; xi) ||^2  (+ sparse matching term)

with per-pixel weight ``w(x) = exp(-gamma * U_r(x))`` where ``U_r`` is the
alpha-blended primitive vacuity (Eq. 8) -- dynamic and ambiguous content is
down-weighted continuously rather than masked.

At the optimum the pose covariance is the Laplace approximation Eq. 10:

    Sigma_xi ~= sigma_hat^2 (J^T W J)^{-1}

with ``J`` the Jacobian of the stacked *weighted* residuals w.r.t. ``xi`` and
``sigma_hat^2`` estimated from the residuals.  The Jacobian is obtained by
forward-mode AD through the renderer (6 JVPs) with a central finite-difference
fallback; either way it reuses the refinement machinery and adds negligible
cost relative to the optimization itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import se3
from .rendering import Camera
from .primitives import EvidentialMap


@dataclass
class RefinementResult:
    T_star: torch.Tensor       # (4, 4) refined pose
    xi_star: torch.Tensor      # (6,) tangent update relative to T0
    Sigma: torch.Tensor        # (6, 6) Laplace covariance in the tangent space at T_star
    psnr: float                # render fidelity at the optimum (for pruning/weighting)
    final_loss: float
    uncertainty_map: torch.Tensor  # (H, W) U_r at the optimum


def _render(emap: EvidentialMap, renderer, T: torch.Tensor, camera: Camera):
    return renderer.render(
        emap.means, emap.covs, emap.colors, emap.opacities, T, camera,
        vacuity=emap.gaussian_vacuity,
    )


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def refine_pose(
    emap: EvidentialMap,
    renderer,
    camera: Camera,
    image_q: torch.Tensor,            # (H, W, 3) query image in [0, 1]
    T0: torch.Tensor,                 # (4, 4) coarse pose hypothesis
    gamma: float = 3.0,
    iters: int = 100,
    lr: float = 3e-3,
    match_term: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    match_weight: float = 0.0,
    compute_covariance: bool = True,
    jacobian_pixels: int = 4096,
) -> RefinementResult:
    """Gradient-based refinement of one hypothesis with vacuity weighting.

    ``match_term``, if given, maps the current pose (4, 4) to a scalar sparse
    matching residual (e.g. LightGlue keypoint distances, Eq. 9's second
    term); it is optional and off by default.
    """
    xi = torch.zeros(6, dtype=image_q.dtype, device=image_q.device, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)

    w_last = None
    for _ in range(iters):
        opt.zero_grad()
        T = se3.apply_tangent(T0, xi)
        out = _render(emap, renderer, T, camera)
        # Weights are treated as fixed at the current iterate (IRLS style):
        # gradients flow through the photometric residual, not through w.
        w = torch.exp(-gamma * out["vacuity"].detach())
        w_last = w
        loss = (w[..., None] * (image_q - out["rgb"]) ** 2).mean()
        if match_term is not None and match_weight > 0:
            loss = loss + match_weight * match_term(T)
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        xi_star = xi.detach().clone()
        T_star = se3.apply_tangent(T0, xi_star)
        out = _render(emap, renderer, T_star, camera)
        w = torch.exp(-gamma * out["vacuity"])
        psnr = _psnr(out["rgb"], image_q)
        final_loss = float((w[..., None] * (image_q - out["rgb"]) ** 2).mean())

    if compute_covariance:
        Sigma = laplace_covariance(
            emap, renderer, camera, image_q, T_star, w.detach(),
            max_pixels=jacobian_pixels,
        )
    else:
        Sigma = torch.eye(6, dtype=image_q.dtype, device=image_q.device)

    return RefinementResult(
        T_star=T_star, xi_star=xi_star, Sigma=Sigma,
        psnr=psnr, final_loss=final_loss, uncertainty_map=out["vacuity"],
    )


def _weighted_residuals(
    emap: EvidentialMap,
    renderer,
    camera: Camera,
    image_q: torch.Tensor,
    T_star: torch.Tensor,
    w: torch.Tensor,
    pix_idx: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns r(xi): (6,) -> (P*3,), the weighted residuals at exp(xi^) T_star."""
    sqrt_w = w.sqrt().reshape(-1)[pix_idx]                      # (P,)
    target = image_q.reshape(-1, 3)[pix_idx]                    # (P, 3)

    def r(xi: torch.Tensor) -> torch.Tensor:
        T = se3.apply_tangent(T_star, xi)
        out = _render(emap, renderer, T, camera)
        pred = out["rgb"].reshape(-1, 3)[pix_idx]
        return (sqrt_w[:, None] * (target - pred)).reshape(-1)

    return r


def residual_jacobian(
    r: Callable[[torch.Tensor], torch.Tensor],
    dim: int = 6,
    mode: str = "auto",
    fd_eps: float = 1e-4,
    dtype=torch.float32,
    device="cpu",
) -> torch.Tensor:
    """Jacobian of a residual function at xi = 0, (P, dim).

    ``mode``: "fwd" (forward-mode AD, 6 JVPs), "fd" (central differences,
    12 evaluations), or "auto" (try fwd, fall back to fd -- some rasterizer
    backends do not implement forward-mode derivatives).
    """
    xi0 = torch.zeros(dim, dtype=dtype, device=device)
    if mode in ("auto", "fwd"):
        try:
            return torch.func.jacfwd(r)(xi0)
        except Exception:
            if mode == "fwd":
                raise
    cols = []
    with torch.no_grad():
        for k in range(dim):
            e = torch.zeros(dim, dtype=dtype, device=device)
            e[k] = fd_eps
            cols.append((r(e) - r(-e)) / (2 * fd_eps))
    return torch.stack(cols, dim=-1)


def laplace_covariance(
    emap: EvidentialMap,
    renderer,
    camera: Camera,
    image_q: torch.Tensor,
    T_star: torch.Tensor,
    w: torch.Tensor,
    max_pixels: int = 4096,
    damping: float = 1e-6,
    mode: str = "auto",
) -> torch.Tensor:
    """Eq. 10: Sigma_xi ~= sigma_hat^2 (J^T J)^{-1} on the weighted residuals.

    The weight matrix W is folded into the residuals as sqrt(w), so J here is
    the Jacobian of the *weighted* residuals and J^T J = J_r^T W J_r.  Pixels
    are subsampled deterministically (strided) to bound cost; sigma_hat^2 is
    the residual variance on the same subset with 6 DoF removed.
    """
    H, W_, _ = image_q.shape
    n_pix = H * W_
    stride = max(1, n_pix // max_pixels)
    pix_idx = torch.arange(0, n_pix, stride, device=image_q.device)

    r = _weighted_residuals(emap, renderer, camera, image_q, T_star, w, pix_idx)
    J = residual_jacobian(r, mode=mode, dtype=image_q.dtype, device=image_q.device)
    with torch.no_grad():
        res = r(torch.zeros(6, dtype=image_q.dtype, device=image_q.device))
        n = res.numel()
        sigma2 = (res @ res) / max(n - 6, 1)
        JtJ = J.T @ J + damping * torch.eye(6, dtype=J.dtype, device=J.device)
        Sigma = sigma2 * torch.linalg.inv(JtJ)
        # Symmetrize against numerical drift.
        Sigma = 0.5 * (Sigma + Sigma.T)
        # Project onto the positive-definite cone. A near-singular J^T W J
        # (a pose direction the render does not constrain, e.g. a textureless
        # frontal wall) can leave tiny negative eigenvalues after inversion,
        # which make logdet and the NLL non-finite. Flooring the spectrum
        # keeps the covariance a valid Gaussian and is a no-op when the
        # Hessian is well conditioned.
        evals, evecs = torch.linalg.eigh(Sigma.double())
        floor = float(evals.max().clamp_min(1e-30)) * 1e-9
        evals = evals.clamp_min(max(floor, 1e-30))
        Sigma = (evecs @ torch.diag(evals) @ evecs.T).to(Sigma.dtype)
        Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma
