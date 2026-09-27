"""Differentiable rendering with an alpha-blended vacuity channel (Eq. 8).

Two interchangeable backends:

* :class:`TorchSplatRenderer` -- a self-contained, pure-PyTorch EWA splatter.
  O(N_gaussians x N_pixels) memory, intended for small images (tests, the
  synthetic demo, and Jacobian evaluation during refinement).  Fully
  differentiable w.r.t. the camera pose and all Gaussian attributes, and
  supports forward-mode AD (used for the Laplace Jacobian).
* :class:`GsplatRenderer` -- wraps `gsplat.rasterization` for full-resolution
  use on real scenes.  The vacuity map costs nothing extra: the per-splat
  vacuity is appended to the color channels and alpha-blended by the same
  kernel, exactly as Eq. 8 prescribes.  The same mechanism carries the
  optional per-Gaussian LOCAL DESCRIPTOR field (:mod:`usgloc.locfeat`): pass
  ``loc_feats`` and one rasterization returns rgb + vacuity + descriptors.

Both return a dict with ``rgb`` (H, W, 3), ``vacuity`` (H, W) and ``alpha``
(H, W); :class:`GsplatRenderer` adds ``depth``, ``xyz``, ``meta`` and
``feats`` on request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            device=device, dtype=dtype,
        )


class TorchSplatRenderer:
    """Pure-PyTorch differentiable Gaussian splatting (EWA approximation)."""

    def __init__(self, near: float = 0.05, alpha_clip: float = 0.995, blur: float = 0.3):
        self.near = near
        self.alpha_clip = alpha_clip
        self.blur = blur  # screen-space dilation (pixels), as in 3DGS

    def render(
        self,
        means: torch.Tensor,        # (N, 3) world
        covs: torch.Tensor,         # (N, 3, 3) world
        colors: torch.Tensor,       # (N, 3)
        opacities: torch.Tensor,    # (N,)
        T_cw: torch.Tensor,         # (4, 4) camera-from-world
        camera: Camera,
        vacuity: Optional[torch.Tensor] = None,  # (N,) per-splat vacuity u_i
    ) -> Dict[str, torch.Tensor]:
        R, t = T_cw[:3, :3], T_cw[:3, 3]
        p_cam = means @ R.T + t                       # (N, 3)
        z = p_cam[:, 2]
        valid = z > self.near
        p, zv = p_cam[valid], z[valid]
        cov_v = covs[valid]
        col_v, op_v = colors[valid], opacities[valid]
        vac_v = vacuity[valid] if vacuity is not None else None

        H, W = camera.height, camera.width
        dev, dt = means.device, means.dtype
        if p.shape[0] == 0:
            out = {
                "rgb": torch.zeros(H, W, 3, device=dev, dtype=dt),
                "vacuity": torch.zeros(H, W, device=dev, dtype=dt),
                "alpha": torch.zeros(H, W, device=dev, dtype=dt),
            }
            return out

        # Project means.
        u = camera.fx * p[:, 0] / zv + camera.cx
        v = camera.fy * p[:, 1] / zv + camera.cy
        mean2d = torch.stack([u, v], dim=-1)          # (M, 2)

        # 2D covariance: Sigma2D = J R Sigma Rt Jt + blur * I.
        # Assembled with stack (not in-place) to stay forward-mode-AD safe.
        zinv = 1.0 / zv
        zero = torch.zeros_like(zinv)
        row0 = torch.stack([camera.fx * zinv, zero, -camera.fx * p[:, 0] * zinv**2], dim=-1)
        row1 = torch.stack([zero, camera.fy * zinv, -camera.fy * p[:, 1] * zinv**2], dim=-1)
        J = torch.stack([row0, row1], dim=-2)
        cov_cam = R @ cov_v @ R.T
        cov2d = J @ cov_cam @ J.transpose(-1, -2)
        cov2d = cov2d + self.blur * torch.eye(2, device=dev, dtype=dt)

        # Inverse of the 2x2 covariance, in closed form.
        a, b = cov2d[:, 0, 0], cov2d[:, 0, 1]
        c = cov2d[:, 1, 1]
        det = (a * c - b * b).clamp_min(1e-8)
        ia, ib, ic = c / det, -b / det, a / det

        # Pixel grid.
        ys = torch.arange(H, device=dev, dtype=dt) + 0.5
        xs = torch.arange(W, device=dev, dtype=dt) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dx = gx[None] - mean2d[:, 0, None, None]      # (M, H, W)
        dy = gy[None] - mean2d[:, 1, None, None]

        # Gaussian falloff and per-splat alpha.
        q = ia[:, None, None] * dx**2 + 2 * ib[:, None, None] * dx * dy + ic[:, None, None] * dy**2
        alpha = (op_v[:, None, None] * torch.exp(-0.5 * q)).clamp(0.0, self.alpha_clip)

        # Front-to-back compositing in depth order.
        order = torch.argsort(zv)
        alpha = alpha[order]
        col_s = col_v[order]
        one_minus = (1.0 - alpha).clamp_min(1e-10)
        trans = torch.cumprod(one_minus, dim=0)
        trans = torch.cat([torch.ones_like(trans[:1]), trans[:-1]], dim=0)
        w = alpha * trans                              # (M, H, W) blend weights

        rgb = (w[..., None] * col_s[:, None, None, :]).sum(dim=0)
        acc = w.sum(dim=0)
        out = {"rgb": rgb, "alpha": acc}
        if vac_v is not None:
            out["vacuity"] = (w * vac_v[order][:, None, None]).sum(dim=0)
        else:
            out["vacuity"] = torch.zeros_like(acc)
        return out


class GsplatRenderer:
    """gsplat-backed renderer; vacuity rendered as a 4th color channel."""

    def __init__(self, near: float = 0.05, eigh_chunk: int = 65536):
        self.near = near
        self.eigh_chunk = eigh_chunk
        # The map is fixed during localization, so its covariance
        # decomposition is computed once and cached (keyed by storage
        # pointer + size); recomputing it per render call would dominate the
        # refinement loop.
        self._cov_cache = None

    @staticmethod
    def _cov_to_quat_scale(covs: torch.Tensor, chunk: int = 65536):
        """Decompose (N, 3, 3) covariances into unit quaternions and scales.

        The eigh runs on CPU in float64: cusolver's batched solver rejects
        large batches outright (CUSOLVER_STATUS_INVALID_VALUE at 64k-330k
        matrices, torch 2.11/cu128), and since render() caches this
        decomposition per map, the one-time CPU cost is irrelevant.
        """
        dev, dt = covs.device, covs.dtype
        evals, evecs = torch.linalg.eigh(covs.detach().cpu().to(torch.float64))
        evals = evals.to(device=dev, dtype=dt)
        evecs = evecs.to(device=dev, dtype=dt)
        # Ensure right-handed rotation matrices.
        det = torch.linalg.det(evecs)
        evecs = evecs * torch.where(det < 0, -1.0, 1.0)[:, None, None]
        scales = evals.clamp_min(1e-12).sqrt()
        R = evecs
        # Rotation matrix -> quaternion (w, x, y, z), batched and branch-free
        # via the trace method with stabilization.
        w = (1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp_min(1e-8).sqrt() / 2.0
        x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
        y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
        z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
        quats = torch.stack([w, x, y, z], dim=-1)
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return quats, scales

    def render(
        self,
        means: torch.Tensor,
        covs: torch.Tensor,
        colors: torch.Tensor,        # (N, 3) diffuse RGB, or (N, K, 3) SH coeffs
        opacities: torch.Tensor,
        T_cw: torch.Tensor,
        camera: Camera,
        vacuity: Optional[torch.Tensor] = None,
        with_depth: bool = False,
        with_xyz: bool = False,
        with_meta: bool = False,
        loc_feats: Optional[torch.Tensor] = None,
        feat_normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Rasterize rgb + vacuity (+ depth / xyz / local descriptors).

        ``loc_feats`` (N, D), when given, is alpha-blended by the SAME
        rasterization as the colour and vacuity channels -- one projection, one
        tile binning -- and returned as ``out["feats"]`` (H, W, D).  With
        ``feat_normalize`` the rendered descriptor map is L2-normalised per
        pixel (STDLoc's convention: normalise per-Gaussian before blending,
        normalise again per pixel after).  ``loc_feats`` must already be on
        ``means``' device and dtype.

        gsplat rasterizes at most ``channel_chunk`` (32 in gsplat 1.5.x)
        channels per ``rasterize_to_pixels`` launch, so a D-dimensional
        descriptor costs ``ceil((4 + D + ...) / 32)`` pixel passes instead of
        one -- the projection and tile binning are shared.  Leaving
        ``loc_feats=None`` reproduces the previous channel layout and the
        previous cost exactly.
        """
        from gsplat import rasterization

        key = (covs.data_ptr(), covs.shape[0])
        if (
            self._cov_cache is not None
            and self._cov_cache[0] == key
            and not covs.requires_grad
        ):
            _, quats, scales = self._cov_cache
        else:
            quats, scales = self._cov_to_quat_scale(covs, self.eigh_chunk)
            if not covs.requires_grad:
                self._cov_cache = (key, quats, scales)
        if vacuity is None:
            vacuity = torch.zeros_like(opacities)
        if colors.dim() == 3:
            # View-dependent appearance: colors is (N, K, 3) spherical-harmonic
            # coefficients instead of (N, 3) diffuse RGB.  We evaluate the SH
            # here, per Gaussian, against this view's direction and hand the
            # resulting (N, 3) to the SAME single rasterization pass, so the
            # vacuity / descriptor / xyz channels keep travelling with the
            # colour exactly as before.  (gsplat's own sh_degree= path cannot be
            # used: it owns the whole `colors` tensor and leaves no room for our
            # extra channels.)
            #
            # A (N, 3) tensor takes the original code path byte for byte, so
            # every existing map and every published number is unaffected.
            from gsplat import spherical_harmonics
            K = colors.shape[1]
            deg = int(round(K ** 0.5)) - 1
            if (deg + 1) ** 2 != K:
                raise ValueError(
                    f"colors has {K} SH coefficients, which is not (deg+1)^2")
            R = T_cw[:3, :3]
            t = T_cw[:3, 3]
            cam_c = -(R.transpose(0, 1) @ t)          # camera centre, world
            dirs = means - cam_c[None]
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rgb = spherical_harmonics(deg, dirs, colors.to(dirs.dtype))
            colors = (rgb + 0.5).clamp(0.0, 1.0)     # 3DGS convention: SH -> RGB
        feats = torch.cat([colors, vacuity[:, None]], dim=-1)  # (N, 4)
        c = 4                                                  # next free channel
        f0 = f1 = None
        if loc_feats is not None:
            if loc_feats.shape[0] != means.shape[0]:
                raise ValueError(
                    f"loc_feats has {loc_feats.shape[0]} rows for "
                    f"{means.shape[0]} gaussians"
                )
            f0, f1 = c, c + loc_feats.shape[-1]
            feats = torch.cat([feats, loc_feats], dim=-1)
            c = f1
        if with_xyz:
            # Alpha-blend the Gaussian world centres as three extra channels.
            # Dividing by alpha gives the expected world point seen at each
            # pixel WITHOUT inverting the camera model through a blended
            # depth, which is where silhouettes invent phantom surfaces.
            x0, c = c, c + 3
            feats = torch.cat([feats, means], dim=-1)          # (N, 7)
        render, alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=feats,
            viewmats=T_cw[None],
            Ks=camera.K(device=means.device, dtype=means.dtype)[None],
            width=camera.width,
            height=camera.height,
            near_plane=self.near,
            render_mode="RGB+ED" if with_depth else "RGB",
            # colors carries [rgb, vacuity] and optionally [loc_feats], [x, y, z]

        )
        out = {
            "rgb": render[0, ..., :3],
            "vacuity": render[0, ..., 3],
            "alpha": alphas[0, ..., 0],
        }
        if with_depth:
            # "RGB+ED": expected (alpha-normalized) z-depth as the last channel.
            out["depth"] = render[0, ..., c]
        if loc_feats is not None:
            fm = render[0, ..., f0:f1]
            if feat_normalize:
                fm = torch.nn.functional.normalize(fm, p=2, dim=-1)
            out["feats"] = fm
        if with_xyz:
            a = alphas[0, ..., 0:1].clamp_min(1e-6)
            out["xyz"] = render[0, ..., x0:x0 + 3] / a
        if with_meta:
            # Tile lists + projected primitives, for usgloc.lift.ray_depths:
            # lets a caller recompute the exact per-ray compositing (hence
            # mode / median / banded depth) for a handful of pixels.
            out["meta"] = _meta
        return out
