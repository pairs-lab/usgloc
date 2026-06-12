"""Instance and scene descriptors (Sec. IV-B/IV-C of the draft).

* :class:`InstanceDescriptorNet` -- Eq. 6: a query instance descriptor
  ``d_j = MLP[ phi(f_j) || psi_x(x_j) || psi_a(a_j) ]`` from the CLIP feature,
  the normalized 2D centroid and the area; reference descriptors are formed
  analogously from ``(f_i, mu_i, Sigma_i)`` projected into the camera frame.
* :class:`SceneAggregator` -- SGLoc-style multi-head self-attention + FFN with
  attention pooling, with the evidential gate injected at the instance level
  so retrieval inherits the uncertainty weighting.
* :func:`info_nce_loss` -- symmetric InfoNCE aligning 2D and 3D scene
  descriptors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class InstanceDescriptorNet(nn.Module):
    """Shared-space instance descriptors for query (2D) and reference (3D).

    The CLIP projection ``phi`` is shared between modalities; the geometric
    encoders differ (2D centroid+area vs. 3D centroid+covariance summary).
    """

    def __init__(self, clip_dim: int, out_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.phi = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, out_dim))
        self.psi_x = _MLP(2, out_dim // 4, hidden // 2)   # normalized centroid
        self.psi_a = _MLP(1, out_dim // 4, hidden // 2)   # area / scale
        self.psi_g3 = _MLP(9, out_dim // 2, hidden)       # 3D mu + Sigma summary
        self.head_q = _MLP(out_dim + out_dim // 2, out_dim, hidden)
        self.head_g = _MLP(out_dim + out_dim // 2, out_dim, hidden)

    def encode_query(
        self, clip_feat: torch.Tensor, centroid: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """(N, d), (N, 2), (N, 1) -> (N, out_dim)."""
        geo = torch.cat([self.psi_x(centroid), self.psi_a(area)], dim=-1)
        d = self.head_q(torch.cat([self.phi(clip_feat), geo], dim=-1))
        return F.normalize(d, dim=-1)

    def encode_reference(
        self, clip_feat: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        """(N, d), (N, 3), (N, 3, 3) -> (N, out_dim).

        The covariance enters through its diagonal and off-diagonal entries
        (6 numbers) concatenated with the centroid -- a cheap pose-agnostic
        geometry summary.
        """
        tri = torch.stack(
            [
                Sigma[..., 0, 0], Sigma[..., 1, 1], Sigma[..., 2, 2],
                Sigma[..., 0, 1], Sigma[..., 0, 2], Sigma[..., 1, 2],
            ],
            dim=-1,
        )
        geo = self.psi_g3(torch.cat([mu, tri], dim=-1))
        d = self.head_g(torch.cat([self.phi(clip_feat), geo], dim=-1))
        return F.normalize(d, dim=-1)


class SceneAggregator(nn.Module):
    """Instance set -> scene descriptor with evidential instance weighting.

    Following SGLoc: ``d_hat = Q + Attn(Q, K, V)``, ``D_hat = d_hat + FFN(d_hat)``,
    attention weights ``W = softmax(MLP(D_hat))`` and ``F = sum_j D_hat_j W_j``.
    The evidential gate ``exp(-lambda_u * u_j)`` multiplies the pooling logits
    so high-vacuity instances contribute weakly to the scene descriptor.
    """

    def __init__(self, dim: int = 256, heads: int = 4, lambda_u: float = 2.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.pool = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
        self.lambda_u = lambda_u

    def forward(
        self, inst: torch.Tensor, vacuity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """(B, N, dim) instances, optional (B, N) vacuities -> (B, dim)."""
        a, _ = self.attn(inst, inst, inst, need_weights=False)
        d_hat = inst + a
        D_hat = d_hat + self.ffn(d_hat)
        logits = self.pool(D_hat).squeeze(-1)  # (B, N)
        if vacuity is not None:
            logits = logits - self.lambda_u * vacuity  # gate in log space
        W = torch.softmax(logits, dim=-1)
        F_scene = (D_hat * W[..., None]).sum(dim=1)
        return F.normalize(F_scene, dim=-1)


def info_nce_loss(
    f2d: torch.Tensor, f3d: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Symmetric InfoNCE between paired 2D/3D scene descriptors (B, D)."""
    logits = f2d @ f3d.T / temperature
    labels = torch.arange(f2d.shape[0], device=f2d.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
