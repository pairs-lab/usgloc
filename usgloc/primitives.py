"""Evidential semantic Gaussian map (Sec. IV-A of the draft).

A reference primitive (Eq. 2) is

    G_i = (mu_i, Sigma_i, f_i, p_i, u_i)

built from a cluster of 3D Gaussians: geometry from the empirical mean and
covariance of member centroids, the CLIP embedding from a vacuity-weighted
average of member embeddings, and the semantic belief from DST fusion of the
member opinions (Eq. 5).  We additionally record the mean pairwise *conflict*
encountered during fusion: Dempster's rule renormalizes conflict away, so for
content whose observations disagree persistently (dynamic objects) the
conflict, not the fused vacuity, carries the signal.  ``effective_vacuity``
exposes ``min(1, u + conflict)`` for consumers that want a single scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from .evidential import FuseRule, fuse_opinions, opinion_to_probability


@dataclass
class EvidentialPrimitive:
    """Object-level evidential primitive G_i = (mu, Sigma, f, p, u)."""

    mu: torch.Tensor          # (3,) centroid
    Sigma: torch.Tensor       # (3, 3) anisotropic geometry covariance
    clip_feat: torch.Tensor   # (d,) vacuity-weighted CLIP embedding, l2-normalized
    prob: torch.Tensor        # (C,) expected semantic probability
    belief: torch.Tensor      # (C,) DST belief mass
    vacuity: torch.Tensor     # () fused vacuity mass u_i
    conflict: torch.Tensor    # () mean pairwise conflict eta accumulated in fusion
    member_idx: torch.Tensor  # (m,) indices of the member Gaussians in the raw map

    @property
    def effective_vacuity(self) -> torch.Tensor:
        """Vacuity inflated by fusion conflict, clamped to [0, 1]."""
        return (self.vacuity + self.conflict).clamp(0.0, 1.0)


@dataclass
class Submap:
    """Overlapping retrieval unit sampled along the mapping trajectory."""

    anchor_pose: torch.Tensor          # (4, 4) camera-from-world pose of the anchor keyframe
    primitive_idx: torch.Tensor        # (k,) indices into EvidentialMap.primitives
    descriptor: Optional[torch.Tensor] = None  # (D,) cached scene descriptor


@dataclass
class EvidentialMap:
    """The reference map M = {G_i} plus raw splats for rendering."""

    primitives: List[EvidentialPrimitive]
    submaps: List[Submap] = field(default_factory=list)
    # Raw 3DGS attributes kept for the rendering-based refinement stage.
    means: Optional[torch.Tensor] = None       # (N, 3)
    covs: Optional[torch.Tensor] = None        # (N, 3, 3)
    colors: Optional[torch.Tensor] = None      # (N, 3)
    opacities: Optional[torch.Tensor] = None   # (N,)
    gaussian_vacuity: Optional[torch.Tensor] = None  # (N,) per-splat vacuity for Eq. 8

    def primitive_tensor(self) -> dict:
        """Stacked per-primitive tensors for vectorized matching."""
        return {
            "mu": torch.stack([g.mu for g in self.primitives]),
            "clip": torch.stack([g.clip_feat for g in self.primitives]),
            "prob": torch.stack([g.prob for g in self.primitives]),
            "vacuity": torch.stack([g.effective_vacuity for g in self.primitives]),
        }

    def assign_gaussian_vacuity(self) -> None:
        """Broadcast each primitive's effective vacuity to its member splats.

        This is the per-splat ``u_i`` that Eq. 8 alpha-blends into the
        per-pixel uncertainty map during refinement.
        """
        if self.means is None:
            raise ValueError("map has no raw splats")
        u = torch.zeros(self.means.shape[0], dtype=self.means.dtype, device=self.means.device)
        for g in self.primitives:
            u[g.member_idx] = g.effective_vacuity.to(u.dtype)
        self.gaussian_vacuity = u


def cluster_gaussians(
    means: torch.Tensor,
    instance_ids: Optional[torch.Tensor] = None,
    distance_threshold: float = 0.25,
) -> List[torch.Tensor]:
    """Group raw Gaussians into object-level clusters.

    If per-Gaussian instance/identity features are available (e.g. from
    Gaussian Grouping or OpenGaussian), pass their argmax ids and clustering is
    exact.  Otherwise a simple single-linkage agglomeration on centroids is
    used as a stand-in (as in Chu & Kim, clustering is by spatial + semantic
    affinity; any instance segmentation of the splats can be substituted).
    """
    if instance_ids is not None:
        ids = instance_ids.unique()
        return [torch.nonzero(instance_ids == i, as_tuple=False).squeeze(-1) for i in ids]

    from scipy.cluster.hierarchy import fcluster, linkage

    pts = means.detach().cpu().numpy()
    if pts.shape[0] == 1:
        return [torch.tensor([0], device=means.device)]
    Z = linkage(pts, method="single")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")
    labels = torch.as_tensor(labels, device=means.device)
    return [torch.nonzero(labels == i, as_tuple=False).squeeze(-1) for i in labels.unique()]


def build_evidential_map(
    means: torch.Tensor,
    covs: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    clip_feats: torch.Tensor,
    beliefs: torch.Tensor,
    vacuities: torch.Tensor,
    instance_ids: Optional[torch.Tensor] = None,
    fuse_rule: FuseRule = "dempster",
    max_members_fused: int = 64,
) -> EvidentialMap:
    """Build the evidential map from per-Gaussian observations.

    Args:
        means/covs/colors/opacities: raw 3DGS attributes, (N, 3)/(N, 3, 3)/(N, 3)/(N,).
        clip_feats: (N, d) per-Gaussian open-vocabulary embeddings.
        beliefs: (N, C) per-Gaussian belief masses from the evidential head.
        vacuities: (N,) per-Gaussian vacuity masses.
        instance_ids: optional (N,) cluster ids; if None, spatial clustering.
        fuse_rule: "dempster" (Eq. 5) or "yager" (conflict -> vacuity).
        max_members_fused: subsample very large clusters for the DST fold,
            which is sequential; geometry/embedding still use all members.
    """
    clusters = cluster_gaussians(means, instance_ids)
    primitives: List[EvidentialPrimitive] = []
    for idx in clusters:
        m = means[idx]
        mu = m.mean(dim=0)
        if idx.numel() > 1:
            centered = m - mu
            Sigma = centered.T @ centered / (idx.numel() - 1)
        else:
            Sigma = covs[idx[0]]

        u_members = vacuities[idx]
        # Vacuity-weighted CLIP average: confident members dominate.
        w = (1.0 - u_members).clamp_min(1e-3)
        f = (clip_feats[idx] * w[:, None]).sum(dim=0)
        f = f / f.norm().clamp_min(1e-8)

        sel = idx
        if sel.numel() > max_members_fused:
            step = sel.numel() // max_members_fused
            sel = sel[::step][:max_members_fused]
        b, u, eta = fuse_opinions(beliefs[sel], vacuities[sel], rule=fuse_rule)
        prob = opinion_to_probability(b, u)

        primitives.append(
            EvidentialPrimitive(
                mu=mu, Sigma=Sigma, clip_feat=f, prob=prob,
                belief=b, vacuity=u, conflict=eta, member_idx=idx,
            )
        )

    emap = EvidentialMap(
        primitives=primitives,
        means=means, covs=covs, colors=colors, opacities=opacities,
    )
    emap.assign_gaussian_vacuity()
    return emap


def make_submaps(
    emap: EvidentialMap,
    anchor_poses: Sequence[torch.Tensor],
    radius: float = 3.0,
    camera=None,
    near: float = 0.05,
) -> None:
    """Partition the map into overlapping submaps along the trajectory.

    Each anchor keyframe pose (camera-from-world) defines a submap containing
    the primitives within ``radius`` of the camera centre, following the
    submap retrieval design of SGLoc.  The anchor pose doubles as the coarse
    pose hypothesis returned by retrieval.

    Args:
        camera: optional :class:`usgloc.rendering.Camera`.  When given, the
            submap is additionally FRUSTUM-CULLED: only primitives whose
            centroid projects into the anchor view (in front of the camera
            and inside the image bounds) are kept.  Without this, in a room
            smaller than ``radius`` every submap contains nearly every
            primitive, all reference descriptors collapse onto one vector,
            and retrieval training is stuck at chance (observed on chess:
            pairwise descriptor cosine 0.9996-1.0 and InfoNCE pinned at
            ln(batch)).  Culling is resolution-independent (only the FOV
            matters), so any resolution of the same intrinsics works.
    """
    mus = torch.stack([g.mu for g in emap.primitives])
    submaps = []
    for T in anchor_poses:
        cam_center = (-T[:3, :3].T @ T[:3, 3:4]).squeeze(-1)
        d = torch.linalg.norm(mus - cam_center, dim=-1)
        keep = d < radius
        if camera is not None:
            p_cam = mus @ T[:3, :3].T + T[:3, 3]
            z = p_cam[:, 2]
            u = camera.fx * p_cam[:, 0] / z.clamp(min=near) + camera.cx
            v = camera.fy * p_cam[:, 1] / z.clamp(min=near) + camera.cy
            keep = keep & (z > near) & (u >= 0) & (u < camera.width) \
                & (v >= 0) & (v < camera.height)
        inside = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if inside.numel() == 0:
            continue
        submaps.append(Submap(anchor_pose=T.clone(), primitive_idx=inside))
    emap.submaps = submaps
