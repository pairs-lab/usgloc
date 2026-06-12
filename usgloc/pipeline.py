"""End-to-end USGLoc inference -- Algorithm 1 of the draft.

    1. Segment + encode the query into evidential instances.
    2. Build query instance/scene descriptors; retrieve top-k submaps by
       gated similarity -> coarse hypotheses {(T0_k, pi_tilde_k)}.
    3. Refine each hypothesis by vacuity-weighted differentiable rendering;
       recover the component covariance by Laplace approximation.
    4. Prune by render fidelity; assemble the SE(3) Gaussian mixture; condense
       to a single Gaussian for the downstream filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from .descriptors import InstanceDescriptorNet, SceneAggregator
from .matching import PoseHypothesis, retrieve_hypotheses
from .posterior import PoseGaussian, PoseMixture
from .primitives import EvidentialMap
from .refinement import refine_pose
from .rendering import Camera


@dataclass
class QueryInstance:
    """One segmented instance of the query image, in the evidential space."""

    clip_feat: torch.Tensor   # (d,) l2-normalized
    centroid: torch.Tensor    # (2,) normalized [0, 1]^2
    area: torch.Tensor        # (1,) normalized mask area
    belief: torch.Tensor      # (C,)
    vacuity: torch.Tensor     # ()


@dataclass
class PipelineConfig:
    lambda_u: float = 2.0          # evidential gate sharpness (Eq. 6)
    gamma: float = 3.0             # per-pixel weight sharpness (Eq. 9)
    top_k: int = 3                 # pose hypotheses from retrieval
    refine_iters: int = 100
    refine_lr: float = 3e-3
    psnr_prune_margin: float = 6.0  # drop hypotheses > margin dB below the best
    psnr_temperature: float = 2.0   # fidelity -> weight softness
    jacobian_pixels: int = 4096
    compute_covariance: bool = True


@dataclass
class LocalizationResult:
    posterior: PoseMixture
    condensed: PoseGaussian
    hypotheses: List[PoseHypothesis] = field(default_factory=list)
    psnrs: List[float] = field(default_factory=list)


def submap_reference_inputs(emap: EvidentialMap, submap):
    """Per-submap reference inputs with geometry in the anchor camera frame.

    The draft forms reference descriptors from ``(f_i, mu_i, Sigma_i)``
    *projected into the camera frame*; expressing the geometry relative to the
    submap anchor makes the 3D side comparable to the view-dependent 2D side.
    """
    prims = [emap.primitives[int(i)] for i in submap.primitive_idx]
    mu = torch.stack([g.mu for g in prims])
    Sigma = torch.stack([g.Sigma for g in prims])
    clip = torch.stack([g.clip_feat for g in prims])
    vac = torch.stack([g.effective_vacuity for g in prims])
    R, t = submap.anchor_pose[:3, :3], submap.anchor_pose[:3, 3]
    mu_cam = mu @ R.T + t
    Sigma_cam = R @ Sigma @ R.T
    return clip, mu_cam, Sigma_cam, vac


class USGLocPipeline:
    def __init__(
        self,
        emap: EvidentialMap,
        renderer,
        camera: Camera,
        descriptor_net: InstanceDescriptorNet,
        aggregator: SceneAggregator,
        config: Optional[PipelineConfig] = None,
    ):
        self.emap = emap
        self.renderer = renderer
        self.camera = camera
        self.desc_net = descriptor_net
        self.aggregator = aggregator
        self.cfg = config or PipelineConfig()

    # ------------------------------------------------------------------ map
    @torch.no_grad()
    def encode_map(self) -> None:
        """Cache a scene descriptor per submap (reference side of retrieval)."""
        for sm in self.emap.submaps:
            clip, mu_cam, Sigma_cam, vac = submap_reference_inputs(self.emap, sm)
            d = self.desc_net.encode_reference(clip, mu_cam, Sigma_cam)[None]
            sm.descriptor = self.aggregator(d, vac[None])[0]

    # ---------------------------------------------------------------- query
    @torch.no_grad()
    def encode_query(self, instances: Sequence[QueryInstance]):
        clip = torch.stack([q.clip_feat for q in instances])
        cen = torch.stack([q.centroid for q in instances])
        area = torch.stack([q.area for q in instances])
        vac = torch.stack([q.vacuity for q in instances])
        inst_desc = self.desc_net.encode_query(clip, cen, area)
        scene_desc = self.aggregator(inst_desc[None], vac[None])[0]
        return inst_desc, scene_desc, vac

    # ------------------------------------------------------------- localize
    def localize(
        self,
        image_q: torch.Tensor,
        instances: Sequence[QueryInstance],
    ) -> LocalizationResult:
        _, scene_desc, _ = self.encode_query(instances)
        hyps = retrieve_hypotheses(self.emap, scene_desc, top_k=self.cfg.top_k)
        if not hyps:
            raise RuntimeError("retrieval returned no hypotheses")

        results = []
        for h in hyps:
            res = refine_pose(
                self.emap, self.renderer, self.camera, image_q, h.T0,
                gamma=self.cfg.gamma,
                iters=self.cfg.refine_iters,
                lr=self.cfg.refine_lr,
                compute_covariance=self.cfg.compute_covariance,
                jacobian_pixels=self.cfg.jacobian_pixels,
            )
            results.append((h, res))

        # Fidelity pruning: keep hypotheses within a PSNR margin of the best.
        best_psnr = max(r.psnr for _, r in results)
        kept = [(h, r) for h, r in results if r.psnr >= best_psnr - self.cfg.psnr_prune_margin]

        # pi_k from retrieval weight x render fidelity (Eq. 12 weights).
        raw = torch.tensor(
            [h.weight for h, _ in kept], dtype=image_q.dtype, device=image_q.device
        )
        fid = torch.tensor(
            [r.psnr for _, r in kept], dtype=image_q.dtype, device=image_q.device
        )
        weights = raw * torch.softmax(fid / self.cfg.psnr_temperature, dim=0)

        components = [PoseGaussian(T_mode=r.T_star, Sigma=r.Sigma) for _, r in kept]
        posterior = PoseMixture(components=components, weights=weights)
        return LocalizationResult(
            posterior=posterior,
            condensed=posterior.condense(),
            hypotheses=[h for h, _ in kept],
            psnrs=[r.psnr for _, r in kept],
        )
