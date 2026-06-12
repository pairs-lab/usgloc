"""In-framework baseline localizers for the comparison/ablation experiments
(Sec. V of the draft, paper/usgloc_ral_draft.tex).

Each baseline shares USGLoc's evidential map and differentiable renderer but
strips away one or more of the paper's contributions, so that table-level
differences are attributable to the contribution rather than to the backbone:

* :func:`icomma_style` -- photometric-only rendering refinement (gamma = 0, no
  vacuity weighting, no covariance): the iComMa-style render-and-compare
  point estimator.
* :func:`top1_point` -- SGLoc-style single-hypothesis retrieval + refinement
  (top-1 submap, no mixture, point estimate).
* :func:`random_anchor` -- a uniformly random submap anchor pose; the sanity
  floor for the result tables.
* :func:`as_pose_gaussian` -- wraps a point baseline in a fixed diagonal
  SE(3) tangent covariance so the calibration metrics (coverage, ECE, AUSE)
  can also be evaluated for point-estimate baselines.

Conventions follow the rest of the package: poses are 4x4 camera-from-world
(``T_cw``); tangent vectors are ``xi = (rho, phi)``, translation first;
images are (H, W, 3) float in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch

from .matching import retrieve_hypotheses
from .pipeline import QueryInstance, USGLocPipeline
from .posterior import PoseGaussian
from .primitives import EvidentialMap
from .refinement import refine_pose
from .rendering import Camera


@dataclass
class BaselineResult:
    """Point estimate produced by one of the baseline localizers.

    Attributes:
        T_est: (4, 4) estimated camera-from-world pose.
        psnr: Render fidelity at the estimate (dB); NaN if not rendered.
        name: Human-readable baseline name (used as the table row label).
        extras: Optional per-baseline diagnostics (e.g. retrieved submap).
    """

    T_est: torch.Tensor
    psnr: float
    name: str
    extras: Dict = field(default_factory=dict)


def icomma_style(
    emap: EvidentialMap,
    renderer,
    camera: Camera,
    image_q: torch.Tensor,
    T0: torch.Tensor,
    iters: int = 100,
    lr: float = 3e-3,
) -> BaselineResult:
    """Photometric-only pose refinement (iComMa-style).

    Runs the same tangent-space rendering refinement as USGLoc (Eq. 9) but
    with ``gamma = 0``, i.e. uniform per-pixel weights: the evidential
    vacuity weighting is disabled, so dynamic/ambiguous content pulls on the
    pose like everything else.  No Laplace covariance is computed -- this is
    a point estimator.

    Args:
        emap: Evidential map providing the splats to render.
        renderer: Differentiable splat renderer (``TorchSplatRenderer``-like).
        camera: Query camera intrinsics.
        image_q: (H, W, 3) query image in [0, 1].
        T0: (4, 4) coarse pose hypothesis (camera-from-world).
        iters: Refinement iterations.
        lr: Adam learning rate.

    Returns:
        BaselineResult with the refined pose, its PSNR, name 'iComMa-style'.
    """
    res = refine_pose(
        emap, renderer, camera, image_q, T0,
        gamma=0.0, iters=iters, lr=lr, compute_covariance=False,
    )
    return BaselineResult(
        T_est=res.T_star,
        psnr=res.psnr,
        name="iComMa-style",
        extras={"final_loss": res.final_loss},
    )


def top1_point(
    pipe: USGLocPipeline,
    image_q: torch.Tensor,
    instances: Sequence[QueryInstance],
    iters: Optional[int] = None,
) -> BaselineResult:
    """SGLoc-style single-hypothesis pipeline (top-1 retrieval, no mixture).

    Encodes the query with the pipeline's descriptor nets, retrieves only the
    single best submap, and refines that one hypothesis with the pipeline's
    configuration (vacuity weighting included, since it is the retrieval
    multi-hypothesis machinery being ablated here).  Returns a point
    estimate: no covariance, no GMM posterior.

    Args:
        pipe: A fully prepared ``USGLocPipeline`` (map encoded, nets trained).
        image_q: (H, W, 3) query image in [0, 1].
        instances: Segmented query instances (front-end output).
        iters: Optional override of ``pipe.cfg.refine_iters``.

    Returns:
        BaselineResult with the refined top-1 pose, name 'top1-point'.
    """
    _, scene_desc, _ = pipe.encode_query(instances)
    hyps = retrieve_hypotheses(pipe.emap, scene_desc, top_k=1)
    if not hyps:
        raise RuntimeError("retrieval returned no hypotheses")
    h = hyps[0]

    cfg = pipe.cfg
    res = refine_pose(
        pipe.emap, pipe.renderer, pipe.camera, image_q, h.T0,
        gamma=cfg.gamma,
        iters=cfg.refine_iters if iters is None else iters,
        lr=cfg.refine_lr,
        compute_covariance=False,
        jacobian_pixels=cfg.jacobian_pixels,
    )
    return BaselineResult(
        T_est=res.T_star,
        psnr=res.psnr,
        name="top1-point",
        extras={"submap_index": h.submap_index, "retrieval_weight": h.weight},
    )


def random_anchor(
    emap: EvidentialMap,
    generator: Optional[torch.Generator] = None,
) -> BaselineResult:
    """A uniformly random submap anchor pose -- the sanity floor.

    Any localizer worth tabulating must beat this; it ignores the query
    entirely and returns the anchor pose of a randomly chosen submap.

    Args:
        emap: Evidential map whose submaps provide the anchor poses.
        generator: Optional torch generator for reproducibility.

    Returns:
        BaselineResult with a random anchor pose, ``psnr = nan``,
        name 'random-anchor'.
    """
    if not emap.submaps:
        raise ValueError("map has no submaps to sample an anchor from")
    idx = int(torch.randint(len(emap.submaps), (1,), generator=generator))
    return BaselineResult(
        T_est=emap.submaps[idx].anchor_pose.clone(),
        psnr=float("nan"),
        name="random-anchor",
        extras={"submap_index": idx},
    )


def as_pose_gaussian(
    b: BaselineResult,
    sigma_t: float = 0.05,
    sigma_r: float = 0.05,
) -> PoseGaussian:
    """Wrap a point baseline in a fixed diagonal SE(3) tangent covariance.

    Point-estimate baselines report no uncertainty; to evaluate them with the
    calibration metrics (coverage, ECE, AUSE) we assign every estimate the
    same diagonal covariance -- ``sigma_t^2`` on the translation block
    ``rho`` (first three tangent coordinates) and ``sigma_r^2`` on the
    rotation block ``phi`` (last three).

    Args:
        b: A baseline point estimate.
        sigma_t: Translation std in metres.
        sigma_r: Rotation std in radians.

    Returns:
        PoseGaussian centred at ``b.T_est`` with the fixed diagonal Sigma.
    """
    dtype, device = b.T_est.dtype, b.T_est.device
    diag = torch.tensor(
        [sigma_t**2] * 3 + [sigma_r**2] * 3, dtype=dtype, device=device
    )
    return PoseGaussian(T_mode=b.T_est.clone(), Sigma=torch.diag(diag))
