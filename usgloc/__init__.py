"""USGLoc: Uncertainty-aware semantic visual localization from 3D Gaussian Splatting.

Implements the method of the RA-L draft (paper/usgloc_ral_draft.tex):

1.  Evidential semantic Gaussian map -- each object-level primitive carries a
    Dempster--Shafer belief over semantics plus an explicit vacuity mass
    (`evidential`, `primitives`).
2.  Evidential query encoding into the same belief space (`query`,
    `descriptors`).
3.  Uncertainty-weighted 2D--3D matching producing a weighted pose-hypothesis
    mixture (`matching`).
4.  Uncertainty-aware differentiable-rendering refinement with per-pixel
    vacuity weights, and a pose covariance via Laplace approximation
    (`rendering`, `refinement`).
5.  Gaussian-mixture pose posterior on SE(3) with single-Gaussian condensation
    for downstream filters (`posterior`).
6.  Calibration-centric evaluation: coverage, AUSE, NLL, ECE (`metrics`).

The full inference pipeline (Algorithm 1 of the draft) lives in `pipeline`.
"""

from . import se3
from .evidential import (
    evidence_to_subjective_opinion,
    dempster_fuse,
    yager_fuse,
    fuse_opinions,
    edl_loss,
)
from .primitives import EvidentialPrimitive, EvidentialMap, build_evidential_map
from .descriptors import InstanceDescriptorNet, SceneAggregator, info_nce_loss
from .matching import gated_similarity, retrieve_hypotheses, PoseHypothesis
from .calibration import (
    fit_temperature,
    fit_temperature_from_poses,
    temper,
    temper_mixture,
    calibrate,
    CalibrationReport,
)
from .baselines import (
    BaselineResult,
    icomma_style,
    top1_point,
    random_anchor,
    as_pose_gaussian,
)
from .datasets import FrameRecord, SevenScenes, load_gaussian_map
from .rendering import TorchSplatRenderer, GsplatRenderer, Camera
from .refinement import refine_pose, laplace_covariance, RefinementResult
from .posterior import PoseGaussian, PoseMixture
from .metrics import (
    pose_errors,
    success_rate,
    coverage,
    ause,
    coverage_ece,
)
from .pipeline import USGLocPipeline, PipelineConfig

__version__ = "0.1.0"
