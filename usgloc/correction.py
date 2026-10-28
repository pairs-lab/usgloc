"""Evidential matching correction (render -> match -> lift -> PnP).

The photometric refinement of Sec. IV-D has a convergence basin of only
~10-35 cm, while retrieval initializes 13-30 cm from the answer and
sometimes picks the wrong region entirely.  This module closes that gap the
way the sub-centimeter 3DGS localizers do (GS-CPR, STDLoc, SplatHLoc): render
the evidential map at a pose hypothesis, match the query against the render
with a dense indoor matcher (LoFTR, ScanNet weights), lift the rendered
keypoints to 3D through the rendered depth, and solve PnP-RANSAC for a
corrected pose.  Its basin is set by the matcher's view overlap (~meters),
so it *rescues* hypotheses the photometric stage cannot reach, and at close
range it *polishes* below the photometric noise floor.

The evidential channel keeps its role: each correspondence is gated by the
rendered vacuity at its render-side keypoint, exactly the same signal that
weights the photometric residual, w(x) = exp(-gamma * U_r(x)).  Matches on
vacuous content (unobserved regions, floaters, ambiguous geometry) are
dropped before RANSAC.

Conventions: poses are T_cw (camera-from-world), the same matrices gsplat
consumes as ``viewmats`` and cv2's PnP returns as [R|t].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import se3
from .lift import ray_depths
from .primitives import EvidentialMap
from .rendering import Camera

# Lift modes that need the rasterizer's tile lists (usgloc.lift.ray_depths).
# 'depth' (alpha-blended expected depth) and 'xyz' keep the old paths.
_RAY_LIFTS = ("maxw", "median", "band", "maxw_d", "median_d", "band_d")


def _sc_of(c, cands, sc):
    """The arbiter's score for the incumbent, or -inf if it is not a candidate."""
    for i, x in enumerate(cands):
        if x is c:
            return sc[i]
    return -float("inf")


@dataclass
class CorrectionResult:
    T: torch.Tensor          # (4, 4) corrected pose (T_cw); == input pose on failure
    success: bool            # >= min_inliers matches survived RANSAC
    inliers: int             # PnP-RANSAC inlier count (0 on failure)
    n_matches: int           # vacuity-gated LoFTR matches fed to RANSAC
    reproj_px: float         # RMS reprojection error of the inliers (px)
    rounds: int              # correction rounds actually run
    obj: Optional[np.ndarray] = None   # (n_inl, 3) world points of the best round's inliers
    img: Optional[np.ndarray] = None   # (n_inl, 2) their query pixels (for re-verification)
    # Vacuity-weighted inlier mass: sum over the RANSAC inliers of the paper's
    # own pixel weight w = exp(-gamma * U_r/alpha) evaluated at the render-side
    # keypoint (Eq. 9).  It is the evidential form of the inlier count -- an
    # inlier on content the map is unsure about contributes less -- and is the
    # default score ``search()`` ranks virtual views by.  Always computed; it
    # feeds nothing unless a caller asks for it, so no existing path changes.
    wmass: float = 0.0
    # ---- multi-round accumulation (accumulate != 'none'; all zero/False when off)
    acc_used: bool = False   # the returned pose came from the pooled multi-round solve
    acc_points: int = 0      # correspondences in the pooled set (after dedup)
    acc_rounds: int = 0      # rounds that contributed to the pool
    acc_inl: int = 0         # inliers of the returned pose on the pooled set
    acc_win: bool = False    # pooled pose did not lose pooled inliers vs the best round
    round_T: Optional[list] = None      # acc_debug: every round's pose (4x4 float64)
    round_inl: Optional[list] = None    # acc_debug: every round's inlier count
    round_rms: Optional[list] = None    # acc_debug: every round's inlier RMS (px)
    # ---- photometric arbitration (arbiter != 'none'); None when it did not run
    arb: Optional[dict] = None          # {'scores': [...], 'poses': [...], 'pick': i,
                                        #  'pick_inl': j} -- the candidate NCCs, their
                                        # poses, the arbiter's choice and the choice the
                                        # inlier rule would have made, so a caller can
                                        # score the arbiter against ground truth


@dataclass
class SearchResult:
    """Outcome of ``EvidentialMatchCorrector.search`` (virtual-view search)."""

    T: torch.Tensor            # best pose found (== the input pose on failure)
    T_view: torch.Tensor       # the virtual view the winner was rendered from
    best: CorrectionResult     # the winner's correction (after the refine pass)
    score: float               # winning score
    base_score: float          # score of sample 0, the UNPERTURBED view
    n_views: int               # views actually evaluated (<= n with stop_score)
    n_success: int             # views whose PnP succeeded
    best_index: int            # which sample won (0 = the unperturbed one)
    improved: bool             # a PERTURBED view won
    scores: Optional[list] = None


def _to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) in [0, 1] -> (H, W) luma."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)


# MASt3R is not pip-installable: it is used the way its authors ship it, as a
# source tree (mast3r/ next to a dust3r/ checkout that carries croco/).  The
# GS-CPR checkout used for the matcher comparison is the fallback location;
# set $MAST3R_ROOT to point elsewhere.
_MAST3R_ROOT_FALLBACK = "/home/thanhnc/gauss_ws/third_party"
_MAST3R_HF_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
_MAST3R_CKPT_FALLBACK = "/home/thanhnc/gauss_ws/third_party/mast3r_ckpt"

# JamMa (CVPR'25) is SplatHLoc's fine matcher: a learned coarse+sub-pixel
# matcher. It ships as a source tree, not a package, so it is used the way its
# authors do -- point JAMMA_ROOT at a checkout that holds JamMa/ and its
# configs. Weights are fetched by its own torch.hub call on first use.
_JAMMA_ROOT_FALLBACK = "/home/thanhnc/gauss_ws/src/usgloc/SplatHLoc"


def _load_jamma(root, device):
    """Import the JamMa sources and build the model with official weights."""
    import os
    import sys

    root = root or os.environ.get("JAMMA_ROOT") or _JAMMA_ROOT_FALLBACK
    jm = os.path.join(root, "JamMa")
    for pth in (root, jm):
        if os.path.isdir(pth) and pth not in sys.path:
            sys.path.insert(0, pth)
    try:
        from src.config.default import get_cfg_defaults
        from src.lightning.lightning_jamma import PL_JamMa
    except ImportError as e:
        raise ImportError(
            "matcher='jamma' needs the JamMa sources (a checkout holding "
            f"JamMa/src/...); set JAMMA_ROOT. Tried {root!r}. ({e})") from e
    cfg = get_cfg_defaults()
    cfgf = os.path.join(jm, "configs/jamma/indoor/test.py")
    if os.path.isfile(cfgf):
        cfg.merge_from_file(cfgf)
    model = PL_JamMa(cfg, pretrained_ckpt="official").eval().to(device)
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def _load_mast3r(root: Optional[str], ckpt: Optional[str], device):
    """Import the MASt3R sources and load the checkpoint.

    Returns ``(model, fast_reciprocal_NNs)``.  ``ckpt`` may be a Hugging Face
    repo id, a local snapshot directory (config.json + model.safetensors) or
    a ``.pth`` from the MASt3R release page.
    """
    import os
    import sys

    root = root or os.environ.get("MAST3R_ROOT") or _MAST3R_ROOT_FALLBACK
    if os.path.isdir(os.path.join(root, "mast3r")) and root not in sys.path:
        sys.path.insert(0, root)
    # croco/models/croco.py sets torch.backends.cuda.matmul.allow_tf32 = True
    # at import time, which would silently move every later fp32 matmul of
    # the refinement and the Laplace Jacobian to TF32. Import, then restore.
    _tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    try:
        from mast3r.model import AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "matcher='mast3r' needs the MASt3R sources (mast3r/ + dust3r/ with "
            "croco/) on sys.path; set MAST3R_ROOT to such a checkout, e.g. the "
            "GS-CPR repo, and `pip install roma huggingface_hub`") from e
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = _tf32[0], _tf32[1]
        torch.set_float32_matmul_precision(_tf32[2])
    ckpt = ckpt or os.environ.get("MAST3R_CKPT")
    if ckpt is None:
        # local snapshot (config.json + model.safetensors) if present, else
        # the Hugging Face id (2.75 GB download into ~/.cache/huggingface)
        ckpt = _MAST3R_CKPT_FALLBACK if os.path.isdir(_MAST3R_CKPT_FALLBACK) else _MAST3R_HF_ID
    model = AsymmetricMASt3R.from_pretrained(ckpt)
    return model.to(device).eval(), fast_reciprocal_NNs


class EvidentialMatchCorrector:
    """LoFTR + vacuity-gated PnP correction against the rendered map.

    Requires the gsplat backend (rendered depth).  The matcher runs at
    ``camera`` resolution -- use the full 640x480 frame: LoFTR's indoor
    weights are trained on ScanNet at exactly that size.

    ``matcher`` selects what the query is associated with: the rendered RGB
    (``'loftr'``, the default, and ``'mast3r'`` / ``'roma'``) or the map's own
    rasterised per-Gaussian descriptor field (``'feat'``, STDLoc's dense
    coarse-to-fine mechanism; ``'pooled'`` feeds both match sets to one PnP).
    All of them hand the SAME correspondence dict to the shared gate / lift /
    solver, so nothing downstream changes with the choice.
    """

    def __init__(
        self,
        emap: EvidentialMap,
        renderer,
        camera: Camera,
        gamma: float = 3.0,          # same sharpness as the pixel weight (Eq. 9)
        u_max: float = 0.9,          # drop matches on near-fully-vacuous content
        conf_min: float = 0.20,      # LoFTR coarse confidence; pushed INTO kornia.
                                     # 0.05 was tested end to end and is worse on
                                     # stairs/office and ~50% slower (11th refuted).
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.0,  # success also needs inliers >= ratio * fed matches;
                                        # 0 for LoFTR (its matches are sparse and clean),
                                        # 0.5 for MASt3R/RoMa, whose thousands of dense
                                        # candidates make min_inliers=8 vacuous
        reproj_thresh_px: float = 2.0,
        rounds: int = 2,
        min_depth: float = 0.10,
        min_alpha: float = 0.5,
        edge_rel: float = 0.10,      # drop matches near depth discontinuities
        solver: str = 'cv2_iter',    # cv2_iter | cv2_epnp | poselib
        lift: str = 'depth',         # depth | xyz (blended world position)
                                     # | maxw | median | band  -- single-surface
                                     #   depth from the rasterizer's own per-ray
                                     #   Gaussian list (usgloc.lift.ray_depths),
                                     #   sampled at the nearest pixel centre
                                     # | maxw_d | median_d | band_d  -- the same
                                     #   statistic applied as a DELTA on top of
                                     #   the bilinear expected depth, which keeps
                                     #   the sub-pixel interpolation
        lift_band: float = 0.02,     # 'band' half-width, relative to the depth
        use_guess: bool = False,     # seed PnP with the current pose
        gate_vacuity: bool = True,   # ablation switch
        matcher: str = 'loftr',      # loftr | mast3r | roma | feat | pooled
                                     # 'feat'   : STDLoc's dense feature matching
                                     #            against the RASTERISED
                                     #            per-Gaussian descriptor field
                                     #            (needs emap.loc_feats)
                                     # 'pooled' : LoFTR *and* 'feat' matches
                                     #            concatenated before the gate
                                     # 'stage'  : LoFTR for the first
                                     #            ``feat_stage`` rounds, the
                                     #            field for the rest -- LoFTR's
                                     #            wide basin to get close, the
                                     #            field's precision to polish
        feat_stage: int = 1,         # 'stage': how many leading LoFTR rounds.  Those
                                     # rounds move the hypothesis but do NOT compete
                                     # for the best-of-rounds result (they have ~2.4x
                                     # more inliers and would always win it); the
                                     # first successful field round takes over, and a
                                     # LoFTR round is returned only if no field round
                                     # ever succeeds.
        stage_warm: str = 'loftr',   # 'stage': which matcher runs the WARM-UP rounds.
                                     # The warm-up only has to land inside the field's
                                     # ~1 cm basin, so what matters is its basin, not
                                     # its precision -- and on the perturbed train-side
                                     # floor MASt3R is far better at that than LoFTR
                                     # (Stairs 0.68 vs 2.17, Office 0.91 vs 1.41 cm).
                                     # LoFTR leaves Stairs at ~2.2 cm, outside the
                                     # basin, which is why staged LoFTR loses there.
        arbiter: str = 'none',       # none | stage | rounds -- how the returned pose is
                                     # chosen.  'none' keeps the inlier-count rule above.
                                     # 'stage' re-renders the best warm-up pose and the
                                     # best field pose and keeps whichever explains the
                                     # query better (_photo_score), so an unconverged
                                     # warm-up can no longer hand a hopeless hypothesis
                                     # to the field and have it accepted on faith.
                                     # 'rounds' arbitrates over EVERY successful round.
        arb_min_cover: float = 0.05,  # a pose rendering less than this fraction of the
                                     # frame is unscorable, not merely bad
        arb_margin: float = 0.0,     # NCC the challenger must beat the incumbent by
        stop_dt: float = 0.0,        # >0: stop once a round moves the camera centre less
                                     # than this (metres) AND less than stop_dr degrees.
                                     # The field iteration converges monotonically but
                                     # slowly -- it is still improving at round 20 -- so
                                     # a useful round budget is large and most frames
                                     # reach their plateau long before it.  The gate is
                                     # on the UPDATE, so it costs nothing in accuracy on
                                     # the frames that are still moving.
        stop_dr: float = 0.0,        # companion rotation threshold (degrees)
        feat_gain: float = 1.0,      # >1: over-relax the FIELD rounds.  The field
                                     # iteration is monotone but slow -- it is still
                                     # closing on the answer at round 20 -- which is the
                                     # signature of a contraction whose rate is near 1.
                                     # Taking alpha times each round's tangent update
                                     # is the standard acceleration for that, and costs
                                     # nothing per round.  Capped by feat_gain_max so a
                                     # large early update is never amplified.
        feat_gain_max: float = 0.02,  # metres: updates above this are applied unscaled
        # ---- 'feat' / 'pooled': the rasterised descriptor field --------------
        # Mechanism: STDLoc (Zhai et al., CVPR'25) -- cited, not claimed.  The
        # query's dense SuperPoint descriptors are correlated with the
        # descriptor map the map itself rasterises at the hypothesis pose:
        # mutual-NN + dual-softmax on the 1/8 coarse grid, then a dual-softmax
        # argmax inside each matched cell's win x win window at full
        # resolution.  The result enters the SAME adapter contract as LoFTR's
        # (keypoints0 = query px, keypoints1 = render px, confidence), so the
        # vacuity gate, the depth-edge filter, the bilinear depth lift,
        # PnP-RANSAC and the multi-round logic downstream are untouched.
        feat_temp: float = 0.1,      # dual-softmax temperature (STDLoc: 0.1)
        feat_win: int = 8,           # fine window = the encoder's stride
        feat_coarse_thr: float = -1.0,   # minimum coarse score (-1 = keep every MNN pair)
        feat_conf_min: float = 0.0,  # minimum FINE dual-softmax score.  The field
                                     # has its own threshold because ``conf_min``
                                     # is LoFTR's coarse confidence and must keep
                                     # acting on LoFTR alone in 'pooled' -- the
                                     # field adapter therefore filters here and
                                     # reports confidence 1.0 for the survivors.
        feat_fine: str = 'argmax',   # how the fine window is read out:
                                     # 'argmax' one match per matched coarse cell
                                     #          (the window's single best pair)
                                     # 'mnn'    STDLoc's own choice: every MUTUAL
                                     #          nearest-neighbour pair inside the
                                     #          window, so one cell can contribute
                                     #          several correspondences and a cell
                                     #          with no mutual pair contributes none
        feat_fine_thr: float = 0.0,  # 'mnn' only: minimum fine dual-softmax score
                                     # (STDLoc's dense.fine_threshold = 0)
        feat_max: int = 20000,       # safety cap on the number of field matches fed
                                     # to the gate: the 'argmax' read-out is bounded
                                     # by the coarse grid (4800 cells at 640x480) but
                                     # 'mnn' is not -- a window of near-duplicate
                                     # descriptors (the fine map is a bilinear
                                     # upsample of a 1/8 map, so it happens) ties at
                                     # the row and column maxima and admits up to
                                     # win^4 pairs.  Keeps the highest-scoring ones.
        feat_top: float = 0.0,       # 0 < f < 1: keep only the top f fraction of the
                                     # field's matches by fine dual-softmax score
                                     # (its confident matches are much cleaner than
                                     # its median one, and in 'pooled' the field
                                     # otherwise dilutes LoFTR's cleaner set)
        feat_subpix: float = 0.0,    # >0: soft-argmax temperature for a sub-pixel
                                     # render-side position inside the 3x3
                                     # neighbourhood of the fine argmax (0 = the
                                     # integer STDLoc recipe)
        mast3r_ckpt: Optional[str] = None,   # HF id, local snapshot dir or .pth;
                                             # default: $MAST3R_CKPT or the HF id
        jamma_root: Optional[str] = None,    # dir holding JamMa/ sources; default
                                             # $JAMMA_ROOT or _JAMMA_ROOT_FALLBACK
        mast3r_root: Optional[str] = None,   # dir holding mast3r/ + dust3r/ sources;
                                             # default: $MAST3R_ROOT or _MAST3R_ROOT_FALLBACK
        mast3r_size: int = 512,      # long-side working resolution (multiple of 16)
        mast3r_subsample: int = 8,   # fast_reciprocal_NNs seed stride (MASt3R demo: 8)
        mast3r_conf: str = 'sim',    # sim | desc_conf  -- the `confidence` the gate sees
        mast3r_border: int = 3,      # drop matches within this many working-res px of the edge
        mast3r_block: int = 2 ** 14, # fast_reciprocal_NNs block: identical matches for any
                                     # value; 2**13 (demo) 0.77 s, 2**14 0.41 s (+0.1 GB),
                                     # >= 2**15 takes the unblocked path, 0.10 s but +2.3 GB
        mast3r_subpix: int = 0,      # 0: integer matches (MASt3R recipe); r > 0: refine the
                                     # render-side position by a soft-argmax of descriptor
                                     # similarity over a (2r+1)^2 window (sub-pixel)
        mast3r_subpix_tau: float = 0.02,  # soft-argmax temperature on cosine similarity
        roma_model: str = 'indoor',  # roma weights: indoor (ScanNet) | outdoor (MegaDepth)
        roma_num: int = 5000,        # matches drawn from RoMa's dense warp per round
        roma_seed: int = 0,          # seed of RoMa's multinomial match sampler (RNG forked)
        roma_coarse_res: int = 560,  # RoMa coarse (DINOv2) working resolution (square)
        roma_upsample_res: int = 864,  # RoMa fine (refined warp) resolution (square)
        # ---- multi-round accumulation (default OFF: the historical behaviour is
        # to keep the single best-inlier round and discard every other round).
        accumulate: str = 'none',    # none | union | weighted
        acc_dedup: str = 'mean',     # all | mean | best -- a query pixel matched in
                                     # several rounds yields several world points
        acc_solve: str = 'lm',       # lm | ransac -- how the pooled set is solved
        acc_rms_k: float = 2.0,      # 'weighted': drop a round whose inlier reprojection
                                     # RMS exceeds k * the best round's RMS
        acc_guard: bool = True,      # keep the pooled pose only if it does not lose
                                     # inliers on the pooled set vs the best round's pose
        acc_jitter: float = 0.0,     # metres of render-pose jitter on rounds >= 1: the
        acc_jitter_deg: float = 0.0, # degrees      pool then averages over genuinely
        acc_jitter_seed: int = 0,    # different render viewpoints, not near-identical ones
        acc_err_q: float = 0.0,      # >0: keep only pooled correspondences whose own
                                     # round reprojection error is below this quantile
        acc_thresh_px: float = 0.0,  # >0: gate the pooled set at this (tighter) threshold
                                     # instead of reproj_thresh_px -- pooling buys enough
                                     # correspondences to afford a stricter gate
        acc_refine: str = 'lm',      # lm | huber | cauchy -- pooled refinement (_irls)
        acc_min_seen: int = 0,       # >1: keep only query pixels matched (and kept as
                                     # an inlier) in at least this many rounds -- the
                                     # rounds vote on which correspondences to trust
        acc_spread_q: float = 0.0,   # 0<q<1, with acc_dedup='mean': of the pixels seen
                                     # in >= 2 rounds, keep the fraction q whose world
                                     # points agree best across rounds (small cross-round
                                     # 3D spread = a depth lift that survives re-rendering)
        acc_debug: bool = False,     # also return every round's pose / inliers / RMS
                                     # (diagnostic: are the rounds independent at all?)
        # ---- final refinement (default 'lm': the historical solvePnPRefineLM) ----
        refine: str = 'lm',          # lm | none | huber | cauchy | lm+huber | lm+cauchy
                                     # | l2 (our own Gauss-Newton L2, = 'lm' unless
                                     #   vacuity weights are on) | plb (poselib's own
                                     #   robust bundle over the inlier set)
                                     # NOTE 'lm' overwrites whatever the solver
                                     # produced with a plain L2 Levenberg-Marquardt;
                                     # with solver='poselib' that DISCARDS poselib's
                                     # Cauchy bundle adjustment.  Use 'none' to let
                                     # the poselib LO-RANSAC bundle stand.
        refine_sigma: float = 1.0,   # IRLS robust scale, pixels
        refine_iters: int = 10,      # IRLS Gauss-Newton iterations
        refine_all: bool = False,    # IRLS over every fed match instead of the
                                     # RANSAC inliers (the robust loss rejects)
        ransac_iters: int = 2000,    # cv2.solvePnPRansac iterationsCount
        # ---- poselib LO-RANSAC / bundle budget (defaults = the historical call) ----
        pl_max_iters: int = 10000,   # poselib ransac_opt max_iterations
        pl_min_iters: int = 1000,    # poselib ransac_opt min_iterations
        pl_loss: str = 'CAUCHY',     # poselib bundle_opt loss_type (poselib default)
        pl_loss_scale: float = 1.0,  # poselib bundle_opt loss_scale, px
        pl_bundle_iters: int = 100,  # poselib bundle_opt max_iterations
        # ---- vacuity as a weight instead of a boolean gate -------------------
        # The rendered vacuity already gates correspondences at u_max; these two
        # let the SAME signal act continuously, through the paper's own pixel
        # weight w_i = exp(-gamma * U_r/alpha) raised to an exponent.
        wvac: float = 0.0,           # >0: multiply the IRLS weight of each
                                     # correspondence by w_i ** wvac (0 = off)
        wvac_thr: float = 0.0,       # >0: re-select the inlier / refinement set
                                     # with a per-correspondence threshold
                                     # thr * (w_i / median w) ** wvac_thr, i.e. a
                                     # soft version of the u_max gate inside the
                                     # geometric verification
        wvac_clip: float = 3.0,      # clip that threshold ratio to [1/c, c]
        loftr_dir: str = 'qr',       # which image LoFTR refines to sub-pixel:
                                     # 'qr'  query = image0 (historical default):
                                     #       the RENDER-side keypoint is the
                                     #       refined one, so the matcher's
                                     #       localisation error enters the 3D
                                     #       point laterally through the lift
                                     # 'rq'  render = image0: the QUERY-side
                                     #       keypoint is refined instead and the
                                     #       lift happens on the coarse grid
                                     # 'sym' both passes, matches concatenated
    ) -> None:
        self.emap = emap
        self.renderer = renderer
        self.camera = camera
        self.gamma = gamma
        self.u_max = u_max
        self.conf_min = conf_min
        self.min_inliers = min_inliers
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.reproj_thresh_px = reproj_thresh_px
        self.rounds = rounds
        self.min_depth = min_depth
        self.min_alpha = min_alpha
        self.edge_rel = edge_rel
        self.solver = solver
        self.lift = lift
        self.use_guess = use_guess
        self.gate_vacuity = gate_vacuity
        self.matcher_name = matcher
        if accumulate not in ("none", "union", "weighted"):
            raise ValueError(f"accumulate must be none | union | weighted, got {accumulate!r}")
        if acc_dedup not in ("all", "mean", "best", "berr"):
            raise ValueError(f"acc_dedup must be all | mean | best | berr, got {acc_dedup!r}")
        if acc_solve not in ("lm", "ransac"):
            raise ValueError(f"acc_solve must be lm | ransac, got {acc_solve!r}")
        if acc_refine not in ("lm", "huber", "cauchy"):
            raise ValueError(f"acc_refine must be lm | huber | cauchy, got {acc_refine!r}")
        self.accumulate = accumulate
        self.acc_dedup = acc_dedup
        self.acc_solve = acc_solve
        self.acc_rms_k = float(acc_rms_k)
        self.acc_guard = bool(acc_guard)
        self.acc_jitter = float(acc_jitter)
        self.acc_jitter_deg = float(acc_jitter_deg)
        self.acc_jitter_seed = int(acc_jitter_seed)
        self.acc_err_q = float(acc_err_q)
        self.acc_thresh_px = float(acc_thresh_px)
        self.acc_refine = acc_refine
        self.acc_min_seen = int(acc_min_seen)
        self.acc_spread_q = float(acc_spread_q)
        self.acc_debug = bool(acc_debug)
        if refine not in ("lm", "none", "huber", "cauchy", "lm+huber", "lm+cauchy",
                          "l2", "plb", "lm+plb", "plb+cauchy"):
            raise ValueError(f"unknown refine {refine!r}")
        self.refine = refine
        self.refine_sigma = float(refine_sigma)
        self.refine_iters = int(refine_iters)
        self.refine_all = bool(refine_all)
        self.ransac_iters = int(ransac_iters)
        self.pl_max_iters = int(pl_max_iters)
        self.pl_min_iters = int(pl_min_iters)
        self.pl_loss = str(pl_loss)
        self.pl_loss_scale = float(pl_loss_scale)
        self.pl_bundle_iters = int(pl_bundle_iters)
        self.wvac = float(wvac)
        self.wvac_thr = float(wvac_thr)
        self.wvac_clip = float(wvac_clip)
        self.lift_band = float(lift_band)
        if loftr_dir not in ("qr", "rq", "sym"):
            raise ValueError(f"loftr_dir must be qr | rq | sym, got {loftr_dir!r}")
        self.loftr_dir = loftr_dir
        self.feat_temp = float(feat_temp)
        self.feat_win = int(feat_win)
        self.feat_coarse_thr = float(feat_coarse_thr)
        self.feat_conf_min = float(feat_conf_min)
        if feat_fine not in ("argmax", "mnn"):
            raise ValueError(f"feat_fine must be argmax | mnn, got {feat_fine!r}")
        self.feat_fine = feat_fine
        self.feat_fine_thr = float(feat_fine_thr)
        self.feat_max = int(feat_max)
        self.feat_stage = int(feat_stage)
        if stage_warm not in ("loftr", "mast3r", "jamma"):
            raise ValueError(
                f"stage_warm must be loftr | mast3r | jamma, got {stage_warm!r}")
        self.stage_warm = stage_warm
        if arbiter not in ("none", "stage", "rounds"):
            raise ValueError(f"arbiter must be none | stage | rounds, got {arbiter!r}")
        self.arbiter = arbiter
        self.arb_min_cover = float(arb_min_cover)
        self.arb_margin = float(arb_margin)
        self.stop_dt = float(stop_dt)
        self.stop_dr = float(stop_dr)
        self.feat_gain = float(feat_gain)
        self.feat_gain_max = float(feat_gain_max)
        self.feat_top = float(feat_top)
        self.feat_subpix = float(feat_subpix)
        # (N, D) rasterisable descriptor field; None on every RGB-matching path,
        # in which case render() takes the byte-for-byte original code path.
        self._lf = None
        dev = emap.means.device
        if matcher not in ("loftr", "mast3r", "roma", "feat", "pooled", "stage",
                           "jamma"):
            raise ValueError(
                f"unknown matcher {matcher!r} "
                "(loftr | mast3r | roma | feat | pooled | stage | jamma)")
        _needs_loftr = (matcher in ("loftr", "pooled")
                        or (matcher == "stage" and stage_warm == "loftr"))
        _needs_mast3r = (matcher == "mast3r"
                         or (matcher == "stage" and stage_warm == "mast3r"))
        _needs_jamma = (matcher == "jamma"
                        or (matcher == "stage" and stage_warm == "jamma"))
        if _needs_jamma:
            self.matcher = _load_jamma(jamma_root, dev)
        if _needs_loftr:
            import copy

            from kornia.feature import LoFTR
            from kornia.feature.loftr.loftr import default_cfg as _loftr_cfg

            # kornia's LoFTR applies its OWN coarse threshold (default 0.2)
            # inside the matcher, before any match is returned, so a smaller
            # conf_min here would silently be dead code. Push our threshold into
            # the matcher config so the stated value is the one that acts, and
            # let the geometric verifier do the rejecting -- the practice the
            # recent feature-matching localizers follow.
            cfg = copy.deepcopy(_loftr_cfg)
            cfg["match_coarse"]["thr"] = float(conf_min)
            self.matcher = LoFTR(pretrained="indoor_new", config=cfg).eval().to(dev)
            assert abs(self.matcher.coarse_matching.thr - conf_min) < 1e-9
        if _needs_mast3r:
            self.matcher, self._fast_nn = _load_mast3r(mast3r_root, mast3r_ckpt, dev)
            self.mast3r_subsample = int(mast3r_subsample)
            self.mast3r_conf = mast3r_conf
            self.mast3r_border = int(mast3r_border)
            self.mast3r_block = int(mast3r_block)
            self.mast3r_subpix = int(mast3r_subpix)
            self.mast3r_subpix_tau = float(mast3r_subpix_tau)
            # Working resolution: long side `mast3r_size`, aspect preserved,
            # both sides a multiple of the 16-px patch (640x480 -> 512x384).
            s = float(mast3r_size) / max(camera.width, camera.height)
            self._m_hw = (max(16, int(round(camera.height * s / 16)) * 16),
                          max(16, int(round(camera.width * s / 16)) * 16))
        if matcher == "roma":
            # RoMa (Edstedt et al., CVPR 2024): dense warp + certainty from a
            # frozen DINOv2-L/14 coarse encoder and a VGG19 fine refiner.
            # Images are resized to the square coarse / fine resolutions
            # internally and the warp comes back in normalized [-1, 1]
            # coordinates, so nothing here depends on 640x480.  The default
            # thresholded-balanced sampler (sample_thresh 0.05, attenuate_cert)
            # is the one the authors evaluate pose estimation with.
            import romatch

            if roma_model not in ("indoor", "outdoor"):
                raise ValueError(f"roma_model must be indoor | outdoor, got {roma_model!r}")
            from romatch.models.model_zoo import weight_urls

            builder = romatch.roma_indoor if roma_model == "indoor" else romatch.roma_outdoor
            # Load the checkpoints on the CPU (romatch's default maps them
            # straight onto the GPU, which doubles the transient footprint of
            # the 1.2 GB DINOv2 state dict on a shared card).
            w = torch.hub.load_state_dict_from_url(weight_urls["romatch"][roma_model],
                                                   map_location="cpu")
            w_dino = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                        map_location="cpu")
            # romatch defaults to a custom CUDA local-correlation kernel
            # (package `local_corr`, not on PyPI); its pure-PyTorch fallback
            # computes the same correlation, only slower.
            try:
                import local_corr  # noqa: F401  (pip package fused-local-corr)
                custom_corr = True
            except Exception:      # missing, or built for another torch/CUDA
                custom_corr = False
            self.matcher = builder(device=dev, weights=w, dinov2_weights=w_dino,
                                   coarse_res=int(roma_coarse_res),
                                   upsample_res=int(roma_upsample_res),
                                   use_custom_corr=custom_corr).eval()
            self.roma_custom_corr = custom_corr
            del w, w_dino
            self.roma_num = int(roma_num)
            self.roma_seed = int(roma_seed)
        # 'feat' needs no RGB matcher at all; the field is set up below. The
        # name was validated at the top of this block, so there is no else here:
        # each backend is built independently because 'stage' needs two of them.
        if matcher in ("feat", "pooled", "stage"):
            # The descriptor field must already be on the map: build it with
            # scripts/analysis/train_locfeat.py and either bake it into the .pt
            # or attach a sidecar with usgloc.locfeat.attach_loc_feats.
            if emap.loc_feats is None:
                raise ValueError(
                    f"matcher={matcher!r} needs a per-Gaussian descriptor field: "
                    "emap.loc_feats is None (build it with "
                    "scripts/analysis/train_locfeat.py and attach it with "
                    "usgloc.locfeat.attach_loc_feats)")
            from .superpoint import SuperPoint

            self._sp = SuperPoint().to(dev).eval()
            self._lf = emap.loc_feats.to(dev, emap.means.dtype)
            self._pca = None
            if emap.loc_feat_pca_basis is not None:
                # The query descriptors MUST get the map's own projection.
                self._pca = (emap.loc_feat_pca_mean.to(dev),
                             emap.loc_feat_pca_basis.to(dev))
            self._q_coarse = self._q_fine = None
        self._K = camera.K(device="cpu", dtype=torch.float64).numpy()
        # poselib consumes the intrinsics as a dict (PINHOLE fx, fy, cx, cy).
        self._pl_cam = {"model": "PINHOLE", "width": camera.width,
                        "height": camera.height,
                        "params": [self._K[0, 0], self._K[1, 1],
                                   self._K[0, 2], self._K[1, 2]]}

    @torch.no_grad()
    def _render(self, T: torch.Tensor):
        return self.renderer.render(
            self.emap.means, self.emap.covs, self.emap.colors,
            self.emap.opacities, T, self.camera,
            vacuity=self.emap.gaussian_vacuity, with_depth=True,
            with_xyz=(self.lift == "xyz"),
            with_meta=(self.lift in _RAY_LIFTS),
            loc_feats=self._lf,
        )

    @torch.no_grad()
    def _photo_score(self, T: torch.Tensor, gray_q: torch.Tensor) -> float:
        """How well does pose ``T`` explain the query, according to the map?

        Zero-mean normalised cross-correlation between the query luma and the
        render at ``T``, over the pixels the map actually covers.  Unlike an
        inlier count this does not depend on WHICH matcher proposed the pose,
        so a LoFTR round and a descriptor-field round become comparable: the
        field feeds ~2.4x fewer correspondences than LoFTR and would lose every
        count-based comparison regardless of whether it is right.  Higher is
        better; -inf when the pose renders almost nothing.
        """
        out = self._render(T)
        r = _to_gray(out["rgb"])
        m = out["alpha"] >= self.min_alpha
        if int(m.sum()) < self.arb_min_cover * m.numel():
            return -float("inf")
        a = r[m].double()
        b = gray_q.to(a.device, torch.float32)[m].double()
        a = a - a.mean()
        b = b - b.mean()
        den = (a.norm() * b.norm()).clamp_min(1e-12)
        return float((a * b).sum() / den)

    @staticmethod
    @torch.no_grad()
    def _tiny_desc(img: torch.Tensor) -> torch.Tensor:
        """Coarse global appearance descriptor (the classic tiny-image place
        descriptor): average-pool the luma to 24x32, remove the mean, L2
        normalise.

        Deliberately blind to fine alignment.  It exists to decide WHICH
        virtual view is worth a correction -- a 5-50 cm, which-part-of-the-room
        question -- and not to rank poses that already agree to a millimetre.
        The photometric arbiter measured that full-resolution NCC fails at the
        latter (it picks the best candidate 30-35% of the time), but that says
        nothing about this much coarser decision, where the competing views
        show visibly different content.
        """
        g = _to_gray(img) if img.dim() == 3 else img
        t = F.adaptive_avg_pool2d(g[None, None].float(), (24, 32))[0, 0]
        t = t.double().flatten()
        t = t - t.mean()
        return t / t.norm().clamp_min(1e-12)

    # ---- LoFTR adapter ----------------------------------------------------

    @torch.no_grad()
    def _match_loftr(self, img_q: torch.Tensor, img_r: torch.Tensor) -> dict:
        """LoFTR with a selectable direction.

        LoFTR is not symmetric: the coarse stage matches 1/8-resolution
        patches of image0 to image1, and the fine stage then localises the
        centre of each image0 patch to sub-pixel precision IN IMAGE1.  So with
        the historical ``'qr'`` (query = image0) the RENDER keypoint is the
        refined one -- the matcher's localisation error is baked into the ray
        the 3D point is lifted along, i.e. into the 3D point laterally.  With
        ``'rq'`` the roles swap: the lift happens on the exact coarse grid and
        the localisation error stays in the query pixel, where PnP measures
        it.  ``'sym'`` runs both and concatenates.
        """
        d = self.loftr_dir
        if d in ("qr", "sym"):
            c = self.matcher({"image0": img_q, "image1": img_r})
            a = {"keypoints0": c["keypoints0"], "keypoints1": c["keypoints1"],
                 "confidence": c["confidence"]}
            if d == "qr":
                return a
        c = self.matcher({"image0": img_r, "image1": img_q})
        b = {"keypoints0": c["keypoints1"], "keypoints1": c["keypoints0"],
             "confidence": c["confidence"]}
        if d == "rq":
            return b
        return {k: torch.cat([a[k], b[k]], dim=0) for k in a}

    # ---- descriptor-field adapter ('feat') --------------------------------
    # Same contract as the LoFTR call: keypoints0 = query (x, y) pixel indices,
    # keypoints1 = render (x, y) pixel indices at ``camera`` resolution,
    # confidence in [0, 1].  Nothing downstream knows which matcher produced
    # them, so the vacuity gate / depth lift / PnP are shared verbatim.

    @torch.no_grad()
    def _prep_query_feats(self, img: torch.Tensor) -> None:
        """Query-side SuperPoint descriptor maps, computed ONCE per correct().

        They do not depend on the pose hypothesis, so the encoder runs once and
        every correction round re-uses the two maps: the native 1/8 coarse grid
        (STDLoc's "coarse") and the bilinearly upsampled, re-normalised
        full-resolution map (their "fine").  When the map's descriptors were
        PCA-reduced, the query maps get the map's own projection -- they must,
        or the cosine similarity is meaningless.
        """
        from . import locfeat
        from .superpoint import upsample_descriptors

        desc, _ = self._sp.dense_descriptors(img)
        if self._pca is not None:
            desc = locfeat.project_query_map(desc, self._pca[0], self._pca[1])
        self._q_coarse = F.normalize(desc[0], p=2, dim=0).contiguous()
        self._q_fine = upsample_descriptors(desc, self.camera.height,
                                            self.camera.width)[0].contiguous()

    @torch.no_grad()
    def _fine_mnn(self, q_f: torch.Tensor, r_f: torch.Tensor,
                  cq: torch.Tensor, cr: torch.Tensor):
        """STDLoc's fine stage verbatim: a MUTUAL-NN inside each matched window.

        ``usgloc.locfeat.refine_match_windows`` keeps only the single best pair
        of each window, which is always one of the mutual pairs but throws the
        others away -- one coarse cell then yields exactly one correspondence
        and no cell is ever rejected.  STDLoc instead runs their ``mnn_match``
        on the (m, win^2, win^2) fine correlation matrix, so a window can
        contribute several correspondences (up to ``win^2``) and a window whose
        rows and columns never agree contributes none.  Same arguments and
        return convention as ``refine_match_windows``.
        """
        win = self.feat_win
        D, H, W = q_f.shape
        w_c = W // win
        ww = win * win

        def _win(x, idx):
            u = F.unfold(x[None], (win, win), stride=win).reshape(1, D, ww, -1)
            return u[0, :, :, idx].permute(2, 1, 0)          # (m, ww, D)

        sim = _win(q_f, cq).float() @ _win(r_f, cr).float().transpose(-2, -1)
        if self.feat_temp > 0:
            s = sim / self.feat_temp
            sim = F.softmax(s, dim=-2) * F.softmax(s, dim=-1)
        mask = sim > self.feat_fine_thr
        mask &= sim == sim.max(dim=-1, keepdim=True).values
        mask &= sim == sim.max(dim=-2, keepdim=True).values
        b, i, j = torch.nonzero(mask, as_tuple=True)
        base_q = torch.stack([(cq[b] % w_c) * win, (cq[b] // w_c) * win], dim=-1)
        base_r = torch.stack([(cr[b] % w_c) * win, (cr[b] // w_c) * win], dim=-1)
        off_q = torch.stack([i % win, i // win], dim=-1)
        off_r = torch.stack([j % win, j // win], dim=-1)
        return base_q + off_q, base_r + off_r, sim[b, i, j]

    @torch.no_grad()
    def _match_feat(self, out: dict) -> dict:
        """STDLoc's coarse-to-fine dense matching against the RASTERISED field.

        ``out`` is the render of this round; ``out["feats"]`` is the
        alpha-blended, per-pixel L2-normalised descriptor map (one rasterization
        with the colour and vacuity channels, so the render is not repeated).
        The coarse stage is a mutual-NN with dual-softmax on the 1/8 grid, the
        fine stage a dual-softmax argmax inside each matched cell's window.

        ``confidence`` is 1.0 for every returned match: the field's own score
        threshold is ``feat_conf_min``, applied here, because ``conf_min`` is
        LoFTR's coarse confidence and has to keep meaning exactly that for the
        LoFTR half of ``matcher='pooled'``.
        """
        from . import locfeat

        dev = self._q_fine.device
        fm = out["feats"].permute(2, 0, 1).contiguous()          # (D, H, W)
        q_c, q_f = self._q_coarse, self._q_fine
        r_c = F.normalize(F.interpolate(fm[None], size=q_c.shape[-2:],
                                        mode="bilinear", align_corners=False)[0],
                          p=2, dim=0)
        cq, cr, _ = locfeat.dense_feature_match(
            q_c, r_c, thr=self.feat_coarse_thr, dual_softmax_temp=self.feat_temp)
        if cq.numel() == 0:
            z = torch.zeros(0, 2, device=dev)
            return {"keypoints0": z, "keypoints1": z.clone(),
                    "confidence": torch.zeros(0, device=dev)}
        wc = q_c.shape[-1]
        fq, fr = cq[:, 1] * wc + cq[:, 0], cr[:, 1] * wc + cr[:, 0]
        if self.feat_fine == "mnn":
            uq, ur, sc = self._fine_mnn(q_f, fm, fq, fr)
        else:
            uq, ur, sc = locfeat.refine_match_windows(
                q_f, fm, fq, fr, win=self.feat_win, temp=self.feat_temp)
        if self.feat_conf_min > 0:
            k = sc >= self.feat_conf_min
            uq, ur, sc = uq[k], ur[k], sc[k]
        if 0.0 < self.feat_top < 1.0 and sc.numel():
            k = torch.topk(sc, max(1, int(round(self.feat_top * sc.numel())))).indices
            uq, ur, sc = uq[k], ur[k], sc[k]
        if 0 < self.feat_max < sc.numel():
            k = torch.topk(sc, self.feat_max).indices
            uq, ur, sc = uq[k], ur[k], sc[k]
        kq, kr = uq.float(), ur.float()
        if self.feat_subpix > 0 and uq.numel():
            # The fine stage returns integer pixels; localise the render-side
            # position to sub-pixel by a soft-argmax of the descriptor
            # similarity over the 3x3 neighbourhood (the same device as
            # ``mast3r_subpix``).
            H, W = self.camera.height, self.camera.width
            f0 = q_f[:, uq[:, 1], uq[:, 0]].T                    # (m, D)
            off = torch.arange(-1, 2, device=dev)
            oy, ox = torch.meshgrid(off, off, indexing="ij")
            oy, ox = oy.reshape(-1), ox.reshape(-1)
            yy = (ur[:, 1, None] + oy[None]).clamp(0, H - 1)
            xx = (ur[:, 0, None] + ox[None]).clamp(0, W - 1)
            s = (fm[:, yy, xx].permute(1, 2, 0) * f0[:, None]).sum(-1)   # (m, 9)
            wgt = torch.softmax(s / self.feat_subpix, dim=1)
            kr = kr + torch.stack([(wgt * ox[None].float()).sum(1),
                                   (wgt * oy[None].float()).sum(1)], dim=-1)
        return {"keypoints0": kq, "keypoints1": kr,
                "confidence": torch.ones_like(sc, dtype=torch.float32)}

    # ---- MASt3R adapter ---------------------------------------------------
    # Same contract as the LoFTR call: keypoints0 = query (x, y) pixels,
    # keypoints1 = render (x, y) pixels, confidence in [0, 1] -- all at the
    # ``camera`` resolution, pixel-index convention (the +0.5 pixel-centre
    # shift is applied later, exactly as for LoFTR).

    @torch.no_grad()
    def _mast3r_prep(self, rgb: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) in [0, 1] -> (1, 3, h, w) in [-1, 1] at the working
        resolution (dust3r's ImgNorm; antialiased bicubic stands in for the
        PIL LANCZOS downscale of dust3r.utils.image.load_images)."""
        x = rgb.to(torch.float32).clamp(0.0, 1.0).permute(2, 0, 1)[None]
        if tuple(x.shape[-2:]) != self._m_hw:
            x = F.interpolate(x, size=self._m_hw, mode="bicubic",
                              align_corners=False, antialias=True).clamp(0.0, 1.0)
        return (x - 0.5) / 0.5

    def _jamma_prep(self, rgb: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) in [0, 1] -> (1, 3, H, W) in [0, 1].

        SplatHLoc feeds JamMa the raw [0, 1] RGB of the query and of the render
        with no further normalisation, so this reproduces that exactly. JamMa's
        backbone strides by 8, which 640x480 already satisfies.
        """
        return rgb.to(torch.float32).clamp(0.0, 1.0).permute(2, 0, 1)[None]

    @torch.no_grad()
    def _match_jamma(self, img0: torch.Tensor, rgb_render: torch.Tensor) -> dict:
        """One JamMa forward on (query, render), mapped to the shared contract.

        JamMa does its own coarse match and then a LEARNED sub-pixel stage; the
        latter is the component whose absence our fixed-window field refinement
        is limited by (the field puts only ~66% of its correspondences inside
        2 px against LoFTR's ~94%). ``confidence`` is JamMa's own ``mconf_f``.
        """
        dev = rgb_render.device
        data = {"imagec_0": img0.to(dev), "imagec_1": self._jamma_prep(rgb_render)}
        self.matcher.backbone(data)
        self.matcher.matcher(data, mode="test")
        k0 = data.get("mkpts0_f")
        if k0 is None or k0.numel() == 0:
            z = torch.zeros(0, 2, device=dev)
            return {"keypoints0": z, "keypoints1": z.clone(),
                    "confidence": torch.zeros(0, device=dev)}
        k1 = data["mkpts1_f"]
        conf = data.get("mconf_f")
        if conf is None or conf.numel() != k0.shape[0]:
            conf = torch.ones(k0.shape[0], device=dev)
        return {"keypoints0": k0.to(dev, torch.float32),
                "keypoints1": k1.to(dev, torch.float32),
                "confidence": conf.to(dev, torch.float32)}

    @torch.no_grad()
    def _match_mast3r(self, img0: torch.Tensor, rgb_render: torch.Tensor) -> dict:
        """One MASt3R forward on (query, render) + fast reciprocal NNs on the
        24-d descriptors (MASt3R demo / GS-CPR recipe), mapped back to
        ``camera`` pixels.

        ``confidence`` is, with ``mast3r_conf='sim'`` (default), the cosine
        similarity of the two unit descriptors of the reciprocal pair; with
        ``'desc_conf'`` it is c / (1 + c) of the smaller of the two per-pixel
        matchability confidences the model predicts (c in (0, inf)).
        """
        dev = img0.device
        H, W = self.camera.height, self.camera.width
        h, w = self._m_hw
        img1 = self._mast3r_prep(rgb_render)
        ts = torch.tensor([[h, w]], dtype=torch.int32)
        view1 = {"img": img0, "true_shape": ts, "idx": 0, "instance": ["0"]}
        view2 = {"img": img1, "true_shape": ts, "idx": 1, "instance": ["1"]}
        pred1, pred2 = self.matcher(view1, view2)
        d1, d2 = pred1["desc"][0], pred2["desc"][0]          # (h, w, 24), unit norm
        m0, m1 = self._fast_nn(d1, d2, subsample_or_initxy1=self.mast3r_subsample,
                               device=dev, dist="dot", block_size=self.mast3r_block)
        # merge_corres hands back reversed (negative-stride) views -> copy.
        m0 = torch.as_tensor(np.ascontiguousarray(m0), device=dev, dtype=torch.long)  # (N, 2) x, y
        m1 = torch.as_tensor(np.ascontiguousarray(m1), device=dev, dtype=torch.long)
        if m0.numel() == 0:
            z = torch.zeros(0, 2, device=dev)
            return {"keypoints0": z, "keypoints1": z.clone(),
                    "confidence": torch.zeros(0, device=dev)}
        b = self.mast3r_border
        ok = ((m0[:, 0] >= b) & (m0[:, 0] < w - b) & (m0[:, 1] >= b) & (m0[:, 1] < h - b)
              & (m1[:, 0] >= b) & (m1[:, 0] < w - b) & (m1[:, 1] >= b) & (m1[:, 1] < h - b))
        m0, m1 = m0[ok], m1[ok]
        f0 = d1[m0[:, 1], m0[:, 0]]
        f1 = d2[m1[:, 1], m1[:, 0]]
        if self.mast3r_conf == "desc_conf":
            c = torch.minimum(pred1["desc_conf"][0][m0[:, 1], m0[:, 0]],
                              pred2["desc_conf"][0][m1[:, 1], m1[:, 0]])
            conf = c / (1.0 + c)
        else:
            conf = (f0 * f1).sum(-1).clamp(0.0, 1.0)
        m1f = m1.float()
        if self.mast3r_subpix > 0:
            # Reciprocal NNs are integer working-res pixels (1.25 camera px at
            # 512).  Localise the query descriptor f0 to sub-pixel precision
            # in the render's descriptor field: soft-argmax of the cosine
            # similarity over a (2r+1)^2 window centred on the integer match.
            r = self.mast3r_subpix
            off = torch.arange(-r, r + 1, device=dev)
            oy, ox = torch.meshgrid(off, off, indexing="ij")
            oy, ox = oy.reshape(-1), ox.reshape(-1)
            yy = (m1[:, 1, None] + oy[None]).clamp(0, h - 1)
            xx = (m1[:, 0, None] + ox[None]).clamp(0, w - 1)
            s = (d2[yy, xx] * f0[:, None]).sum(-1)              # (N, K)
            wgt = torch.softmax(s / self.mast3r_subpix_tau, dim=1)
            m1f = m1f + torch.stack([(wgt * ox[None].float()).sum(1),
                                     (wgt * oy[None].float()).sum(1)], dim=-1)
        # Working-res pixel index -> camera-res pixel index through the pixel
        # centre: (x + 0.5) * W / w - 0.5, so the +0.5 added downstream lands
        # on the exact centre of the matched working-res pixel.
        sc = torch.tensor([W / w, H / h], device=dev, dtype=torch.float32)
        kq = (m0.float() + 0.5) * sc - 0.5
        kr = (m1f + 0.5) * sc - 0.5
        return {"keypoints0": kq, "keypoints1": kr, "confidence": conf.float()}

    # ---- RoMa adapter -----------------------------------------------------
    # Same contract again: keypoints0 = query (x, y), keypoints1 = render
    # (x, y), both at ``camera`` resolution in the pixel-index convention,
    # confidence in [0, 1].

    @staticmethod
    def _roma_pil(rgb: torch.Tensor):
        """(H, W, 3) in [0, 1] -> 8-bit RGB PIL image (romatch's own input
        path: PIL -> bicubic resize -> ImageNet normalisation)."""
        from PIL import Image

        a = (rgb.detach().to(torch.float32).clamp(0.0, 1.0).cpu().numpy() * 255.0)
        return Image.fromarray(np.round(a).astype(np.uint8), mode="RGB")

    @torch.no_grad()
    def _match_roma(self, pil_q, rgb_render: torch.Tensor) -> dict:
        """RoMa dense warp (query -> render and render -> query, symmetric)
        + the authors' thresholded-balanced sampler, mapped to pixels.

        ``confidence`` is the certainty romatch's ``sample`` returns: 1.0 for
        every sample whose certainty exceeds ``sample_thresh`` (0.05), the raw
        certainty otherwise -- so the downstream ``conf >= conf_min`` gate
        only drops the few low-certainty samples the balanced sampler admits.
        The sampler is multinomial; its RNG is forked and seeded so the
        correction is deterministic and the caller's RNG state is untouched.
        """
        dev = self.emap.means.device
        H, W = self.camera.height, self.camera.width
        pil_r = self._roma_pil(rgb_render)
        warp, cert = self.matcher.match(pil_q, pil_r, device=dev)  # (h, 2w, 4), (h, 2w)
        devs = [dev] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self.roma_seed)
            m, c = self.matcher.sample(warp, cert, num=self.roma_num)
        if m.numel() == 0:
            z = torch.zeros(0, 2, device=dev)
            return {"keypoints0": z, "keypoints1": z.clone(),
                    "confidence": torch.zeros(0, device=dev)}
        kq, kr = self.matcher.to_pixel_coordinates(m, H, W, H, W)
        # romatch's pixel coordinates put the centre of pixel j at j + 0.5;
        # the lift below adds +0.5 to pixel indices, so shift back by 0.5.
        return {"keypoints0": (kq - 0.5).float(), "keypoints1": (kr - 0.5).float(),
                "confidence": c.float()}

    # ---- multi-round accumulation ----------------------------------------
    # Every round renders the map from a slightly different pose and produces
    # its own set of gated 2D-3D correspondences whose world points are lifted
    # through THAT round's render pose, so they are all expressed in the world
    # frame and are directly poolable.  The default behaviour keeps only the
    # round with the most inliers; with ``accumulate`` set, the inliers of all
    # rounds are pooled and one PnP is solved over the union.

    def _jitter(self, T: torch.Tensor, rnd: int) -> torch.Tensor:
        """Render pose for round ``rnd``: ``exp(xi) T`` with a fixed-magnitude,
        pseudo-random-direction xi (deterministic in ``rnd`` and the seed)."""
        from . import se3

        g = torch.Generator().manual_seed(1000003 * self.acc_jitter_seed + 7919 * rnd + 17)
        xi = torch.zeros(6)
        if self.acc_jitter > 0:
            d = torch.randn(3, generator=g)
            xi[:3] = d / d.norm().clamp_min(1e-9) * self.acc_jitter
        if self.acc_jitter_deg > 0:
            d = torch.randn(3, generator=g)
            xi[3:] = d / d.norm().clamp_min(1e-9) * (self.acc_jitter_deg * math.pi / 180.0)
        out = se3.apply_tangent(T.to(torch.float32),
                                xi.to(T.device, torch.float32))
        return out.to(T.dtype)

    def _reproj_err(self, obj, img, rvec, tvec):
        import cv2

        p, _ = cv2.projectPoints(obj,
                                 np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1),
                                 np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1),
                                 self._K, None)
        return np.linalg.norm(p.reshape(-1, 2) - img, axis=1)

    def _irls(self, obj, img, rvec, tvec, loss: str, pw=None):
        """Iteratively-reweighted Gauss-Newton on (rvec, tvec).

        ``cv2.projectPoints`` hands back the analytic Jacobian of the 2N
        residuals w.r.t. the Rodrigues vector (cols 0:3) and the translation
        (cols 3:6), so the normal equations of the robust problem

            min_x sum_i rho(|r_i(x)|),   w_i = rho'(e)/e

        cost one projectPoints call per iteration -- the same parameterization
        ``solvePnPRefineLM`` uses, only with a Huber / Cauchy weight in place
        of the plain square.  Returns (rvec, tvec); a non-finite step aborts
        and the last good iterate is returned.

        ``pw`` is an optional per-correspondence PRIOR weight (the rendered
        vacuity weight, see ``wvac``), multiplied into the robust weight, so
        the evidence the map itself carries scales the residual the same way
        it scales the photometric residual in Eq. 9.
        """
        import cv2

        s = max(self.refine_sigma, 1e-6)
        rvec = np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1).copy()
        tvec = np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1).copy()
        for _ in range(self.refine_iters):
            proj, J = cv2.projectPoints(obj, rvec, tvec, self._K, None)
            r = (proj.reshape(-1, 2) - img)                       # (n, 2)
            e = np.linalg.norm(r, axis=1)
            if loss == "huber":
                w = np.where(e <= s, 1.0, s / np.maximum(e, 1e-12))
            elif loss == "cauchy":
                w = 1.0 / (1.0 + (e / s) ** 2)
            else:
                w = np.ones_like(e)
            if pw is not None:
                w = w * pw
            sw = np.sqrt(w).repeat(2)[:, None]                    # (2n, 1)
            A = np.asarray(J[:, :6], dtype=np.float64) * sw
            b = -r.reshape(-1, 1) * sw
            H = A.T @ A
            g = A.T @ b
            H = H + 1e-9 * np.trace(H) / 6.0 * np.eye(6)
            try:
                dx = np.linalg.solve(H, g).reshape(6)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dx)):
                break
            rvec = rvec + dx[:3].reshape(3, 1)
            tvec = tvec + dx[3:].reshape(3, 1)
            if np.linalg.norm(dx) < 1e-12:
                break
        return rvec, tvec

    def _pl_refine(self, obj, img, rvec, tvec):
        """poselib's own non-linear bundle (robust loss, ``pl_loss``) from the
        current iterate -- the refinement poselib's LO-RANSAC itself ends with,
        made available to the cv2 branch as well."""
        import cv2
        import poselib

        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        p0 = poselib.CameraPose()
        p0.R = np.ascontiguousarray(R, dtype=np.float64)
        p0.t = np.ascontiguousarray(np.asarray(tvec, np.float64).reshape(3))
        pose, _info = poselib.refine_absolute_pose(
            img, obj, p0, self._pl_cam,
            {"loss_type": self.pl_loss, "loss_scale": self.pl_loss_scale,
             "max_iterations": self.pl_bundle_iters, "verbose": False})
        rv, _ = cv2.Rodrigues(np.asarray(pose.R, dtype=np.float64))
        return rv, np.asarray(pose.t, dtype=np.float64).reshape(3, 1)

    def _final_refine(self, obj, img, inl, rvec, tvec, pw=None):
        """Dispatch the configured final refinement over the RANSAC inliers.

        ``pw``, when given, is the per-correspondence vacuity weight
        w_i = exp(-gamma * U_r/alpha) (full length, indexed like ``obj``).  It
        enters only the IRLS modes, and only when ``wvac`` > 0, raised to that
        exponent: ``wvac`` = 1 is the paper's own pixel weight, larger values
        trust low-vacuity correspondences more sharply.
        """
        import cv2

        mode = self.refine
        sub = np.arange(len(obj)) if self.refine_all else inl
        pws = None
        if pw is not None and self.wvac > 0:
            pws = np.ascontiguousarray(np.asarray(pw, np.float64)[sub] ** self.wvac,
                                       dtype=np.float64)
        if mode in ("lm", "lm+huber", "lm+cauchy", "lm+plb"):
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self._K, None,
                                              rvec, tvec)
        if mode in ("plb", "lm+plb", "plb+cauchy"):
            rvec, tvec = self._pl_refine(obj[inl], img[inl], rvec, tvec)
        if mode in ("huber", "cauchy", "lm+huber", "lm+cauchy", "l2", "plb+cauchy"):
            loss = ("cauchy" if mode.endswith("cauchy")
                    else "huber" if mode.endswith("huber") else "l2")
            rvec, tvec = self._irls(obj[sub], img[sub], rvec, tvec, loss, pw=pws)
        return (np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1),
                np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1))

    def _wvac_reselect(self, obj, img, rvec, tvec, wv, inl):
        """Vacuity-weighted inlier re-selection (``wvac_thr`` > 0).

        Each correspondence gets its own reprojection gate,
        ``thr * clip((w_i / median w) ** wvac_thr, 1/c, c)``: a correspondence
        the map is confident about (low rendered vacuity, large w) is allowed a
        looser gate than one on vacuous content.  This is the one place the
        continuous evidence can reach the geometric verification -- neither
        cv2's RANSAC nor poselib's exposes per-point weights or thresholds --
        and it is also the set the refinement is then run on.  Returns the new
        index array (or ``inl`` unchanged if too few survive).
        """
        if wv is None or self.wvac_thr <= 0:
            return inl
        e = self._reproj_err(obj, img, rvec, tvec)
        med = float(np.median(wv)) if len(wv) else 1.0
        r = (wv / max(med, 1e-12)) ** self.wvac_thr
        c = max(self.wvac_clip, 1.0)
        thr = self.reproj_thresh_px * np.clip(r, 1.0 / c, c)
        sel = np.where(e <= thr)[0]
        return sel if len(sel) >= self.min_inliers else inl

    def _pool(self, rounds):
        """rounds: list of (obj, img, rms, err) -> (obj, img, n_rounds_used).

        ``err`` is each correspondence's reprojection error under its own
        round's refined pose, so both the per-round filter (``accumulate =
        'weighted'``, by the round's RMS) and the per-correspondence filter
        (``acc_err_q``) act on measured quality, not on a guess.
        """
        rms = np.asarray([r[2] for r in rounds], dtype=np.float64)
        use = np.arange(len(rounds))
        if self.accumulate == "weighted" and self.acc_rms_k > 0:
            use = use[rms[use] <= self.acc_rms_k * float(rms.min())]
            if len(use) < 2:
                return None
        obj = np.concatenate([rounds[i][0] for i in use], axis=0)
        img = np.concatenate([rounds[i][1] for i in use], axis=0)
        err = np.concatenate([rounds[i][3] for i in use], axis=0)
        src = np.concatenate([np.full(len(rounds[i][0]), i, dtype=np.int64) for i in use])
        if 0.0 < self.acc_err_q < 1.0:
            m = err <= np.quantile(err, self.acc_err_q)
            obj, img, err, src = obj[m], img[m], err[m], src[m]
        # LoFTR's query-side keypoints sit on a fixed 8-px coarse grid, so the
        # same scene point re-matched in a later round lands on the *same*
        # query pixel: bucket by integer query pixel.
        q = np.floor(img).astype(np.int64)
        key = q[:, 1] * 100000 + q[:, 0]
        if self.acc_min_seen > 1:
            # A correspondence the later re-renders failed to reproduce is one
            # the rounds disagree about: let them vote.
            uk, inv = np.unique(key, return_inverse=True)
            m = np.bincount(inv, minlength=len(uk))[inv] >= self.acc_min_seen
            if int(m.sum()) < self.min_inliers:
                return None
            obj, img, err, src, key = obj[m], img[m], err[m], src[m], key[m]
        if self.acc_dedup == "all":
            return obj, img, int(len(use))
        if self.acc_dedup == "mean":
            uk, inv = np.unique(key, return_inverse=True)
            cnt = np.bincount(inv, minlength=len(uk)).astype(np.float64)
            # 'weighted' also weights the average of a repeated query pixel by
            # each instance's own reprojection error (0.5 px floor, so a
            # perfectly-fitting instance does not take the whole weight).
            w = (1.0 / (err ** 2 + 0.25) if self.accumulate == "weighted"
                 else np.ones(len(err)))
            ws = np.bincount(inv, weights=w, minlength=len(uk))
            o = np.stack([np.bincount(inv, weights=w * obj[:, k], minlength=len(uk))
                          for k in range(3)], axis=1) / ws[:, None]
            m = np.stack([np.bincount(inv, weights=w * img[:, k], minlength=len(uk))
                          for k in range(2)], axis=1) / ws[:, None]
            if 0.0 < self.acc_spread_q < 1.0:
                # How far apart are the rounds' world points for this pixel?
                d2 = ((obj - o[inv]) ** 2).sum(1)
                sp = np.sqrt(np.bincount(inv, weights=d2, minlength=len(uk))
                             / np.maximum(cnt, 1.0))
                rep = cnt >= 2
                if int(rep.sum()) >= self.min_inliers:
                    lim = float(np.quantile(sp[rep], self.acc_spread_q))
                    keep = ~rep | (sp <= lim)
                    if int(keep.sum()) >= self.min_inliers:
                        o, m = o[keep], m[keep]
            return o, m, int(len(use))
        # 'best': the instance from the round with the lowest inlier RMS;
        # 'berr': the instance with the lowest error of its own.
        rank = err if self.acc_dedup == "berr" else rms[src]
        order = np.lexsort((rank, key))
        ks = key[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = ks[1:] != ks[:-1]
        sel = order[first]
        return obj[sel], img[sel], int(len(use))

    def _pooled_solve(self, rounds, best: CorrectionResult) -> CorrectionResult:
        import cv2

        pooled = self._pool(rounds)
        if pooled is None:
            return best
        obj, img, n_used = pooled
        obj = np.ascontiguousarray(obj, dtype=np.float64)
        img = np.ascontiguousarray(img, dtype=np.float64)
        if len(obj) < self.min_inliers:
            return best
        thr = self.acc_thresh_px if self.acc_thresh_px > 0 else self.reproj_thresh_px
        rv0, _ = cv2.Rodrigues(np.ascontiguousarray(best.T[:3, :3].double().cpu().numpy()))
        tv0 = np.ascontiguousarray(best.T[:3, 3].double().cpu().numpy()).reshape(3, 1)
        n_best = int((self._reproj_err(obj, img, rv0, tv0) <= thr).sum())
        if self.acc_solve == "ransac":
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, self._K, None,
                reprojectionError=thr, iterationsCount=self.ransac_iters,
                confidence=0.9999, flags=cv2.SOLVEPNP_ITERATIVE,
                rvec=rv0.copy(), tvec=tv0.copy(), useExtrinsicGuess=True)
            if not ok or inl is None or len(inl) < self.min_inliers:
                return best
            sel = np.asarray(inl).reshape(-1)
        else:
            # The pooled points are already every round's RANSAC inliers, so a
            # second RANSAC mostly burns time; gate them once at the same
            # threshold from the best round's pose and refine.
            sel = np.where(self._reproj_err(obj, img, rv0, tv0) <= thr)[0]
            if len(sel) < self.min_inliers:
                return best
            rvec, tvec = rv0.copy(), tv0.copy()
        rvec = np.ascontiguousarray(np.asarray(rvec, dtype=np.float64)).reshape(3, 1).copy()
        tvec = np.ascontiguousarray(np.asarray(tvec, dtype=np.float64)).reshape(3, 1).copy()
        rvec, tvec = cv2.solvePnPRefineLM(obj[sel], img[sel], self._K, None, rvec, tvec)
        if self.acc_refine != "lm":
            rvec, tvec = self._irls(obj[sel], img[sel], rvec, tvec, self.acc_refine)
            rvec = np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1)
            tvec = np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1)
        n_pool = int((self._reproj_err(obj, img, rvec, tvec) <= thr).sum())
        win = n_pool >= n_best
        if self.acc_guard and not win:
            return replace(best, acc_used=False, acc_points=int(len(obj)),
                           acc_rounds=n_used, acc_inl=n_best, acc_win=False)
        e = self._reproj_err(obj[sel], img[sel], rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T_new = torch.eye(4, dtype=best.T.dtype, device=best.T.device)
        T_new[:3, :3] = torch.as_tensor(R, dtype=best.T.dtype, device=best.T.device)
        T_new[:3, 3] = torch.as_tensor(tvec.reshape(3), dtype=best.T.dtype,
                                       device=best.T.device)
        return replace(best, T=T_new, reproj_px=float(np.sqrt(np.mean(e ** 2))),
                       obj=np.asarray(obj[sel], dtype=np.float64),
                       img=np.asarray(img[sel], dtype=np.float64),
                       acc_used=True, acc_points=int(len(obj)), acc_rounds=n_used,
                       acc_inl=n_pool, acc_win=win)

    @torch.no_grad()
    def correct(self, query: torch.Tensor, T0: torch.Tensor,
                rounds: Optional[int] = None) -> CorrectionResult:
        """``query``: (H, W) luma or (H, W, 3) RGB in [0, 1] at ``camera``
        resolution.  LoFTR consumes the luma (RGB is converted with the same
        formula the runner uses); MASt3R consumes RGB (a luma query is
        replicated to three channels).

        ``rounds`` overrides ``self.rounds`` for this call only (None = the
        configured value, i.e. every existing caller is unchanged).  It exists
        so ``search()`` can evaluate a virtual view with a single
        render/match/PnP instead of the full iteration.
        """
        import cv2

        n_rounds = self.rounds if rounds is None else int(rounds)

        dev = self.emap.means.device
        H, W = self.camera.height, self.camera.width
        if query.dim() == 3:
            rgb_q = query.to(dev, torch.float32)
            gray_q = _to_gray(rgb_q)
        else:
            rgb_q, gray_q = None, query
        assert gray_q.shape == (H, W), f"query {tuple(gray_q.shape)} != camera {(H, W)}"
        img0 = None
        _stage_m3 = self.matcher_name == "stage" and self.stage_warm == "mast3r"
        _stage_jm = self.matcher_name == "stage" and self.stage_warm == "jamma"
        if self.matcher_name == "jamma" or _stage_jm:
            # JamMa takes raw [0, 1] RGB; the field rounds ignore img0.
            if rgb_q is None:
                rgb_q = gray_q.to(dev, torch.float32)[..., None].expand(H, W, 3)
            img0 = self._jamma_prep(rgb_q)
        elif _stage_m3:
            # The warm-up is MASt3R, so img0 must be its prepared RGB; the field
            # rounds read self._q_coarse / _q_fine and do not use img0 at all.
            if rgb_q is None:
                rgb_q = gray_q.to(dev, torch.float32)[..., None].expand(H, W, 3)
            img0 = self._mast3r_prep(rgb_q)
        elif self.matcher_name in ("loftr", "pooled", "stage"):
            img0 = gray_q.to(dev, torch.float32)[None, None]
        elif self.matcher_name != "feat":
            if rgb_q is None:
                rgb_q = gray_q.to(dev, torch.float32)[..., None].expand(H, W, 3)
            if self.matcher_name == "roma":
                img0 = self._roma_pil(rgb_q)      # constant across rounds
            else:
                img0 = self._mast3r_prep(rgb_q)
        if self.matcher_name in ("feat", "pooled", "stage"):
            # Pose-independent: the encoder runs once per query, not per round.
            self._prep_query_feats(rgb_q if rgb_q is not None else gray_q)

        T = T0.clone()
        best = CorrectionResult(T=T0, success=False, inliers=0,
                                n_matches=0, reproj_px=float("inf"), rounds=0)
        pool = []            # (obj_inl, img_inl, rms, err) per successful round
        dbg = []             # acc_debug: (pose, inliers, rms) per successful round
        best_is_feat = False  # matcher='stage': did `best` come from a field round?
        best_warm = None      # best warm-up (LoFTR) result, for arbiter='stage'
        cands = []            # every successful round's pose, for arbiter='rounds'
        for rnd in range(n_rounds):
            # matcher='stage': a leading LoFTR round, run only to move the
            # hypothesis into the field's basin.
            warm = self.matcher_name == "stage" and rnd < self.feat_stage
            # The pose the map is RENDERED and LIFTED from; it equals the
            # current hypothesis unless render-pose jitter is on.
            T_r = T
            if rnd > 0 and (self.acc_jitter > 0 or self.acc_jitter_deg > 0):
                T_r = self._jitter(T, rnd)
            out = self._render(T_r)
            if self.matcher_name == "loftr":
                img1 = _to_gray(out["rgb"])[None, None].to(torch.float32)
                corr = self._match_loftr(img0, img1)
            elif self.matcher_name == "jamma":
                corr = self._match_jamma(img0, out["rgb"])
            elif self.matcher_name == "feat":
                corr = self._match_feat(out)
            elif self.matcher_name == "stage":
                # LoFTR's wide basin first, the field's precision afterwards:
                # the field is the better matcher AT the answer and the worse
                # one far from it, so let each do what it is good at.
                if warm:
                    if self.stage_warm == "mast3r":
                        corr = self._match_mast3r(img0, out["rgb"])
                    elif self.stage_warm == "jamma":
                        corr = self._match_jamma(img0, out["rgb"])
                    else:
                        corr = self._match_loftr(
                            img0, _to_gray(out["rgb"])[None, None].to(torch.float32))
                else:
                    corr = self._match_feat(out)
            elif self.matcher_name == "pooled":
                # The two matchers fail differently: LoFTR is outlier-free but
                # photometric, the field is descriptor-based but noisier.  Pool
                # their correspondences into ONE PnP -- the gate, the lift and
                # the solver do not care where a match came from.
                a = self._match_loftr(
                    img0, _to_gray(out["rgb"])[None, None].to(torch.float32))
                b = self._match_feat(out)
                corr = {k: torch.cat([a[k].float(), b[k].float()], dim=0) for k in a}
            elif self.matcher_name == "roma":
                corr = self._match_roma(img0, out["rgb"])
            else:
                corr = self._match_mast3r(img0, out["rgb"])
            kq = corr["keypoints0"]      # (M, 2) x, y in query pixels
            kr = corr["keypoints1"]      # (M, 2) x, y in render pixels
            conf = corr["confidence"]
            if kq.numel() == 0:
                break

            # Sample render-side depth / alpha / vacuity at the matches.
            xi = kr[:, 0].round().long().clamp(0, W - 1)
            yi = kr[:, 1].round().long().clamp(0, H - 1)
            depth = out["depth"]
            d = depth[yi, xi]
            a = out["alpha"][yi, xi]
            u = out["vacuity"][yi, xi]
            # Expected depth blends fore/background into phantom surfaces at
            # silhouettes: drop matches near a depth discontinuity (3x3
            # local spread relative to depth).
            dm = depth[None, None]
            spread = (F.max_pool2d(dm, 3, 1, 1) + F.max_pool2d(-dm, 3, 1, 1))[0, 0]
            keep = (
                (conf >= self.conf_min)
                & (a >= self.min_alpha)
                & (d >= self.min_depth)
                & (spread[yi, xi] <= self.edge_rel * d)
            )
            # The rendered vacuity channel is alpha-blended without
            # normalization, so it is bounded by alpha; both the gate and the
            # continuous weight act on the alpha-normalized vacuity U_r / alpha.
            u_n = u / a.clamp_min(1e-6)
            if self.gate_vacuity:
                keep &= u_n <= self.u_max
            if int(keep.sum()) < 6:
                break
            kq, kr = kq[keep], kr[keep]
            # Paper's own pixel weight, Eq. 9, as a per-correspondence weight.
            wv = None
            if self.wvac > 0 or self.wvac_thr > 0:
                wv = torch.exp(-self.gamma * u_n[keep].double().clamp(0.0, 1.0))
                wv = np.ascontiguousarray(wv.cpu().numpy(), dtype=np.float64)

            # Bilinear depth at the subpixel keypoint. Pixel (i, j) is the
            # ray through K-coordinate (j + 0.5, i + 0.5) -- the gsplat
            # pixel-center convention.
            gx = 2.0 * (kr[:, 0] + 0.5) / W - 1.0
            gy = 2.0 * (kr[:, 1] + 0.5) / H - 1.0
            grid = torch.stack([gx, gy], dim=-1)[None, None].to(dm.dtype)
            d_bil = F.grid_sample(dm, grid, mode="bilinear",
                                  align_corners=False)[0, 0, 0]

            if self.lift == "xyz":
                # Read the alpha-normalized blended world position straight
                # off the rasterizer: no camera-model inversion, and the
                # silhouette artefacts of blended depth do not reappear as
                # phantom rays.
                gx2 = 2.0 * (kr[:, 0] + 0.5) / W - 1.0
                gy2 = 2.0 * (kr[:, 1] + 0.5) / H - 1.0
                g2 = torch.stack([gx2, gy2], dim=-1)[None, None]
                xyz = out["xyz"].permute(2, 0, 1)[None]
                X_w = F.grid_sample(xyz, g2.to(xyz.dtype), mode="bilinear",
                                    align_corners=False)[0, :, 0].T.double()
            else:
                d_use = d_bil
                if self.lift in _RAY_LIFTS:
                    # Single-surface depth: recompute the rasterizer's exact
                    # front-to-back compositing for these pixels only and take
                    # the mode / median / banded-mean of the weight profile
                    # instead of its mean.  The expected depth is biased
                    # towards the camera (the blend integrates through the
                    # front half of every Gaussian) and invents a surface
                    # wherever two depths mix.
                    xg = kr[:, 0].round().long().clamp(0, W - 1)
                    yg = kr[:, 1].round().long().clamp(0, H - 1)
                    st = ray_depths(out["meta"], xg.float() + 0.5, yg.float() + 0.5,
                                    band=self.lift_band)
                    key = self.lift.split("_")[0]
                    ok_ray = st["w_sum"] > 1e-6
                    d_rob = torch.where(ok_ray, st[key], depth[yg, xg])
                    if self.lift.endswith("_d"):
                        # Keep the bilinear sub-pixel interpolation and apply
                        # only the (mode - mean) correction of the pixel the
                        # keypoint falls in.
                        d_use = d_bil + torch.where(ok_ray, d_rob - st["exp"],
                                                    torch.zeros_like(d_rob))
                    else:
                        d_use = d_rob
                # Lift render keypoints to world: X_cam = d * K^-1 [u, v, 1].
                K = torch.as_tensor(self._K, device=dev, dtype=torch.float64)
                uv1 = torch.cat([kr.double() + 0.5,
                                 torch.ones_like(kr[:, :1]).double()], dim=1)
                x_cam = (torch.linalg.inv(K) @ uv1.T).T * d_use.double()[:, None]
                T_wc = torch.linalg.inv(T_r.double())
                X_w = (T_wc[:3, :3] @ x_cam.T).T + T_wc[:3, 3]

            obj = np.ascontiguousarray(X_w.cpu().numpy(), dtype=np.float64)
            img = np.ascontiguousarray(kq.double().cpu().numpy() + 0.5,
                                       dtype=np.float64)
            if self.solver == "poselib":
                import poselib
                K = self._K
                cam = {"model": "PINHOLE", "width": W, "height": H,
                       "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
                pose, info = poselib.estimate_absolute_pose(
                    img, obj, cam,
                    {"max_iterations": self.pl_max_iters,
                     "min_iterations": self.pl_min_iters,
                     "max_reproj_error": self.reproj_thresh_px,
                     "success_prob": 0.9999},
                    {"loss_type": self.pl_loss, "loss_scale": self.pl_loss_scale,
                     "max_iterations": self.pl_bundle_iters, "verbose": False})
                n_inl = int(info["num_inliers"])
                if n_inl < self.min_inliers:
                    break
                inl = np.where(np.asarray(info["inliers"]))[0]
                Rm = np.asarray(pose.R, dtype=np.float64)
                tv = np.asarray(pose.t, dtype=np.float64).reshape(3, 1)
                rvec, _ = cv2.Rodrigues(Rm)
                tvec = tv
            else:
                flag = (cv2.SOLVEPNP_EPNP if self.solver == "cv2_epnp"
                        else cv2.SOLVEPNP_ITERATIVE)
                guess = {}
                if self.use_guess:
                    rv0, _ = cv2.Rodrigues(
                        np.ascontiguousarray(T[:3, :3].double().cpu().numpy()))
                    guess = dict(rvec=rv0,
                                 tvec=np.ascontiguousarray(
                                     T[:3, 3].double().cpu().numpy()).reshape(3, 1),
                                 useExtrinsicGuess=True)
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    obj, img, self._K, None,
                    reprojectionError=self.reproj_thresh_px,
                    iterationsCount=self.ransac_iters, confidence=0.9999,
                    flags=flag, **guess,
                )
                n_inl = 0 if inl is None else int(len(inl))
                if not ok or n_inl < self.min_inliers:
                    break
                inl = inl.reshape(-1)
            rvec = np.ascontiguousarray(np.asarray(rvec, dtype=np.float64)).copy()
            tvec = np.ascontiguousarray(np.asarray(tvec, dtype=np.float64)).reshape(3, 1).copy()
            if self.wvac_thr > 0:
                inl = self._wvac_reselect(obj, img, rvec, tvec, wv, inl)
                n_inl = int(len(inl))
            rvec, tvec = self._final_refine(obj, img, inl, rvec, tvec, pw=wv)

            R, _ = cv2.Rodrigues(rvec)
            T_new = torch.eye(4, dtype=T.dtype, device=dev)
            T_new[:3, :3] = torch.as_tensor(R, dtype=T.dtype, device=dev)
            T_new[:3, 3] = torch.as_tensor(tvec.reshape(3), dtype=T.dtype, device=dev)

            # Over-relax a small field update: the field iteration is a
            # contraction with a rate close to 1, so the same step taken alpha
            # times lands where several more rounds would have.  Only small
            # updates (the converging tail) are scaled; a large one is a coarse
            # correction that must be applied as solved.
            if self.feat_gain != 1.0 and not warm:
                d = T_new.double() @ torch.linalg.inv(T_r.double())
                if float(d[:3, 3].norm()) < self.feat_gain_max:
                    xi = se3.relative_tangent(T_r.double(), T_new.double())
                    T_new = se3.apply_tangent(T_r.double(),
                                              xi * self.feat_gain).to(T.dtype)
                    R = np.ascontiguousarray(
                        T_new[:3, :3].double().cpu().numpy(), dtype=np.float64)
                    rvec, _ = cv2.Rodrigues(R)
                    tvec = np.ascontiguousarray(
                        T_new[:3, 3].double().cpu().numpy().reshape(3, 1))

            proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, self._K, None)
            e_inl = np.sqrt(np.sum((proj.reshape(-1, 2) - img[inl]) ** 2, axis=1))
            rms = float(np.sqrt(np.mean(e_inl ** 2)))
            # Vacuity-weighted inlier mass (CorrectionResult.wmass): the same
            # w = exp(-gamma * U_r/alpha) the photometric residual is weighted
            # by, summed over the inliers.  Read-only diagnostic/score.
            _ii = torch.as_tensor(np.ascontiguousarray(np.asarray(inl).reshape(-1)),
                                  device=u_n.device, dtype=torch.long)
            wmass = float(torch.exp(-self.gamma
                                    * u_n[keep][_ii].double().clamp(0.0, 1.0)).sum())
            # Iterate from the new pose, but only promote a round to `best`
            # when it improves -- a later round can regress (degenerate
            # match set on the re-render) and must not overwrite a better
            # earlier solution.
            T = T_new
            if self.acc_debug:
                dbg.append((T_new.double().cpu().numpy().copy(), n_inl, rms))
            if self.accumulate != "none":
                pool.append((np.asarray(obj[inl], dtype=np.float64),
                             np.asarray(img[inl], dtype=np.float64), rms,
                             np.asarray(e_inl, dtype=np.float64)))
            promote = (not best.success or n_inl > best.inliers
                       or (n_inl == best.inliers and rms < best.reproj_px))
            if self.matcher_name == "stage":
                # The two matchers are not comparable by inlier count (LoFTR
                # feeds ~2.4x more correspondences), so the warm-up rounds must
                # not win on volume: a field round always displaces a LoFTR one,
                # and a LoFTR round is kept only while no field round has
                # succeeded.
                promote = (promote and not best_is_feat) if warm \
                    else (promote or not best_is_feat)
            res = CorrectionResult(T=T_new, success=True, inliers=n_inl,
                                   n_matches=int(len(obj)), reproj_px=rms,
                                   rounds=rnd + 1,
                                   obj=np.asarray(obj[inl], dtype=np.float64),
                                   img=np.asarray(img[inl], dtype=np.float64),
                                   wmass=wmass)
            if self.arbiter == "rounds":
                cands.append(res)
            if warm and (best_warm is None or n_inl > best_warm.inliers
                         or (n_inl == best_warm.inliers
                             and rms < best_warm.reproj_px)):
                best_warm = res
            if promote:
                best = res
                best_is_feat = not warm
            # Converged: this round barely moved the camera, so the remaining
            # rounds would re-render the same view and re-solve the same PnP.
            if self.stop_dt > 0 and not warm:
                dT = T_new.double() @ torch.linalg.inv(T_r.double())
                d_t = float(dT[:3, 3].norm())
                c = ((dT[0, 0] + dT[1, 1] + dT[2, 2] - 1) / 2).clamp(-1, 1)
                d_r = float(torch.rad2deg(torch.arccos(c)))
                if d_t < self.stop_dt and d_r < self.stop_dr:
                    break
        # ---- photometric arbitration -------------------------------------
        # The inlier rule above can only rank poses proposed by the SAME
        # matcher.  Ask the map instead: render each candidate and keep the one
        # that explains the query best.  Two extra renders (~0.02 s) against a
        # 0.36 s correction.
        if self.arbiter == "rounds" and len(cands) >= 2:
            sc = [self._photo_score(c.T, gray_q) for c in cands]
            k = int(np.argmax(sc))
            inc = best
            if sc[k] > -float("inf") and sc[k] > _sc_of(best, cands, sc) + self.arb_margin:
                best = cands[k]
            best = replace(best, arb=dict(
                scores=[float(x) for x in sc],
                poses=[c.T.double().cpu().numpy().tolist() for c in cands],
                pick=k, pick_inl=next((i for i, c in enumerate(cands) if c is inc), -1)))
        elif (self.arbiter == "stage" and best.success and best_is_feat
                and best_warm is not None):
            cands = [best_warm, best]           # 0 = warm-up/LoFTR, 1 = field
            sc = [self._photo_score(c.T, gray_q) for c in cands]
            k = 1 if sc[1] >= sc[0] - self.arb_margin else 0
            best = replace(cands[k], arb=dict(
                scores=[float(x) for x in sc],
                poses=[c.T.double().cpu().numpy().tolist() for c in cands],
                pick=k, pick_inl=1))
        if self.accumulate != "none" and best.success and len(pool) >= 2:
            best = self._pooled_solve(pool, best)
        if self.acc_debug and dbg:
            best = replace(best, round_T=[d[0].tolist() for d in dbg],
                           round_inl=[int(d[1]) for d in dbg],
                           round_rms=[float(d[2]) for d in dbg])
        # Inlier-ratio gate on the FINAL result only. Applied per round it
        # stops the re-render/re-match iteration after a low-overlap first
        # round from a distant anchor, which is exactly where the later
        # rounds recover; measured on Chess test frames that cost MASt3R
        # 37% of its snaps and 0.47 -> 0.77 cm.
        if (best.success and self.min_inlier_ratio > 0
                and best.inliers < self.min_inlier_ratio * max(best.n_matches, 1)):
            best = CorrectionResult(T=T0, success=False, inliers=best.inliers,
                                    n_matches=best.n_matches, reproj_px=best.reproj_px,
                                    rounds=best.rounds)
        return best

    @torch.no_grad()
    def fuse_across_starts(self, results) -> Optional[CorrectionResult]:
        """One PnP over the correspondences of SEVERAL corrected hypotheses.

        Each :class:`CorrectionResult` carries the inlier correspondences its
        own correction survived with: ``obj`` in WORLD coordinates and ``img``
        in the pixels of the SAME query image.  So every hypothesis constrains
        the identical unknown pose, and their correspondence sets can simply be
        concatenated and solved once -- which is a strictly better-conditioned
        problem than any one of them, because the anchors look at the scene from
        geometrically different viewpoints.

        The pipeline otherwise corrects each retrieval anchor independently and
        keeps only the winner, discarding the others' correspondences entirely.
        This recovers them.  Returns None when fewer than two hypotheses
        succeeded with correspondences, in which case the caller keeps its own
        best.

        Reuses ``_pooled_solve``, which is the same machinery the multi-ROUND
        accumulation uses, so the gate, the threshold and the refinement are
        identical to the single-hypothesis path.
        """
        usable = [c for c in results
                  if c is not None and c.success
                  and c.obj is not None and len(c.obj) >= 4]
        if len(usable) < 2:
            return None
        best = max(usable, key=lambda c: c.inliers)
        # (obj, img, rms, per-correspondence error) -- the tuple _pool expects.
        rounds = []
        for c in usable:
            e = self._reproj_err(
                np.ascontiguousarray(c.obj, dtype=np.float64),
                np.ascontiguousarray(c.img, dtype=np.float64),
                *self._rt_of(c.T))
            rounds.append((np.ascontiguousarray(c.obj, dtype=np.float64),
                           np.ascontiguousarray(c.img, dtype=np.float64),
                           float(c.reproj_px), e))
        out = self._pooled_solve(rounds, best)
        return replace(out, acc_rounds=len(usable))

    def _rt_of(self, T):
        """(rvec, tvec) of a (4, 4) camera-from-world pose, for cv2."""
        import cv2
        R = np.ascontiguousarray(T[:3, :3].double().cpu().numpy())
        rv, _ = cv2.Rodrigues(R)
        tv = np.ascontiguousarray(T[:3, 3].double().cpu().numpy()).reshape(3, 1)
        return rv, tv

    # ---- adaptive virtual-view search ------------------------------------
    # Mechanism: SplatHLoc (adaptive virtual-view retrieval) -- cited, not
    # claimed.  When the support for a hypothesis is weak, synthesise virtual
    # views by perturbing the candidate pose, match the query against each, and
    # keep the one that matches best.  Because the map is a 3DGS map, a virtual
    # view costs exactly one rasterisation, and -- unlike retrieval -- the
    # sampled poses are not restricted to poses some anchor proposed, which is
    # the structural limitation of an anchor-based front end.
    #
    # Everything inside the loop is the EXISTING correct() block: render,
    # match, vacuity gate, depth-edge filter, lift, PnP-RANSAC.  Only the
    # selection is new.

    @staticmethod
    def _score_of(c: "CorrectionResult", how: str) -> float:
        """Rank a virtual view by the evidence its correction produced."""
        if not c.success:
            return -float("inf")
        if how == "inliers":
            return float(c.inliers)
        if how == "ratio":
            return float(c.inliers) / max(float(c.n_matches), 1.0)
        if how == "rms":
            return -float(c.reproj_px)
        if how == "wmass":
            return float(c.wmass)
        raise ValueError(f"score must be wmass | inliers | ratio | rms, got {how!r}")

    @staticmethod
    def _perturb(T: torch.Tensor, dc: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Virtual view ``dc`` metres and ``phi`` radians away from ``T`` (T_cw).

        The perturbation is applied to the CAMERA CENTRE and the camera
        rotation directly, not through se3_exp, so ``|dc|`` is exactly the
        camera-centre displacement in metres and ``|phi|`` exactly the rotation
        angle in radians -- the radius study needs both to mean what they say
        (a staircase tread is ~30 cm; a sampling radius has to be comparable to
        it to reach the neighbouring tread).  ``dc`` is expressed in the camera
        axes of ``T``.

            R' = exp(phi) R,   c' = c + R^T dc   =>   t' = exp(phi) (t - dc)
        """
        from . import se3

        R, t = T[:3, :3], T[:3, 3]
        dR = se3.so3_exp(phi.to(T.dtype))
        out = torch.eye(4, dtype=T.dtype, device=T.device)
        out[:3, :3] = dR @ R
        out[:3, 3] = dR @ (t - dc.to(T.dtype))
        return out

    @staticmethod
    def _sample_xi(n: int, sigma_t: float, sigma_r_deg: float, dist: str,
                   seed: int):
        """(n, 3) centre offsets in metres and (n, 3) rotation vectors in rad.

        Sample 0 is always ZERO, so the unperturbed pose is a candidate and the
        search can only be a no-op, never a regression, at equal score.  The
        RNG is a private CPU generator seeded explicitly, so the caller's
        torch/numpy streams are untouched (the pattern ``_match_roma`` uses).

        ``dist``:
          ``normal``  per-axis N(0, sigma) -- SplatHLoc's own choice; the
                      centre displacement magnitude is then Maxwell with mean
                      ~1.596 * sigma_t, so sigma_t is NOT the reach.
          ``uniform`` uniform inside the ball of radius sigma_t (radius
                      r = sigma_t * u^(1/3), direction uniform on S^2), so
                      sigma_t IS the maximum reach and the mass sits at large
                      radii where the normal has almost none.
          ``shell``   fixed magnitude sigma_t, uniform direction: every sample
                      is spent at exactly the radius of interest.
        """
        g = torch.Generator().manual_seed(int(seed))
        sr = float(sigma_r_deg) * math.pi / 180.0

        def _draw(scale):
            if n <= 1 or scale <= 0:
                return torch.zeros(n, 3)
            x = torch.randn(n, 3, generator=g)
            if dist == "normal":
                v = x * scale
            else:
                d = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
                if dist == "uniform":
                    u = torch.rand(n, 1, generator=g) ** (1.0 / 3.0)
                elif dist == "shell":
                    u = torch.ones(n, 1)
                else:
                    raise ValueError(
                        f"dist must be normal | uniform | shell, got {dist!r}")
                v = d * u * scale
            v[0] = 0.0
            return v

        return _draw(float(sigma_t)), _draw(sr)

    @torch.no_grad()
    def search(self, query: torch.Tensor, T: torch.Tensor, n: int = 32,
               sigma_t: float = 0.25, sigma_r: float = 5.0, *,
               seed: int = 0, dist: str = "normal", score: str = "wmass",
               rounds: int = 1, refine: bool = True,
               refine_rounds: Optional[int] = None,
               stop_score: float = 0.0, prescreen: int = 0) -> "SearchResult":
        """Render/match/gate ``n`` virtual views around ``T`` and keep the best.

        ``T`` is a camera-from-world pose (the current hypothesis).  ``n``
        virtual views are drawn around it -- sample 0 is ``T`` itself -- and
        each is put through ``correct(..., rounds=rounds)``, i.e. the existing
        render -> match -> vacuity gate -> lift -> PnP block.  The view with the
        largest ``score`` (default: the vacuity-weighted inlier mass) wins, and
        with ``refine`` the winner is then iterated to convergence with the
        configured number of rounds.

        ``stop_score`` > 0 stops as soon as a view's score exceeds it (the
        search is only worth running if it is rare AND short).

        ``prescreen`` > 0 makes the search two-stage: all ``n`` views are
        RENDERED and ranked by the coarse global appearance descriptor
        (``_tiny_desc``), and only the best ``prescreen`` of them -- sample 0
        always among them -- are put through the correction.  A render is
        roughly an order of magnitude cheaper than a render-match-PnP round, so
        this buys a much wider search for the same budget, which is what a
        self-similar scene needs: the right tread is not near the wrong one, so
        the sampling has to be wide enough to contain it before any scoring
        rule can pick it.
        """
        if n < 1:
            raise ValueError(f"search(): n must be >= 1, got {n}")
        dcs, phis = self._sample_xi(n, sigma_t, sigma_r, dist, seed)
        dev = T.device
        dcs, phis = dcs.to(dev), phis.to(dev)
        order = list(range(n))
        pre_sims = None
        if 0 < prescreen < n:
            q = self._tiny_desc(query.to(dev))
            pre_sims = []
            for i in range(n):
                T_i = T if i == 0 else self._perturb(T, dcs[i], phis[i])
                pre_sims.append(float(self._tiny_desc(self._render(T_i)["rgb"]) @ q))
            # sample 0 is the incumbent and must stay in the race, so that the
            # search can always decline to move.
            rank = sorted(range(1, n), key=lambda i: -pre_sims[i])
            order = [0] + rank[: max(prescreen - 1, 0)]
        best_i, best_c, best_T, best_s = -1, None, T, -float("inf")
        base_s, scores, n_ok = -float("inf"), [], 0
        for i in order:
            T_i = T if i == 0 else self._perturb(T, dcs[i], phis[i])
            c = self.correct(query, T_i, rounds=rounds)
            s = self._score_of(c, score)
            scores.append(s)
            n_ok += int(c.success)
            if i == 0:
                base_s = s
            if s > best_s:
                best_i, best_c, best_T, best_s = i, c, T_i, s
            if stop_score > 0 and s >= stop_score:
                break
        n_views = len(scores)
        if best_c is None or not best_c.success:
            return SearchResult(T=T, T_view=T, best=CorrectionResult(
                T=T, success=False, inliers=0, n_matches=0,
                reproj_px=float("inf"), rounds=0),
                score=-float("inf"), base_score=base_s, n_views=n_views,
                n_success=n_ok, best_index=best_i, improved=False, scores=scores)
        out = best_c
        if refine:
            # The winner was scored after ONE round; give it the full
            # iteration, starting from the pose that round produced.
            n_ref = self.rounds if refine_rounds is None else int(refine_rounds)
            rr = self.correct(query, best_c.T, rounds=n_ref)
            # matcher='stage': the scoring pass ran `rounds` rounds, so it ends
            # on a warm (MASt3R/LoFTR/JamMa) round whenever rounds <=
            # feat_stage, while the refine runs the full cascade and ends on
            # the field.  Inlier counts do not compare across matchers -- the
            # field returns far fewer -- so the >= test discards every refine.
            # Apply the main path's rule instead (`best_is_feat` in correct()):
            # a field result always displaces a warm one.
            cross = (self.matcher_name == "stage"
                     and rounds <= self.feat_stage < n_ref)
            if rr.success and (cross or rr.inliers >= best_c.inliers):
                out = rr
        return SearchResult(T=out.T, T_view=best_T, best=out, score=best_s,
                            base_score=base_s, n_views=n_views, n_success=n_ok,
                            best_index=best_i, improved=best_i > 0, scores=scores)
