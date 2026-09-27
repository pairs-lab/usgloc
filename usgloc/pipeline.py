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
    # Coarse-to-fine pyramid: (scale, iters) stages run before the final
    # full-config refinement; () disables. Two coarse stages widen the
    # convergence basin from ~10 cm (single stage) to ~20-35 cm.
    pyramid: tuple = ((0.25, 200), (0.5, 150))
    triage_k: int = 8               # candidates refined at the FIRST pyramid
                                    # stage only; best top_k by PSNR continue
    # Evidential matching correction (usgloc.correction): hypotheses snapped
    # by LoFTR + vacuity-gated PnP skip the pyramid (their start is ~1 cm off)
    # and use corr_refine_iters at the final stage; failed hypotheses fall
    # back to the photometric pyramid path unchanged.
    corr_refine_iters: int = 60
    corr_dedup_t: float = 0.01      # merge snapped hypotheses whose camera
    corr_dedup_r: float = 2.0       # centres are < t (m) AND rotations < r (deg)
    corr_polish: bool = True        # re-correct at the refined pose, keep if
                                    # render fidelity improves
    polish_refine_iters: int = 15   # short refine after an accepted polish so
                                    # Eq. 10 is evaluated at a stationary point
    # Mixture weighting. The retrieval score is a property of the ANCHOR, not
    # of the corrected pose, so it is demoted to a weak tie-break (power 0
    # removes it entirely) and the weights come from evidence measured at the
    # pose itself: render fidelity and the correction's inlier support.
    weight_by_inliers: bool = True
    inlier_floor: float = 0.05      # never zero a component outright
    retrieval_prior_power: float = 0.0
    # Point estimate: the highest-weight mode, not the mixture mean. For a
    # multi-modal posterior the mean lies between the modes, at a pose the
    # model itself considers unlikely; the reported covariance is still the
    # moment-matched one, so ambiguity widens it as before.
    report_mode: bool = True
    corr_verify: bool = True        # a snapped pose must beat its own anchor
    report_geometric: bool = False  # after an accepted polish, report the PnP
                                    # pose (geometric optimum) instead of the
                                    # short-refined photometric one; Sigma is
                                    # still the Laplace at the refined point
    support_power: float = 0.0      # weight *= (number of anchors whose snap
                                    # merged into this hypothesis) ** power
    # --- Evidence beyond the inlier COUNT (every default 0 => the weights are
    # bit-for-bit the ones above). On a repetitive scene a wrong-tread snap
    # can match the right one on inliers, so the count alone cannot separate
    # them; these terms score the SAME correspondences by where they sit and
    # how well they reproject. See _corr_stats().
    # Which snapped candidates get the top_k MIXTURE SLOTS, and what happens
    # to the ones that do not. The slots have always been filled by render
    # fidelity while the weights come from the inlier count -- two different
    # criteria deciding the same question, so a snap the weights would have
    # preferred can be evicted before it is ever weighted.
    snap_rank: str = "psnr"         # psnr (historical) | inliers | inl_cov
    snap_overflow: str = "anchor"   # a snap beyond the last slot falls back to
                                    # the photometric path from its ANCHOR
                                    # (historical, which discards the
                                    # correction) or from the CORRECTED pose
    coverage_power: float = 0.0     # weight *= (8x8 grid cells with an inlier) ** p
    cov_w_power: float = 0.0        # ... using the evenness-softened coverage
    ratio_power: float = 0.0        # weight *= (inliers / fed matches) ** p
    rms_gamma: float = 0.0          # weight *= exp(-gamma * inlier RMS px)
    geo_verify_refine: bool = False # after the polish, keep the short-refined pose
                                    # only if the polish correspondences still
                                    # reproject within threshold at least as often
                                    # as at the PnP pose; otherwise report the PnP
                                    # pose (Sigma stays the Laplace at the refined
                                    # point)
    geo_verify_ratio: float = 0.95
    # --- Support-gated retrieval escalation (default OFF) ----------------
    # The correction's PnP inlier count is a near-binary indicator that the
    # registration is the RIGHT one, not merely a self-consistent one.
    # Measured on Stairs train frames (leave-one-sequence-out anchors, 61
    # corrections): a snap with >= 1500 inliers is within 5 cm 97% of the
    # time, one below 1000 inliers is within 5 cm 4% of the time.  When NO
    # candidate clears the bar the answer is simply not in the hypothesis
    # set -- on a self-similar scene the global descriptor ranks the correct
    # anchor outside the shortlist -- and every downstream criterion
    # (fidelity, inliers, anchor agreement, a semantic verifier) can then
    # only choose between wrong poses.  With escalate_inliers > 0 the
    # pipeline consumes the NEXT tier of retrieval candidates and corrects
    # those too, stopping as soon as one clears the bar.  Cost is paid only
    # on queries that need it.
    escalate_inliers: int = 0       # 0 = off; else the absolute support bar
    escalate_ratio: float = 0.0     # 0 = off; else the same bar expressed as
                                    # inliers / fed matches, which is scene
                                    # independent (at the ground-truth pose it
                                    # is 0.90-0.92 on chess, office and stairs,
                                    # while the absolute inlier count there
                                    # ranges 1622-2814). A candidate clears the
                                    # gate when it meets EVERY bar that is set;
                                    # escalation runs while none does.
    escalate_tiers: int = 2         # extra candidate tiers to try
    escalate_batch: int = 8         # candidates per extra tier
    escalate_tier1: int = 0         # restrict the FIRST tier to this many entries
                                    # of the caller's shortlist, escalating into
                                    # the rest. With escalate_tier1 = the
                                    # shortlist size the option used before,
                                    # the first tier is bit-for-bit the old
                                    # behaviour and every difference is the
                                    # escalation itself; 0 reranks the whole
                                    # (now longer) shortlist into tier 1, which
                                    # changes the baseline too.
    escalate_order: str = "gated"   # order the escalation tiers are consumed in:
                                    # 'gated'     - the evidential gated similarity
                                    #               (the same ranking tier 1 uses)
                                    # 'shortlist' - the order the caller's global
                                    #               shortlist supplied. Measured on
                                    #               Stairs the gated rerank is the
                                    #               WORSE of the two at surfacing a
                                    #               geometrically close anchor
                                    #               (median rank 8 vs 4), so the
                                    #               shortlist order reaches it sooner
                                    # 'spread'    - farthest-point sampling over the
                                    #               anchor centres: each next
                                    #               candidate is the one furthest
                                    #               from every anchor already tried.
                                    #               Escalation fires exactly when the
                                    #               descriptor has proved uninformative
                                    #               for this query, so covering the map
                                    #               can beat following that same
                                    #               ranking further down.
    escalate_min_dist: float = 0.0  # skip an escalation candidate whose anchor
                                    # centre is within this many metres of an
                                    # already-corrected anchor (spatial
                                    # non-maximum suppression: a repetitive
                                    # scene ranks many near-identical views
                                    # together, so the next tier by score
                                    # alone is often more of the same place)
                                    # on render fidelity before it takes a
                                    # mixture slot; snaps rank by fidelity,
                                    # not by inlier count (inliers can be
                                    # large and consistent on a repeated
                                    # structure, e.g. the wrong stair tread)
    # --- Adaptive virtual-view search (default OFF) -----------------------
    # Mechanism: SplatHLoc's adaptive virtual-view retrieval -- cited, not
    # claimed.  Escalation above can only ever try poses SOME ANCHOR PROPOSED;
    # held-out queries sit 13-15 cm off the mapping trajectory whatever the
    # anchor spacing, so on a self-similar scene the correct pose may simply
    # never be in the hypothesis set.  Rendering from a 3DGS map removes that
    # restriction: a virtual view costs one rasterisation, so when no candidate
    # clears the support bar the corrector samples poses AROUND the best
    # candidate, matches the query against each, and keeps the one with the
    # largest vacuity-weighted inlier mass.  It fires only on the queries whose
    # support is weak, which is where every failure lives.
    vsearch_n: int = 0              # 0 = off; virtual views per search,
                                    # INCLUDING the unperturbed pose (sample 0),
                                    # so n=1 is a no-op by construction
    vsearch_inliers: int = 0        # trigger: fire while no corrected candidate
                                    # reaches this many PnP inliers
    vsearch_ratio: float = 0.0      # trigger: ... this inlier/match ratio
    vsearch_psnr: float = 0.0       # trigger: ... this render PSNR (dB)
    vsearch_sigma_t: float = 0.25   # camera-centre sampling scale, metres
    vsearch_sigma_r: float = 5.0    # rotation sampling scale, degrees
    vsearch_dist: str = "normal"    # normal | uniform | shell (see _sample_xi)
    vsearch_rounds: int = 1         # correction rounds spent per virtual view
    vsearch_seed: int = 0           # explicit; the RNG is forked in search()
    vsearch_score: str = "wmass"    # wmass | inliers | ratio | rms
    vsearch_prescreen: int = 0      # >0: two-stage search. All vsearch_n views are
                                    # RENDERED and ranked by a coarse global
                                    # appearance descriptor, and only this many get a
                                    # correction. A render costs ~10x less than a
                                    # render-match-PnP round, so the same budget buys a
                                    # much WIDER sample -- which is the thing a
                                    # self-similar scene needs, since the correct tread
                                    # is not near the wrong one and no scoring rule can
                                    # pick a pose the sampling never drew.
    vsearch_around: str = "snap"    # centre the sampling on the best CORRECTED
                                    # pose ('snap') or on the top anchor
                                    # ('anchor'); with no successful snap both
                                    # fall back to the top anchor
    vsearch_verify: bool = True     # the winner must pass the same render-
                                    # fidelity check every snap passes
    vsearch_gain: float = 1.0       # accept the winner only if its score is at
                                    # least this multiple of the unperturbed
                                    # view's score (1.0 = any improvement)
    vsearch_stop: float = 0.0       # >0: stop sampling once a view scores this
                                    # high (cost control)
    vsearch_priority: bool = True   # the accepted search pose takes the FIRST
                                    # mixture slot. Without it the candidate
                                    # list is sorted by render fidelity and
                                    # truncated at top_k, so on the very scenes
                                    # the search exists for -- where the wrong
                                    # pose renders as well as the right one --
                                    # the found pose can be evicted by the
                                    # candidates that just failed the bar.
    vsearch_support: int = 1        # anchor-agreement count credited to the
                                    # accepted search pose (support_power acts
                                    # on it). 1 = treat it as one more anchor's
                                    # snap; larger values stop a wrong pose that
                                    # several anchors agree on from outvoting a
                                    # much better-supported one the search found.
    # --- Prior-free landmark initialisation (default OFF) -----------------
    # Every hypothesis above comes from a retrieval ANCHOR, so every one of
    # them starts 13-16 cm and 19-47 deg from the query -- and on a
    # self-similar scene the correction can then register confidently on the
    # wrong repeated structure.  usgloc.landmarks.LandmarkCorrector produces
    # an absolute pose from NO anchor at all: query SuperPoint keypoints are
    # matched straight against the per-Gaussian descriptor field and solved by
    # PnP-RANSAC, with no render and no pose prior (~0.05 s).  Measured on
    # held-out train frames it lands 0.9-1.8 cm / 0.27-0.44 deg from the truth,
    # i.e. an order of magnitude closer than the best admissible anchor.
    # With landmark_init the resulting pose is ADDED to the hypothesis set as
    # one more candidate (submap_index = LANDMARK_SUBMAP): the fidelity check,
    # the de-duplication, the inlier weighting and the anchor-agreement term
    # all still decide, so it can only displace an anchor that renders worse.
    # Nothing in the evidential channel, the vacuity gate, the SE(3) mixture
    # or the Laplace covariance is touched.
    landmark_init: bool = False     # add the prior-free landmark hypothesis
    landmark_sole_inliers: int = 0  # >0: when the landmark PnP reaches this
                                    # many inliers the landmark hypothesis
                                    # becomes the SOLE candidate and no anchor
                                    # is corrected at all (the cheap variant:
                                    # one sparse solve + one correction)
    landmark_weight: str = "max"    # retrieval weight handed to the landmark
                                    # hypothesis: 'max' (the largest anchor
                                    # weight, so retrieval_prior_power cannot
                                    # penalise a candidate retrieval never
                                    # proposed) | 'mean' | 'one'


#: ``submap_index`` of the hypothesis produced by the prior-free landmark
#: stage; it belongs to no submap, and -1 is already the prior-refinement
#: sentinel.
LANDMARK_SUBMAP = -2


@dataclass
class LocalizationResult:
    posterior: PoseMixture
    condensed: PoseGaussian
    hypotheses: List[PoseHypothesis] = field(default_factory=list)
    psnrs: List[float] = field(default_factory=list)
    snapped: List[bool] = field(default_factory=list)   # matching correction hit
    inliers: List[int] = field(default_factory=list)    # PnP inliers per kept hyp.
    n_corrections: int = 0      # correct() calls spent on candidates
    n_escalations: int = 0      # extra retrieval tiers consumed (0 unless
                                # PipelineConfig.escalate_inliers > 0)
    n_vviews: int = 0           # virtual views rendered+matched by the adaptive
                                # search (0 unless PipelineConfig.vsearch_n > 1
                                # AND the support trigger fired on this query)
    vsearch_fired: bool = False   # the trigger fired
    vsearch_used: bool = False    # a PERTURBED view was accepted into the mixture
    landmark_ok: bool = False     # the prior-free landmark solve succeeded
                                  # (False unless PipelineConfig.landmark_init)
    landmark_inliers: int = 0     # its sparse PnP inlier count
    landmark_sole: bool = False   # it cleared landmark_sole_inliers, so no
                                  # retrieval anchor was corrected
    landmark_kept: bool = False   # a component of the reported posterior came
                                  # from the landmark hypothesis
    landmark_sec: float = 0.0     # seconds spent in the landmark solve


def _count_inliers(obj, img, T_cw, camera, thresh_px: float) -> int:
    """Number of (world point, pixel) pairs that reproject within thresh_px
    under T_cw (camera-from-world). Used to verify a refined pose against
    the correspondences that produced the PnP pose it started from."""
    import numpy as np

    T = T_cw.detach().double().cpu().numpy()
    X = obj @ T[:3, :3].T + T[:3, 3]
    z = X[:, 2]
    ok = z > 1e-6
    u = camera.fx * X[:, 0] / np.where(ok, z, 1.0) + camera.cx
    v = camera.fy * X[:, 1] / np.where(ok, z, 1.0) + camera.cy
    err = np.hypot(u - img[:, 0], v - img[:, 1])
    return int(np.sum(ok & (err <= thresh_px)))


def _corr_stats(c, camera, grid: int = 8) -> dict:
    """Summarize one correction beyond its inlier COUNT.

    A snap onto a repeated structure (the wrong stair tread) can carry as many
    inliers as the right one, but those inliers are geometrically special:
    they sit on the repeated structure, so they cluster in a part of the image
    instead of covering it. ``cov`` is the fraction of an 8x8 image grid that
    holds at least one inlier, ``cov_w`` the same thing softened by how evenly
    the inliers are distributed over the occupied cells (normalized
    perplexity), and ``spread`` their RMS radius about their centroid in image
    widths. Cheap: a bincount over the inliers already returned by PnP.
    """
    import numpy as np

    st = dict(inliers=int(c.inliers), n_matches=int(c.n_matches),
              reproj_px=float(c.reproj_px), cov=0.0, cov_w=0.0, spread=0.0)
    if c.img is None or len(c.img) == 0:
        return st
    W, H = max(int(camera.width), 1), max(int(camera.height), 1)
    u = np.clip((c.img[:, 0] / W * grid).astype(int), 0, grid - 1)
    v = np.clip((c.img[:, 1] / H * grid).astype(int), 0, grid - 1)
    cells = np.bincount(v * grid + u, minlength=grid * grid).astype(float)
    st["cov"] = float((cells > 0).mean())
    p = cells / max(cells.sum(), 1.0)
    nz = p[p > 0]
    st["cov_w"] = float(np.exp(-float((nz * np.log(nz)).sum())) / (grid * grid))
    d = c.img - c.img.mean(axis=0, keepdims=True)
    st["spread"] = float(np.sqrt(float((d ** 2).sum(axis=1).mean())) / W)
    return st


def _pose_close(Ta: torch.Tensor, Tb: torch.Tensor,
                t_th: float, r_th_deg: float) -> bool:
    """Proximity of two camera-from-world poses: camera-centre distance AND
    rotation angle (the translation column t = -R c is NOT a distance)."""
    ca = -Ta[:3, :3].T @ Ta[:3, 3]
    cb = -Tb[:3, :3].T @ Tb[:3, 3]
    if (ca - cb).norm() >= t_th:
        return False
    cos = (((Ta[:3, :3].T @ Tb[:3, :3]).diagonal().sum() - 1) / 2).clamp(-1, 1)
    return bool(torch.rad2deg(torch.arccos(cos)) < r_th_deg)


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
        corrector=None,   # Optional[EvidentialMatchCorrector]; needs gray_full
        landmark_corrector=None,  # Optional[usgloc.landmarks.LandmarkCorrector];
                                  # used only when cfg.landmark_init is set
    ):
        self.emap = emap
        self.renderer = renderer
        self.camera = camera
        self.desc_net = descriptor_net
        self.aggregator = aggregator
        self.cfg = config or PipelineConfig()
        self.corrector = corrector
        self.landmark_corrector = landmark_corrector
        # Diagnostics only: when True, localize() stores a dict of every
        # candidate pose it saw (anchor, snap, kept component) in
        # ``self.last_debug``. Default OFF -- no effect on any result.
        self.debug = False
        self.last_debug = None

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
        candidates: Optional[Sequence[int]] = None,
        gray_full: Optional[torch.Tensor] = None,
        prior: Optional[torch.Tensor] = None,
    ) -> LocalizationResult:
        """``candidates``: optional submap-index shortlist (hierarchical
        retrieval); the evidential gated similarity reranks within it.
        ``gray_full``: full-resolution query luma for the matching
        correction (required when a corrector is attached)."""
        if self.corrector is not None and gray_full is None:
            raise ValueError(
                "localize(): gray_full is required when a corrector is "
                "attached (pass the full-resolution query luma, or build "
                "the pipeline without corrector=)")
        if prior is not None:
            # Prior-refinement setting (the regime GS-CPR, STDLoc, UGS-Loc and
            # SplatHLoc operate in): an external pose replaces our retrieval
            # front end, and everything downstream -- evidential correction,
            # weighted refinement, tempered Laplace covariance -- is unchanged.
            hyps = [PoseHypothesis(T0=prior.to(image_q.dtype), weight=1.0,
                                   submap_index=-1)]
            pending: List[PoseHypothesis] = []
        else:
            _, scene_desc, _ = self.encode_query(instances)
            n_first = max(self.cfg.top_k, self.cfg.triage_k)
            n_total = n_first
            if ((self.cfg.escalate_inliers > 0 or self.cfg.escalate_ratio > 0)
                    and self.corrector is not None):
                n_total += self.cfg.escalate_tiers * self.cfg.escalate_batch
            if (n_total > n_first and self.cfg.escalate_tier1 > 0
                    and candidates is not None):
                # Tier 1 sees exactly the shortlist the run without escalation
                # would have seen, so the two differ only by the escalation.
                cl = [int(c) for c in candidates]
                head, tail = cl[:self.cfg.escalate_tier1], cl[self.cfg.escalate_tier1:]
                hyps = retrieve_hypotheses(self.emap, scene_desc, top_k=n_first,
                                           candidates=head)
                pending = (retrieve_hypotheses(self.emap, scene_desc,
                                               top_k=len(tail), candidates=tail)
                           if tail else [])
                if not hyps:
                    raise RuntimeError("retrieval returned no hypotheses")
            else:
                ranked = retrieve_hypotheses(self.emap, scene_desc, top_k=n_total,
                                             candidates=candidates)
                if not ranked:
                    raise RuntimeError("retrieval returned no hypotheses")
                hyps, pending = ranked[:n_first], ranked[n_first:]
            if (pending and self.cfg.escalate_order == "shortlist"
                    and candidates is not None):
                # Consume the escalation tiers in the ORDER THE SHORTLIST GAVE
                # (global CLIP similarity), keeping each candidate's gated
                # weight; only the order of the later tiers changes.
                pos = {int(c): j for j, c in enumerate(candidates)}
                pending.sort(key=lambda h: pos.get(int(h.submap_index), 1 << 30))

        # --- Prior-free landmark initialisation (Sec. IV-E, optional) ------
        # One sparse solve against the per-Gaussian descriptor field. It uses
        # no anchor, no render and no prior, so its pose is independent of the
        # retrieval front end; it enters as one more hypothesis and every
        # downstream criterion still decides which mode survives.
        lm_ok, lm_inl, lm_sole, lm_sec = False, 0, False, 0.0
        if self.cfg.landmark_init and self.landmark_corrector is not None:
            import time as _time
            if gray_full is None:
                raise ValueError(
                    "localize(): gray_full is required when "
                    "PipelineConfig.landmark_init is set (the landmark stage "
                    "matches SuperPoint keypoints of the full-resolution query)")
            _t0 = _time.time()
            # T0 is a fall-back only: LandmarkCorrector.correct() ignores it
            # unless the solve fails (verifiable -- the pose is bit-identical
            # for any T0).
            lm = self.landmark_corrector.correct(gray_full, hyps[0].T0)
            lm_sec = _time.time() - _t0
            if lm.success:
                lm_ok, lm_inl = True, int(lm.inliers)
                w = [float(h.weight) for h in hyps]
                if self.cfg.landmark_weight == "mean":
                    lw = sum(w) / max(len(w), 1)
                elif self.cfg.landmark_weight == "one":
                    lw = 1.0
                else:
                    lw = max(w) if w else 1.0
                lm_hyp = PoseHypothesis(T0=lm.T.to(image_q.dtype), weight=lw,
                                        submap_index=LANDMARK_SUBMAP)
                if (self.cfg.landmark_sole_inliers > 0
                        and lm_inl >= self.cfg.landmark_sole_inliers):
                    # The cheap variant: the sparse support is strong enough
                    # that the anchors have nothing to add, so skip them.
                    hyps, pending, lm_sole = [lm_hyp], [], True
                else:
                    hyps = [lm_hyp] + list(hyps)

        # Coarse-to-fine pyramid: low-resolution stages have much wider
        # convergence basins (measured on chess: a 15 cm start fails at
        # 320x240 alone; 80x60 -> 160x120 -> 320x240 recovers 20-35 cm
        # starts), which test-time anchors need -- held-out sequences run
        # ~15 cm off the mapping trajectory.
        stages = []
        for s, it in (self.cfg.pyramid or ()):
            cam_s = Camera(
                fx=self.camera.fx * s, fy=self.camera.fy * s,
                cx=self.camera.cx * s, cy=self.camera.cy * s,
                width=int(round(self.camera.width * s)),
                height=int(round(self.camera.height * s)),
            )
            img_s = torch.nn.functional.interpolate(
                image_q.permute(2, 0, 1)[None],
                size=(cam_s.height, cam_s.width),
                mode="bilinear", align_corners=False,
            )[0].permute(1, 2, 0).contiguous()
            stages.append((cam_s, img_s, it))

        # --- Evidential matching correction (Sec. IV-E): snap every
        # candidate with LoFTR + vacuity-gated PnP against the rendered map.
        # Its basin (~60 cm) covers the anchor-to-query gap, so snapped
        # hypotheses skip the pyramid; failures keep the photometric path.
        snapped: List[tuple] = []   # (hyp, T_snap, inliers, support, stats)
        failed: List[PoseHypothesis] = []
        _want_stats = bool(
            self.debug or self.cfg.coverage_power or self.cfg.cov_w_power
            or self.cfg.ratio_power or self.cfg.rms_gamma
            or self.cfg.snap_rank == "inl_cov")
        if self.corrector is not None:
            import dataclasses

            from .refinement import _psnr, _render

            def _fidelity(T):
                with torch.no_grad():
                    out = _render(self.emap, self.renderer,
                                  T.to(image_q.dtype), self.camera)
                    return _psnr(out["rgb"], image_q)

            scored = []
            dbg_snaps = []
            tried: List[torch.Tensor] = []   # anchors already corrected

            def _correct_batch(batch):
                """Correct every hypothesis in ``batch``; fill scored/failed."""
                for h in batch:
                    tried.append(h.T0)
                    c = self.corrector.correct(gray_full, h.T0)
                    if self.debug:
                        dbg_snaps.append(dict(
                            T0=h.T0.detach().cpu(), weight=float(h.weight),
                            submap=int(h.submap_index), success=bool(c.success),
                            T=c.T.detach().cpu(), inliers=int(c.inliers),
                            n_matches=int(c.n_matches),
                            reproj_px=float(c.reproj_px),
                            **{("st_" + k): v for k, v in
                               _corr_stats(c, self.corrector.camera).items()}))
                    if not c.success:
                        failed.append(h)
                        continue
                    if self.cfg.corr_verify:
                        # A correction is only trusted when it renders better
                        # than the anchor it started from. On repetitive
                        # geometry PnP can be confident and wrong; fidelity is
                        # the independent check, and it costs one render.
                        psnr_c = _fidelity(c.T)
                        if self.debug:
                            dbg_snaps[-1]["psnr"] = float(psnr_c)
                            dbg_snaps[-1]["psnr_anchor"] = float(_fidelity(h.T0))
                        if psnr_c < _fidelity(h.T0):
                            if self.debug:
                                dbg_snaps[-1]["verify_failed"] = True
                            failed.append(h)
                            continue
                        scored.append((h, c, psnr_c))
                    else:
                        scored.append((h, c, float(c.inliers)))

            _correct_batch(hyps)
            n_escalations = 0
            if self.cfg.escalate_inliers > 0 or self.cfg.escalate_ratio > 0:
                def _cleared():
                    for _, c, _ in scored:
                        if int(c.inliers) < self.cfg.escalate_inliers:
                            continue
                        if (self.cfg.escalate_ratio > 0 and int(c.inliers)
                                < self.cfg.escalate_ratio * max(int(c.n_matches), 1)):
                            continue
                        return True
                    return False

                def _centre(T):
                    return -T[:3, :3].T @ T[:3, 3]

                while pending and not _cleared():
                    batch = []
                    if self.cfg.escalate_order == "spread":
                        ref = [_centre(T) for T in tried]
                        while pending and len(batch) < self.cfg.escalate_batch:
                            cs = torch.stack([_centre(h.T0) for h in pending])
                            d = torch.cdist(cs[None].float(),
                                            torch.stack(ref)[None].float())[0]
                            dmin = d.min(dim=1).values
                            j = int(torch.argmax(dmin))
                            if float(dmin[j]) < self.cfg.escalate_min_dist:
                                break   # nothing genuinely new left to try
                            h = pending.pop(j)
                            ref.append(_centre(h.T0))
                            batch.append(h)
                    else:
                        while pending and len(batch) < self.cfg.escalate_batch:
                            h = pending.pop(0)
                            if self.cfg.escalate_min_dist > 0 and any(
                                    _pose_close(h.T0, Tt, self.cfg.escalate_min_dist,
                                                1e9) for Tt in tried):
                                continue   # same place as something already tried
                            batch.append(h)
                    if not batch:
                        break
                    n_escalations += 1
                    _correct_batch(batch)

            # --- Adaptive virtual-view search (SplatHLoc's mechanism) ------
            # Fires only while NO corrected candidate clears the support bar,
            # i.e. exactly on the queries whose answer is not in the anchor
            # set. The search is free to propose poses no anchor proposed.
            n_vviews = 0
            vs_fired = vs_used = False
            vs_entry = None
            if (self.cfg.vsearch_n > 1 and (self.cfg.vsearch_inliers > 0
                                            or self.cfg.vsearch_ratio > 0
                                            or self.cfg.vsearch_psnr > 0)):
                def _vs_cleared():
                    for _, c, p in scored:
                        if int(c.inliers) < self.cfg.vsearch_inliers:
                            continue
                        if (self.cfg.vsearch_ratio > 0 and int(c.inliers)
                                < self.cfg.vsearch_ratio * max(int(c.n_matches), 1)):
                            continue
                        if self.cfg.vsearch_psnr > 0:
                            pv = p if self.cfg.corr_verify else _fidelity(c.T)
                            if float(pv) < self.cfg.vsearch_psnr:
                                continue
                        return True
                    return False

                if not _vs_cleared():
                    vs_fired = True
                    # Centre the sampling on the best evidence we have.
                    if scored and self.cfg.vsearch_around == "snap":
                        h_c, c_c, _ = max(scored, key=lambda t: int(t[1].inliers))
                        T_c = c_c.T
                    else:
                        h_c, T_c = hyps[0], hyps[0].T0
                    sr = self.corrector.search(
                        gray_full, T_c, n=self.cfg.vsearch_n,
                        sigma_t=self.cfg.vsearch_sigma_t,
                        sigma_r=self.cfg.vsearch_sigma_r,
                        seed=self.cfg.vsearch_seed, dist=self.cfg.vsearch_dist,
                        score=self.cfg.vsearch_score,
                        rounds=self.cfg.vsearch_rounds,
                        stop_score=self.cfg.vsearch_stop,
                        prescreen=self.cfg.vsearch_prescreen)
                    n_vviews = sr.n_views
                    c = sr.best
                    ok = (c.success and sr.improved
                          and (sr.base_score <= 0.0
                               or sr.score >= self.cfg.vsearch_gain * sr.base_score))
                    psnr_c = None
                    if ok and self.cfg.corr_verify and self.cfg.vsearch_verify:
                        # Same independent check every snap passes: the pose
                        # must render the query at least as well as the pose
                        # the search started from.
                        psnr_c = _fidelity(c.T)
                        ok = bool(psnr_c >= _fidelity(T_c))
                    if ok:
                        vs_used = True
                        key = (psnr_c if self.cfg.corr_verify
                               else float(c.inliers))
                        if key is None:
                            key = _fidelity(c.T)
                        vs_entry = (dataclasses.replace(h_c, T0=sr.T_view), c, key)
                        scored.append(vs_entry)
                    if self.debug:
                        dbg_snaps.append(dict(
                            T0=T_c.detach().cpu(), weight=float(h_c.weight),
                            submap=int(h_c.submap_index), success=bool(c.success),
                            T=c.T.detach().cpu(), inliers=int(c.inliers),
                            n_matches=int(c.n_matches),
                            reproj_px=float(c.reproj_px), vsearch=True,
                            vs_score=float(sr.score), vs_base=float(sr.base_score),
                            vs_index=int(sr.best_index), vs_views=int(sr.n_views),
                            vs_accepted=bool(vs_used)))
            def _slot_key(hcp):
                """What a snap must win to take one of the top_k slots."""
                _h, c, s = hcp
                if self.cfg.snap_rank == "inliers":
                    return -float(c.inliers)
                if self.cfg.snap_rank == "inl_cov":
                    return -float(c.inliers) * float(
                        _corr_stats(c, self.corrector.camera)["cov"])
                return -s        # render fidelity (or the inlier count when
                                 # corr_verify is off): the historical order

            scored.sort(key=_slot_key)
            if vs_entry is not None and self.cfg.vsearch_priority:
                # The search fired because NOTHING cleared the support bar, so
                # the ranking that follows is a ranking of poses already judged
                # unsupported; do not let it evict the one pose that is.
                for _j, _t in enumerate(scored):
                    if _t is vs_entry:      # identity: the tuples hold tensors
                        scored.insert(0, scored.pop(_j))
                        break
            for h, c, _ in scored:
                # Near-duplicate poses waste mixture slots: merge the
                # candidate's retrieval mass into the kept duplicate so no
                # probability is silently dropped.
                dup = next((j for j, (_, Ts, _, _, _) in enumerate(snapped)
                            if _pose_close(c.T, Ts, self.cfg.corr_dedup_t,
                                           self.cfg.corr_dedup_r)), None)
                if dup is not None:
                    hk, Tk, ik, sk, stk = snapped[dup]
                    snapped[dup] = (
                        dataclasses.replace(hk, weight=hk.weight + h.weight),
                        Tk, ik, sk + 1, stk)
                    continue
                if len(snapped) == self.cfg.top_k:
                    # Overflow: keep it on the photometric path rather than
                    # dropping it. 'anchor' restarts from the retrieval pose
                    # and throws the correction away; 'snap' keeps the
                    # corrected pose, so a good snap that lost the slot race
                    # is still refined from where the correction put it.
                    failed.append(dataclasses.replace(h, T0=c.T.to(h.T0.dtype))
                                  if self.cfg.snap_overflow == "snap" else h)
                    continue
                snapped.append((h, c.T, c.inliers, 1,
                                _corr_stats(c, self.corrector.camera)
                                if _want_stats else None))
            if vs_entry is not None and self.cfg.vsearch_support > 1:
                # Anchor agreement is a vote among the anchors, and the search
                # fires exactly when that vote is uninformative: several anchors
                # can agree on the same WRONG pose (the same wrong stair tread
                # is what their descriptors retrieved), and with support_power
                # on that agreement outweighs a lone, far better supported pose.
                for _j, (hk, Tk, ik, sk, stk) in enumerate(snapped):
                    if _pose_close(vs_entry[1].T, Tk, self.cfg.corr_dedup_t,
                                   self.cfg.corr_dedup_r):
                        snapped[_j] = (hk, Tk, ik,
                                       max(sk, self.cfg.vsearch_support), stk)
                        break
        else:
            failed = list(hyps)

        # seeds: (hyp, T_init, was_snapped, inliers, support, corr stats);
        # snapped first.
        seeds = [(h, T, True, inl, sup, st) for h, T, inl, sup, st in snapped]
        n_free = self.cfg.top_k - len(seeds)
        if (self.cfg.escalate_inliers > 0 or self.cfg.escalate_ratio > 0):
            # Escalation only ever ADDS candidates that failed to snap, and the
            # photometric triage below refines every one of them. Keep that
            # fallback the size it was without escalation (the tier-1
            # candidates, which come first in `failed`): the escalation tiers
            # exist to produce a snap, not to enlarge the photometric search.
            failed = failed[:max(self.cfg.triage_k, self.cfg.top_k)]
        if n_free > 0 and failed:
            # Coarse triage: run only the FIRST (cheapest) pyramid stage on
            # the remaining candidates, keep the best by render fidelity,
            # and spend the rest of the pyramid + the final stage on those.
            if stages and len(failed) > n_free:
                cam_s, img_s, it = stages[0]
                triaged = []
                for h in failed:
                    r0 = refine_pose(
                        self.emap, self.renderer, cam_s, img_s, h.T0,
                        gamma=self.cfg.gamma, iters=it, lr=self.cfg.refine_lr,
                        compute_covariance=False,
                    )
                    triaged.append((h, r0))
                triaged.sort(key=lambda hr: -hr[1].psnr)
                pyr_seeds = [(h, r0.T_star) for h, r0 in triaged[:n_free]]
                rest = stages[1:]
            else:
                pyr_seeds = [(h, h.T0) for h in failed[:n_free]]
                rest = stages
            for h, T0 in pyr_seeds:
                for cam_s, img_s, it in rest:
                    T0 = refine_pose(
                        self.emap, self.renderer, cam_s, img_s, T0,
                        gamma=self.cfg.gamma, iters=it, lr=self.cfg.refine_lr,
                        compute_covariance=False,
                    ).T_star
                seeds.append((h, T0, False, 0, 1, None))

        results = []
        for h, T0, was_snapped, inl, sup, st in seeds:
            res = refine_pose(
                self.emap, self.renderer, self.camera, image_q, T0,
                gamma=self.cfg.gamma,
                iters=self.cfg.corr_refine_iters if was_snapped
                      else self.cfg.refine_iters,
                lr=self.cfg.refine_lr,
                compute_covariance=self.cfg.compute_covariance,
                jacobian_pixels=self.cfg.jacobian_pixels,
            )
            results.append((h, res, was_snapped, inl, sup, st))

        # Fidelity pruning: keep hypotheses within a PSNR margin of the best.
        best_psnr = max(r.psnr for _, r, _, _, _, _ in results)
        kept = [hr for hr in results
                if hr[1].psnr >= best_psnr - self.cfg.psnr_prune_margin]

        # Polish: re-correct at the refined pose. A short refinement then
        # re-establishes stationarity of the weighted photometric energy at
        # the polished pose, so the Laplace approximation \eqref is evaluated
        # at an actual optimum. The polish is accepted only when render
        # fidelity does not degrade.
        if self.corrector is not None and self.cfg.corr_polish:
            polished = []
            for h, r, was_snapped, inl, sup, st in kept:
                c = self.corrector.correct(gray_full, r.T_star)
                if c.success:
                    r2 = refine_pose(
                        self.emap, self.renderer, self.camera, image_q,
                        c.T.to(image_q.dtype),
                        gamma=self.cfg.gamma,
                        iters=self.cfg.polish_refine_iters,
                        lr=self.cfg.refine_lr,
                        compute_covariance=self.cfg.compute_covariance,
                        jacobian_pixels=self.cfg.jacobian_pixels,
                    )
                    if r2.psnr >= r.psnr:
                        r, was_snapped, inl = r2, True, c.inliers
                        if _want_stats:
                            # The accepted polish is the correction the kept
                            # component now rests on; score THAT one.
                            st = _corr_stats(c, self.corrector.camera)
                        if self.cfg.report_geometric:
                            # The PnP solution is the geometric optimum;
                            # the short refinement only re-establishes
                            # stationarity for the Laplace covariance.
                            r = dataclasses.replace(
                                r2, T_star=c.T.to(image_q.dtype))
                        elif self.cfg.geo_verify_refine and c.obj is not None:
                            # Geometric verification of the photometric
                            # refinement: re-count the polish inliers under
                            # the refined pose. A refinement that leaves the
                            # geometric optimum loses inliers; report the PnP
                            # pose in that case.
                            n_ref = _count_inliers(c.obj, c.img, r2.T_star, self.camera,
                                                   self.corrector.reproj_thresh_px)
                            if n_ref < self.cfg.geo_verify_ratio * c.inliers:
                                r = dataclasses.replace(
                                    r2, T_star=c.T.to(image_q.dtype))
                polished.append((h, r, was_snapped, inl, sup, st))
            kept = polished

        # pi_k from the evidence observed AT the corrected pose (Eq. 12).
        # The retrieval score describes the anchor a hypothesis started from,
        # not where the correction moved it, so on a repetitive scene it is
        # uninformative about which mode is correct and, being a hard prior,
        # it overrides the terms that are: on Stairs, selecting by retrieval
        # weight gives a 12.9 cm median against 2.9 cm by inlier count, with
        # an oracle of 2.5 cm. We therefore weight by render fidelity and by
        # the geometric support of the correction, and keep the retrieval
        # score only as a weak tie-break.
        dt, dv = image_q.dtype, image_q.device
        fid = torch.tensor([r.psnr for _, r, _, _, _, _ in kept], dtype=dt, device=dv)
        inl = torch.tensor([float(i) for _, _, _, i, _, _ in kept], dtype=dt, device=dv)
        raw = torch.tensor([h.weight for h, _, _, _, _, _ in kept], dtype=dt, device=dv)
        sup = torch.tensor([float(s) for _, _, _, _, s, _ in kept], dtype=dt, device=dv)
        if self.cfg.weight_by_inliers and float(inl.max()) > 0:
            # Geometric support alone is the sharpest discriminator between
            # modes. Render fidelity is a weaker one -- two poses one stair
            # tread apart render almost equally well -- and because a softmax
            # over PSNR is peaked, mixing it in actively degrades the choice
            # (measured on Stairs: inliers 2.9 cm, PSNR 5.5 cm, their product
            # 3.4 cm, oracle 2.1 cm). PSNR is therefore kept as the pruning
            # criterion above, not as a weight.
            weights = (inl / inl.max()).clamp_min(self.cfg.inlier_floor)
        else:
            weights = torch.softmax(fid / self.cfg.psnr_temperature, dim=0)
        weights = weights * raw.clamp_min(1e-6) ** self.cfg.retrieval_prior_power
        # Anchor agreement: a pose that several anchors' corrections
        # converged to is corroborated; a wrong-tread snap is usually
        # specific to the anchor it started from.
        weights = weights * sup ** self.cfg.support_power
        # Evidence the inlier COUNT cannot express: where the inliers sit and
        # how tightly they close. Every power/gamma defaults to 0, in which
        # case each factor is exactly 1 and the weights are unchanged.
        if (self.cfg.coverage_power or self.cfg.cov_w_power
                or self.cfg.ratio_power or self.cfg.rms_gamma):
            def _col(key, default):
                return torch.tensor(
                    [float(st[key]) if st else default for _, _, _, _, _, st in kept],
                    dtype=dt, device=dv)

            if self.cfg.coverage_power:
                weights = weights * _col("cov", 0.0).clamp_min(1e-3) ** self.cfg.coverage_power
            if self.cfg.cov_w_power:
                weights = weights * _col("cov_w", 0.0).clamp_min(1e-3) ** self.cfg.cov_w_power
            if self.cfg.ratio_power:
                nm = _col("n_matches", 1.0).clamp_min(1.0)
                ratio = (_col("inliers", 0.0) / nm).clamp(0.0, 1.0)
                weights = weights * ratio.clamp_min(1e-3) ** self.cfg.ratio_power
            if self.cfg.rms_gamma:
                rms = _col("reproj_px", 0.0)
                rms = torch.where(torch.isfinite(rms), rms, torch.zeros_like(rms))
                weights = weights * torch.exp(-self.cfg.rms_gamma
                                              * (rms - rms.min()).clamp_min(0.0))

        components = [PoseGaussian(T_mode=r.T_star, Sigma=r.Sigma)
                      for _, r, _, _, _, _ in kept]
        posterior = PoseMixture(components=components, weights=weights)
        if self.debug:
            self.last_debug = dict(
                snaps=dbg_snaps if self.corrector is not None else [],
                n_snapped=len(snapped),
                n_escalations=n_escalations if self.corrector is not None else 0,
                n_vviews=n_vviews if self.corrector is not None else 0,
                kept=[dict(T=r.T_star.detach().cpu(), psnr=float(r.psnr),
                           inliers=int(i), support=int(s), snapped=bool(sn),
                           weight=float(w), Sigma=c.Sigma.detach().cpu(),
                           stats=st)
                      for (_, r, sn, i, s, st), w, c in
                      zip(kept, weights.tolist(), components)],
            )
        return LocalizationResult(
            posterior=posterior,
            condensed=posterior.condense(at_mode=self.cfg.report_mode),
            hypotheses=[h for h, _, _, _, _, _ in kept],
            psnrs=[r.psnr for _, r, _, _, _, _ in kept],
            snapped=[s for _, _, s, _, _, _ in kept],
            inliers=[i for _, _, _, i, _, _ in kept],
            n_corrections=(len(tried) if self.corrector is not None else 0),
            n_escalations=(n_escalations if self.corrector is not None else 0),
            n_vviews=(n_vviews if self.corrector is not None else 0),
            vsearch_fired=(vs_fired if self.corrector is not None else False),
            vsearch_used=(vs_used if self.corrector is not None else False),
            landmark_ok=lm_ok,
            landmark_inliers=lm_inl,
            landmark_sole=lm_sole,
            landmark_kept=any(int(h.submap_index) == LANDMARK_SUBMAP
                              for h, _, _, _, _, _ in kept),
            landmark_sec=lm_sec,
        )
