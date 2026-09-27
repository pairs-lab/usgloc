"""Pose-free 2D-3D association: query pixels matched DIRECTLY to Gaussians.

Motivation
----------
:mod:`usgloc.correction` renders the map at a pose hypothesis, matches the
query against the rendered IMAGE and lifts the render-side keypoint to 3D
through the rendered DEPTH.  The 3D point is therefore a read-out of an
alpha-blended depth buffer: biased at silhouettes, biased along grazing
surfaces, and undefined where the blend mixes two surfaces.  It also needs a
pose hypothesis to render from, so every correspondence inherits whatever
anchor the retrieval front end supplied.

This module removes both dependencies.  Query keypoints are described with the
frozen SuperPoint encoder and matched against the descriptors of a sampled set
of LANDMARK Gaussians (:mod:`usgloc.locfeat`'s field); the 3D point of a
correspondence is that Gaussian's MEAN -- a map parameter.  No rendering, no
depth buffer, no pose prior.  This is STDLoc's sparse stage (Zhai et al.,
CVPR'25, ``stdloc.py::loc_sparse`` + ``train_detector.py``); the mechanism is
theirs and is cited as such, nothing here is claimed as novel matching.

What is ours is where the evidential channel acts.  In the rendering path the
vacuity gate can only see an alpha-blended vacuity image; here the same signal
is available per landmark, as a map parameter, so it can select the landmark
set (``vac='gate'``), bias the matching-oriented sampling (``vac='score'``) or
weight the PnP residual, all without a pose.

Landmark selection
------------------
STDLoc samples ~16k landmarks per scene by "matching-oriented sampling":

1. :func:`match_score_field` -- for every training view, project every
   Gaussian, keep the ones the map's own rendered depth says are visible, and
   score each by the cosine between its own descriptor and the query-image
   descriptor at its projection.  Average over views.  This is a
   REPEATABILITY / keypoint-ness signal (``train_detector.py::
   calculate_match_score``): it is high exactly for the Gaussians whose
   descriptor an unseen view will reproduce, which is what the sparse matcher
   needs, and it is NOT the observation count (that was measured and does not
   work).
2. :func:`mo_sample` -- ``train_detector.py::random_knn_score``: draw ``num``
   random Gaussians, and from each one's ``k`` spatial neighbours keep the
   highest-scoring not-yet-taken Gaussian.  The random seeds spread the
   landmarks over the whole cloud while the kNN step snaps each to the locally
   most matchable Gaussian.

:func:`build_landmarks` assembles the resulting set, :func:`landmark_matches`
produces the ``(object points, image points)`` float64 arrays
:mod:`usgloc.correction` feeds to PnP -- same convention (image points are
pixel CENTRES, i.e. index + 0.5), so the two association mechanisms are
interchangeable in front of the same solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from . import locfeat
from .rendering import Camera
from .superpoint import SuperPoint

#: Keys a match-score sidecar checkpoint carries (``scripts/analysis/
#: train_landmarks.py``).  Small: 4 * N bytes per field.
LANDMARK_SCORE_KEYS = ("lm_score", "lm_det", "lm_score_obs", "lm_score_n",
                       "lm_score_base", "lm_score_feats", "lm_score_stride",
                       "lm_score_holdout", "lm_score_dim")


# ---------------------------------------------------------------------------
# 1. the keypoint-ness / repeatability field
# ---------------------------------------------------------------------------

@torch.no_grad()
def match_score_field(
    emap,
    ds,
    camera: Camera,
    encoder: SuperPoint,
    renderer,
    frame_ids: Sequence[int],
    occl_rel: float = 0.05,
    occl_abs: float = 0.02,
    min_alpha: float = 0.2,
    near: float = 0.05,
    log_every: int = 200,
) -> Dict[str, torch.Tensor]:
    """Per-Gaussian mean descriptor agreement with the training views.

    STDLoc's ``calculate_match_score`` with our own occlusion test: a Gaussian
    counts in a view when it is in the frustum, the map's own RENDERED alpha at
    its projection is at least ``min_alpha`` and its camera depth agrees with
    the rendered expected depth to ``occl_abs + occl_rel * z`` (no sensor depth
    anywhere, exactly as :func:`usgloc.locfeat.init_loc_feats`).

    Requires ``emap.loc_feats``; the query descriptors are projected with the
    map's PCA when it has one, so the cosine is computed in the space the
    matcher will use.

    Returns:
        ``score`` (N,) mean cosine (``-1`` for never-observed Gaussians),
        ``det`` (N,) mean SuperPoint DETECTOR score at the projection (a second,
        descriptor-free keypoint-ness candidate), ``obs`` (N,) views counted,
        ``n_frames``.
    """
    if emap.loc_feats is None:
        raise ValueError("match_score_field needs emap.loc_feats (attach a "
                         "descriptor field first, see usgloc.locfeat)")
    dev = emap.means.device
    N = emap.means.shape[0]
    feats = emap.loc_feats.to(dev).float()
    pca = None
    if emap.loc_feat_pca_basis is not None:
        pca = (emap.loc_feat_pca_mean.to(dev), emap.loc_feat_pca_basis.to(dev))
    ssum = torch.zeros(N, device=dev)
    dsum = torch.zeros(N, device=dev)
    obs = torch.zeros(N, device=dev)
    H, W = camera.height, camera.width
    for n, i in enumerate(frame_ids):
        img = ds.load_image(int(i)).to(dev)
        T = ds[int(i)].pose_cw.to(dev, torch.float32)
        out = renderer.render(emap.means, emap.covs, emap.colors, emap.opacities,
                              T, camera, vacuity=emap.gaussian_vacuity,
                              with_depth=True)
        desc, scores = encoder.dense_descriptors(img)
        p = emap.means @ T[:3, :3].T + T[:3, 3]
        z = p[:, 2]
        u = camera.fx * p[:, 0] / z.clamp_min(1e-6) + camera.cx
        v = camera.fy * p[:, 1] / z.clamp_min(1e-6) + camera.cy
        ok = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()
        ok &= out["alpha"][vi, ui] >= min_alpha
        ok &= (z - out["depth"][vi, ui]).abs() <= occl_abs + occl_rel * z
        idx = torch.nonzero(ok, as_tuple=False).squeeze(-1)
        if idx.numel():
            uv = torch.stack([u[idx], v[idx]], dim=-1)
            f = locfeat.sample_query_descriptors(desc, uv, H, W, pca=pca)
            ssum.index_add_(0, idx, (f * feats[idx]).sum(-1))
            dsum.index_add_(0, idx, scores[0][vi[idx], ui[idx]].float())
            obs.index_add_(0, idx, torch.ones_like(idx, dtype=ssum.dtype))
        del out, desc, scores
        if log_every and n % log_every == 0:
            print(f"      match score {n + 1}/{len(frame_ids)}: "
                  f"{int((obs > 0).sum())}/{N} observed", flush=True)
    den = obs.clamp_min(1.0)
    score = ssum / den
    det = dsum / den
    score[obs == 0] = -1.0
    det[obs == 0] = 0.0
    return {"score": score, "det": det, "obs": obs, "n_frames": len(frame_ids)}


# ---------------------------------------------------------------------------
# 2. landmark sampling
# ---------------------------------------------------------------------------

def mo_sample(xyz: torch.Tensor, score: torch.Tensor, num: int = 16384,
              k: int = 32, seed: int = 0,
              candidates: Optional[torch.Tensor] = None) -> torch.Tensor:
    """STDLoc's matching-oriented sampling (``random_knn_score``).

    ``num`` uniformly random seeds among ``candidates``; each seed's ``k``
    spatial nearest neighbours are visited in order of decreasing ``score`` and
    the first not-yet-selected one is taken.  Returns the selected indices INTO
    THE FULL Gaussian array (sorted, unique, <= ``num`` of them).

    The kNN uses a KD-tree instead of the authors' dense ``cdist`` (which is a
    ``num x N`` matrix: 28 GB at our map sizes) -- same neighbours, exactly.
    """
    from scipy.spatial import cKDTree

    if candidates is None:
        candidates = torch.arange(xyz.shape[0], device=xyz.device)
    cand = candidates.to(torch.long)
    pts = xyz[cand].detach().cpu().double().numpy()
    sc = score[cand].detach().cpu().float().numpy()
    m = pts.shape[0]
    if m == 0:
        return cand[:0]
    g = torch.Generator().manual_seed(int(seed))
    seeds = torch.randperm(m, generator=g)[:int(num)].numpy()
    kk = int(min(max(k, 1), m))
    tree = cKDTree(pts)
    knn = tree.query(pts[seeds], k=kk, workers=-1)[1]
    if knn.ndim == 1:
        knn = knn[:, None]
    order = np.argsort(-sc[knn], axis=1)                 # best score first
    knn = np.take_along_axis(knn, order, axis=1)
    taken = np.zeros(m, dtype=bool)
    out = np.empty(len(seeds), dtype=np.int64)
    n = 0
    for row in knn:
        for j in row:
            if not taken[j]:
                taken[j] = True
                out[n] = j
                n += 1
                break
    sel = torch.as_tensor(np.sort(out[:n]), dtype=torch.long)
    return cand[sel.to(cand.device)]


@dataclass
class LandmarkSet:
    """The Gaussians the pose-free sparse stage may match against."""

    idx: torch.Tensor      # (M,) indices into the map's Gaussian arrays
    xyz: torch.Tensor      # (M, 3) float32 Gaussian means (the 3D points)
    feat: torch.Tensor     # (M, D) float32 L2-normalised descriptors
    vac: torch.Tensor      # (M,) per-Gaussian vacuity (0 when the map has none)
    score: torch.Tensor    # (M,) matching-oriented score (-1 if unavailable)
    info: dict             # how the set was built (for the record)


def build_landmarks(
    emap,
    score: Optional[torch.Tensor] = None,
    det: Optional[torch.Tensor] = None,
    num: int = 16384,
    k: int = 32,
    sample: str = "mo",
    seed: int = 0,
    min_obs: float = 1.0,
    score_min: float = -2.0,
    det_min: float = 0.0,
    vac: str = "none",
    u_max: float = 1.0,
    vac_gamma: float = 3.0,
) -> LandmarkSet:
    """Choose the landmark set.

    Args:
        score: (N,) matching-oriented score from :func:`match_score_field`.
            Required for ``sample='mo'`` and ``'top'``.
        det: (N,) mean detector score, for ``det_min`` / ``sample='det'``.
        num: landmark budget (STDLoc: 16384).
        k: kNN width of the matching-oriented sampling (STDLoc: 32).
        sample: ``'mo'`` matching-oriented (STDLoc) | ``'top'`` the ``num``
            globally highest-scoring | ``'det'`` the ``num`` highest detector
            score | ``'random'`` uniform | ``'all'`` every candidate.
        min_obs: keep only Gaussians seen in at least this many field views.
        vac: how the map's own vacuity enters -- ``'none'``, ``'gate'``
            (candidates with vacuity > ``u_max`` are dropped), ``'score'`` (the
            sampling score is multiplied by the evidential weight
            ``exp(-vac_gamma * u)``, the paper's own pixel weight applied to a
            landmark), or ``'both'``.
    """
    if emap.loc_feats is None:
        raise ValueError("build_landmarks needs emap.loc_feats")
    dev = emap.means.device
    N = emap.means.shape[0]
    feats = emap.loc_feats.to(dev).float()
    obs = (emap.loc_feat_obs.to(dev).float() if emap.loc_feat_obs is not None
           else (feats.abs().sum(-1) > 0).float())
    u = (emap.gaussian_vacuity.to(dev).float() if emap.gaussian_vacuity is not None
         else torch.zeros(N, device=dev))
    keep = (obs > max(0.0, min_obs - 0.5)) & (feats.abs().sum(-1) > 0)
    if vac in ("gate", "both") and u_max < 1.0:
        keep &= u <= u_max
    sc = (score.to(dev).float() if score is not None
          else torch.zeros(N, device=dev))
    if score is not None and score_min > -2.0:
        keep &= sc >= score_min
    if det is not None and det_min > 0:
        keep &= det.to(dev).float() >= det_min
    cand = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    sc_eff = sc
    if vac in ("score", "both"):
        # clamp_min(0) first: a negative cosine times a small weight would come
        # out LARGER, i.e. a vacuous bad landmark would be promoted.
        sc_eff = sc.clamp_min(0.0) * torch.exp(-vac_gamma * u.clamp(0.0, 1.0))
    if sample == "all" or cand.numel() <= num:
        idx = cand
    elif sample == "mo":
        if score is None:
            raise ValueError("sample='mo' needs the match-score field")
        idx = mo_sample(emap.means, sc_eff, num=num, k=k, seed=seed,
                        candidates=cand)
    elif sample == "top":
        if score is None:
            raise ValueError("sample='top' needs the match-score field")
        idx = cand[torch.topk(sc_eff[cand], int(num)).indices].sort().values
    elif sample == "det":
        if det is None:
            raise ValueError("sample='det' needs the detector-score field")
        idx = cand[torch.topk(det.to(dev).float()[cand], int(num)).indices].sort().values
    elif sample == "random":
        g = torch.Generator().manual_seed(int(seed))
        pick = torch.randperm(cand.numel(), generator=g)[:int(num)]
        idx = cand[pick.to(cand.device)].sort().values
    else:
        raise ValueError(f"unknown sample {sample!r} "
                         "(mo | top | det | random | all)")
    return LandmarkSet(
        idx=idx, xyz=emap.means[idx].float(), feat=feats[idx], vac=u[idx],
        score=sc[idx],
        info=dict(sample=sample, num=int(num), k=int(k), seed=int(seed),
                  min_obs=float(min_obs), score_min=float(score_min),
                  det_min=float(det_min), vac=vac, u_max=float(u_max),
                  n_candidates=int(cand.numel()), n_landmarks=int(idx.numel()),
                  dim=int(feats.shape[1])),
    )


# ---------------------------------------------------------------------------
# 3. the association itself
# ---------------------------------------------------------------------------

@torch.no_grad()
def landmark_matches(
    query: torch.Tensor,
    landmarks: LandmarkSet,
    encoder: SuperPoint,
    pca: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    detect_num: int = 2048,
    nms: int = 4,
    topk: int = 1,
    mnn: bool = False,
    thr: float = 0.0,
    ratio: float = 0.0,
    chunk: int = 32768,
) -> Dict[str, object]:
    """Query pixels -> Gaussian centres, with no pose and no rendering.

    Args:
        query: (H, W) luma or (H, W, 3) RGB in [0, 1] -- the full-resolution
            query image.
        landmarks: from :func:`build_landmarks`.
        pca: the map's ``(loc_feat_pca_mean, loc_feat_pca_basis)``, which the
            query descriptors MUST be projected with when the field was reduced.
        topk: landmarks kept per query keypoint (STDLoc's config: 1, one-way
            nearest neighbour -- they do NOT use mutual NN for the sparse
            stage, and the geometric verification does the rejecting).
        mnn: use mutual-NN instead of the one-way top-k.
        thr: cosine floor (STDLoc: 0).
        ratio: Lowe ratio test on the query side, mutual-NN only.

    Returns:
        ``obj`` (M, 3) float64 world points (Gaussian means), ``img`` (M, 2)
        float64 query pixel CENTRES -- byte-compatible with what
        :meth:`usgloc.correction.EvidentialMatchCorrector.correct` assembles --
        plus ``score`` (M,), ``lm`` (M,) landmark row indices, ``n_keypoints``.
    """
    desc, scores = encoder.dense_descriptors(query)
    H, W = scores.shape[-2:]
    kp = locfeat.detect_keypoints(scores[0], detect_num, nms)
    if kp.numel() == 0:
        return dict(obj=np.zeros((0, 3)), img=np.zeros((0, 2)),
                    score=np.zeros(0), lm=np.zeros(0, dtype=np.int64),
                    n_keypoints=0)
    qd = locfeat.sample_query_descriptors(desc, kp, H, W, pca=pca)   # (Q, D)
    ref = landmarks.feat
    if mnn:
        qi, gi, sc = locfeat.mnn_match(qd, ref, chunk=chunk, thr=thr, ratio=ratio)
    else:
        # STDLoc's topk_match: for every query keypoint keep its `topk` best
        # landmarks.  No mutuality -- with ~1e4-1e5 near-duplicate landmark
        # descriptors the mutual test rejects most CORRECT matches.
        # Chunked over the landmarks so a large set (sample='all' is ~2.5e5)
        # does not materialise a Q x M score matrix (2048 x 2.5e5 = 2 GB).
        kq = int(min(max(topk, 1), ref.shape[0]))
        val = idx = None
        for s in range(0, ref.shape[0], max(chunk, 1)):
            v, j = torch.topk(qd @ ref[s:s + chunk].T,
                              min(kq, ref[s:s + chunk].shape[0]), dim=1)
            j = j + s
            if val is None:
                val, idx = v, j
            else:
                v = torch.cat([val, v], dim=1)
                j = torch.cat([idx, j], dim=1)
                val, sel = torch.topk(v, kq, dim=1)
                idx = torch.gather(j, 1, sel)
        q_ar = torch.arange(qd.shape[0], device=qd.device)[:, None].expand_as(idx)
        keep = val > thr
        qi, gi, sc = q_ar[keep], idx[keep], val[keep]
    obj = landmarks.xyz[gi].double().cpu().numpy()
    img = (kp[qi].double() + 0.5).cpu().numpy()
    return dict(obj=np.ascontiguousarray(obj, dtype=np.float64),
                img=np.ascontiguousarray(img, dtype=np.float64),
                score=sc.float().cpu().numpy(),
                lm=gi.cpu().numpy().astype(np.int64),
                n_keypoints=int(kp.shape[0]))


# ---------------------------------------------------------------------------
# 4. the corrector: same contract as EvidentialMatchCorrector
# ---------------------------------------------------------------------------

def _reproj_err(obj, img, rvec, tvec, K):
    import cv2

    p, _ = cv2.projectPoints(obj,
                             np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1),
                             np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1),
                             K, None)
    return np.linalg.norm(p.reshape(-1, 2) - img, axis=1)


def _irls(obj, img, K, rvec, tvec, loss="huber", sigma=1.0, iters=10, pw=None):
    """Iteratively-reweighted Gauss-Newton on (rvec, tvec).

    Same parameterisation and weights as
    :meth:`usgloc.correction.EvidentialMatchCorrector._irls` (kept separate so
    the render-matching path is not touched by this module); ``pw`` is the
    optional per-correspondence PRIOR weight -- here the landmark's own
    evidential weight ``exp(-gamma * u_i)``, a map parameter rather than a
    rendered pixel.
    """
    import cv2

    s = max(float(sigma), 1e-6)
    rvec = np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1).copy()
    tvec = np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1).copy()
    for _ in range(int(iters)):
        proj, J = cv2.projectPoints(obj, rvec, tvec, K, None)
        r = proj.reshape(-1, 2) - img
        e = np.linalg.norm(r, axis=1)
        if loss == "huber":
            w = np.where(e <= s, 1.0, s / np.maximum(e, 1e-12))
        elif loss == "cauchy":
            w = 1.0 / (1.0 + (e / s) ** 2)
        else:
            w = np.ones_like(e)
        if pw is not None:
            w = w * pw
        sw = np.sqrt(w).repeat(2)[:, None]
        A = np.asarray(J[:, :6], dtype=np.float64) * sw
        b = -r.reshape(-1, 1) * sw
        Hm = A.T @ A
        g = A.T @ b
        Hm = Hm + 1e-9 * np.trace(Hm) / 6.0 * np.eye(6)
        try:
            dx = np.linalg.solve(Hm, g).reshape(6)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break
        rvec = rvec + dx[:3].reshape(3, 1)
        tvec = tvec + dx[3:].reshape(3, 1)
        if np.linalg.norm(dx) < 1e-12:
            break
    return rvec, tvec


class LandmarkCorrector:
    """Absolute pose from pose-free landmark correspondences + PnP-RANSAC.

    Drop-in for :class:`usgloc.correction.EvidentialMatchCorrector`: the
    constructor takes ``(emap, renderer, camera, ...)`` and ``correct(query,
    T0)`` returns a :class:`usgloc.correction.CorrectionResult`.  ``renderer``
    and ``rounds`` are accepted and IGNORED -- nothing is rendered and there is
    nothing to iterate, which is the point: the returned pose does not depend on
    ``T0`` in any way (verifiable: the result is bit-identical for any ``T0``),
    so it is an initialiser, not a correction.

    Defaults follow STDLoc's sparse-stage config (``configs/stdloc_7scenes.yaml``):
    2048 keypoints, NMS 4, one-way nearest neighbour, threshold 0, poselib
    LO-RANSAC with a 12 px reprojection threshold and a 1e5 iteration budget.
    """

    def __init__(
        self,
        emap,
        renderer=None,                # ignored (no rendering)
        camera: Optional[Camera] = None,
        rounds: int = 1,              # ignored (single pass)
        # ---- landmark set -------------------------------------------------
        landmark_score: Optional[str] = None,   # match-score sidecar .pt
        landmark_num: int = 16384,
        landmark_k: int = 32,
        landmark_sample: str = "mo",
        landmark_seed: int = 0,
        landmark_min_obs: float = 1.0,
        landmark_score_min: float = -2.0,
        landmark_det_min: float = 0.0,
        landmark_vac: str = "none",   # none | gate | score | both
        landmark_u_max: float = 1.0,
        # ---- association ---------------------------------------------------
        landmark_detect: int = 2048,
        landmark_nms: int = 4,
        landmark_topk: int = 1,
        landmark_mnn: bool = False,
        landmark_thr: float = 0.0,
        landmark_ratio: float = 0.0,
        landmark_cull: float = 0.0,   # >0: before matching, keep only the
                                      # landmarks that project inside the image
                                      # (scaled by this margin, 1.0 = exactly the
                                      # image) at the START pose.  This USES T0 --
                                      # the result is then a correction with a
                                      # basin, not a prior-free pose -- but it
                                      # still renders nothing, and it removes the
                                      # distractors the query cannot see.
        # ---- geometry ------------------------------------------------------
        solver: str = "poselib",
        reproj_thresh_px: float = 12.0,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.0,
        ransac_iters: int = 100000,
        pl_min_iters: int = 1000,
        pl_loss: str = "CAUCHY",
        pl_loss_scale: float = 1.0,
        pl_bundle_iters: int = 100,
        refine: str = "none",         # none | lm | huber | cauchy | lm+huber | lm+cauchy
        refine_sigma: float = 1.0,
        refine_iters: int = 10,
        gamma: float = 3.0,           # evidential weight sharpness (Eq. 9)
        wvac: float = 0.0,            # >0: weight the IRLS residual of each
                                      # correspondence by exp(-gamma*u) ** wvac
    ) -> None:
        if camera is None:
            raise ValueError("LandmarkCorrector needs a camera")
        if emap.loc_feats is None:
            raise ValueError(
                "LandmarkCorrector needs a per-Gaussian descriptor field: "
                "emap.loc_feats is None (build it with "
                "scripts/analysis/train_locfeat.py, attach it with "
                "usgloc.locfeat.attach_loc_feats)")
        self.emap = emap
        self.camera = camera
        self.gamma = float(gamma)
        self.wvac = float(wvac)
        self.solver = solver
        self.reproj_thresh_px = float(reproj_thresh_px)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ransac_iters = int(ransac_iters)
        self.pl_min_iters = int(pl_min_iters)
        self.pl_loss = str(pl_loss)
        self.pl_loss_scale = float(pl_loss_scale)
        self.pl_bundle_iters = int(pl_bundle_iters)
        self.refine = str(refine)
        self.refine_sigma = float(refine_sigma)
        self.refine_iters = int(refine_iters)
        self.detect = int(landmark_detect)
        self.nms = int(landmark_nms)
        self.topk = int(landmark_topk)
        self.mnn = bool(landmark_mnn)
        self.thr = float(landmark_thr)
        self.ratio = float(landmark_ratio)
        self.cull = float(landmark_cull)
        dev = emap.means.device
        score = det = None
        self.score_path = landmark_score
        if landmark_score is not None:
            sc = torch.load(str(landmark_score), map_location="cpu",
                            weights_only=False)
            n = emap.means.shape[0]
            if int(sc.get("lm_score_n", sc["lm_score"].shape[0])) != n:
                raise ValueError(
                    f"{landmark_score}: match-score field has "
                    f"{sc['lm_score'].shape[0]} rows for a map of {n} gaussians")
            score = sc["lm_score"].to(dev).float()
            det = sc.get("lm_det")
            det = None if det is None else det.to(dev).float()
        self.landmarks = build_landmarks(
            emap, score=score, det=det, num=landmark_num, k=landmark_k,
            sample=landmark_sample, seed=landmark_seed,
            min_obs=landmark_min_obs, score_min=landmark_score_min,
            det_min=landmark_det_min, vac=landmark_vac, u_max=landmark_u_max,
            vac_gamma=gamma)
        self._sp = SuperPoint().to(dev).eval()
        self._pca = None
        if emap.loc_feat_pca_basis is not None:
            self._pca = (emap.loc_feat_pca_mean.to(dev),
                         emap.loc_feat_pca_basis.to(dev))
        self._K = camera.K(device="cpu", dtype=torch.float64).numpy()
        self._pl_cam = {"model": "PINHOLE", "width": camera.width,
                        "height": camera.height,
                        "params": [self._K[0, 0], self._K[1, 1],
                                   self._K[0, 2], self._K[1, 2]]}
        # Evidential weight of every landmark, exp(-gamma * u): the paper's own
        # pixel weight (Eq. 9) evaluated on the MAP PARAMETER instead of on an
        # alpha-blended vacuity image.
        self._lw = np.ascontiguousarray(
            torch.exp(-self.gamma * self.landmarks.vac.clamp(0.0, 1.0).double()
                      ).cpu().numpy(), dtype=np.float64)

    # ------------------------------------------------------------------
    def _visible(self, T_cw: torch.Tensor):
        """The landmarks whose centres project inside the image at ``T_cw``.

        Returns ``(LandmarkSet, rows)``; ``rows`` indexes the FULL landmark
        array, so per-landmark quantities (the vacuity weight) can be looked up.
        """
        cam, ls = self.camera, self.landmarks
        T = T_cw.to(ls.xyz.device, torch.float32)
        p = ls.xyz @ T[:3, :3].T + T[:3, 3]
        z = p[:, 2]
        u = cam.fx * p[:, 0] / z.clamp_min(1e-6) + cam.cx
        v = cam.fy * p[:, 1] / z.clamp_min(1e-6) + cam.cy
        mx = 0.5 * (self.cull - 1.0) * cam.width
        my = 0.5 * (self.cull - 1.0) * cam.height
        keep = ((z > 0.05) & (u >= -mx) & (u < cam.width + mx)
                & (v >= -my) & (v < cam.height + my))
        rows = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        return LandmarkSet(idx=ls.idx[rows], xyz=ls.xyz[rows], feat=ls.feat[rows],
                           vac=ls.vac[rows], score=ls.score[rows],
                           info=dict(ls.info, culled=int(rows.numel()))), rows

    @torch.no_grad()
    def match(self, query: torch.Tensor, T0: Optional[torch.Tensor] = None) -> dict:
        """The (object points, image points) the solver consumes.

        ``T0`` is used only when ``landmark_cull`` > 0, to restrict the landmark
        set to what that pose can see.
        """
        ls, rows = self.landmarks, None
        if self.cull > 0 and T0 is not None:
            ls, rows = self._visible(T0)
            if ls.idx.numel() == 0:
                return dict(obj=np.zeros((0, 3)), img=np.zeros((0, 2)),
                            score=np.zeros(0), lm=np.zeros(0, dtype=np.int64),
                            n_keypoints=0, n_landmarks=0)
        m = landmark_matches(
            query, ls, self._sp, pca=self._pca,
            detect_num=self.detect, nms=self.nms, topk=self.topk,
            mnn=self.mnn, thr=self.thr, ratio=self.ratio)
        m["n_landmarks"] = int(ls.idx.numel())
        if rows is not None and len(m["lm"]):
            # `lm` indexes the culled set; re-express it in the full landmark
            # array so per-landmark quantities (the vacuity weight) line up.
            m["lm"] = rows.cpu().numpy()[m["lm"]]
        return m

    @torch.no_grad()
    def correct(self, query: torch.Tensor, T0: torch.Tensor, matches=None):
        """``query``: (H, W) luma or (H, W, 3) RGB in [0, 1] at ``camera``
        resolution.  ``T0`` is used ONLY as the fall-back pose of a failed
        solve and for the output dtype/device -- never by the estimator.
        ``matches``: an already-computed :meth:`match` result (the association
        is deterministic, so passing it back only saves the recomputation)."""
        import cv2

        from .correction import CorrectionResult

        m = self.match(query, T0) if matches is None else matches
        obj, img = m["obj"], m["img"]
        if len(obj) < max(4, self.min_inliers):
            return CorrectionResult(T=T0, success=False, inliers=0,
                                    n_matches=int(len(obj)),
                                    reproj_px=float("inf"), rounds=1)
        pw = self._lw[m["lm"]] if self.wvac > 0 else None
        if self.solver == "poselib":
            import poselib

            pose, info = poselib.estimate_absolute_pose(
                img, obj, self._pl_cam,
                {"max_iterations": self.ransac_iters,
                 "min_iterations": self.pl_min_iters,
                 "max_reproj_error": self.reproj_thresh_px,
                 "success_prob": 0.99999},
                {"loss_type": self.pl_loss, "loss_scale": self.pl_loss_scale,
                 "max_iterations": self.pl_bundle_iters, "verbose": False})
            n_inl = int(info["num_inliers"])
            if n_inl < self.min_inliers:
                return CorrectionResult(T=T0, success=False, inliers=n_inl,
                                        n_matches=int(len(obj)),
                                        reproj_px=float("inf"), rounds=1)
            inl = np.where(np.asarray(info["inliers"]))[0]
            rvec, _ = cv2.Rodrigues(np.asarray(pose.R, dtype=np.float64))
            tvec = np.asarray(pose.t, dtype=np.float64).reshape(3, 1)
        else:
            flag = (cv2.SOLVEPNP_EPNP if self.solver == "cv2_epnp"
                    else cv2.SOLVEPNP_ITERATIVE)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, self._K, None,
                reprojectionError=self.reproj_thresh_px,
                iterationsCount=self.ransac_iters, confidence=0.99999,
                flags=flag)
            n_inl = 0 if inl is None else int(len(inl))
            if not ok or n_inl < self.min_inliers:
                return CorrectionResult(T=T0, success=False, inliers=n_inl,
                                        n_matches=int(len(obj)),
                                        reproj_px=float("inf"), rounds=1)
            inl = np.asarray(inl).reshape(-1)
        rvec = np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1).copy()
        tvec = np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1).copy()
        if self.refine in ("lm", "lm+huber", "lm+cauchy"):
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self._K, None,
                                              rvec, tvec)
        if self.refine in ("huber", "cauchy", "lm+huber", "lm+cauchy"):
            loss = "cauchy" if self.refine.endswith("cauchy") else "huber"
            rvec, tvec = _irls(obj[inl], img[inl], self._K, rvec, tvec, loss=loss,
                               sigma=self.refine_sigma, iters=self.refine_iters,
                               pw=None if pw is None else pw[inl] ** self.wvac)
        rvec = np.ascontiguousarray(np.asarray(rvec, np.float64)).reshape(3, 1)
        tvec = np.ascontiguousarray(np.asarray(tvec, np.float64)).reshape(3, 1)
        e = _reproj_err(obj[inl], img[inl], rvec, tvec, self._K)
        R, _ = cv2.Rodrigues(rvec)
        T = torch.eye(4, dtype=T0.dtype, device=T0.device)
        T[:3, :3] = torch.as_tensor(R, dtype=T0.dtype, device=T0.device)
        T[:3, 3] = torch.as_tensor(tvec.reshape(3), dtype=T0.dtype, device=T0.device)
        res = CorrectionResult(
            T=T, success=True, inliers=int(len(inl)), n_matches=int(len(obj)),
            reproj_px=float(np.sqrt(np.mean(e ** 2))), rounds=1,
            obj=np.asarray(obj[inl], dtype=np.float64),
            img=np.asarray(img[inl], dtype=np.float64))
        if (self.min_inlier_ratio > 0
                and res.inliers < self.min_inlier_ratio * max(res.n_matches, 1)):
            return CorrectionResult(T=T0, success=False, inliers=res.inliers,
                                    n_matches=res.n_matches,
                                    reproj_px=res.reproj_px, rounds=1)
        return res
