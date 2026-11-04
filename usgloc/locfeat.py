"""Per-Gaussian LOCAL descriptor field (the feature-field association option).

Motivation
----------
The RGB-matching correction (:mod:`usgloc.correction`) renders the map at a
pose hypothesis, matches the query against the RENDERED IMAGE, and lifts the
render-side keypoint to 3D through the RENDERED DEPTH.  The 3D point is then a
read-out of an alpha-blended depth buffer, which is biased at silhouettes and
along grazing surfaces -- the "correction floor".

STDLoc (Zhai et al., CVPR'25) and its successors instead attach a learned local
descriptor to every Gaussian and match the query's dense descriptors against
that field, so a correspondence's 3D point is the Gaussian MEAN, i.e. a map
parameter.  This module builds exactly that field for a usgloc map.  The
matching mechanism is theirs and is cited as such; nothing here is claimed as
novel.  The evidential channel is untouched: ``EvidentialMap.loc_feats`` is an
additional per-Gaussian attribute that the vacuity gate, the SE(3) mixture
posterior and the Laplace covariance act on unchanged.

Pipeline
--------
1. :func:`init_loc_feats` -- project every Gaussian centre into the training
   views, occlusion-test against the RENDERED expected depth (never the sensor
   depth), and average the frozen SuperPoint descriptor sampled at the
   projection.
2. :func:`finetune_loc_feats` -- optimise ONLY the descriptors (geometry,
   colour, opacity frozen) so that the alpha-blended rendered descriptor map
   reproduces the per-view SuperPoint map, exactly STDLoc's feature loss.
3. :func:`pca_fit` / :func:`pca_apply` -- optional per-scene PCA to a smaller
   dimension.  The basis travels with the map because the QUERY descriptors
   must be projected with it too (:func:`project_query_map`).

Consuming the field
-------------------
:func:`detect_keypoints` + :func:`sample_query_descriptors` give the query-side
sparse descriptors; :func:`mnn_match` associates them with Gaussians (2D-3D,
pose-free) and :func:`dense_feature_match` associates them with a RENDERED
descriptor map (2D-2D, STDLoc's dense stage).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .rendering import Camera
from .superpoint import SUPERPOINT_DIM, SUPERPOINT_STRIDE, SuperPoint, sample_descriptors


#: Keys that carry the descriptor field inside a map checkpoint.  A SIDECAR
#: checkpoint (``scripts/analysis/train_locfeat.py --feats-only``) holds exactly these
#: plus ``loc_feat_base`` / ``loc_feat_n``, and costs ~2 * N * D bytes instead
#: of a full copy of the map's geometry.
LOC_FEAT_KEYS = ("loc_feats", "loc_feat_obs", "loc_feat_pca_mean",
                 "loc_feat_pca_basis", "loc_feat_dim", "loc_feat_pca_ev",
                 "loc_feat_source", "loc_feat_iters", "loc_feat_stride",
                 "loc_feat_holdout", "loc_feat_res")


def attach_loc_feats(emap, path, device=None):
    """Merge a sidecar descriptor-field checkpoint into an already-loaded map.

    Equivalent to having loaded a map whose ``.pt`` carried the ``loc_feats``
    keys, so a consumer can hold ONE copy of the geometry on disk while
    comparing several fields.

    Args:
        emap: an :class:`usgloc.primitives.EvidentialMap`, mutated in place.
        path: sidecar ``.pt`` from ``train_locfeat.py --feats-only``.
        device: optional device for the descriptor tensors.

    Returns:
        The sidecar's raw dict (its ``loc_feat_*`` metadata is useful to the
        caller, e.g. the held-out frame split).

    Raises:
        ValueError: if the sidecar was built for a different number of
            Gaussians than ``emap`` has.
    """
    import torch as _t
    sc = _t.load(str(path), map_location="cpu", weights_only=False)
    n = emap.means.shape[0]
    if int(sc.get("loc_feat_n", sc["loc_feats"].shape[0])) != n:
        raise ValueError(
            f"{path}: descriptor field has {sc['loc_feats'].shape[0]} rows for a "
            f"map of {n} gaussians (built against {sc.get('loc_feat_base')!r})")
    for k, attr in (("loc_feats", "loc_feats"), ("loc_feat_obs", "loc_feat_obs"),
                    ("loc_feat_pca_mean", "loc_feat_pca_mean"),
                    ("loc_feat_pca_basis", "loc_feat_pca_basis")):
        v = sc.get(k)
        if v is not None and device is not None:
            v = v.to(device)
        setattr(emap, attr, v)
    return sc


# ----------------------------------------------------------------------------
# query side: detection + descriptor sampling
# ----------------------------------------------------------------------------

def simple_nms(scores: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """SuperPoint's iterative max-pool non-maximum suppression on a (H, W) map."""
    if radius <= 0:
        return scores
    s = scores[None, None]

    def _pool(x):
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(s)
    max_mask = s == _pool(s)
    for _ in range(2):
        supp_mask = _pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, s)
        new_max_mask = supp_scores == _pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, s, zeros)[0, 0]


@torch.no_grad()
def detect_keypoints(scores: torch.Tensor, num: int = 2048, nms: int = 4
                     ) -> torch.Tensor:
    """(H, W) SuperPoint score map -> (M, 2) keypoint pixel INDICES (x, y).

    The frozen SuperPoint detector head stands in for STDLoc's separately
    trained landmark detector: same NMS radius (4) and budget (2048), no extra
    training.
    """
    H, W = scores.shape
    s = simple_nms(scores, nms).flatten()
    k = min(int(num), s.numel())
    val, idx = torch.topk(s, k)
    idx = idx[val > 0]
    return torch.stack([idx % W, idx // W], dim=-1)


@torch.no_grad()
def sample_query_descriptors(
    desc: torch.Tensor, kp: torch.Tensor, height: int, width: int,
    pca: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """(1, D_src, h, w) SuperPoint map + (M, 2) keypoints -> (M, D) descriptors.

    ``pca`` is the ``(mean, basis)`` pair stored on the map when its
    descriptors were reduced; it MUST be applied to the query too.
    """
    q = sample_descriptors(desc, kp, height, width)
    if pca is not None:
        q = pca_apply(q, pca[0], pca[1])
    return q


@torch.no_grad()
def project_query_map(desc: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor
                      ) -> torch.Tensor:
    """Apply the map's PCA to a dense (B, D_src, h, w) query descriptor map."""
    B, D, h, w = desc.shape
    flat = desc.permute(0, 2, 3, 1).reshape(-1, D)
    out = pca_apply(flat, mean, basis)
    return out.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------

@torch.no_grad()
def pca_fit(feats: torch.Tensor, dim: int, max_rows: int = 200_000, seed: int = 0
            ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Per-scene PCA of the (N, D_src) descriptor field.

    Returns ``(mean (D_src,), basis (D_src, dim), explained_variance)`` --
    tensors float32 on ``feats``' device.  Fitted on at most ``max_rows``
    randomly chosen rows (pass only OBSERVED rows; all-zero rows would bias
    the mean).
    """
    x = feats.float()
    if x.shape[0] > max_rows:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(x.shape[0], generator=g)[:max_rows].to(x.device)
        x = x[sel]
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    try:
        evals, evecs = torch.linalg.eigh(cov.double())
    except RuntimeError:
        # cuSOLVER cannot create its handle once fine-tuning a large map has
        # filled the card; the (D_src, D_src) problem is tiny on the host.
        evals, evecs = (t.to(cov.device) for t in torch.linalg.eigh(cov.double().cpu()))
    basis = evecs[:, -int(dim):].flip(-1).float()
    kept = float(evals[-int(dim):].sum() / evals.clamp_min(0).sum())
    return mean.float(), basis.contiguous(), kept


@torch.no_grad()
def pca_apply(feats: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor
              ) -> torch.Tensor:
    """Project (M, D_src) onto the PCA basis and re-L2-normalise the rows."""
    out = (feats.float() - mean.to(feats.device).float()) @ basis.to(feats.device).float()
    return F.normalize(out, p=2, dim=-1)


# ----------------------------------------------------------------------------
# matching
# ----------------------------------------------------------------------------

@torch.no_grad()
def mnn_match(
    query: torch.Tensor, ref: torch.Tensor, chunk: int = 32768,
    thr: float = -1.0, ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mutual-nearest-neighbour cosine matching, chunked over ``ref``.

    Args:
        query: (M, D) L2-normalised query descriptors.
        ref: (N, D) L2-normalised reference descriptors (the Gaussian field, or
            a flattened rendered descriptor map).  N may be ~1e6.
        thr: minimum cosine similarity.
        ratio: Lowe-style ratio test on the query side (0 = off): the match
            survives only when ``second_best < ratio * best`` over all of
            ``ref``, with the second-best tracked across chunks.

    Returns:
        ``(q_idx, r_idx, score)`` of the mutual matches.
    """
    M = query.shape[0]
    dev = query.device
    q = query.float()
    best_v = torch.full((M,), -2.0, device=dev)
    best_i = torch.zeros(M, dtype=torch.long, device=dev)
    second_v = torch.full((M,), -2.0, device=dev)
    ref_best_q = torch.empty(ref.shape[0], dtype=torch.long, device=dev)
    for s in range(0, ref.shape[0], chunk):
        r = ref[s:s + chunk].float()
        sim = q @ r.T                                    # (M, c)
        if sim.shape[1] >= 2:
            top2 = sim.topk(2, dim=1)
            v, i = top2.values[:, 0], top2.indices[:, 0]
            v2 = top2.values[:, 1]
        else:
            v, i = sim.max(dim=1)
            v2 = torch.full_like(v, -2.0)
        # running best / second-best over all chunks
        better = v > best_v
        second_v = torch.where(better, torch.maximum(best_v, v2),
                               torch.maximum(second_v, v))
        best_i = torch.where(better, i + s, best_i)
        best_v = torch.where(better, v, best_v)
        ref_best_q[s:s + chunk] = sim.argmax(dim=0)
    keep = best_v > thr
    if ratio > 0:
        keep &= second_v < ratio * best_v
    q_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    r_idx = best_i[q_idx]
    mutual = ref_best_q[r_idx] == q_idx
    q_idx, r_idx = q_idx[mutual], r_idx[mutual]
    return q_idx, r_idx, best_v[q_idx]


@torch.no_grad()
def dense_feature_match(
    query_map: torch.Tensor, render_map: torch.Tensor, thr: float = -1.0,
    dual_softmax_temp: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """STDLoc's coarse dense stage: MNN between two descriptor GRIDS.

    The full (h*w, h*w) score matrix is materialised, so this is for the COARSE
    (1/8) grid only: 80x60 -> 4800^2 scores (92 MB).  Full resolution is not
    representable.

    Args:
        query_map: (D, h, w) L2-normalised per pixel.
        render_map: (D, h, w) L2-normalised per pixel (the rasterised field).
        dual_softmax_temp: > 0 applies STDLoc's dual-softmax to the score
            matrix before the MNN (their default temperature is 0.1).

    Returns:
        ``(uv_q (m, 2), uv_r (m, 2), score (m,))`` -- pixel INDICES (x, y) on
        the grid the two maps live on.
    """
    D, h, w = query_map.shape
    a = query_map.reshape(D, -1).T
    b = render_map.reshape(D, -1).T
    sim = a.float() @ b.float().T                        # (hw, hw)
    if dual_softmax_temp > 0:
        s = sim / dual_softmax_temp
        sim = F.softmax(s, dim=0) * F.softmax(s, dim=1)
    # Mutual nearest neighbour BY INDEX, not by value equality.  Testing
    # `sim == sim.max(...)` accepts every tie, and a degenerate render ties
    # everywhere: when the hypothesis pose renders almost nothing the blended
    # descriptors are all zero, so every similarity is equal and the mask comes
    # back all-True.  That yielded hw^2 = 23M "matches" on Stairs and asked the
    # fine stage for a 351 GiB gather (4800^2 x win^2 x D).  argmax picks one
    # partner per cell, so the result is at most hw pairs and is identical to
    # the old code whenever the maxima are unique, i.e. on every non-degenerate
    # frame.
    hw = sim.shape[0]
    nn_r = sim.argmax(dim=1)                             # best render cell per query cell
    nn_q = sim.argmax(dim=0)                             # best query cell per render cell
    qi = torch.arange(hw, device=sim.device)
    keep = nn_q[nn_r] == qi
    if thr > -float("inf"):
        keep &= sim[qi, nn_r] > thr
    qi, ri = qi[keep], nn_r[keep]
    uv_q = torch.stack([qi % w, qi // w], dim=-1)
    uv_r = torch.stack([ri % w, ri // w], dim=-1)
    return uv_q, uv_r, sim[qi, ri]


@torch.no_grad()
def refine_match_windows(
    query_fine: torch.Tensor, render_fine: torch.Tensor,
    cq: torch.Tensor, cr: torch.Tensor, win: int = SUPERPOINT_STRIDE,
    temp: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """STDLoc's fine dense stage: localise coarse cell matches to a pixel.

    Args:
        query_fine / render_fine: (D, H, W) descriptor maps at FULL resolution,
            L2-normalised per pixel.
        cq / cr: (m,) flat indices into the coarse (H/win, W/win) grid, as
            returned by :func:`dense_feature_match` (``y * w + x``).
        temp: dual-softmax temperature inside the window (STDLoc: 0.1).

    Returns:
        ``(uv_q (m, 2), uv_r (m, 2), score (m,))`` full-resolution pixel
        INDICES (x, y).
    """
    D, H, W = query_fine.shape
    # `cq`/`cr` index the COARSE grid, which is the encoder's 1/8 grid, and the
    # unfold below strides by `win`.  The two only line up when win equals that
    # stride: any other value silently mis-indexes and faults inside the gather
    # (observed as an 'unspecified launch failure'), so the window size is not a
    # free parameter and cannot be used to widen the matcher's basin.
    if win != SUPERPOINT_STRIDE:
        raise ValueError(
            f"refine_match_windows: win must equal the encoder stride "
            f"{SUPERPOINT_STRIDE}, got {win}. The fine window is tied to the "
            f"coarse grid the match indices come from; widening the basin needs "
            f"a search over neighbouring CELLS, not a larger window.")
    w_c = W // win
    ww = win * win

    def _win(x, idx):
        u = F.unfold(x[None], (win, win), stride=win).reshape(1, D, ww, -1)
        return u[0, :, :, idx].permute(2, 1, 0)          # (m, ww, D)

    qw, rw = _win(query_fine, cq), _win(render_fine, cr)
    sim = qw.float() @ rw.float().transpose(-2, -1)      # (m, ww, ww)
    if temp > 0:
        s = sim / temp
        sim = F.softmax(s, dim=-2) * F.softmax(s, dim=-1)
    flat = sim.reshape(sim.shape[0], -1)
    val, arg = flat.max(dim=-1)
    i, j = arg // ww, arg % ww
    base_q = torch.stack([(cq % w_c) * win, (cq // w_c) * win], dim=-1)
    base_r = torch.stack([(cr % w_c) * win, (cr // w_c) * win], dim=-1)
    off_q = torch.stack([i % win, i // win], dim=-1)
    off_r = torch.stack([j % win, j // win], dim=-1)
    return base_q + off_q, base_r + off_r, val


# ----------------------------------------------------------------------------
# field construction
# ----------------------------------------------------------------------------

def _render_geom(renderer, emap, T_cw: torch.Tensor, camera: Camera):
    """rgb + expected depth + alpha at ``T_cw`` (no descriptor channels)."""
    return renderer.render(
        emap.means, emap.covs, emap.colors, emap.opacities, T_cw, camera,
        vacuity=emap.gaussian_vacuity, with_depth=True,
    )


@torch.no_grad()
def init_loc_feats(
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
    score_weight: bool = False,
    log_every: int = 100,
) -> Dict[str, torch.Tensor]:
    """Initialise the field by projection-averaging frozen SuperPoint features.

    Mirrors :func:`usgloc.mapping.distill_clip_features`' projection +
    occlusion loop, with two deliberate differences: the descriptor is sampled
    bilinearly from SuperPoint's dense 1/8 map instead of a mask, and the
    occlusion test runs against the map's own RENDERED expected depth, so no
    sensor-depth scale/noise question enters the field.

    A Gaussian is accepted in a view when it is inside the frustum, the
    rendered alpha at its projection is at least ``min_alpha``, and its camera
    depth agrees with the rendered expected depth to
    ``occl_abs + occl_rel * z``.

    Returns a dict with ``loc_feats`` (N, 256) float32 L2-normalised rows
    (zero for never-observed Gaussians), ``obs_count`` (N,) float32, and
    ``n_frames``.
    """
    dev = emap.means.device
    N = emap.means.shape[0]
    fsum = torch.zeros(N, SUPERPOINT_DIM, device=dev)
    obs = torch.zeros(N, device=dev)
    H, W = camera.height, camera.width
    for n, i in enumerate(frame_ids):
        img = ds.load_image(int(i)).to(dev)
        T = ds[int(i)].pose_cw.to(dev, torch.float32)
        out = _render_geom(renderer, emap, T, camera)
        depth, alpha = out["depth"], out["alpha"]
        desc, scores = encoder.dense_descriptors(img)

        p = emap.means @ T[:3, :3].T + T[:3, 3]
        z = p[:, 2]
        u = camera.fx * p[:, 0] / z.clamp_min(1e-6) + camera.cx
        v = camera.fy * p[:, 1] / z.clamp_min(1e-6) + camera.cy
        ok = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()
        d_at = depth[vi, ui]
        a_at = alpha[vi, ui]
        ok &= a_at >= min_alpha
        ok &= (z - d_at).abs() <= occl_abs + occl_rel * z
        idx = torch.nonzero(ok, as_tuple=False).squeeze(-1)
        if idx.numel():
            uv = torch.stack([u[idx], v[idx]], dim=-1)
            f = sample_descriptors(desc, uv, H, W)
            if score_weight:
                w = scores[0][vi[idx], ui[idx]].clamp_min(1e-6)
                fsum.index_add_(0, idx, f * w[:, None])
                obs.index_add_(0, idx, w)
            else:
                fsum.index_add_(0, idx, f)
                obs.index_add_(0, idx, torch.ones_like(idx, dtype=fsum.dtype))
        del out, desc, scores
        if log_every and n % log_every == 0:
            print(f"      locfeat init {n + 1}/{len(frame_ids)}: "
                  f"{int((obs > 0).sum())}/{N} gaussians observed", flush=True)
    feats = F.normalize(fsum, p=2, dim=-1)
    feats[obs == 0] = 0.0
    return {"loc_feats": feats, "obs_count": obs, "n_frames": len(frame_ids)}


def _gt_feature_map(encoder: SuperPoint, img: torch.Tensor, h: int, w: int
                    ) -> torch.Tensor:
    """(D, h, w) SuperPoint target: dense map bilinearly resized to the render
    resolution and re-L2-normalised (STDLoc's train.py convention)."""
    with torch.no_grad():
        desc, _ = encoder.dense_descriptors(img)
        tgt = F.interpolate(desc, size=(h, w), mode="bilinear", align_corners=False)
        return F.normalize(tgt, p=2, dim=1)[0]


class _LeanNormalize(torch.autograd.Function):
    """``F.normalize(x, dim=-1)`` whose backward saves only the row norms and
    writes the gradient into the incoming one in place, in row chunks.  At 2.3M
    Gaussians x 256 every (N, D) copy is 2.3 GB, and the autograd backward of
    F.normalize holds two of them on top of the parameter and its gradient."""

    _EPS, _CH = 1e-12, 65536

    @staticmethod
    def forward(ctx, x):
        nrm = x.norm(dim=-1, keepdim=True).clamp_min(_LeanNormalize._EPS)
        ctx.save_for_backward(x, nrm)
        return x / nrm

    @staticmethod
    def backward(ctx, g):
        x, nrm = ctx.saved_tensors
        g = g.contiguous()
        for a in range(0, x.shape[0], _LeanNormalize._CH):
            b = min(a + _LeanNormalize._CH, x.shape[0])
            n, d = x[a:b] / nrm[a:b], nrm[a:b]
            gc = g[a:b]
            # d(x/|x|) = (g - n (n.g)) / |x|; a clamped norm is a constant.
            proj = (n * gc).sum(dim=-1, keepdim=True) * (d > _LeanNormalize._EPS)
            gc.sub_(n * proj).div_(d)
        return g


def finetune_loc_feats(
    emap,
    ds,
    camera: Camera,
    encoder: SuperPoint,
    renderer,
    frame_ids: Sequence[int],
    init: torch.Tensor,
    iters: int = 3000,
    lr: float = 1e-3,
    lr_final: Optional[float] = None,
    min_alpha: float = 0.2,
    loss: str = "l1",
    beta1: float = 0.9,
    seed: int = 0,
    log_every: int = 200,
    cpu_opt_state: bool = False,
) -> Dict[str, torch.Tensor]:
    """Optimise ONLY the per-Gaussian descriptors against the SuperPoint maps.

    Geometry, colour, opacity and every evidential attribute are frozen (they
    are not even passed to an optimiser), so the Gaussian set, the vacuity
    field and therefore every existing result stay valid.

    Args:
        camera: the resolution the descriptor field is RENDERED at during
            training.  The SuperPoint target is computed on the full-resolution
            image and resized to it.
        init: (N, D) initial descriptors (from :func:`init_loc_feats`).
        loss: ``"l1"`` (STDLoc's) or ``"cos"`` (1 - cosine per pixel).
        beta1: Adam's momentum.  0.9 is STDLoc's (torch's) default, but the
            gradient here is EXTREMELY sparse -- one view per step touches a
            few percent of the Gaussians -- and with momentum a Gaussian keeps
            stepping for ~10 iterations after its last gradient, so
            descriptors drift while they are invisible.  ``beta1=0`` makes the
            numerator the current gradient, so a Gaussian with no gradient
            takes no step, while Adam's per-parameter scaling is kept.

    Returns a dict with ``loc_feats`` (N, D) float32 L2-normalised, and the
    loss trace under ``trace``.
    """
    if loss not in ("l1", "cos"):
        raise ValueError(f"loss must be l1 | cos, got {loss!r}")
    dev = emap.means.device
    h, w = camera.height, camera.width
    lf = torch.nn.Parameter(init.to(dev, torch.float32, copy=True))
    opt = torch.optim.Adam([lf], lr=lr, betas=(float(beta1), 0.999), eps=1e-15)
    # 12Scenes maps reach 1.04M Gaussians, where Adam's m and v cost 2.1 GB on
    # top of the 1 GB parameter and 1 GB gradient and the card runs out. Holding
    # the state on the host and applying the SAME update there is numerically
    # identical to torch.optim.Adam (no amsgrad, no weight decay), so the field
    # stays comparable with the 7Scenes ones.  The host step is torch's fused
    # CPU Adam kernel (the one behind Adam(fused=True)) on a pinned host copy of
    # the parameter: ~5x faster than chunked tensor ops at 2.3M Gaussians.
    _cpu = bool(cpu_opt_state)
    if _cpu:
        opt = None
        _p = torch.empty(lf.shape, dtype=torch.float32, pin_memory=True)
        _p.copy_(lf.detach())
        _g = torch.empty_like(_p, pin_memory=True)
        _m = torch.zeros_like(_p)
        _v = torch.zeros_like(_p)
        _step = torch.zeros((), dtype=torch.float32)
        _b1, _b2, _eps = float(beta1), 0.999, 1e-15

    def _cpu_adam(cur_lr):
        _g.copy_(lf.grad)
        _step.add_(1)
        torch._fused_adam_([_p], [_g], [_m], [_v], [], [_step], lr=float(cur_lr), beta1=_b1, beta2=_b2,
                           weight_decay=0.0, eps=_eps, amsgrad=False, maximize=False)
        lf.data.copy_(_p)
    sched = None
    _gamma = 1.0
    if lr_final is not None and iters > 1:
        _gamma = (lr_final / lr) ** (1.0 / (iters - 1))
        if not _cpu:                      # no torch optimiser to attach to
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=_gamma)
    g = torch.Generator().manual_seed(seed)
    ids = list(int(i) for i in frame_ids)
    trace: List[float] = []
    for it in range(iters):
        # The SuperPoint target is recomputed per step rather than cached: a
        # (256, 480, 640) half target is 157 MB, so a cache over ~1e3 views
        # does not fit anywhere, and the encoder pass is <10 ms against a
        # 256-channel rasterisation.
        i = ids[int(torch.randint(len(ids), (1,), generator=g))]
        img = ds.load_image(i).to(dev)
        tgt = _gt_feature_map(encoder, img, h, w)
        T = ds[i].pose_cw.to(dev, torch.float32)
        out = renderer.render(
            emap.means, emap.covs, emap.colors, emap.opacities, T, camera,
            vacuity=emap.gaussian_vacuity,
            loc_feats=_LeanNormalize.apply(lf) if _cpu else F.normalize(lf, p=2, dim=-1),
            feat_normalize=True,
        )
        fm = out["feats"].permute(2, 0, 1)                 # (D, h, w)
        m = (out["alpha"] >= min_alpha)[None]
        if loss == "l1":
            per = (fm - tgt).abs().mean(dim=0, keepdim=True)
        else:
            per = 1.0 - (fm * tgt).sum(dim=0, keepdim=True)
        denom = m.sum().clamp_min(1)
        L = (per * m).sum() / denom
        if _cpu:
            L.backward()
            _cpu_adam(lr * (_gamma ** it))
            lf.grad = None                # do not carry 1 (N, D) copy into the next forward
        else:
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
        if sched is not None:
            sched.step()
        trace.append(float(L.detach()))
        del out, fm
        if log_every and it % log_every == 0:
            print(f"      locfeat ft {it}/{iters}  loss {trace[-1]:.5f}  "
                  f"lr {(lr * (_gamma ** it)) if _cpu else opt.param_groups[0]['lr']:.2e}", flush=True)
    with torch.no_grad():
        final = F.normalize(lf.detach(), p=2, dim=-1)
    return {"loc_feats": final, "trace": trace}
