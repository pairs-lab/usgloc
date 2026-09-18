"""Real-data map training: 3DGS fitting, CLIP distillation, EDL head training.

This module provides the offline stages that turn a posed RGB-D sequence
(7Scenes conventions via :class:`usgloc.datasets.SevenScenes`) into the
evidential map checkpoint that :func:`usgloc.datasets.load_gaussian_map`
ingests (Sec. IV-A of the draft):

1. :func:`backproject_init_points` -- seed Gaussians from the depth maps
   (7Scenes ships depth, so no SfM point cloud is needed).
2. :func:`fit_gaussians` -- photometric 3DGS fit with the installed ``gsplat``
   (``rasterization`` + ``DefaultStrategy`` densification), plain RGB colors.
3. :class:`MaskClipEncoder` + :func:`distill_clip_features` -- FastSAM masks
   with CLIP crop embeddings, accumulated onto the Gaussians each mask covers
   (visibility + depth-occlusion tested), yielding per-Gaussian open-vocabulary
   features and observation counts.
4. :func:`clip_text_prototypes` / :func:`pseudo_labels` /
   :func:`train_edl_head` -- the shared evidential head (Sec. IV-B: one head
   for map AND query) trained on mask embeddings with CLIP zero-shot
   pseudo-labels under the EDL loss (Eq. 7).
5. :func:`gaussian_opinions` -- head evidence scaled by observation count
   (few observations => little evidence => high vacuity), then Eq. 3.
6. :func:`assign_instance_ids` -- voxel x semantic-class quantization, the
   cheap instance segmentation ``cluster_gaussians`` accepts.

Heavy dependencies (gsplat, ultralytics, open_clip) are imported lazily so
the module imports on CPU-only test machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .datasets import SevenScenes
from .evidential import EvidentialHead, edl_loss, evidence_to_subjective_opinion
from .rendering import Camera

# ~25 indoor classes covering 7Scenes content (office, kitchen, stairs ...).
# C = len(INDOOR_VOCABULARY) is the evidential class count of the paper.
INDOOR_VOCABULARY: Tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "window",
    "chair", "sofa", "table", "desk", "shelf",
    "cabinet", "monitor", "keyboard", "mouse", "book",
    "cup", "bottle", "plant", "lamp", "stairs",
    "curtain", "box", "bag", "kitchen appliance", "sink",
)

DEPTH_INVALID = 65535  # 7Scenes sentinel for missing depth (uint16 mm)


# --------------------------------------------------------------------------
# Stage 1: initialization from depth
# --------------------------------------------------------------------------

def read_depth(path) -> torch.Tensor:
    """7Scenes ``frame-XXXXXX.depth.png`` -> (H, W) float32 metres, NaN invalid."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        d = torch.from_numpy(np.asarray(im, dtype=np.int64))
    depth = d.to(torch.float32) / 1000.0
    depth[(d == DEPTH_INVALID) | (d == 0)] = float("nan")
    return depth


def backproject_init_points(
    ds: SevenScenes,
    camera: Camera,
    frame_stride: int = 25,
    pixel_stride: int = 8,
    voxel: float = 0.02,
    max_points: int = 200_000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Back-project depth into a voxel-deduplicated world point cloud.

    Returns (P, 3) points and (P, 3) RGB colors in [0, 1].  Uses the dataset's
    ``pose_c2w`` (camera-to-world); with SfM pseudo-GT poses the ~1 cm
    DSLAM/SfM discrepancy is irrelevant for initialization -- the photometric
    fit corrects geometry.

    ``camera`` must be the DEPTH intrinsics (``ds.depth_camera()``); colors
    are sampled from the color image with shape-based rescaling, so datasets
    whose color resolution differs from depth (12Scenes: 1296x968 vs
    640x480) work unchanged.
    """
    pts, cols = [], []
    for i in range(0, len(ds), frame_stride):
        rec = ds[i]
        depth = read_depth(rec.depth_path)
        rgb = ds.load_image(i)
        H, W = depth.shape
        ys = torch.arange(0, H, pixel_stride)
        xs = torch.arange(0, W, pixel_stride)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        z = depth[gy, gx]
        ok = torch.isfinite(z) & (z > 0.2) & (z < 8.0)
        if not ok.any():
            continue
        gy, gx, z = gy[ok].float(), gx[ok].float(), z[ok]
        x = (gx - camera.cx) / camera.fx * z
        y = (gy - camera.cy) / camera.fy * z
        p_cam = torch.stack([x, y, z], dim=-1)
        T = rec.pose_c2w
        p_world = p_cam @ T[:3, :3].T + T[:3, 3]
        pts.append(p_world)
        ry = (gy * (rgb.shape[0] / H)).long().clamp(0, rgb.shape[0] - 1)
        rx = (gx * (rgb.shape[1] / W)).long().clamp(0, rgb.shape[1] - 1)
        cols.append(rgb[ry, rx])
    if not pts:
        raise ValueError("no valid depth pixels found for initialization")
    pts = torch.cat(pts)
    cols = torch.cat(cols)

    # Voxel deduplication: keep the first point per voxel.
    key = torch.floor(pts / voxel).to(torch.int64)
    _, inverse = torch.unique(key, dim=0, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    inv_sorted = inverse[order]
    is_first = torch.ones_like(inv_sorted, dtype=torch.bool)
    is_first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    keep = order[is_first]
    pts, cols = pts[keep], cols[keep]

    if pts.shape[0] > max_points:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(pts.shape[0], generator=g)[:max_points]
        pts, cols = pts[sel], cols[sel]
    return pts, cols


# --------------------------------------------------------------------------
# Stage 2: photometric 3DGS fit (gsplat)
# --------------------------------------------------------------------------

def _inverse_sigmoid(x: float) -> float:
    import math

    return math.log(x / (1.0 - x))


def fit_gaussians(
    ds: SevenScenes,
    camera: Camera,
    init_points: torch.Tensor,
    init_colors: torch.Tensor,
    iters: int = 7000,
    frame_stride: int = 10,
    device: str = "cuda",
    lr_means: float = 1.6e-4,
    opacity_init: float = 0.1,
    seed: int = 0,
    log_every: int = 500,
) -> Dict[str, torch.Tensor]:
    """Fit a plain-RGB 3DGS scene with gsplat's DefaultStrategy densification.

    Trains at the resolution of ``camera`` (pass e.g. ``ds.scene_camera(320,
    240)`` for speed) on every ``frame_stride``-th train frame, one random
    view per iteration, L1 photometric loss.

    Returns the raw-map tensors of the export contract: ``means`` (N, 3),
    ``covs`` (N, 3, 3) rebuilt as R diag(s^2) R^T, ``colors`` (N, 3) in
    [0, 1], ``opacities`` (N,) in [0, 1] -- all detached, on ``device``.
    """
    from gsplat import DefaultStrategy, rasterization

    dev = torch.device(device)
    P = init_points.shape[0]
    # Isotropic initial scale from the local point density (mean 8-NN spacing
    # estimated on a subsample, clamped to sane indoor bounds).
    sub = init_points[torch.randperm(P)[: min(P, 4000)]].to(dev)
    d2 = torch.cdist(sub, sub)
    d2.fill_diagonal_(float("inf"))
    knn = d2.topk(4, largest=False).values.mean().clamp(5e-3, 0.1)
    scene_scale = float((init_points.max(0).values - init_points.min(0).values).norm() / 2)

    g = torch.Generator(device="cpu").manual_seed(seed)
    quats0 = torch.randn(P, 4, generator=g)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(init_points.to(dev).clone()),
            "quats": torch.nn.Parameter(quats0.to(dev)),
            "scales": torch.nn.Parameter(torch.full((P, 3), float(knn.log()), device=dev)),
            "opacities": torch.nn.Parameter(
                torch.full((P,), _inverse_sigmoid(opacity_init), device=dev)
            ),
            "colors": torch.nn.Parameter(
                torch.logit(init_colors.to(dev).clamp(1e-3, 1 - 1e-3))
            ),
        }
    )
    lrs = {
        "means": lr_means * scene_scale,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": [v], "lr": lrs[k], "name": k}], eps=1e-15)
        for k, v in params.items()
    }
    strategy = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=min(iters // 2 + 500, 15_000),
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    frame_ids = list(range(0, len(ds), frame_stride))
    # Preload the training views once (resized to the training resolution).
    import torch.nn.functional as F

    views = []
    for i in frame_ids:
        img = ds.load_image(i).permute(2, 0, 1)[None]
        img = F.interpolate(
            img, size=(camera.height, camera.width), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).contiguous()
        views.append((img, ds[i].pose_cw))
    K = camera.K(device=dev)[None]

    for step in range(iters):
        img, T_cw = views[int(torch.randint(len(views), (1,), generator=g))]
        gt = img.to(dev, non_blocking=True)
        render, _, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.sigmoid(params["colors"]),
            viewmats=T_cw.to(dev)[None],
            Ks=K,
            width=camera.width,
            height=camera.height,
            packed=False,
        )
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss = (render[0] - gt).abs().mean()
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if log_every and step % log_every == 0:
            print(
                f"      fit step {step:5d}: L1 {float(loss):.4f}, "
                f"{params['means'].shape[0]} gaussians"
            )

    with torch.no_grad():
        quats = params["quats"] / params["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = quats.unbind(-1)
        R = torch.stack(
            [
                1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        S2 = torch.exp(params["scales"]) ** 2
        covs = R @ torch.diag_embed(S2) @ R.transpose(-1, -2)
        return {
            "means": params["means"].detach(),
            "covs": covs.detach(),
            "colors": torch.sigmoid(params["colors"]).detach(),
            "opacities": torch.sigmoid(params["opacities"]).detach(),
        }


# --------------------------------------------------------------------------
# Stage 3: FastSAM + CLIP mask features distilled onto Gaussians
# --------------------------------------------------------------------------

class MaskClipEncoder:
    """FastSAM instance masks + batched CLIP crop embeddings for one image.

    The same CLIP model/preprocess as :class:`usgloc.query.RealQueryEncoder`
    (ViT-B-16 laion2b), so distilled map features and query features live in
    the same embedding space -- required for the shared evidential head.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        clip_pretrained: str = "laion2b_s34b_b88k",
        fastsam_weights: str = "FastSAM-s.pt",
        device: str = "cuda",
        min_mask_area: float = 0.002,
    ):
        import open_clip
        from ultralytics import FastSAM

        self.device = device
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=device
        )
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.sam = FastSAM(fastsam_weights)
        self.min_mask_area = min_mask_area

    @torch.no_grad()
    def encode_frame(self, image_bgr) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """numpy HxWx3 uint8 BGR -> list of ((H, W) bool mask, (d,) CLIP feat)."""
        import numpy as np
        from PIL import Image

        results = self.sam(image_bgr, device=self.device, retina_masks=True, verbose=False)
        masks = results[0].masks
        if masks is None:
            return []
        crops, keep = [], []
        for m in masks.data:
            mask = m.bool().cpu()
            if float(mask.float().mean()) < self.min_mask_area:
                continue
            ys, xs = mask.nonzero(as_tuple=True)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            crop = image_bgr[y0:y1, x0:x1, ::-1]
            crops.append(self.preprocess(Image.fromarray(np.ascontiguousarray(crop))))
            keep.append(mask)
        if not crops:
            return []
        x = torch.stack(crops).to(self.device)
        f = self.clip.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return list(zip(keep, f.cpu()))

    @torch.no_grad()
    def text_prototypes(self, classes: Sequence[str]) -> torch.Tensor:
        """(C, d) l2-normalized CLIP text embeddings for zero-shot labels."""
        prompts = [f"a photo of a {c} in a room" for c in classes]
        tokens = self.tokenizer(prompts).to(self.device)
        t = self.clip.encode_text(tokens)
        return (t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)).cpu()


def distill_clip_features(
    ds: SevenScenes,
    camera: Camera,
    means: torch.Tensor,
    encoder: MaskClipEncoder,
    frame_stride: int = 20,
    occlusion_tol: float = 0.15,
    near: float = 0.05,
    log_every: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate mask CLIP features onto the Gaussians each mask covers.

    For every ``frame_stride``-th train frame: run FastSAM+CLIP, project all
    Gaussian centres with the frame's pose, keep those in front of the camera,
    inside the image and depth-consistent (|z - depth| < tol; frames with
    invalid depth at that pixel are accepted), and add each covering mask's
    feature to those Gaussians.

    Args:
        camera: intrinsics at the ORIGINAL image resolution (640 x 480) --
            masks are computed on full-resolution images.

    Returns:
        clip_feats: (N, d) l2-normalized accumulated features (unobserved
            Gaussians get the mean observed direction as a neutral stand-in;
            their zero observation count keeps their evidence at zero).
        obs_count: (N,) number of mask observations per Gaussian.
        mask_feats: (M, d) all mask features seen -- the EDL head train set.
    """
    dev = means.device
    N = means.shape[0]
    feat_dim: Optional[int] = None
    feat_sum: Optional[torch.Tensor] = None
    obs_count = torch.zeros(N, device=dev)
    bank: List[torch.Tensor] = []

    frame_ids = list(range(0, len(ds), frame_stride))
    for n_done, i in enumerate(frame_ids):
        rec = ds[i]
        img = ds.load_image(i)
        bgr = (img.numpy() * 255).astype("uint8")[:, :, ::-1]
        pairs = encoder.encode_frame(bgr)
        if not pairs:
            continue
        if feat_sum is None:
            feat_dim = pairs[0][1].shape[0]
            feat_sum = torch.zeros(N, feat_dim, device=dev)

        T = rec.pose_cw.to(dev)
        p_cam = means @ T[:3, :3].T + T[:3, 3]
        z = p_cam[:, 2]
        u = camera.fx * p_cam[:, 0] / z.clamp_min(1e-6) + camera.cx
        v = camera.fy * p_cam[:, 1] / z.clamp_min(1e-6) + camera.cy
        H, W = img.shape[0], img.shape[1]
        valid = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        depth = read_depth(rec.depth_path).to(dev)
        # Depth may be lower-resolution than color (12Scenes): rescale indices.
        di = (v * (depth.shape[0] / H)).clamp(0, depth.shape[0] - 1).long()
        dj = (u * (depth.shape[1] / W)).clamp(0, depth.shape[1] - 1).long()
        d_at = depth[di, dj]
        occl_ok = torch.isnan(d_at) | ((z - d_at).abs() < occlusion_tol)
        valid = valid & occl_ok
        ui = u.clamp(0, W - 1).long()
        vi = v.clamp(0, H - 1).long()

        vidx = valid.nonzero(as_tuple=False).squeeze(-1)
        if vidx.numel():
            uv_v, vv_v = ui[vidx], vi[vidx]
            for mask, feat in pairs:
                covered = mask.to(dev)[vv_v, uv_v]
                gi = vidx[covered]
                if gi.numel():
                    feat_sum[gi] += feat.to(dev)
                    obs_count[gi] += 1
        bank.extend(f for _, f in pairs)
        if log_every and n_done % log_every == 0:
            seen = int((obs_count > 0).sum())
            print(f"      distill frame {n_done + 1}/{len(frame_ids)}: "
                  f"{seen}/{N} gaussians observed, {len(bank)} mask feats")

    if feat_sum is None:
        raise ValueError("FastSAM produced no masks on any distillation frame")
    observed = obs_count > 0
    clip_feats = feat_sum
    mean_dir = clip_feats[observed].mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm().clamp_min(1e-8)
    clip_feats[~observed] = mean_dir
    clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return clip_feats, obs_count, torch.stack(bank)


# --------------------------------------------------------------------------
# Stage 4: the shared EDL head
# --------------------------------------------------------------------------

def pseudo_labels(
    feats: torch.Tensor, prototypes: torch.Tensor, margin: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CLIP zero-shot labels with a confidence filter.

    Returns (labels, keep): argmax-cosine class per feature, and a mask that
    keeps only features whose top1-top2 similarity margin exceeds ``margin``
    (ambiguous masks are excluded from head training; the head then assigns
    them little evidence at inference, which is exactly the desired vacuity).
    """
    sims = feats @ prototypes.T
    top2 = sims.topk(2, dim=-1).values
    keep = (top2[:, 0] - top2[:, 1]) > margin
    return sims.argmax(dim=-1), keep


def build_head_training_set(
    mask_feats: torch.Tensor,
    gauss_feats: torch.Tensor,
    gauss_obs: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float = 0.01,
    max_gauss: int = 20_000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Training set for the shared head covering BOTH of its input domains.

    The head is applied to raw mask-crop CLIP features (query side) and to
    per-Gaussian features averaged over many overlapping masks (map side).
    Averaging blurs features off the raw-mask manifold, and a head trained
    only on raw masks assigns the averages near-zero evidence (observed
    empirically: median map vacuity 0.997 despite 95% mask coverage), which
    collapses the vacuity spread the gate needs.  Training on the union of
    both domains -- each zero-shot pseudo-labeled with the same confidence
    margin -- keeps map and query vacuities on one scale, which is the whole
    point of sharing the head (Sec. IV-B).

    Returns (feats, labels) ready for :func:`train_edl_head`.
    """
    m_lab, m_keep = pseudo_labels(mask_feats, prototypes, margin=margin)
    observed = (gauss_obs > 0).nonzero(as_tuple=False).squeeze(-1)
    if observed.numel() > max_gauss:
        g = torch.Generator().manual_seed(seed)
        observed = observed[torch.randperm(observed.numel(), generator=g)[:max_gauss]]
    gf = gauss_feats[observed].cpu()
    g_lab, g_keep = pseudo_labels(gf, prototypes, margin=margin)
    feats = torch.cat([mask_feats[m_keep], gf[g_keep]])
    labels = torch.cat([m_lab[m_keep], g_lab[g_keep]])
    return feats, labels


def train_edl_head(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    hidden: int = 256,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    kl_ramp_epochs: int = 10,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> EvidentialHead:
    """Train the shared evidential head on (CLIP feature, pseudo-label) pairs.

    The Eq. 7 loss with the standard KL annealing ramp ``lambda_t = min(1,
    epoch / kl_ramp_epochs)``.  Train ONCE on mask features pooled over
    scenes and reuse the checkpoint everywhere: the head only sees CLIP space,
    which is scene-independent, and sharing it keeps map and query vacuities
    comparable (Sec. IV-B).

    Note: classification accuracy converges early, but the *uncertainty
    structure* (ambiguous features earning higher vacuity than clean ones)
    only emerges with longer training -- empirically absent at ~20 epochs and
    present at 40-60 on toy data -- hence the default of 40.
    """
    head = EvidentialHead(in_dim=feats.shape[-1], num_classes=num_classes,
                          hidden=hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    feats = feats.to(device)
    labels = labels.to(device)
    for epoch in range(epochs):
        perm = torch.randperm(feats.shape[0], generator=g).to(device)
        kl_w = min(1.0, epoch / max(1, kl_ramp_epochs))
        total, nb = 0.0, 0
        for s in range(0, perm.numel(), batch_size):
            idx = perm[s : s + batch_size]
            evidence = head(feats[idx])
            loss = edl_loss(evidence, labels[idx], kl_weight=kl_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        if verbose:
            print(f"      edl epoch {epoch + 1}/{epochs}: loss {total / max(nb, 1):.4f} "
                  f"(kl_w {kl_w:.2f})")
    return head.eval()


@torch.no_grad()
def gaussian_opinions(
    head: EvidentialHead,
    clip_feats: torch.Tensor,
    obs_count: torch.Tensor,
    full_evidence_obs: float = 4.0,
    batch_size: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-Gaussian (beliefs, vacuities) from the head, evidence scaled by
    observation count: ``e_i <- e_i * min(1, n_i / full_evidence_obs)``.

    Evidence accumulates with observations, so rarely-seen Gaussians are
    honestly vacuous (n = 0 gives u = 1 exactly), and cluster-level Yager
    fusion later turns any *disagreement* between members into extra vacuity.
    """
    dev = next(head.parameters()).device
    beliefs, vacuities = [], []
    scale = (obs_count / full_evidence_obs).clamp(0.0, 1.0)
    for s in range(0, clip_feats.shape[0], batch_size):
        f = clip_feats[s : s + batch_size].to(dev)
        e = head(f) * scale[s : s + batch_size, None].to(dev)
        _, b, u = evidence_to_subjective_opinion(e)
        beliefs.append(b.cpu())
        vacuities.append(u.cpu())
    return torch.cat(beliefs), torch.cat(vacuities)


# --------------------------------------------------------------------------
# Stage 5: instance ids + export
# --------------------------------------------------------------------------

def assign_instance_ids(
    means: torch.Tensor, class_ids: torch.Tensor, voxel: float = 0.4
) -> torch.Tensor:
    """Voxel x semantic-class quantization as a cheap instance segmentation.

    Groups Gaussians that are spatially close (same ``voxel`` cell) AND agree
    semantically -- the granularity ``cluster_gaussians`` expects when exact
    instance features are unavailable (its docstring: "any instance
    segmentation of the splats can be substituted").
    """
    key = torch.cat(
        [torch.floor(means / voxel).to(torch.int64), class_ids[:, None].to(torch.int64)],
        dim=-1,
    )
    _, ids = torch.unique(key, dim=0, return_inverse=True)
    return ids


def export_map(
    path,
    raw: Dict[str, torch.Tensor],
    clip_feats: torch.Tensor,
    beliefs: torch.Tensor,
    vacuities: torch.Tensor,
    instance_ids: torch.Tensor,
    obs_count: Optional[torch.Tensor] = None,
) -> None:
    """torch.save the dict :func:`usgloc.datasets.load_gaussian_map` ingests.

    ``obs_count`` is stored as an extra key (ignored by the loader) so a new
    evidential head can be re-applied later without re-running distillation.
    """
    ckpt = {
        "means": raw["means"].detach().cpu().float(),
        "covs": raw["covs"].detach().cpu().float(),
        "colors": raw["colors"].detach().cpu().float(),
        "opacities": raw["opacities"].detach().cpu().float(),
        "clip_feats": clip_feats.detach().cpu().float(),
        "beliefs": beliefs.detach().cpu().float(),
        "vacuities": vacuities.detach().cpu().float(),
        "instance_ids": instance_ids.detach().cpu().long(),
    }
    if obs_count is not None:
        ckpt["obs_count"] = obs_count.detach().cpu().float()
    torch.save(ckpt, path)
