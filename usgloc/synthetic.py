"""Synthetic semi-dynamic room scenes for tests and the end-to-end demo.

Builds a room of Gaussian-splat "objects" on a floor, simulates evidential
semantic observations (static objects agree across views; dynamic objects
yield conflicting opinions, as a moved object does between mapping passes),
constructs the evidential map, and renders semi-dynamic queries in which the
dynamic objects have been displaced after mapping -- the protocol of the
draft's robustness experiments, in miniature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .evidential import evidence_to_subjective_opinion
from .pipeline import QueryInstance
from .primitives import EvidentialMap, build_evidential_map, make_submaps
from .rendering import Camera


@dataclass
class SyntheticScene:
    means: torch.Tensor          # (N, 3)
    covs: torch.Tensor           # (N, 3, 3)
    colors: torch.Tensor         # (N, 3)
    opacities: torch.Tensor      # (N,)
    instance_ids: torch.Tensor   # (N,) object id per splat (0 = floor)
    classes: torch.Tensor        # (N,) semantic class per splat
    is_dynamic: torch.Tensor     # (N,) bool, True if the object moves after mapping
    clip_feats: torch.Tensor     # (N, d)
    beliefs: torch.Tensor        # (N, C)
    vacuities: torch.Tensor      # (N,)
    class_embeddings: torch.Tensor  # (C, d) ground-truth semantic directions
    num_classes: int
    dynamic_offsets: torch.Tensor   # (num_objects+1, 3) displacement applied at query time

    def moved_means(self) -> torch.Tensor:
        """Splat centres at query time (dynamic objects displaced)."""
        return self.means + self.dynamic_offsets[self.instance_ids]


def look_at_pose(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """OpenCV-convention camera-from-world pose looking from eye to target."""
    up = torch.tensor([0.0, 0.0, 1.0], dtype=eye.dtype)
    z = target - eye
    z = z / z.norm()
    x = torch.cross(z, up, dim=0)
    x = x / x.norm().clamp_min(1e-8)
    y = torch.cross(z, x, dim=0)
    R_wc = torch.stack([x, y, z], dim=1)  # columns = camera axes in world
    R = R_wc.T
    t = -R @ eye
    T = torch.eye(4, dtype=eye.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_scene(
    num_objects: int = 8,
    dynamic_fraction: float = 0.25,
    splats_per_object: int = 25,
    clip_dim: int = 32,
    num_classes: int = 10,
    evidence_strength: float = 12.0,
    seed: int = 0,
) -> SyntheticScene:
    g = torch.Generator().manual_seed(seed)

    # Floor: a grid of large flat splats (instance 0, class 0 = "structure").
    grid = torch.linspace(-2.5, 2.5, 8)
    fx, fy = torch.meshgrid(grid, grid, indexing="ij")
    floor_means = torch.stack([fx.flatten(), fy.flatten(), torch.zeros(64)], dim=-1)
    floor_cov = torch.diag(torch.tensor([0.30, 0.30, 0.01])) ** 2
    floor_covs = floor_cov.expand(64, 3, 3).clone()
    floor_colors = 0.35 + 0.10 * torch.rand(64, 3, generator=g)
    n_floor = 64

    # Objects: blobs of splats at random positions, one class + color each.
    means, covs, colors, inst, cls = [floor_means], [floor_covs], [floor_colors], \
        [torch.zeros(n_floor, dtype=torch.long)], [torch.zeros(n_floor, dtype=torch.long)]
    palette = torch.rand(num_objects, 3, generator=g) * 0.7 + 0.25
    for k in range(num_objects):
        center = torch.cat([
            (torch.rand(2, generator=g) - 0.5) * 3.6,
            0.25 + 0.45 * torch.rand(1, generator=g),
        ])
        pts = center + 0.18 * torch.randn(splats_per_object, 3, generator=g)
        means.append(pts)
        covs.append((torch.eye(3) * 0.06**2).expand(splats_per_object, 3, 3).clone())
        colors.append(palette[k].expand(splats_per_object, 3).clone()
                      + 0.03 * torch.randn(splats_per_object, 3, generator=g))
        inst.append(torch.full((splats_per_object,), k + 1, dtype=torch.long))
        cls.append(torch.full((splats_per_object,), 1 + k % (num_classes - 1), dtype=torch.long))

    means = torch.cat(means)
    covs = torch.cat(covs)
    colors = torch.cat(colors).clamp(0.0, 1.0)
    instance_ids = torch.cat(inst)
    classes = torch.cat(cls)
    opacities = torch.full((means.shape[0],), 0.9)

    # Dynamic objects: the last ceil(fraction * num_objects) ones move at query time.
    n_dyn = max(1, int(round(dynamic_fraction * num_objects)))
    dyn_ids = torch.arange(num_objects - n_dyn + 1, num_objects + 1)
    is_dynamic = torch.isin(instance_ids, dyn_ids)
    dynamic_offsets = torch.zeros(num_objects + 1, 3)
    for d in dyn_ids:
        off = torch.cat([(torch.rand(2, generator=g) - 0.5) * 1.6, torch.zeros(1)])
        dynamic_offsets[d] = off

    # Ground-truth semantic directions (well separated random unit vectors).
    class_emb = torch.randn(num_classes, clip_dim, generator=g)
    class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)

    clip_feats = class_emb[classes] + 0.05 * torch.randn(means.shape[0], clip_dim, generator=g)
    clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)

    # Simulated evidential observations.  Static splats: confident evidence on
    # the true class.  Dynamic splats: half of the cluster observes a
    # different class (the content seen at that location changed between
    # passes), so cluster-level DST fusion meets persistent conflict.
    evidence = torch.rand(means.shape[0], num_classes, generator=g) * 0.3
    confident = evidence_strength * (0.8 + 0.4 * torch.rand(means.shape[0], generator=g))
    for i in range(means.shape[0]):
        if is_dynamic[i] and i % 2 == 0:
            wrong = (int(classes[i]) + 3) % num_classes
            evidence[i, wrong] += confident[i]
        else:
            evidence[i, classes[i]] += confident[i]
    _, beliefs, vacuities = evidence_to_subjective_opinion(evidence)

    return SyntheticScene(
        means=means, covs=covs, colors=colors, opacities=opacities,
        instance_ids=instance_ids, classes=classes, is_dynamic=is_dynamic,
        clip_feats=clip_feats, beliefs=beliefs, vacuities=vacuities,
        class_embeddings=class_emb, num_classes=num_classes,
        dynamic_offsets=dynamic_offsets,
    )


def default_camera(width: int = 80, height: int = 60) -> Camera:
    f = 0.9 * width
    return Camera(fx=f, fy=f, cx=width / 2, cy=height / 2, width=width, height=height)


def trajectory_poses(
    n: int, radius: float = 2.6, height: float = 1.6, dtype=torch.float32
) -> List[torch.Tensor]:
    """Anchor poses on a circle, looking at the room centre."""
    poses = []
    target = torch.tensor([0.0, 0.0, 0.4])
    for k in range(n):
        a = 2 * torch.pi * k / n
        eye = torch.tensor([radius * torch.cos(torch.tensor(a)),
                            radius * torch.sin(torch.tensor(a)), height])
        poses.append(look_at_pose(eye, target).to(dtype))
    return poses


def build_map_from_scene(
    scene: SyntheticScene,
    n_anchors: int = 18,
    fuse_rule: str = "yager",
    submap_radius: float = 6.0,
) -> EvidentialMap:
    """Evidential map + submaps from the synthetic scene (mapping-time state).

    ``yager`` is the default here: Dempster's rule (Eq. 5) renormalizes the
    conflict of contradictory observations away, leaving dynamic objects with
    *low* vacuity; Yager's variant routes conflict into vacuity, which is the
    behaviour the refinement weighting needs.  ``EvidentialPrimitive`` records
    the conflict separately either way (see ``effective_vacuity``).

    Submap descriptors are made distinguishable not by content (the radius
    may cover the whole room) but by expressing primitive geometry in the
    anchor camera frame (see ``pipeline.submap_reference_inputs``), so
    retrieval is a viewpoint-matching problem as in SGLoc.
    """
    emap = build_evidential_map(
        means=scene.means, covs=scene.covs, colors=scene.colors,
        opacities=scene.opacities, clip_feats=scene.clip_feats,
        beliefs=scene.beliefs, vacuities=scene.vacuities,
        instance_ids=scene.instance_ids, fuse_rule=fuse_rule,
    )
    make_submaps(emap, trajectory_poses(n_anchors), radius=submap_radius)
    return emap


def query_map(scene: SyntheticScene) -> EvidentialMap:
    """A rendering-only map at query-time geometry (dynamic objects moved).

    Used to synthesize query images; the localizer itself only ever sees the
    mapping-time :func:`build_map_from_scene` result.
    """
    return EvidentialMap(
        primitives=[], means=scene.moved_means(), covs=scene.covs,
        colors=scene.colors, opacities=scene.opacities,
        gaussian_vacuity=torch.zeros_like(scene.opacities),
    )


@torch.no_grad()
def render_query_image(
    scene: SyntheticScene, renderer, camera: Camera, T_cw: torch.Tensor
) -> torch.Tensor:
    qm = query_map(scene)
    out = renderer.render(qm.means, qm.covs, qm.colors, qm.opacities, T_cw, camera)
    return out["rgb"].clamp(0.0, 1.0)


@torch.no_grad()
def make_query_instances(
    scene: SyntheticScene,
    T_cw: torch.Tensor,
    camera: Camera,
    noise: float = 0.05,
    seed: int = 0,
    at_map_time: bool = False,
) -> List[QueryInstance]:
    """Simulate the FastSAM+CLIP+EDL front end.

    Each visible object yields one instance with a noisy class embedding and
    an evidential opinion.  At query time (default) dynamic objects sit at
    their displaced positions and get weaker evidence (their context changed),
    hence higher query-side vacuity; with ``at_map_time=True`` the mapping-time
    geometry and full evidence are used (for retrieval training pairs).
    """
    g = torch.Generator().manual_seed(seed)
    instances: List[QueryInstance] = []
    moved = scene.means if at_map_time else scene.moved_means()
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    num_objects = int(scene.instance_ids.max())
    for obj in range(0, num_objects + 1):
        sel = scene.instance_ids == obj
        center = moved[sel].mean(dim=0)
        p = R @ center + t
        if p[2] < 0.2:
            continue
        u = camera.fx * p[0] / p[2] + camera.cx
        v = camera.fy * p[1] / p[2] + camera.cy
        if not (0 <= u < camera.width and 0 <= v < camera.height):
            continue
        cls = int(scene.classes[sel][0])
        dyn = bool(scene.is_dynamic[sel][0]) and not at_map_time
        f = scene.class_embeddings[cls] + noise * torch.randn(
            scene.class_embeddings.shape[1], generator=g
        )
        f = f / f.norm()
        strength = 4.0 if dyn else 12.0
        evidence = torch.rand(scene.num_classes, generator=g) * 0.3
        evidence[cls] += strength
        _, b, vac = evidence_to_subjective_opinion(evidence[None])
        extent = scene.means[sel].std(dim=0).mean()
        area = (camera.fx * extent / p[2]) ** 2 / (camera.width * camera.height)
        instances.append(
            QueryInstance(
                clip_feat=f,
                centroid=torch.stack([u / camera.width, v / camera.height]),
                area=area.reshape(1).clamp(1e-4, 1.0),
                belief=b[0],
                vacuity=vac[0],
            )
        )
    return instances


def train_retrieval(
    desc_net,
    aggregator,
    scene: SyntheticScene,
    emap: EvidentialMap,
    camera: Camera,
    iters: int = 300,
    lr: float = 1e-3,
    seed: int = 0,
    variants_per_anchor: int = 4,
) -> float:
    """Train descriptor nets + aggregator with symmetric InfoNCE (Sec. IV-C).

    Positive pairs are (instances seen from a submap's anchor at mapping time,
    the submap's reference primitives).  Several noise variants of the query
    side are pre-generated per anchor and sampled per iteration so the
    encoders generalize over feature noise instead of memorizing one draw.
    Returns the final loss.
    """
    from .descriptors import info_nce_loss
    from .pipeline import submap_reference_inputs

    pairs = []
    for k, sm in enumerate(emap.submaps):
        variants = [
            make_query_instances(scene, sm.anchor_pose, camera,
                                 at_map_time=True, seed=seed + 1000 * v + k)
            for v in range(variants_per_anchor)
        ]
        variants = [qi for qi in variants if len(qi) >= 2]
        if variants and sm.primitive_idx.numel() >= 2:
            pairs.append((sm, variants))
    if len(pairs) < 2:
        raise RuntimeError("not enough trainable submap/query pairs")

    params = list(desc_net.parameters()) + list(aggregator.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    loss = torch.tensor(0.0)
    for _ in range(iters):
        opt.zero_grad()
        f2d, f3d = [], []
        for sm, variants in pairs:
            qi = variants[int(torch.randint(len(variants), (1,), generator=g))]
            clip = torch.stack([q.clip_feat for q in qi])
            cen = torch.stack([q.centroid for q in qi])
            area = torch.stack([q.area for q in qi])
            vac_q = torch.stack([q.vacuity for q in qi])
            d2 = desc_net.encode_query(clip, cen, area)
            f2d.append(aggregator(d2[None], vac_q[None])[0])

            c3, m3, S3, v3 = submap_reference_inputs(emap, sm)
            d3 = desc_net.encode_reference(c3, m3, S3)
            f3d.append(aggregator(d3[None], v3[None])[0])
        loss = info_nce_loss(torch.stack(f2d), torch.stack(f3d))
        loss.backward()
        opt.step()
    return float(loss.detach())
