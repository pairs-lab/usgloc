"""CPU tests for the pure-torch parts of usgloc.mapping (no gsplat/FastSAM)."""

from pathlib import Path

import pytest
import torch

from usgloc import mapping
from usgloc.datasets import SevenScenes
from usgloc.evidential import EvidentialHead


def _write_depth_png(path: Path, depth_mm: torch.Tensor) -> None:
    import numpy as np
    from PIL import Image

    Image.fromarray(depth_mm.numpy().astype(np.uint16)).save(path)


def test_read_depth_invalid_handling(tmp_path):
    pytest.importorskip("PIL")
    d = torch.tensor([[1500, 0], [mapping.DEPTH_INVALID, 2500]], dtype=torch.int64)
    p = tmp_path / "d.png"
    _write_depth_png(p, d)
    depth = mapping.read_depth(p)
    assert depth[0, 0] == pytest.approx(1.5)
    assert depth[1, 1] == pytest.approx(2.5)
    assert torch.isnan(depth[0, 1]) and torch.isnan(depth[1, 0])


@pytest.fixture
def depth_scene(tmp_path):
    """One-sequence fake scene with constant-depth frames at identity pose."""
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    scene = tmp_path / "room"
    seq = scene / "seq-01"
    seq.mkdir(parents=True)
    H, W = 48, 64
    for f in range(2):
        T = torch.eye(4)
        lines = [" ".join(f"{v:.6f}" for v in row) for row in T.tolist()]
        (seq / f"frame-{f:06d}.pose.txt").write_text("\n".join(lines) + "\n")
        rgb = np.full((H, W, 3), 128, dtype=np.uint8)
        Image.fromarray(rgb).save(seq / f"frame-{f:06d}.color.png")
        _write_depth_png(seq / f"frame-{f:06d}.depth.png",
                         torch.full((H, W), 2000, dtype=torch.int64))
    (scene / "TrainSplit.txt").write_text("sequence1\n")
    (scene / "TestSplit.txt").write_text("sequence1\n")
    return tmp_path


def test_backproject_init_points(depth_scene):
    ds = SevenScenes(depth_scene, "room", split="train")
    cam = mapping.Camera(fx=60.0, fy=60.0, cx=32.0, cy=24.0, width=64, height=48)
    pts, cols = mapping.backproject_init_points(
        ds, cam, frame_stride=1, pixel_stride=4, voxel=0.01, max_points=10_000
    )
    assert pts.shape[0] > 50 and pts.shape == cols.shape[:1] + (3,)
    # Identity pose + constant 2 m depth: all points on the z=2 plane.
    assert torch.allclose(pts[:, 2], torch.full_like(pts[:, 2], 2.0), atol=1e-4)
    # Principal-point pixel maps to x = y = 0.
    assert float(pts[:, 0].abs().min()) < 0.2
    assert cols.min() >= 0.0 and cols.max() <= 1.0
    # max_points cap respected.
    few, _ = mapping.backproject_init_points(
        ds, cam, frame_stride=1, pixel_stride=4, voxel=0.01, max_points=7
    )
    assert few.shape[0] == 7


def test_pseudo_labels_margin_filter():
    protos = torch.eye(4)
    feats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],       # confident class 0
        [0.7, 0.695, 0.0, 0.0],     # ambiguous 0 vs 1 -> filtered
    ])
    feats = feats / feats.norm(dim=-1, keepdim=True)
    labels, keep = mapping.pseudo_labels(feats, protos, margin=0.05)
    assert labels.tolist() == [0, 0]
    assert keep.tolist() == [True, False]


def test_gaussian_opinions_observation_scaling():
    torch.manual_seed(0)
    head = EvidentialHead(in_dim=8, num_classes=5)
    feats = torch.randn(6, 8)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    obs = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0, 100.0])
    beliefs, vac = mapping.gaussian_opinions(head, feats, obs, full_evidence_obs=4.0)
    assert beliefs.shape == (6, 5) and vac.shape == (6,)
    # Zero observations -> zero evidence -> vacuity exactly 1.
    assert vac[0] == pytest.approx(1.0)
    assert torch.allclose(beliefs[0], torch.zeros(5))
    # More observations of the same feature never increase vacuity.
    same = feats[0].expand(4, 8)
    _, v_same = mapping.gaussian_opinions(
        head, same, torch.tensor([1.0, 2.0, 4.0, 8.0]), full_evidence_obs=4.0
    )
    assert v_same[0] >= v_same[1] >= v_same[2]
    assert v_same[2] == pytest.approx(float(v_same[3]))  # saturated at n >= 4
    # Opinion consistency: u + sum(b) == 1.
    assert torch.allclose(vac + beliefs.sum(-1), torch.ones(6), atol=1e-5)


def test_assign_instance_ids_groups_by_voxel_and_class():
    means = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0],   # same voxel as above
        [2.0, 0.0, 0.0],   # different voxel
    ])
    cls = torch.tensor([3, 3, 3])
    ids = mapping.assign_instance_ids(means, cls, voxel=0.5)
    assert ids[0] == ids[1] and ids[0] != ids[2]
    # Same voxel, different class -> different instance.
    ids2 = mapping.assign_instance_ids(means, torch.tensor([3, 4, 3]), voxel=0.5)
    assert ids2[0] != ids2[1]
    # Ids are contiguous 0..K-1 (torch.unique inverse).
    assert set(ids2.tolist()) == set(range(int(ids2.max()) + 1))


def test_train_edl_head_learns_and_separates():
    torch.manual_seed(1)
    C, d = 4, 16
    protos = torch.linalg.qr(torch.randn(d, C)).Q.T  # orthonormal class directions
    labels = torch.randint(0, C, (400,))
    feats = protos[labels] + 0.05 * torch.randn(400, d)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    head = mapping.train_edl_head(feats, labels, num_classes=C, epochs=60,
                                  device="cpu", verbose=False)
    with torch.no_grad():
        e_in = head(feats)
        acc = (e_in.argmax(-1) == labels).float().mean()
        # Ambiguous features: 50/50 mixtures of two class prototypes -- the
        # operating regime the vacuity gate relies on (Sec. IV-B).
        mix = protos[0] + protos[1]
        mix = (mix / mix.norm()).expand(100, d) + 0.05 * torch.randn(100, d)
        e_mix = head(mix / mix.norm(dim=-1, keepdim=True))
    assert float(acc) > 0.9
    _, b_in, u_in = mapping.evidence_to_subjective_opinion(e_in)
    _, b_mix, u_mix = mapping.evidence_to_subjective_opinion(e_mix)
    # Ambiguity earns more vacuity and much less belief concentration.
    assert float(u_mix.mean()) > float(u_in.mean())
    assert float(b_in.max(-1).values.mean()) > 1.3 * float(b_mix.max(-1).values.mean())


def test_build_head_training_set_covers_both_domains():
    protos = torch.eye(3)
    mask_feats = torch.tensor([[1.0, 0.0, 0.0], [0.577, 0.577, 0.577]])  # 2nd ambiguous
    gauss_feats = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.6, 0.52],  # ambiguous
    ])
    obs = torch.tensor([1.0, 3.0, 2.0])
    feats, labels = mapping.build_head_training_set(
        mask_feats, gauss_feats, obs, protos, margin=0.05
    )
    # Confident mask feat (class 0) + two confident gaussian feats (1, 2).
    assert feats.shape[0] == 3
    assert sorted(labels.tolist()) == [0, 1, 2]
    # Unobserved gaussians are excluded even if confident.
    feats2, _ = mapping.build_head_training_set(
        mask_feats, gauss_feats, torch.zeros(3), protos, margin=0.05
    )
    assert feats2.shape[0] == 1
    # max_gauss caps the gaussian-domain contribution.
    many = torch.eye(3).repeat(100, 1)
    feats3, _ = mapping.build_head_training_set(
        mask_feats, many, torch.ones(300), protos, margin=0.05, max_gauss=10
    )
    assert feats3.shape[0] == 1 + 10


def test_make_submaps_frustum_culling():
    from usgloc.primitives import build_evidential_map, make_submaps

    # Three single-splat instances: ahead, behind, and off to the side.
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [5.0, 0.0, 2.0]])
    n = means.shape[0]
    emap = build_evidential_map(
        means=means, covs=torch.eye(3).expand(n, 3, 3) * 0.01,
        colors=torch.full((n, 3), 0.5), opacities=torch.ones(n),
        clip_feats=torch.nn.functional.normalize(torch.randn(n, 8), dim=-1),
        beliefs=torch.full((n, 4), 0.2), vacuities=torch.full((n,), 0.2),
        instance_ids=torch.arange(n),
    )
    anchor = torch.eye(4)  # camera at origin looking down +z
    cam = mapping.Camera(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width=64, height=48)
    make_submaps(emap, [anchor], radius=10.0)
    assert emap.submaps[0].primitive_idx.numel() == 3  # radius-only: all kept
    make_submaps(emap, [anchor], radius=10.0, camera=cam)
    kept = emap.submaps[0].primitive_idx
    # Only the primitive ahead survives: behind fails z>near, side fails bounds.
    assert kept.numel() == 1
    assert torch.allclose(emap.primitives[int(kept[0])].mu, means[0])
