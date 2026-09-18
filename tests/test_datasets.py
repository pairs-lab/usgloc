"""Tests for the 7Scenes harness (usgloc.datasets) on a fake dataset tree."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from usgloc import se3, synthetic
from usgloc.datasets import (
    KINECT_FX,
    MAP_REQUIRED_KEYS,
    SevenScenes,
    load_gaussian_map,
)
from usgloc.primitives import EvidentialMap

REPO = Path(__file__).resolve().parents[1]


def _random_pose(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    xi = torch.cat([0.5 * torch.randn(3, generator=g), 0.3 * torch.randn(3, generator=g)])
    return se3.se3_exp(xi)


def _write_color_png(path: Path) -> None:
    """A tiny 8x6 PNG via PIL if available, else a zero-byte placeholder."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        path.touch()
        return
    arr = np.zeros((6, 8, 3), dtype=np.uint8)
    arr[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 30  # horizontal ramp
    arr[..., 1] = 128
    Image.fromarray(arr).save(path)


@pytest.fixture
def fake_root(tmp_path):
    """Fake 7Scenes tree: scene 'office', 2 sequences x 3 frames."""
    scene_dir = tmp_path / "office"
    poses = {}
    for s, seq in ((1, "seq-01"), (2, "seq-02")):
        seq_dir = scene_dir / seq
        seq_dir.mkdir(parents=True)
        for f in range(3):
            T_c2w = _random_pose(seed=100 * s + f)
            poses[(seq, f)] = T_c2w
            lines = [" ".join(f"{v:.9f}" for v in row) for row in T_c2w.tolist()]
            (seq_dir / f"frame-{f:06d}.pose.txt").write_text("\n".join(lines) + "\n")
            _write_color_png(seq_dir / f"frame-{f:06d}.color.png")
            (seq_dir / f"frame-{f:06d}.depth.png").touch()
    (scene_dir / "TrainSplit.txt").write_text("sequence1\n")
    (scene_dir / "TestSplit.txt").write_text("sequence2\n")
    return tmp_path, poses


def test_split_parsing_and_len(fake_root):
    root, _ = fake_root
    train = SevenScenes(root, "office", split="train")
    test = SevenScenes(root, "office", split="test")
    assert len(train) == 3 and len(test) == 3
    assert all(r.seq == "seq-01" for r in train.frames)
    assert all(r.seq == "seq-02" for r in test.frames)
    assert [r.frame_id for r in test.frames] == [0, 1, 2]
    rec = test[1]
    assert rec.rgb_path.name == "frame-000001.color.png"
    assert rec.depth_path.name == "frame-000001.depth.png"


def test_bad_split_and_missing_scene(fake_root):
    root, _ = fake_root
    with pytest.raises(ValueError):
        SevenScenes(root, "office", split="validation")
    with pytest.raises(FileNotFoundError):
        SevenScenes(root, "kitchen", split="test")


def test_pose_roundtrip(fake_root):
    root, poses = fake_root
    ds = SevenScenes(root, "office", split="test")
    eye = torch.eye(4)
    for rec in ds.frames:
        assert torch.allclose(rec.pose_c2w, poses[(rec.seq, rec.frame_id)], atol=1e-6)
        # pose_cw is camera-from-world: composing with the stored camera-to-world
        # matrix must give the identity.
        assert torch.allclose(rec.pose_cw @ rec.pose_c2w, eye, atol=1e-5)


def test_load_image(fake_root):
    pytest.importorskip("PIL")
    root, _ = fake_root
    ds = SevenScenes(root, "office", split="train")
    img = ds.load_image(0)
    assert img.shape == (6, 8, 3)
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    # Known fixture content: green channel constant 128/255, red ramp.
    assert torch.allclose(img[..., 1], torch.full((6, 8), 128 / 255))
    assert float(img[0, 7, 0]) == pytest.approx(7 * 30 / 255)


def test_camera_intrinsic_scaling():
    cam = SevenScenes.camera()
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (585.0, 585.0, 320.0, 240.0)
    assert (cam.width, cam.height) == (640, 480)
    half = SevenScenes.camera(width=320, height=240)
    assert half.fx == pytest.approx(KINECT_FX / 2)
    assert half.fy == pytest.approx(585.0 / 2)
    assert half.cx == pytest.approx(160.0)
    assert half.cy == pytest.approx(120.0)
    assert (half.width, half.height) == (320, 240)


def _scene_to_map_dict(scene):
    return {
        "means": scene.means,
        "covs": scene.covs,
        "colors": scene.colors,
        "opacities": scene.opacities,
        "clip_feats": scene.clip_feats,
        "beliefs": scene.beliefs,
        "vacuities": scene.vacuities,
        "instance_ids": scene.instance_ids,
    }


def test_load_gaussian_map_lists_missing_keys(tmp_path):
    scene = synthetic.make_scene(num_objects=2, splats_per_object=6, seed=0)
    d = _scene_to_map_dict(scene)
    del d["beliefs"], d["vacuities"]
    path = tmp_path / "incomplete.pt"
    torch.save(d, path)
    with pytest.raises(ValueError) as ei:
        load_gaussian_map(path)
    msg = str(ei.value)
    assert "beliefs" in msg and "vacuities" in msg
    # Keys that are present must not be reported as missing.
    assert "missing required keys: beliefs, vacuities" in msg


def test_load_gaussian_map_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gaussian_map(tmp_path / "nope.pt")


def test_load_gaussian_map_builds_evidential_map(tmp_path):
    scene = synthetic.make_scene(num_objects=3, splats_per_object=8, seed=1)
    d = _scene_to_map_dict(scene)
    assert set(MAP_REQUIRED_KEYS) <= set(d)
    path = tmp_path / "map.pt"
    torch.save(d, path)
    emap = load_gaussian_map(path)
    assert isinstance(emap, EvidentialMap)
    n = scene.means.shape[0]
    assert emap.means.shape == (n, 3)
    # instance_ids drive the clustering: floor + 3 objects -> 4 primitives.
    assert len(emap.primitives) == int(scene.instance_ids.max()) + 1
    assert emap.gaussian_vacuity is not None
    assert emap.gaussian_vacuity.shape == (n,)
    assert torch.isfinite(emap.gaussian_vacuity).all()


def test_run_7scenes_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_7scenes.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--root", "--map", "--anchor-stride", "--temperature-split", "--encoders"):
        assert flag in proc.stdout


# --------------------------------------------------------------------------
# SfM pseudo-GT mode (Brachmann et al., ICCV'21 pose files)
# --------------------------------------------------------------------------

SFM_FOCAL = 526.22


@pytest.fixture
def sfm_root(fake_root, tmp_path):
    """Adds pgt-style SfM pose files next to the fake 7Scenes tree.

    Covers train seq-01 fully; test seq-02 misses frame 1 (unregistered).
    """
    root, _ = fake_root
    pgt_dir = tmp_path / "pgt" / "sfm" / "7scenes"
    pgt_dir.mkdir(parents=True)
    g = torch.Generator().manual_seed(7)
    cw_poses = {}
    for split, seq, frame_ids in (("train", "seq-01", [0, 1, 2]),
                                  ("test", "seq-02", [0, 2])):
        lines = []
        for f in frame_ids:
            q = torch.randn(4, generator=g, dtype=torch.float64)
            q = q / torch.linalg.norm(q)
            t = torch.randn(3, generator=g, dtype=torch.float64)
            T_cw = torch.eye(4, dtype=torch.float64)
            T_cw[:3, :3] = se3.quat_to_rotmat(q)
            T_cw[:3, 3] = t
            cw_poses[(seq, f)] = T_cw.to(torch.float32)
            vals = " ".join(f"{v:.9f}" for v in (*q.tolist(), *t.tolist()))
            lines.append(f"{seq}/frame-{f:06d}.color.png {vals} {SFM_FOCAL}")
        (pgt_dir / f"office_{split}.txt").write_text("\n".join(lines) + "\n")
    return root, pgt_dir, cw_poses


def test_sfm_poses_override_dslam(sfm_root):
    root, pgt_dir, cw_poses = sfm_root
    test = SevenScenes(root, "office", split="test", sfm_pose_dir=pgt_dir)
    # frame 1 is not in the SfM file -> dropped and counted.
    assert [r.frame_id for r in test.frames] == [0, 2]
    assert test.n_unregistered == 1
    assert test.focal == pytest.approx(SFM_FOCAL)
    for rec in test.frames:
        # pose_cw must equal the world-to-camera pose written to the file.
        assert torch.allclose(rec.pose_cw, cw_poses[(rec.seq, rec.frame_id)],
                              atol=1e-5)
    train = SevenScenes(root, "office", split="train", sfm_pose_dir=pgt_dir)
    assert len(train) == 3 and train.n_unregistered == 0


def test_sfm_scene_camera_uses_pgt_focal(sfm_root):
    root, pgt_dir, _ = sfm_root
    ds = SevenScenes(root, "office", split="test", sfm_pose_dir=pgt_dir)
    cam = ds.scene_camera(320, 240)
    assert cam.fx == pytest.approx(SFM_FOCAL * 0.5)
    assert cam.fy == pytest.approx(SFM_FOCAL * 0.5)
    assert (cam.cx, cam.cy) == (160.0, 120.0)
    # Without SfM mode, scene_camera falls back to the Kinect constants.
    plain = SevenScenes(root, "office", split="test")
    assert plain.scene_camera(640, 480).fx == KINECT_FX


def test_sfm_missing_or_malformed_file(sfm_root):
    root, pgt_dir, _ = sfm_root
    with pytest.raises(FileNotFoundError):
        SevenScenes(root, "office", split="test", sfm_pose_dir=pgt_dir / "nope")
    bad = pgt_dir / "office_test.txt"
    bad.write_text("seq-02/frame-000000.color.png 1 0 0 0 0 0 0\n")  # 8 fields
    with pytest.raises(ValueError):
        SevenScenes(root, "office", split="test", sfm_pose_dir=pgt_dir)


# --------------------------------------------------------------------------
# 12Scenes layout (TwelveScenes)
# --------------------------------------------------------------------------

@pytest.fixture
def twelve_root(tmp_path):
    """Fake 12Scenes tree: apt1/kitchen, 4 frames (first 2 = test)."""
    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image

    from usgloc.datasets import TwelveScenes  # noqa: F401  (import check)

    scene = tmp_path / "apt1" / "kitchen"
    data = scene / "data"
    data.mkdir(parents=True)
    (scene / "split.txt").write_text("sequence0 [frames=2]\nsequence1 [frames=2]\n")
    (scene / "info.txt").write_text(
        "m_colorWidth = 128\nm_colorHeight = 96\n"
        "m_depthWidth = 64\nm_depthHeight = 48\n"
        "m_calibrationColorIntrinsic = 120 0 64 0 0 120 48 0 0 0 1 0 0 0 0 1\n"
        "m_calibrationDepthIntrinsic = 60 0 32 0 0 60 24 0 0 0 1 0 0 0 0 1\n"
    )
    poses = {}
    for f in range(4):
        T = se3.se3_exp(0.1 * torch.randn(6, generator=torch.Generator().manual_seed(f)))
        poses[f] = T
        lines = [" ".join(f"{v:.9f}" for v in row) for row in T.tolist()]
        (data / f"frame-{f:06d}.pose.txt").write_text("\n".join(lines) + "\n")
        Image.fromarray(np.full((96, 128, 3), 100, dtype=np.uint8)).save(
            data / f"frame-{f:06d}.color.jpg")
        Image.fromarray(np.full((48, 64), 1500, dtype=np.uint16)).save(
            data / f"frame-{f:06d}.depth.png")
    return tmp_path, poses


def test_twelvescenes_split_and_intrinsics(twelve_root):
    from usgloc.datasets import TwelveScenes

    root, poses = twelve_root
    test = TwelveScenes(root, "apt1/kitchen", split="test")
    train = TwelveScenes(root, "apt1/kitchen", split="train")
    assert [r.frame_id for r in test.frames] == [0, 1]
    assert [r.frame_id for r in train.frames] == [2, 3]
    assert torch.allclose(test[0].pose_c2w, poses[0], atol=1e-6)
    assert test.color_width == 128 and test.color_height == 96
    cam = test.scene_camera()  # native color intrinsics from info.txt
    assert (cam.fx, cam.cx, cam.cy) == (120.0, 64.0, 48.0)
    dcam = test.depth_camera()
    assert (dcam.fx, dcam.width, dcam.height) == (60.0, 64, 48)
    img = test.load_image(0)
    assert img.shape == (96, 128, 3)


def test_twelvescenes_sfm_mode(twelve_root):
    from usgloc.datasets import TwelveScenes

    root, _ = twelve_root
    pgt = root / "pgt12"
    pgt.mkdir()
    g = torch.Generator().manual_seed(9)
    lines = []
    cw = {}
    for f in (0,):  # frame 1 left unregistered
        q = torch.randn(4, generator=g, dtype=torch.float64)
        t = torch.randn(3, generator=g, dtype=torch.float64)
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3] = se3.quat_to_rotmat(q / q.norm())
        T[:3, 3] = t
        cw[f] = T.to(torch.float32)
        vals = " ".join(f"{v:.9f}" for v in (*(q / q.norm()).tolist(), *t.tolist()))
        lines.append(f"data/frame-{f:06d}.color.jpg {vals} 118.5")
    (pgt / "apt1_kitchen_test.txt").write_text("\n".join(lines) + "\n")
    ds = TwelveScenes(root, "apt1/kitchen", split="test", sfm_pose_dir=pgt)
    assert len(ds) == 1 and ds.n_unregistered == 1
    assert ds.focal == pytest.approx(118.5)
    assert torch.allclose(ds[0].pose_cw, cw[0], atol=1e-5)
    cam = ds.scene_camera(64, 48)  # half the native 128x96
    assert cam.fx == pytest.approx(118.5 * 0.5)
