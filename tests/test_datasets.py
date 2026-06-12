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
