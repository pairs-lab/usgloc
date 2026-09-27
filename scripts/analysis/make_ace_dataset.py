#!/usr/bin/env python3
"""Build the ACE/marepo dataset layout from our 7Scenes copy + SfM pgt poses.

ACE and marepo both read  <root>/<split>/{rgb,poses,calibration}/  with one
file per frame and matching sort order. We symlink the RGB (no copying of
21 GB) and write the pgt pose + per-scene SfM focal length, so any baseline
built on that loader is evaluated on exactly the ground truth we report.

    python3 make_ace_dataset.py chess [outdir]
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
import numpy as np  # noqa: E402
from usgloc.datasets import SevenScenes  # noqa: E402

PGT = "/home/thanhnc/gauss_ws/src/usgloc/data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
ROOT = "/home/thanhnc/gauss_ws/src/usgloc/data/7scenes_source"


def build(scene, out_base):
    for split in ("train", "test"):
        ds = SevenScenes(ROOT, scene, split=split, sfm_pose_dir=PGT)
        d = Path(out_base) / f"7scenes_{scene}" / split
        for sub in ("rgb", "poses", "calibration"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        for i in range(len(ds)):
            fr = ds[i]
            stem = f"{fr.seq}-frame-{fr.frame_id:06d}"
            link = d / "rgb" / f"{stem}.color.png"
            if not link.exists():
                link.symlink_to(Path(fr.rgb_path).resolve())
            # ACE expects camera-to-world
            np.savetxt(d / "poses" / f"{stem}.pose.txt",
                       fr.pose_c2w.numpy(), fmt="%.10f")
            with open(d / "calibration" / f"{stem}.calibration.txt", "w") as f:
                f.write(f"{ds.focal:.6f}\n")
        print(f"  {scene}/{split}: {len(ds)} frames  focal {ds.focal:.2f}")


if __name__ == "__main__":
    scene = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else \
        "/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad/ace_data"
    build(scene, out)
