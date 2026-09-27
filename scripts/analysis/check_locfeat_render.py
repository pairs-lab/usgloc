#!/usr/bin/env python3
"""The descriptor channels must not perturb rgb / vacuity / depth / alpha / xyz.

If they do not, the loc_feats=None path (which is the ORIGINAL code, unchanged)
is a fortiori bit-identical, so every LoFTR result still reproduces.
"""
import os
import sys

import pathlib

import torch
import torch.nn.functional as F

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from usgloc.datasets import SevenScenes, load_gaussian_map
from usgloc.rendering import GsplatRenderer

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
scene = sys.argv[1] if len(sys.argv) > 1 else "office"
suffix = sys.argv[2] if len(sys.argv) > 2 else "_r1"
dev = "cuda"
emap = load_gaussian_map(f"maps/{scene}{suffix}.pt")
for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
    setattr(emap, f, getattr(emap, f).to(dev))
ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
cam = ds.scene_camera(640, 480)
rend = GsplatRenderer()
N = emap.means.shape[0]
bad = 0
for i in (0, 500, 1234):
    T = ds[i].pose_cw.to(dev, torch.float32)
    with torch.no_grad():
        for with_xyz in (False, True):
            a = rend.render(emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
                            vacuity=emap.gaussian_vacuity, with_depth=True,
                            with_xyz=with_xyz)
            for D in (64, 256):
                lf = F.normalize(torch.randn(N, D, device=dev), dim=-1)
                b = rend.render(emap.means, emap.covs, emap.colors, emap.opacities, T,
                                cam, vacuity=emap.gaussian_vacuity, with_depth=True,
                                with_xyz=with_xyz, loc_feats=lf)
                for k in ("rgb", "vacuity", "alpha", "depth") + (("xyz",) if with_xyz else ()):
                    same = bool(torch.equal(a[k], b[k]))
                    if not same:
                        bad += 1
                        print(f"  MISMATCH frame {i} xyz={with_xyz} D={D} {k}: "
                              f"max|diff| {float((a[k]-b[k]).abs().max()):.3e}")
                assert set(b) - set(a) == {"feats"}, sorted(set(b) - set(a))
                assert b["feats"].shape == (480, 640, D)
                nrm = b["feats"].norm(dim=-1)
                m = b["alpha"] > 0.1
                assert float((nrm[m] - 1).abs().max()) < 1e-5, "feats not unit-norm"
                del lf, b
            del a
            torch.cuda.empty_cache()
print("RENDER IDENTITY CHECK:", "FAILED" if bad else
      "rgb/vacuity/alpha/depth/xyz bit-identical with and without descriptor channels")
sys.exit(1 if bad else 0)
