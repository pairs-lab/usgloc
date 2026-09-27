#!/usr/bin/env python3
"""Select the map-fitting resolution PER SCENE on TRAIN-side frames.

Fitting at the matcher's resolution helps small scenes (geometry converges)
and hurts large ones (more Gaussians, imperfect geometry, sharp-but-wrong
detail). Rather than fix one recipe or tune on test, we choose per scene by
the criterion that actually matters -- the correction floor, i.e. how well
the stage localizes when started from the true pose -- measured on TRAIN
frames that are disjoint from the test split.

    python3 select_map.py            # all scenes
    python3 select_map.py chess fire # a subset
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
import os
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402
from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
N_FRAMES = 20


def err(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    return float(d[:3, 3].norm()) * 100


def score(scene, suffix, ds, idx, cam, rend, dev):
    mp = Path(f"maps/{scene}{suffix}.pt")
    if not mp.is_file():
        return None
    emap = load_gaussian_map(str(mp))
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    corr = EvidentialMatchCorrector(emap, rend, cam, rounds=5)
    fl, ps = [], []
    for i in idx:
        img = ds.load_image(i).to(dev)
        T = ds[i].pose_cw.to(dev, torch.float32)
        out = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                          T, cam, vacuity=emap.gaussian_vacuity, with_depth=True)
        ps.append(float(10 * torch.log10(
            1.0 / torch.mean((out["rgb"] - img) ** 2).clamp_min(1e-12))))
        r = corr.correct(_to_gray(img), T)
        if r.success:
            fl.append(err(r.T, T))
    if not fl:
        return dict(floor=float("inf"), ok=0, psnr=float(np.mean(ps)),
                    splats=int(emap.means.shape[0]))
    # rank by success rate first, then median floor: a map the matcher cannot
    # register on is useless however precise it is on the frames it does.
    return dict(floor=float(np.median(fl)), ok=len(fl) / len(idx),
                psnr=float(np.mean(ps)), splats=int(emap.means.shape[0]))


def main():
    scenes = sys.argv[1:] or SCENES
    dev = "cuda"
    rend = GsplatRenderer()
    print(f"{'scene':<11}{'variant':<9}{'splats':>9}{'PSNR@GT':>9}"
          f"{'floor cm':>10}{'ok':>7}   pick")
    picks = {}
    for s in scenes:
        ds = SevenScenes("data/7scenes_source", s, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        rows = {}
        for suf, name in (("", "base"), ("_hq", "hq")):
            r = score(s, suf, ds, idx, cam, rend, dev)
            if r:
                rows[suf] = r
        if not rows:
            continue
        # prefer the variant that registers more often; break ties by floor
        best = min(rows, key=lambda k: (-round(rows[k]["ok"], 2), rows[k]["floor"]))
        picks[s] = best
        for suf, r in rows.items():
            mark = "  <== " if suf == best else ""
            print(f"{s:<11}{('base' if suf=='' else 'hq'):<9}{r['splats']:9d}"
                  f"{r['psnr']:9.2f}{r['floor']:10.3f}{100*r['ok']:6.0f}%{mark}")
    print("\nSELECTION (train-side, test never touched):")
    for s, suf in picks.items():
        print(f"  {s:<11} -> maps/{s}{suf}.pt")


if __name__ == "__main__":
    main()
