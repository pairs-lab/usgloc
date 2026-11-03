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
    python3 select_map.py --dataset 12scenes --maps maps/rb0 apt2/luke
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
import os
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from usgloc.datasets import SevenScenes, TwelveScenes, load_gaussian_map  # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402
from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm"
DATASETS = {  # loader, root, scenes
    "7scenes": (SevenScenes, "data/7scenes_source",
                ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]),
    "12scenes": (TwelveScenes, "data/12scenes_source",
                 ["apt1/kitchen", "apt1/living", "apt2/bed", "apt2/kitchen", "apt2/living",
                  "apt2/luke", "office1/gates362", "office1/gates381", "office1/lounge",
                  "office1/manolis", "office2/5a", "office2/5b"]),
}
N_FRAMES = 20


def err(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    return float(d[:3, 3].norm()) * 100


def score(scene, suffix, ds, idx, cam, rend, dev, maps="maps"):
    mp = Path(maps) / f"{scene.replace('/', '_')}{suffix}.pt"
    if not mp.is_file():
        return None
    emap = load_gaussian_map(str(mp))
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    corr = EvidentialMatchCorrector(emap, rend, cam, rounds=5)
    fl, ps = [], []
    for i in idx:
        img = ds.load_image(i).to(dev)
        if img.shape[:2] != (cam.height, cam.width):  # 12Scenes colour is 1296x968
            img = torch.nn.functional.interpolate(
                img.permute(2, 0, 1)[None], size=(cam.height, cam.width),
                mode="bilinear", align_corners=False)[0].permute(1, 2, 0).contiguous()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="7scenes")
    ap.add_argument("--maps", default="maps", help="folder holding <scene>{,_hq}.pt")
    args = ap.parse_args()
    DS, root, all_scenes = DATASETS[args.dataset]
    scenes = args.scenes or all_scenes
    dev = "cuda"
    rend = GsplatRenderer()
    print(f"{'scene':<11}{'variant':<9}{'splats':>9}{'PSNR@GT':>9}"
          f"{'floor cm':>10}{'ok':>7}   pick")
    picks = {}
    for s in scenes:
        ds = DS(root, s, split="train", sfm_pose_dir=f"{PGT}/{args.dataset}")
        idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        rows = {}
        for suf, name in (("", "base"), ("_hq", "hq")):
            r = score(s, suf, ds, idx, cam, rend, dev, args.maps)
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
        print(f"  {s:<11} -> {args.maps}/{s.replace('/', '_')}{suf}.pt")


if __name__ == "__main__":
    main()
