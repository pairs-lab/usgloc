#!/usr/bin/env python3
"""Rendered-depth accuracy of a map at the ground-truth pose.

The correction lifts render-side keypoints through the RENDERED depth, so the
floor is set by that depth, not by how pretty the RGB render is.  This prints,
for each ``--map-suffix``, the PSNR of the colour render AND the error of the
rendered depth against the sensor depth on the same 20 train frames the floor
uses -- so a variant that gains PSNR while losing depth accuracy is visible.

    python3 scripts/analysis/gsfit_depth.py chess --suffixes _ft _gsS

The sensor depth is in metres while the map lives in the SfM pseudo-GT frame
(~0.9x); the scale is estimated per map as the median rendered/sensor ratio
over the frames, and the reported errors are of the SCALE-CORRECTED depth, so
they measure the shape of the depth map, not the frame convention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.mapping import read_depth                        # noqa: E402
from usgloc.rendering import GsplatRenderer                   # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
N_FRAMES = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--suffixes", nargs="+", default=["_ft"])
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--out", default="results/gsfit")
    args = ap.parse_args()
    os.chdir(ROOT)
    dev = "cuda"
    rend = GsplatRenderer()
    rows = []
    print(f"{'scene':<10}{'map':<14}{'PSNR':>7}{'depth MAE cm':>14}"
          f"{'p90 cm':>9}{'<1cm %':>8}{'scale':>8}{'cover':>7}")
    for s in args.scenes:
        ds = SevenScenes("data/7scenes_source", s, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        for suf in args.suffixes:
            p = Path(f"maps/{s}{suf}.pt")
            if not p.is_file():
                continue
            emap = load_gaussian_map(str(p))
            for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
                setattr(emap, f, getattr(emap, f).to(dev))
            ratios, errs, psnrs, covs = [], [], [], []
            with torch.no_grad():
                for i in idx:
                    img = ds.load_image(i).to(dev)
                    T = ds[i].pose_cw.to(dev, torch.float32)
                    o = rend.render(emap.means, emap.covs, emap.colors,
                                    emap.opacities, T, cam,
                                    vacuity=emap.gaussian_vacuity, with_depth=True)
                    psnrs.append(float(10 * torch.log10(
                        1.0 / ((o["rgb"] - img) ** 2).mean().clamp_min(1e-12))))
                    d_s = read_depth(ds[i].depth_path).to(dev)
                    if d_s.shape != (cam.height, cam.width):
                        d_s = torch.nn.functional.interpolate(
                            d_s[None, None], size=(cam.height, cam.width),
                            mode="nearest")[0, 0]
                    ok = (torch.isfinite(d_s) & (d_s > 0.2) & (d_s < 8.0)
                          & (o["alpha"] >= args.alpha_min) & (o["depth"] > 0))
                    covs.append(float(ok.float().mean()))
                    if int(ok.sum()) < 100:
                        continue
                    ratios.append((o["depth"][ok] / d_s[ok]).median())
                    errs.append((o["depth"][ok], d_s[ok]))
                    del o
            scale = float(torch.stack(ratios).median()) if ratios else 1.0
            res = torch.cat([(a - scale * b).abs() for a, b in errs]) if errs else \
                torch.zeros(1, device=dev)
            mae = float(res.mean()) * 100
            p90 = float(res.quantile(0.9)) * 100
            frac = float((res < 0.01).float().mean()) * 100
            psnr = float(np.mean(psnrs))
            cov = float(np.mean(covs)) * 100
            rows.append(dict(scene=s, map=str(p), psnr=psnr, depth_mae_cm=mae,
                             depth_p90_cm=p90, frac_under_1cm=frac,
                             depth_scale=scale, coverage_pct=cov))
            print(f"{s:<10}{p.stem:<14}{psnr:7.2f}{mae:14.2f}{p90:9.2f}"
                  f"{frac:8.1f}{scale:8.4f}{cov:6.0f}%")
            del emap
            torch.cuda.empty_cache()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "depth_quality.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
