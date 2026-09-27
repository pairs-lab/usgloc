#!/usr/bin/env python3
r"""Is the test-view rendering deficit APPEARANCE or GEOMETRY?

Our maps store one diffuse RGB per Gaussian (SH degree 0), so they cannot
reproduce view-dependent appearance from directions the fit never saw.  Every
experiment today hit the same wall: components that work on train frames lose
most of their benefit on test frames, and test-view PSNR sits ~4 dB below
train-view PSNR.

This separates the two possible causes with an ORACLE that is deliberately
cheating: optimise ONLY the per-Gaussian colours against the TEST views
(geometry, opacity and every evidential attribute frozen) and report PSNR
before and after.

  - If the oracle recovers most of the deficit, the limit is APPEARANCE, and a
    view-dependent colour model (spherical harmonics, as standard 3DGS and
    SplatHLoc use) should close a large part of it legitimately.
  - If PSNR barely moves, the limit is GEOMETRY, and no colour model will help.

The fitted colours are NEVER saved -- this measures a ceiling, it is not a
method.

    python3 scripts/analysis/diag_appearance.py office --iters 600
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.rendering import GsplatRenderer                  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def psnr(a, b):
    return float(10 * torch.log10(1.0 / torch.mean((a - b) ** 2).clamp_min(1e-12)))


def frames(ds, n):
    idx = torch.linspace(0, len(ds) - 1, n).round().long().unique().tolist()
    return idx


def measure(scene, split, n, iters, lr, dev="cuda"):
    ds = SevenScenes("data/7scenes_source", scene, split=split, sfm_pose_dir=PGT)
    cam = ds.scene_camera(640, 480)
    emap = load_gaussian_map(f"maps/{scene}_r1.pt")
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    rend = GsplatRenderer()
    idx = frames(ds, n)
    views = [(ds.load_image(i).to(dev), ds[i].pose_cw.to(dev, torch.float32)) for i in idx]

    def mean_psnr():
        tot = 0.0
        with torch.no_grad():
            for img, T in views:
                o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                T, cam, vacuity=emap.gaussian_vacuity, with_depth=False)
                tot += psnr(o["rgb"], img)
        return tot / len(views)

    before = mean_psnr()
    # ORACLE: optimise only the colours, on these very views
    col = emap.colors.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([col], lr=lr)
    g = torch.Generator(device="cpu").manual_seed(0)
    for it in range(iters):
        j = int(torch.randint(len(views), (1,), generator=g))
        img, T = views[j]
        o = rend.render(emap.means, emap.covs, col, emap.opacities, T, cam,
                        vacuity=emap.gaussian_vacuity, with_depth=False)
        loss = (o["rgb"] - img).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            col.clamp_(0.0, 1.0)
    emap.colors = col.detach()
    after = mean_psnr()
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    print(f"{'scene':<12}{'split':<7}{'PSNR now':>9}{'colour-oracle':>15}{'gain':>7}")
    for s in a.scenes:
        for split in ("train", "test"):
            b, c = measure(s, split, a.n, a.iters, a.lr)
            print(f"{s:<12}{split:<7}{b:9.2f}{c:15.2f}{c-b:7.2f}", flush=True)


if __name__ == "__main__":
    main()
