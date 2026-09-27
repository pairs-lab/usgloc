#!/usr/bin/env python3
"""Cost of the descriptor channels in GsplatRenderer.render.

gsplat rasterizes at most ``channel_chunk`` channels (32 in gsplat 1.5.x) per
``rasterize_to_pixels`` launch; the projection, the 2D covariances and the tile
binning are shared across the chunks.  So a D-dimensional descriptor field adds
``ceil((4 + D + depth) / 32) - 1`` extra pixel passes, not extra projections.
This measures what that actually costs, forward (localisation) and
forward+backward (the descriptor fine-tune), at several resolutions.

    python3 scripts/analysis/bench_locfeat_render.py office --out results/locfeat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / n * 1e3,
            torch.cuda.max_memory_allocated() / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="office")
    ap.add_argument("--map-suffix", default="_r1")
    ap.add_argument("--dims", nargs="+", type=int, default=[64, 256])
    ap.add_argument("--res", nargs="+", default=["640x480", "320x240", "80x60"])
    ap.add_argument("--out", default="results/locfeat")
    args = ap.parse_args()
    os.chdir(ROOT)
    dev = "cuda"
    import gsplat
    emap = load_gaussian_map(f"maps/{args.scene}{args.map_suffix}.pt")
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    N = emap.means.shape[0]
    ds = SevenScenes("data/7scenes_source", args.scene, split="train", sfm_pose_dir=PGT)
    T = ds[0].pose_cw.to(dev, torch.float32)
    rend = GsplatRenderer()
    from gsplat.rendering import rasterization
    import inspect
    chunk = inspect.signature(rasterization).parameters["channel_chunk"].default
    rows = []
    print(f"{args.scene}{args.map_suffix}: {N} gaussians, gsplat {gsplat.__version__}, "
          f"channel_chunk={chunk}")
    for spec in args.res:
        W, Hh = (int(v) for v in spec.split("x"))
        cam = ds.scene_camera(W, Hh)
        with torch.no_grad():
            ms, gb = bench(lambda: rend.render(
                emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
                vacuity=emap.gaussian_vacuity, with_depth=True))
            rows.append(dict(res=spec, dim=0, mode="forward", ms=ms, peak_gb=gb,
                             passes=1))
            print(f"  {spec:<9} rgb+vac+depth (baseline)        {ms:8.2f} ms  {gb:5.2f} GB")
            for D in args.dims:
                lf = F.normalize(torch.randn(N, D, device=dev), dim=-1)
                ms, gb = bench(lambda lf=lf, cam=cam: rend.render(
                    emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
                    vacuity=emap.gaussian_vacuity, with_depth=True, loc_feats=lf))
                p = -(-(4 + D + 1) // chunk)
                rows.append(dict(res=spec, dim=D, mode="forward", ms=ms, peak_gb=gb,
                                 passes=p))
                print(f"  {spec:<9} + {D:3d}-d feats ({p} chunks)          "
                      f"{ms:8.2f} ms  {gb:5.2f} GB")
                del lf
                torch.cuda.empty_cache()
    for spec in args.res:
        W, Hh = (int(v) for v in spec.split("x"))
        cam = ds.scene_camera(W, Hh)
        for D in args.dims:
            lf = torch.nn.Parameter(F.normalize(torch.randn(N, D, device=dev), dim=-1))
            tgt = F.normalize(torch.randn(D, Hh, W, device=dev), dim=0)

            def step(lf=lf, cam=cam, tgt=tgt):
                o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                T, cam, vacuity=emap.gaussian_vacuity,
                                loc_feats=F.normalize(lf, dim=-1))
                (o["feats"].permute(2, 0, 1) - tgt).abs().mean().backward()
                lf.grad = None
            try:
                ms, gb = bench(step, n=5, warm=2)
                rows.append(dict(res=spec, dim=D, mode="train", ms=ms, peak_gb=gb))
                print(f"  {spec:<9} {D:3d}-d TRAIN step (fwd+bwd)      "
                      f"{ms:8.2f} ms  {gb:5.2f} GB")
            except torch.OutOfMemoryError:
                rows.append(dict(res=spec, dim=D, mode="train", ms=None, peak_gb=None,
                                 oom=True))
                print(f"  {spec:<9} {D:3d}-d TRAIN step: OOM")
            del lf, tgt
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"render_cost_{args.scene}{args.map_suffix}.json"
    with open(p, "w") as f:
        json.dump(dict(scene=args.scene, map=f"maps/{args.scene}{args.map_suffix}.pt",
                       n_gaussians=N, gsplat=gsplat.__version__,
                       channel_chunk=int(chunk), rows=rows), f, indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
