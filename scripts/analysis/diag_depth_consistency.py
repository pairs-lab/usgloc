#!/usr/bin/env python3
"""Diagnosis: is the 7Scenes sensor depth consistent with the map geometry?

The correction lifts render-side keypoints to 3D THROUGH THE RENDERED DEPTH,
so the correction floor is capped by how metrically right that depth is.  The
map, however, is fitted with a photometric L1 loss only -- nothing constrains
its depth.  Before adding a depth term to the fine-tune we must know whether
the sensor depth is even in the same units/scale as the map: the pseudo-GT
poses come from an SfM model calibrated at focal ~526 while the Kinect depth
was captured at focal 585, so a systematic scale factor is plausible.

For N train frames spread evenly over the split, render the map at the
ground-truth pose and compare the alpha-normalised expected z-depth with the
sensor depth on valid pixels (both finite, 0.2-8 m, alpha >= 0.5):

    python3 scripts/analysis/diag_depth_consistency.py chess office stairs \
        --out results/depthsup/diag

Reports per scene the median ratio rendered/sensor, the median |difference|,
and the 10/90 percentiles of both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.mapping import read_depth                        # noqa: E402
from usgloc.rendering import GsplatRenderer                  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["chess", "office", "stairs"])
    ap.add_argument("--suffix", default="ft", help="maps/<scene>_<suffix>.pt")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--alpha-min", type=float, default=0.5)
    ap.add_argument("--edge-rel", type=float, default=None,
                    help="also drop pixels whose 3x3 rendered-depth spread exceeds "
                         "this fraction of the depth (silhouette / phantom-surface "
                         "pixels, as the corrector's edge gate does)")
    ap.add_argument("--out", default="results/depthsup/diag")
    args = ap.parse_args()

    import os
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rend = GsplatRenderer()
    rows = []
    for s in args.scenes:
        ds = SevenScenes("data/7scenes_source", s, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, args.frames).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        emap = load_gaussian_map(f"maps/{s}_{args.suffix}.pt")
        for f in ("means", "covs", "colors", "opacities"):
            setattr(emap, f, getattr(emap, f).to(dev))
        ratios, diffs, per_frame = [], [], []
        rend_all, sens_all = [], []
        for i in idx:
            rec = ds[i]
            T = rec.pose_cw.to(dev, torch.float32)
            with torch.no_grad():
                o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                T, cam, with_depth=True)
            dr = o["depth"].float()
            al = o["alpha"].float()
            dsens = read_depth(rec.depth_path).to(dev)
            if dsens.shape != dr.shape:
                dsens = torch.nn.functional.interpolate(
                    dsens[None, None], size=dr.shape, mode="nearest")[0, 0]
            ok = (torch.isfinite(dsens) & (dsens > 0.2) & (dsens < 8.0)
                  & torch.isfinite(dr) & (dr > 0.2) & (dr < 8.0) & (al >= args.alpha_min))
            if args.edge_rel is not None:
                dm = dr[None, None]
                hi = torch.nn.functional.max_pool2d(dm, 3, 1, 1)[0, 0]
                lo = -torch.nn.functional.max_pool2d(-dm, 3, 1, 1)[0, 0]
                ok = ok & ((hi - lo) <= args.edge_rel * dr)
            if ok.sum() < 100:
                continue
            a, b = dr[ok].cpu().numpy(), dsens[ok].cpu().numpy()
            r = a / b
            d = a - b
            ratios.append(r)
            diffs.append(d)
            rend_all.append(a)
            sens_all.append(b)
            per_frame.append(dict(frame=int(i), n=int(ok.sum()),
                                  med_ratio=float(np.median(r)),
                                  med_absdiff=float(np.median(np.abs(d)))))
            print(f"  {s:<9} frame {i:5d}  n={int(ok.sum()):6d}  "
                  f"ratio {np.median(r):.4f}  |diff| {np.median(np.abs(d))*100:6.2f} cm  "
                  f"signed diff {np.median(d)*100:+6.2f} cm", flush=True)
        R = np.concatenate(ratios)
        D = np.concatenate(diffs)
        A = np.concatenate(rend_all)      # rendered depth
        B = np.concatenate(sens_all)      # sensor depth
        # A single global scale per scene: does it explain the offset?  The
        # robust scale is the median ratio; the residual after rescaling the
        # sensor depth into the map's (pseudo-GT/SfM) frame is what is left
        # for depth supervision to fix.
        scale = float(np.median(R))
        Dres = A - scale * B
        rec = dict(
            scene=s, suffix=args.suffix, frames=len(per_frame), n_pixels=int(R.size),
            ratio_median=float(np.median(R)),
            ratio_p10=float(np.percentile(R, 10)), ratio_p90=float(np.percentile(R, 90)),
            absdiff_median_m=float(np.median(np.abs(D))),
            absdiff_p10_m=float(np.percentile(np.abs(D), 10)),
            absdiff_p90_m=float(np.percentile(np.abs(D), 90)),
            signed_median_m=float(np.median(D)),
            signed_p10_m=float(np.percentile(D, 10)), signed_p90_m=float(np.percentile(D, 90)),
            # global robust scale s (median ratio) and the residual left after
            # mapping the sensor depth into the map's frame with it.
            scale_median_ratio=scale,
            scale_focal_over_585=float(ds.focal / 585.0) if ds.focal else None,
            res_absmedian_m=float(np.median(np.abs(Dres))),
            res_p10_m=float(np.percentile(Dres, 10)),
            res_p90_m=float(np.percentile(Dres, 90)),
            per_frame=per_frame,
        )
        rows.append(rec)
        print(f"==> {s:<9} ratio med {rec['ratio_median']:.4f} "
              f"[{rec['ratio_p10']:.4f}, {rec['ratio_p90']:.4f}]  "
              f"|diff| med {100*rec['absdiff_median_m']:.2f} cm "
              f"[{100*rec['absdiff_p10_m']:.2f}, {100*rec['absdiff_p90_m']:.2f}]  "
              f"| after scale {scale:.4f} (f/585={rec['scale_focal_over_585']:.4f}): "
              f"|res| med {100*rec['res_absmedian_m']:.2f} cm "
              f"[{100*rec['res_p10_m']:+.2f}, {100*rec['res_p90_m']:+.2f}]",
              flush=True)
        del emap
        torch.cuda.empty_cache()
    tag = f"{args.suffix}_a{args.alpha_min:g}"
    if args.edge_rel is not None:
        tag += f"_e{args.edge_rel:g}"
    with open(out / f"depth_consistency_{tag}.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{'scene':<9}{'ratio':>8}{'p10':>8}{'p90':>8}{'|d| cm':>9}{'p10':>8}{'p90':>8}"
          f"{'signed cm':>11}{'f/585':>8}{'|res| cm':>10}{'resp10':>9}{'resp90':>9}")
    for r in rows:
        print(f"{r['scene']:<9}{r['ratio_median']:8.4f}{r['ratio_p10']:8.4f}"
              f"{r['ratio_p90']:8.4f}{100*r['absdiff_median_m']:9.2f}"
              f"{100*r['absdiff_p10_m']:8.2f}{100*r['absdiff_p90_m']:8.2f}"
              f"{100*r['signed_median_m']:11.2f}{r['scale_focal_over_585']:8.4f}"
              f"{100*r['res_absmedian_m']:10.2f}{100*r['res_p10_m']:9.2f}"
              f"{100*r['res_p90_m']:9.2f}")


if __name__ == "__main__":
    main()
