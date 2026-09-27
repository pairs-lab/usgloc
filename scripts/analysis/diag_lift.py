#!/usr/bin/env python3
"""How much does the 2D-3D lift actually move when the blended depth is
replaced by a single-surface depth?

The correction lifts every gated render-side keypoint through the
alpha-blended (expected) depth.  This script renders maps/<scene>_ft.pt at the
ground-truth pose of the same 20 train frames exp_floor_matcher.py uses, runs
LoFTR against the real frame, applies the *same* gate the corrector applies
(confidence, alpha, min depth, 3x3 depth-spread, vacuity), and then, for the
surviving matches only, recomputes the rasterizer's exact per-ray compositing
(usgloc.lift.ray_depths) to report

  * the fraction of gated matches that still sit within 2 px of a depth
    discontinuity (3x3 relative depth spread above a threshold),
  * |expected depth - mode depth| and |expected - median| there and elsewhere,
  * the implied 3D displacement of the lifted point, in cm.

    python3 scripts/analysis/diag_lift.py chess office stairs \
        --out results/estimator_lift/diag

Read-only: it never writes a map and never touches the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc.correction import EvidentialMatchCorrector, _to_gray   # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map          # noqa: E402
from usgloc.lift import ray_depths                                  # noqa: E402
from usgloc.rendering import GsplatRenderer                         # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
N_FRAMES = 20


def q(a, p):
    return float(np.quantile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["chess", "office", "stairs"])
    ap.add_argument("--map-suffix", default="_ft")
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--disc-rel", type=float, default=0.02,
                    help="relative 5x5 depth spread above which a gated match "
                         "counts as sitting near a depth discontinuity")
    ap.add_argument("--out", default="results/estimator_lift/diag")
    args = ap.parse_args()

    import os
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rend = GsplatRenderer()
    rows = []
    for scene in args.scenes:
        print(f"[{scene}] loading map", flush=True)
        emap = load_gaussian_map(f"maps/{scene}{args.map_suffix}.pt")
        for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
            setattr(emap, f, getattr(emap, f).to(dev))
        ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        H, W = cam.height, cam.width
        print(f"[{scene}] map loaded, building LoFTR", flush=True)
        corr = EvidentialMatchCorrector(emap, rend, cam, rounds=1)
        print(f"[{scene}] LoFTR ready, {len(idx)} frames", flush=True)
        acc = {k: [] for k in ("disc", "d_exp", "d_max", "d_med", "d_band", "n_hit")}
        for i in idx:
            print(f"  [{scene}] frame {i}", flush=True)
            img = ds.load_image(i).to(dev)
            T = ds[i].pose_cw.to(dev, torch.float32)
            with torch.no_grad():
                o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                T, cam, vacuity=emap.gaussian_vacuity,
                                with_depth=True, with_meta=True)
                g0 = _to_gray(img)[None, None].to(torch.float32)
                g1 = _to_gray(o["rgb"])[None, None].to(torch.float32)
                m = corr.matcher({"image0": g0, "image1": g1})
            kr = m["keypoints1"]
            if kr.numel() == 0:
                continue
            xi = kr[:, 0].round().long().clamp(0, W - 1)
            yi = kr[:, 1].round().long().clamp(0, H - 1)
            depth, alpha, vac = o["depth"], o["alpha"], o["vacuity"]
            d, a, u = depth[yi, xi], alpha[yi, xi], vac[yi, xi]
            dm = depth[None, None]
            spread3 = (F.max_pool2d(dm, 3, 1, 1) + F.max_pool2d(-dm, 3, 1, 1))[0, 0]
            spread5 = (F.max_pool2d(dm, 5, 1, 2) + F.max_pool2d(-dm, 5, 1, 2))[0, 0]
            keep = ((m["confidence"] >= corr.conf_min) & (a >= corr.min_alpha)
                    & (d >= corr.min_depth)
                    & (spread3[yi, xi] <= corr.edge_rel * d)
                    & ((u / a.clamp_min(1e-6)) <= corr.u_max))
            if int(keep.sum()) == 0:
                continue
            xi, yi, d = xi[keep], yi[keep], d[keep]
            st = ray_depths(o["meta"], xi.float() + 0.5, yi.float() + 0.5, band=args.band)
            # 5x5 spread == "within 2 px of a depth discontinuity"
            acc["disc"].append((spread5[yi, xi] / d.clamp_min(1e-6)
                                > args.disc_rel).float().cpu().numpy())
            acc["d_exp"].append(st["exp"].cpu().numpy())
            acc["d_max"].append(st["maxw"].cpu().numpy())
            acc["d_med"].append(st["median"].cpu().numpy())
            acc["d_band"].append(st["band"].cpu().numpy())
            acc["n_hit"].append(st["n_hit"].cpu().numpy())
            del o
        A = {k: np.concatenate(v) if v else np.zeros(0) for k, v in acc.items()}
        disc = A["disc"] > 0.5
        row = dict(scene=scene, map=f"maps/{scene}{args.map_suffix}.pt",
                   n_gated=int(len(disc)), frac_near_disc=float(disc.mean()),
                   median_n_contrib=q(A["n_hit"], 0.5),
                   median_depth_m=q(A["d_exp"], 0.5))
        for name, key in (("maxw", "d_max"), ("median", "d_med"), ("band", "d_band")):
            off = A[key] - A["d_exp"]                     # metres, + = further away
            for tag, msk in (("all", np.ones_like(disc)), ("disc", disc), ("flat", ~disc)):
                o_ = off[msk]
                row[f"{name}_{tag}_med_cm"] = 100 * q(o_, 0.5)
                row[f"{name}_{tag}_medabs_cm"] = 100 * q(np.abs(o_), 0.5)
                row[f"{name}_{tag}_p90abs_cm"] = 100 * q(np.abs(o_), 0.9)
            # lateral 3D shift is zero (same ray); the whole move is along z
            row[f"{name}_frac_move_gt_1cm"] = float((np.abs(off) > 0.01).mean())
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)
        del corr, emap
        torch.cuda.empty_cache()
    with open(out.with_suffix(".json") if out.suffix else out / "diag_lift.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{'scene':<10}{'gated':>7}{'near-disc':>11}{'maxw all':>10}{'maxw disc':>11}"
          f"{'med all':>10}{'med disc':>10}   (median |offset| cm)")
    for r in rows:
        print(f"{r['scene']:<10}{r['n_gated']:7d}{100*r['frac_near_disc']:10.1f}%"
              f"{r['maxw_all_medabs_cm']:10.3f}{r['maxw_disc_medabs_cm']:11.3f}"
              f"{r['median_all_medabs_cm']:10.3f}{r['median_disc_medabs_cm']:10.3f}")


if __name__ == "__main__":
    main()
