#!/usr/bin/env python3
"""Score one map's correction floor from the ground truth AND from the
20 cm / 10 deg perturbation in ONE process, with the map loaded once.

This is scripts/analysis/exp_floor_matcher.py's protocol verbatim -- it
imports and calls that module's ``run()``, so the numbers are identical to
running it twice -- but it pays the ~1 min covariance eigendecomposition and
the LoFTR/map load once instead of twice, and takes the shared GPU lock once
instead of twice.  On a card shared by several jobs that halves the queueing.

It writes exactly the files exp_floor_matcher.py would write
(``floor_<scene>_<spec><map-suffix>[_p0.2m10deg].json``) plus a combined
``score_<scene>_<suffix>.json``.

    python3 scripts/analysis/gx_score.py chess --map-suffix _r1 \
        --frames 60 --out results/densify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import exp_floor_matcher as E                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--map-suffix", required=True)
    ap.add_argument("--matcher", default="loftr")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--perturb", nargs=2, type=float, default=[0.2, 10.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-psnr", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = E.SevenScenes("data/7scenes_source", args.scene, split="train",
                       sfm_pose_dir=E.PGT)
    idx = torch.linspace(0, len(ds) - 1, args.frames).round().long().unique().tolist()
    cam = ds.scene_camera(640, 480)
    rend = E.GsplatRenderer()
    spec_tag = args.matcher.replace(":", "_").replace("=", "")
    stag = "" if args.map_suffix == "_ft" else f"_{args.map_suffix.lstrip('_')}"
    combined = {}
    for name, pert in (("gt", None),
                       ("pert", (args.perturb[0], args.perturb[1]))):
        summ = E.run(args.scene, args.matcher, ds, idx, cam, rend, "cuda",
                     args.rounds, pert, suffix=args.map_suffix,
                     psnr=not args.no_psnr, cache_map=True)
        ptag = "" if name == "gt" else f"_p{pert[0]:g}m{pert[1]:g}deg"
        (out / f"floor_{args.scene}_{spec_tag}{stag}{ptag}.json").write_text(
            json.dumps(summ, indent=1))
        combined[name] = summ
        print(f"==> {args.scene:<9}{args.matcher:<22}{name:<5} "
              f"floor {summ['floor_cm']:.3f} cm / {summ['floor_deg']:.3f} deg  "
              f"reg {100 * summ['reg_rate']:.0f}%  inl {summ['median_inliers']:.0f}  "
              f"{summ['sec_median']:.2f} s/corr  peak {summ['peak_gpu_gb']:.2f} GB"
              + ("" if summ["psnr_gt"] is None else f"  PSNR {summ['psnr_gt']:.2f}"),
              flush=True)
    (out / f"score_{args.scene}_{args.map_suffix.lstrip('_')}.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "frames"}
                    for k, v in combined.items()}, indent=1))


if __name__ == "__main__":
    main()
