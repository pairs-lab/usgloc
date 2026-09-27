#!/usr/bin/env python3
"""Build the per-Gaussian MATCH-SCORE field that selects landmarks.

The pose-free sparse stage (:mod:`usgloc.landmarks`) matches query keypoints
directly against the descriptors of a sampled set of Gaussians.  STDLoc chooses
that set by "matching-oriented sampling": score every Gaussian by how well its
own descriptor agrees with the training images at its projection, then sample
spatially with a kNN argmax over that score (``train_detector.py::
matching_oriented_sample`` / ``random_knn_score``).

This script computes the score field only -- a tiny sidecar (4 * N bytes per
channel, ~2 MB) next to the map:

    python3 scripts/analysis/train_landmarks.py --scene office \
        --map maps/office_r1.pt --feats maps/office_lfinit.pt \
        --out maps/office_lmsc_init.pt

The sampling itself is done at load time inside
``usgloc.landmarks.build_landmarks``, so the landmark budget, the kNN width and
the vacuity mode are experiment options, not baked artefacts.  Nothing existing
is modified: the input map and field are read-only and the output is a new file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc import landmarks, locfeat                          # noqa: E402
from usgloc.datasets import SevenScenes, TwelveScenes, load_gaussian_map  # noqa: E402
from usgloc.rendering import GsplatRenderer                    # noqa: E402
from usgloc.superpoint import SuperPoint                       # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/7scenes_source")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir", default=PGT)
    ap.add_argument("--map", required=True, help="geometry map .pt (never modified)")
    ap.add_argument("--feats", default=None,
                    help="descriptor-field SIDECAR .pt (train_locfeat.py "
                         "--feats-only); omit when --map already carries the field")
    ap.add_argument("--out", required=True, help="score sidecar .pt (must not exist)")
    ap.add_argument("--stride", type=int, default=5,
                    help="score on every STRIDE-th train frame (the same views the "
                         "field was built from)")
    ap.add_argument("--holdout", type=int, default=2,
                    help="frames with i %% stride == HOLDOUT are never used")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--min-alpha", type=float, default=0.2)
    ap.add_argument("--occl-rel", type=float, default=0.05)
    ap.add_argument("--occl-abs", type=float, default=0.02)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import os
    os.chdir(ROOT)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --out")
    dev = args.device
    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train", sfm_pose_dir=args.sfm_pose_dir)
    if args.holdout % args.stride == 0:
        raise SystemExit("--holdout must not be 0 mod --stride (it would leak)")
    train_ids = [i for i in range(len(ds)) if i % args.stride == 0]
    cam = ds.scene_camera(args.width, args.height)
    emap = load_gaussian_map(args.map)
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    if args.feats:
        locfeat.attach_loc_feats(emap, args.feats, device=dev)
    if emap.loc_feats is None:
        raise SystemExit(f"{args.map} carries no loc_feats (pass --feats <sidecar.pt>)")
    N = emap.means.shape[0]
    print(f"{args.scene}: {N} gaussians, D={emap.loc_feats.shape[1]}, "
          f"{len(train_ids)} scoring views @ {args.width}x{args.height}", flush=True)
    rend = GsplatRenderer()
    enc = SuperPoint().to(dev).eval()
    t0 = time.time()
    res = landmarks.match_score_field(
        emap, ds, cam, enc, rend, train_ids, occl_rel=args.occl_rel,
        occl_abs=args.occl_abs, min_alpha=args.min_alpha)
    sec = time.time() - t0
    sc, det, obs = res["score"], res["det"], res["obs"]
    seen = obs > 0
    ck = {
        "lm_score": sc.float().cpu(),
        "lm_det": det.float().cpu(),
        "lm_score_obs": obs.float().cpu(),
        "lm_score_n": int(N),
        "lm_score_base": str(args.map),
        "lm_score_feats": str(args.feats),
        "lm_score_stride": int(args.stride),
        "lm_score_holdout": int(args.holdout),
        "lm_score_dim": int(emap.loc_feats.shape[1]),
    }
    torch.save(ck, out)
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=sc.device)
    report = dict(scene=args.scene, map=args.map, feats=args.feats, out=str(out),
                  n_gaussians=int(N), n_views=len(train_ids), sec=sec,
                  observed=int(seen.sum()), observed_frac=float(seen.float().mean()),
                  score_quantiles=[float(v) for v in
                                   torch.quantile(sc[seen].double(), q.double())],
                  det_quantiles=[float(v) for v in
                                 torch.quantile(det[seen].double(), q.double())],
                  score_mean=float(sc[seen].mean()), det_mean=float(det[seen].mean()))
    print(f"  {report['observed']}/{N} observed ({100*report['observed_frac']:.1f}%), "
          f"{sec:.0f}s; match score q10/50/90 = "
          f"{report['score_quantiles'][0]:.3f}/{report['score_quantiles'][2]:.3f}/"
          f"{report['score_quantiles'][4]:.3f}")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
