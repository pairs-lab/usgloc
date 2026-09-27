#!/usr/bin/env python3
"""Build the per-Gaussian LOCAL descriptor field on top of an existing map.

Adds an optional ``loc_feats`` (N, D) attribute to a map checkpoint WITHOUT
touching any other tensor: the Gaussian set, its geometry, colour, opacity and
every evidential attribute are copied through unchanged, so every existing
result still reproduces from the input map and the RGB-matching path never
sees the new key.

    # 256-d field on the stabilised office re-fit, held-out frames = i % 5 == 2
    python3 scripts/analysis/train_locfeat.py --scene office --map maps/office_r1.pt \
        --out maps/office_lf256.pt --stride 5 --holdout 2 --iters 4000

    # per-scene PCA of that same field down to 64-d (no re-training)
    python3 scripts/analysis/train_locfeat.py --scene office --map maps/office_lf256.pt \
        --out maps/office_lf64.pt --pca-only --pca-dim 64

    # bake a sidecar field (--feats-only output) into a standalone map
    python3 scripts/analysis/train_locfeat.py --scene office --map maps/office_r1.pt \
        --feats maps/office_lf256.pt --out maps/office_lf.pt --merge-only

Stage 1 (init) projects every Gaussian into the training views and averages the
frozen SuperPoint descriptor at its projection, occlusion-tested against the
map's own RENDERED expected depth.  Stage 2 (fine-tune) optimises ONLY the
descriptors so the alpha-blended rendered descriptor map reproduces the
per-view SuperPoint map -- STDLoc's feature loss, geometry frozen.
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

from usgloc import locfeat                                    # noqa: E402
from usgloc.datasets import SevenScenes, TwelveScenes, load_gaussian_map  # noqa: E402
from usgloc.rendering import GsplatRenderer                   # noqa: E402
from usgloc.superpoint import SuperPoint                      # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def frame_split(n: int, stride: int, holdout: int):
    """Training views (i % stride == 0) and held-out views (i % stride == holdout)."""
    if holdout % stride == 0:
        raise SystemExit("--holdout must not be 0 mod --stride (it would leak)")
    train = [i for i in range(n) if i % stride == 0]
    held = [i for i in range(n) if i % stride == holdout % stride]
    return train, held


def _save(ckpt, out, args, n):
    """Write the full map, or (``--feats-only``) just the descriptor-field keys."""
    if args.feats_only:
        sc = {k: ckpt[k] for k in locfeat.LOC_FEAT_KEYS if k in ckpt}
        sc["loc_feat_base"] = str(args.map)
        sc["loc_feat_n"] = int(n)
        torch.save(sc, out)
    else:
        torch.save(ckpt, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/7scenes_source")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir", default=PGT)
    ap.add_argument("--map", required=True, help="input map .pt (never modified)")
    ap.add_argument("--out", required=True, help="output map .pt (must not exist)")
    ap.add_argument("--stride", type=int, default=5,
                    help="use every STRIDE-th train frame to build the field")
    ap.add_argument("--holdout", type=int, default=2,
                    help="frames with i %% stride == HOLDOUT are never used, so the "
                         "sanity script can measure on unseen views")
    ap.add_argument("--iters", type=int, default=4000,
                    help="descriptor-only fine-tune steps (0 = projection init only)")
    ap.add_argument("--lr", type=float, default=1e-3, help="STDLoc's loc_feature_lr")
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--loss", choices=("l1", "cos"), default="l1")
    ap.add_argument("--adam-beta1", type=float, default=0.9,
                    help="0.9 = STDLoc's/torch's default; 0 removes momentum, so a "
                         "Gaussian with no gradient this step does not drift (the "
                         "gradient is one-view sparse)")
    ap.add_argument("--fit-width", type=int, default=640)
    ap.add_argument("--fit-height", type=int, default=480)
    ap.add_argument("--min-alpha", type=float, default=0.2,
                    help="rendered-alpha floor for a pixel to carry supervision and "
                         "for a Gaussian to be accepted in a view")
    ap.add_argument("--occl-rel", type=float, default=0.05)
    ap.add_argument("--occl-abs", type=float, default=0.02)
    ap.add_argument("--score-weight", action="store_true",
                    help="weight the init average by the SuperPoint detector score")
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="reduce the field to this dimension with a per-scene PCA "
                         "(0 = keep the encoder's native 256-d)")
    ap.add_argument("--pca-only", action="store_true",
                    help="skip init+fine-tune: re-PCA the loc_feats already in --map "
                         "(or in --feats, when a sidecar is given)")
    ap.add_argument("--feats", default=None,
                    help="read the INPUT descriptor field from this sidecar instead "
                         "of from --map (for --pca-only / --merge-only)")
    ap.add_argument("--merge-only", action="store_true",
                    help="write --map plus the --feats sidecar as one standalone map "
                         "checkpoint and do nothing else (no GPU needed)")
    ap.add_argument("--feats-only", action="store_true",
                    help="write a SIDECAR holding only the loc_feat_* keys (~2*N*D "
                         "bytes) instead of a full copy of the map; load it with "
                         "usgloc.locfeat.attach_loc_feats. The map's geometry is "
                         "then stored once on disk no matter how many fields you "
                         "compare.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os
    os.chdir(ROOT)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --out (existing maps are never "
                         f"overwritten)")
    dev = args.device
    ckpt = torch.load(args.map, map_location="cpu", weights_only=False)
    N = ckpt["means"].shape[0]
    if args.feats:
        sc = torch.load(args.feats, map_location="cpu", weights_only=False)
        if int(sc.get("loc_feat_n", sc["loc_feats"].shape[0])) != N:
            raise SystemExit(f"{args.feats} was built for a different map than {args.map}")
        for k in locfeat.LOC_FEAT_KEYS:
            if k in sc:
                ckpt[k] = sc[k]
    t0 = time.time()
    report = dict(scene=args.scene, map_in=args.map, map_out=args.out,
                  n_gaussians=N, iters=args.iters, stride=args.stride,
                  holdout=args.holdout, fit=[args.fit_width, args.fit_height],
                  min_alpha=args.min_alpha, loss=args.loss, lr=args.lr,
                  lr_final=args.lr_final, pca_dim=args.pca_dim)

    if args.merge_only:
        if "loc_feats" not in ckpt:
            raise SystemExit("--merge-only needs a --feats sidecar (or a --map that "
                             "already carries loc_feats)")
        if args.feats_only:
            raise SystemExit("--merge-only and --feats-only are contradictory")
        torch.save(ckpt, out)
        d = ckpt["loc_feats"].shape[1]
        report.update(stage="merge_only", dim=int(d))
        print(f"saved {out}: {args.map} + {args.feats} -> {N} x {d} loc_feats, "
              f"{time.time() - t0:.0f}s")
        print(json.dumps(report))
        return

    if args.pca_only:
        if "loc_feats" not in ckpt:
            raise SystemExit(f"{args.map} has no loc_feats to reduce")
        if args.pca_dim <= 0:
            raise SystemExit("--pca-only needs --pca-dim > 0")
        if "loc_feat_pca_basis" in ckpt:
            raise SystemExit(f"{args.map} is already PCA-reduced; reduce the native map")
        feats = ckpt["loc_feats"].to(dev).float()
        obs = ckpt.get("loc_feat_obs")
        rows = feats if obs is None else feats[obs.to(dev) > 0]
        mean, basis, ev = locfeat.pca_fit(rows, args.pca_dim, seed=args.seed)
        red = locfeat.pca_apply(feats, mean, basis)
        red[(obs.to(dev) == 0) if obs is not None else (feats.abs().sum(-1) == 0)] = 0.0
        ckpt["loc_feats"] = red.half().cpu()
        ckpt["loc_feat_pca_mean"] = mean.cpu()
        ckpt["loc_feat_pca_basis"] = basis.cpu()
        ckpt["loc_feat_dim"] = int(args.pca_dim)
        ckpt["loc_feat_pca_ev"] = float(ev)
        report.update(explained_variance=float(ev), stage="pca_only")
        _save(ckpt, out, args, N)
        print(f"saved {out}: {N} x {args.pca_dim} loc_feats "
              f"(PCA of {feats.shape[1]}-d, {100*ev:.1f}% variance kept), "
              f"{time.time() - t0:.0f}s")
        print(json.dumps(report))
        return

    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train", sfm_pose_dir=args.sfm_pose_dir)
    train_ids, held_ids = frame_split(len(ds), args.stride, args.holdout)
    cam = ds.scene_camera(args.fit_width, args.fit_height)
    print(f"{args.scene}: {N} gaussians, {len(ds)} train frames -> "
          f"{len(train_ids)} field views, {len(held_ids)} held out "
          f"@ {args.fit_width}x{args.fit_height}")

    emap = load_gaussian_map(args.map)
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    rend = GsplatRenderer()
    enc = SuperPoint().to(dev).eval()

    t_init = time.time()
    ini = locfeat.init_loc_feats(
        emap, ds, cam, enc, rend, train_ids, occl_rel=args.occl_rel,
        occl_abs=args.occl_abs, min_alpha=args.min_alpha,
        score_weight=args.score_weight)
    feats, obs = ini["loc_feats"], ini["obs_count"]
    seen = int((obs > 0).sum())
    report.update(init_sec=time.time() - t_init, observed=seen,
                  observed_frac=seen / N,
                  obs_median=float(obs[obs > 0].median()) if seen else 0.0)
    print(f"  init: {seen}/{N} ({100*seen/N:.1f}%) observed, median "
          f"{report['obs_median']:.0f} views/gaussian, {report['init_sec']:.0f}s")

    if args.iters > 0:
        t_ft = time.time()
        ft = locfeat.finetune_loc_feats(
            emap, ds, cam, enc, rend, train_ids, feats, iters=args.iters,
            lr=args.lr, lr_final=args.lr_final, min_alpha=args.min_alpha,
            loss=args.loss, beta1=args.adam_beta1, seed=args.seed)
        feats = ft["loc_feats"]
        tr = ft["trace"]
        k = max(1, len(tr) // 20)
        report.update(ft_sec=time.time() - t_ft,
                      loss_first=float(sum(tr[:k]) / k),
                      loss_last=float(sum(tr[-k:]) / k),
                      trace_every_200=[float(v) for v in tr[::200]])
        print(f"  fine-tune: loss {report['loss_first']:.5f} -> "
              f"{report['loss_last']:.5f} in {args.iters} iters, "
              f"{report['ft_sec']:.0f}s "
              f"({1000*report['ft_sec']/args.iters:.0f} ms/iter)")
        # Never-observed Gaussians stay at the zero vector: they cannot be
        # matched, and a random direction would invent correspondences.
        feats = feats.clone()
        feats[obs == 0] = 0.0

    ckpt["loc_feat_obs"] = obs.half().cpu()
    ckpt["loc_feat_source"] = "superpoint_v1"
    ckpt["loc_feat_iters"] = int(args.iters)
    ckpt["loc_feat_stride"] = int(args.stride)
    ckpt["loc_feat_holdout"] = int(args.holdout)
    ckpt["loc_feat_res"] = [int(args.fit_width), int(args.fit_height)]
    if args.pca_dim > 0:
        mean, basis, ev = locfeat.pca_fit(feats[obs > 0], args.pca_dim, seed=args.seed)
        red = locfeat.pca_apply(feats, mean, basis)
        red[obs == 0] = 0.0
        ckpt["loc_feats"] = red.half().cpu()
        ckpt["loc_feat_pca_mean"] = mean.cpu()
        ckpt["loc_feat_pca_basis"] = basis.cpu()
        ckpt["loc_feat_dim"] = int(args.pca_dim)
        ckpt["loc_feat_pca_ev"] = float(ev)
        report.update(explained_variance=float(ev))
        print(f"  PCA 256 -> {args.pca_dim}: {100*ev:.1f}% variance kept")
    else:
        ckpt["loc_feats"] = feats.half().cpu()
        ckpt["loc_feat_dim"] = int(feats.shape[1])
    _save(ckpt, out, args, N)
    d = ckpt["loc_feats"].shape[1]
    print(f"saved {out}: {N} x {d} float16 loc_feats "
          f"(+{N * d * 2 / 1e6:.0f} MB), {time.time() - t0:.0f}s total")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
