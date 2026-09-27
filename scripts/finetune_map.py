#!/usr/bin/env python3
"""Photometric fine-tune of an existing map at a higher resolution, with the
Gaussian set frozen in size and order.

The 640x480 fits from scratch lose fidelity on the large scenes because
densification over-grows them (Finding 7 of the report: twice the Gaussians,
5 dB less). Starting from the converged 320x240 map and refining only the
existing Gaussians' means, covariances, colours and opacities keeps the
geometry that registers well and adds the detail the matcher needs. Because
no Gaussian is added or removed, the map's distilled CLIP features, opinions,
vacuities and instance ids remain valid, so no re-distillation, head
re-application or encoder retraining is needed. Selection between the
original and the fine-tuned map stays train-side (select_map.py).

    python3 scripts/finetune_map.py --scene office --map maps/office_sel.pt \
        --out maps/office_ft.pt --iters 5000 --fit-width 640 --fit-height 480 --fit-stride 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usgloc import mapping  # noqa: E402
from usgloc.datasets import SevenScenes, TwelveScenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/7scenes_source")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir", default="data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--fit-width", type=int, default=640)
    ap.add_argument("--fit-height", type=int, default=480)
    ap.add_argument("--fit-stride", type=int, default=5)
    ap.add_argument("--lr-means", type=float, default=1.6e-5,
                    help="10x below the from-scratch value: the geometry is already converged")
    # Optional recipe knobs (all default to the adopted recipe's behaviour).
    ap.add_argument("--lr-quats", type=float, default=None,
                    help="override fit_gaussians' fixed quats lr (default 1e-3)")
    ap.add_argument("--lr-scales", type=float, default=None,
                    help="override fit_gaussians' fixed scales lr (default 5e-3)")
    ap.add_argument("--lr-opacities", type=float, default=None,
                    help="override fit_gaussians' fixed opacities lr (default 5e-2)")
    ap.add_argument("--lr-colors", type=float, default=None,
                    help="override fit_gaussians' fixed colors lr (default 2.5e-3)")
    ap.add_argument("--densify-window", nargs=3, type=int, default=None,
                    metavar=("START", "STOP", "EVERY"),
                    help="run gsplat's DefaultStrategy for iterations [START, STOP) every "
                         "EVERY steps and freeze the Gaussian set afterwards. This CHANGES "
                         "the number of Gaussians, which invalidates the map's distilled "
                         "semantics (re-run train_map.py's distillation, reapply_head.py "
                         "and train_encoders.py before using such a map end to end).")
    ap.add_argument("--views-uint8", action="store_true",
                    help="keep the preloaded training views in 8-bit on the host (exact for "
                         "8-bit sources, 4x less RAM) so small --fit-stride values fit")
    ap.add_argument("--depth-weight", type=float, default=0.0,
                    help="weight of an L1 depth term against the sensor depth maps "
                         "(0 = off, the photometric-only recipe). The correction lifts "
                         "keypoints through the RENDERED depth, which a photometric loss "
                         "alone never constrains.")
    ap.add_argument("--depth-scale", default="auto",
                    help="'auto' (median rendered/sensor depth ratio of the input map over "
                         "20 views) or a float: factor mapping the metric sensor depth into "
                         "the map's frame. Only used with --depth-weight > 0.")
    ap.add_argument("--depth-alpha-min", type=float, default=0.5,
                    help="minimum rendered alpha for a pixel to carry depth supervision")
    # Reference-3DGS fidelity knobs (all default OFF = the adopted recipe).
    ap.add_argument("--ssim-lambda", type=float, default=0.0,
                    help="weight of the D-SSIM term: loss = (1-lam)*L1 + lam*(1-SSIM); "
                         "reference 3DGS uses 0.2 (0 = the pure-L1 recipe)")
    ap.add_argument("--sh-degree", type=int, default=0,
                    help="fit spherical-harmonic colour up to this degree (0 = the "
                         "plain-RGB recipe). The exported map keeps the usgloc (N,3) "
                         "'colors' contract: the DC band is BAKED and the view-dependent "
                         "residual discarded (kept under 'sh_coeffs' with --save-sh)")
    ap.add_argument("--sh-increase-every", type=int, default=1000,
                    help="activate one more SH band every K iterations")
    ap.add_argument("--save-sh", action="store_true",
                    help="also store the raw SH coefficients in the checkpoint under "
                         "'sh_coeffs' (ignored by the loader; for measuring the bake)")
    ap.add_argument("--lr-means-final", type=float, default=None,
                    help="exponentially decay the means lr from --lr-means to this "
                         "value over the run (reference 3DGS: 1.6e-4 -> 1.6e-6)")
    ap.add_argument("--psnr-frames", type=int, default=0,
                    help="evaluate the fit PSNR on this many training views at the end")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train", sfm_pose_dir=args.sfm_pose_dir)
    cam_fit = ds.scene_camera(args.fit_width, args.fit_height)
    ckpt = torch.load(args.map, map_location="cpu", weights_only=False)
    n = ckpt["means"].shape[0]
    print(f"fine-tuning {args.map} ({n} gaussians) for {args.iters} iters @ "
          f"{args.fit_width}x{args.fit_height}, every {args.fit_stride}th frame, densification off")
    lr_over = {k: getattr(args, f"lr_{k}") for k in ("quats", "scales", "opacities", "colors")}
    lr_over = {k: v for k, v in lr_over.items() if v is not None}
    dens = args.densify_window is not None
    strat = None
    if dens:
        s, e, every = args.densify_window
        strat = dict(refine_start_iter=s, refine_stop_iter=e, refine_every=every)
        print(f"  densification ON for iters [{s}, {e}) every {every} -- the Gaussian set "
              f"may change, which invalidates the distilled semantics")
    if lr_over:
        print(f"  lr overrides: {lr_over}")
    t0 = time.time()
    raw = mapping.fit_gaussians(
        ds, cam_fit, ckpt["means"], ckpt["colors"], iters=args.iters,
        frame_stride=args.fit_stride, device=args.device, lr_means=args.lr_means,
        seed=args.seed,
        init_state={k: ckpt[k] for k in ("means", "covs", "colors", "opacities")},
        densify=dens,
        lr_overrides=lr_over or None,
        strategy_kwargs=strat,
        store_views_uint8=args.views_uint8,
        depth_weight=args.depth_weight,
        depth_scale=(None if str(args.depth_scale).lower() == "auto"
                     else float(args.depth_scale)),
        depth_alpha_min=args.depth_alpha_min,
        ssim_lambda=args.ssim_lambda,
        sh_degree=args.sh_degree,
        sh_increase_every=args.sh_increase_every,
        lr_means_final=args.lr_means_final,
        return_sh=args.save_sh,
        psnr_frames=args.psnr_frames,
    )
    n_out = raw["means"].shape[0]
    if not dens:
        assert n_out == n, "the Gaussian set must be unchanged"
    elif n_out != n:
        # The distilled per-Gaussian attributes no longer line up: drop them so
        # nothing downstream silently reads a stale tensor.
        for k in ("clip_feats", "beliefs", "vacuities", "instance_ids", "obs_count"):
            ckpt.pop(k, None)
        ckpt["semantics_invalid"] = True
        print(f"  gaussian count {n} -> {n_out}: distilled semantics DROPPED from the "
              f"checkpoint (needs train_map.py distillation + reapply_head.py + "
              f"train_encoders.py to be usable end to end)")
    for k in ("means", "covs", "colors", "opacities"):
        ckpt[k] = raw[k].cpu()
    if "sh" in raw:
        ckpt["sh_coeffs"] = raw["sh"].cpu()
    if "psnr_fit" in raw:
        ckpt["psnr_fit"] = raw["psnr_fit"]
        ckpt["psnr_fit_baked"] = raw["psnr_fit_baked"]
    torch.save(ckpt, args.out)
    print(f"saved {args.out}: {n_out} gaussians "
          f"({'unchanged set' if n_out == n else f'was {n}'}), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
