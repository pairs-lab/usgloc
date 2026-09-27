#!/usr/bin/env python3
"""3DGS fit with FULL control of the densification schedule -- from scratch or
warm-started from an existing map -- exported as a map the floor experiment can
score.

scripts/finetune_map.py can only switch densification on for a window
(--densify-window START STOP EVERY); it exposes no DefaultStrategy thresholds,
and scripts/train_map.py always runs the distillation stages.  This script is
the measurement-side equivalent that exposes both:

  * ``--init-map`` warm-starts from an existing map's Gaussians (the _r1
    recipe's starting point); without it the Gaussians are initialised from
    the back-projected depth, i.e. train_map.py stages 1-2 from scratch.
  * ``--densify/--no-densify`` plus refine_start/stop/every, reset_every,
    grow_grad2d, grow_scale3d, prune_opa: gsplat's DefaultStrategy knobs.
  * ``--donor`` copies a donor map's distilled semantics onto the resulting
    Gaussian set by nearest neighbour, so the correction floor is measurable
    without re-running the FastSAM+CLIP distillation.  A map patched that way
    is a MEASUREMENT SCAFFOLD: its semantics belong to the donor's Gaussian
    set, so before end-to-end use the distillation (train_map.py stages 4-5),
    reapply_head.py and train_encoders.py must be re-run.

    # (b) the stabilised recipe + a densification window, from the current map
    python3 scripts/analysis/gx_fit.py --scene chess --out maps/chess_gxd1.pt \
        --init-map maps/chess_ft.pt --donor maps/chess_ft.pt \
        --iters 20000 --fit-stride 1 --ssim-lambda 0.2 \
        --lr-means 1.6e-5 --lr-means-final 1.6e-7 \
        --densify --refine-start-iter 500 --refine-stop-iter 10000 \
        --views-uint8 --psnr-frames 20

    # (c) reference-3DGS from scratch at the correction resolution
    python3 scripts/analysis/gx_fit.py --scene chess --out maps/chess_gxsc.pt \
        --donor maps/chess_ft.pt --iters 30000 --fit-stride 1 \
        --ssim-lambda 0.2 --lr-means 1.6e-4 --lr-means-final 1.6e-6 \
        --densify --refine-stop-iter 15000 --views-uint8 --psnr-frames 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from usgloc import mapping                                  # noqa: E402
from usgloc.datasets import SevenScenes, TwelveScenes       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/7scenes_source")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir",
                    default="data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-map", default=None,
                    help="warm start from this map's Gaussians instead of the "
                         "back-projected depth seeds")
    ap.add_argument("--donor", default=None,
                    help="map whose distilled semantics are copied by nearest "
                         "neighbour when the Gaussian count changes (scaffold)")
    ap.add_argument("--always-transfer", action="store_true",
                    help="transfer from --donor even when the count is unchanged")
    ap.add_argument("--half-clip-over", type=int, default=800_000,
                    help="store the transferred clip_feats as float16 above this "
                         "Gaussian count (disk economy; unused by the corrector)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--fit-width", type=int, default=640)
    ap.add_argument("--fit-height", type=int, default=480)
    ap.add_argument("--fit-stride", type=int, default=1)
    ap.add_argument("--init-stride", type=int, default=25)
    ap.add_argument("--max-init-points", type=int, default=200_000)
    ap.add_argument("--init-intrinsics", choices=("sfm", "kinect"), default="sfm")
    ap.add_argument("--lr-means", type=float, default=1.6e-5)
    ap.add_argument("--lr-means-final", type=float, default=None)
    ap.add_argument("--lr-quats", type=float, default=None)
    ap.add_argument("--lr-scales", type=float, default=None)
    ap.add_argument("--lr-opacities", type=float, default=None)
    ap.add_argument("--lr-colors", type=float, default=None)
    ap.add_argument("--ssim-lambda", type=float, default=0.0)
    ap.add_argument("--densify", action="store_true")
    ap.add_argument("--refine-start-iter", type=int, default=None)
    ap.add_argument("--refine-stop-iter", type=int, default=None)
    ap.add_argument("--refine-every", type=int, default=None)
    ap.add_argument("--reset-every", type=int, default=None)
    ap.add_argument("--grow-grad2d", type=float, default=None,
                    help="DefaultStrategy densification gradient threshold "
                         "(gsplat default 2e-4; lower = more splits/clones)")
    ap.add_argument("--grow-scale3d", type=float, default=None)
    ap.add_argument("--prune-opa", type=float, default=None)
    ap.add_argument("--grow-scale2d", type=float, default=None)
    ap.add_argument("--pause-refine-after-reset", type=int, default=None)
    ap.add_argument("--depth-weight", type=float, default=0.0)
    ap.add_argument("--depth-scale", default="auto")
    ap.add_argument("--views-uint8", action="store_true")
    ap.add_argument("--psnr-frames", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train", sfm_pose_dir=args.sfm_pose_dir)
    cam_fit = ds.scene_camera(args.fit_width, args.fit_height)

    init_state = None
    ckpt_in = None
    if args.init_map:
        ckpt_in = torch.load(args.init_map, map_location="cpu", weights_only=False)
        init_state = {k: ckpt_in[k] for k in ("means", "covs", "colors", "opacities")}
        pts, cols = ckpt_in["means"], ckpt_in["colors"]
        print(f"[1/3] warm start from {args.init_map}: {pts.shape[0]} gaussians")
    else:
        print(f"[1/3] {args.scene}: {len(ds)} train frames; init from depth "
              f"(every {args.init_stride}th frame, {args.init_intrinsics} intrinsics)")
        pts, cols = mapping.backproject_init_points(
            ds,
            (ds.depth_camera() if args.init_intrinsics == "sfm"
             else ds.camera(ds.color_width, ds.color_height)),
            frame_stride=args.init_stride, max_points=args.max_init_points,
            seed=args.seed,
        )
        print(f"      {pts.shape[0]} seed points")
    n_in = int(pts.shape[0])

    strat = {k: v for k, v in (
        ("refine_start_iter", args.refine_start_iter),
        ("refine_stop_iter", args.refine_stop_iter),
        ("refine_every", args.refine_every),
        ("reset_every", args.reset_every),
        ("grow_grad2d", args.grow_grad2d),
        ("grow_scale3d", args.grow_scale3d),
        ("prune_opa", args.prune_opa),
        ("grow_scale2d", args.grow_scale2d),
        ("pause_refine_after_reset", args.pause_refine_after_reset),
    ) if v is not None}
    lr_over = {k: getattr(args, f"lr_{k}")
               for k in ("quats", "scales", "opacities", "colors")}
    lr_over = {k: v for k, v in lr_over.items() if v is not None}
    print(f"[2/3] fit {args.iters} iters @ {args.fit_width}x{args.fit_height}, every "
          f"{args.fit_stride}th frame, densify={args.densify} {strat} "
          f"lr_overrides={lr_over}")
    raw = mapping.fit_gaussians(
        ds, cam_fit, pts, cols, iters=args.iters, frame_stride=args.fit_stride,
        device=args.device, seed=args.seed, lr_means=args.lr_means,
        lr_means_final=args.lr_means_final, ssim_lambda=args.ssim_lambda,
        init_state=init_state, densify=args.densify,
        strategy_kwargs=strat or None, lr_overrides=lr_over or None,
        store_views_uint8=args.views_uint8, psnr_frames=args.psnr_frames,
        depth_weight=args.depth_weight,
        depth_scale=(None if str(args.depth_scale).lower() == "auto"
                     else float(args.depth_scale)),
    )
    n = int(raw["means"].shape[0])
    fit_s = time.time() - t0
    print(f"      fitted {n} gaussians (was {n_in}) in {fit_s:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = dict(ckpt_in) if ckpt_in is not None else {}
    for k in ("means", "covs", "colors", "opacities"):
        ckpt[k] = raw[k].detach().cpu().float()
    for k in ("psnr_fit", "psnr_fit_baked"):
        if k in raw:
            ckpt[k] = float(raw[k])
    ckpt["fit_seconds"] = fit_s
    ckpt["n_gaussians_in"] = n_in
    changed = n != n_in or ckpt_in is None
    if changed:
        for k in ("clip_feats", "beliefs", "vacuities", "instance_ids", "obs_count"):
            ckpt.pop(k, None)
        ckpt["semantics_invalid"] = True
    torch.save(ckpt, out)
    del raw, ckpt
    torch.cuda.empty_cache()
    print(f"[3/3] saved {out} ({'semantics dropped' if changed else 'semantics kept'})")
    info = None
    if args.donor and (changed or args.always_transfer):
        from gx_nn_transfer import transfer
        info = transfer(out, Path(args.donor), args.half_clip_over)
        print(f"      nn-transferred semantics from {args.donor}: {info}")
    print(f"done: {n} gaussians, fit {fit_s:.0f}s, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
