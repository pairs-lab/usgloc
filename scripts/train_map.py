#!/usr/bin/env python3
"""Train the evidential 3DGS map for one 7Scenes scene (Sec. IV-A).

Produces the ``--map`` checkpoint that ``scripts/run_7scenes.py`` consumes,
via four stages (see ``usgloc/mapping.py``):

1. init Gaussians from back-projected depth,
2. photometric 3DGS fit (gsplat, DefaultStrategy densification),
3. FastSAM+CLIP mask features distilled onto the Gaussians,
4. shared EDL head (train once, reuse across scenes via --head-ckpt) ->
   per-Gaussian beliefs/vacuities scaled by observation count,
then voxel x class instance ids and export.

Usage (chess, SfM pseudo-GT protocol):
    python3 scripts/train_map.py --root data/7scenes_source --scene chess \\
        --sfm-pose-dir data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes \\
        --out maps/chess.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usgloc import mapping
from usgloc.datasets import SevenScenes, TwelveScenes
from usgloc.evidential import EvidentialHead


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True, help="scene name, e.g. chess")
    ap.add_argument("--sfm-pose-dir", default=None,
                    help="pgt/sfm/7scenes dir (SfM pseudo-GT; Table I protocol)")
    ap.add_argument("--out", default=None,
                    help="output map path (default maps/<scene>.pt)")
    ap.add_argument("--head-ckpt", default="ckpts/edl_head.pt",
                    help="shared EDL head checkpoint: loaded if it exists, else "
                         "trained on this scene's mask features and saved here")
    ap.add_argument("--iters", type=int, default=7000, help="3DGS fit iterations")
    ap.add_argument("--fit-width", type=int, default=320)
    ap.add_argument("--fit-height", type=int, default=240)
    ap.add_argument("--fit-stride", type=int, default=10,
                    help="use every K-th train frame for the photometric fit")
    ap.add_argument("--init-intrinsics", choices=("sfm", "kinect"), default="sfm",
                    help="intrinsics used to back-project the depth maps: 'sfm' (the "
                         "per-scene SfM colour focal, i.e. depth registered to colour) or "
                         "'kinect' (the native 585 depth focal)")
    ap.add_argument("--init-stride", type=int, default=25,
                    help="every K-th frame feeds the depth initialization")
    ap.add_argument("--max-init-points", type=int, default=200_000)
    ap.add_argument("--distill-stride", type=int, default=20,
                    help="every K-th train frame gets FastSAM+CLIP distillation")
    ap.add_argument("--fastsam-weights", default="FastSAM-s.pt")
    ap.add_argument("--full-evidence-obs", type=float, default=4.0,
                    help="observations needed for full evidence (Sec. IV-A)")
    ap.add_argument("--instance-voxel", type=float, default=0.4,
                    help="voxel size (m) of the voxel x class instance ids")
    ap.add_argument("--head-mode", choices=("edl", "softmax"), default="edl",
                    help="'softmax': CE-trained head + entropy vacuity (the "
                         "ablation baseline); pair with a separate --head-ckpt "
                         "and --out")
    ap.add_argument("--label-margin", type=float, default=0.01,
                    help="CLIP top1-top2 margin filter for head pseudo-labels")
    # Reference-3DGS fidelity knobs for stage 3 (all default OFF = current recipe).
    ap.add_argument("--ssim-lambda", type=float, default=0.0,
                    help="D-SSIM term: loss = (1-lam)*L1 + lam*(1-SSIM); reference "
                         "3DGS uses 0.2 (0 = the pure-L1 recipe)")
    ap.add_argument("--sh-degree", type=int, default=0,
                    help="fit spherical-harmonic colour up to this degree (0 = plain "
                         "RGB). The exported map keeps the (N,3) 'colors' contract: "
                         "the DC band is baked, the view-dependent residual discarded")
    ap.add_argument("--sh-increase-every", type=int, default=1000)
    ap.add_argument("--lr-means", type=float, default=1.6e-4,
                    help="means learning rate before the scene-extent scaling")
    ap.add_argument("--lr-means-final", type=float, default=None,
                    help="exponentially decay the means lr to this value over the fit "
                         "(reference 3DGS: 1.6e-4 -> 1.6e-6)")
    ap.add_argument("--refine-start-iter", type=int, default=None)
    ap.add_argument("--refine-stop-iter", type=int, default=None,
                    help="default min(iters//2 + 500, 15000); reference 3DGS refines "
                         "until 15k of 30k")
    ap.add_argument("--refine-every", type=int, default=None)
    ap.add_argument("--reset-every", type=int, default=None)
    ap.add_argument("--grow-grad2d", type=float, default=None,
                    help="DefaultStrategy densification gradient threshold (gsplat "
                         "default 2e-4)")
    ap.add_argument("--grow-scale3d", type=float, default=None)
    ap.add_argument("--prune-opa", type=float, default=None)
    ap.add_argument("--views-uint8", action="store_true",
                    help="keep preloaded fit views in 8-bit (4x less host RAM) so "
                         "small --fit-stride values fit")
    ap.add_argument("--psnr-frames", type=int, default=0,
                    help="evaluate the fit PSNR on this many training views at the end")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else Path("maps") / f"{args.scene}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/5] loading {args.scene} train split "
          f"({'SfM pseudo-GT' if args.sfm_pose_dir else 'DSLAM GT'})")
    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train",
            sfm_pose_dir=args.sfm_pose_dir)
    cam_full = ds.scene_camera(ds.color_width, ds.color_height)
    cam_fit = ds.scene_camera(args.fit_width, args.fit_height)
    print(f"      {len(ds)} frames, focal {cam_full.fx:.2f}")

    print("[2/5] initializing from depth")
    pts, cols = mapping.backproject_init_points(
        ds, (ds.depth_camera() if args.init_intrinsics == "sfm"
            else ds.camera(ds.color_width, ds.color_height)),
        frame_stride=args.init_stride,
        max_points=args.max_init_points, seed=args.seed,
    )
    print(f"      {pts.shape[0]} seed points")

    print(f"[3/5] photometric 3DGS fit ({args.iters} iters @ "
          f"{args.fit_width}x{args.fit_height}, every {args.fit_stride}th frame)")
    strat = {k: v for k, v in (
        ("refine_start_iter", args.refine_start_iter),
        ("refine_stop_iter", args.refine_stop_iter),
        ("refine_every", args.refine_every),
        ("reset_every", args.reset_every),
        ("grow_grad2d", args.grow_grad2d),
        ("grow_scale3d", args.grow_scale3d),
        ("prune_opa", args.prune_opa),
    ) if v is not None}
    if strat:
        print(f"      densification overrides: {strat}")
    raw = mapping.fit_gaussians(
        ds, cam_fit, pts, cols, iters=args.iters,
        frame_stride=args.fit_stride, device=args.device, seed=args.seed,
        lr_means=args.lr_means, lr_means_final=args.lr_means_final,
        ssim_lambda=args.ssim_lambda, sh_degree=args.sh_degree,
        sh_increase_every=args.sh_increase_every,
        strategy_kwargs=strat or None, store_views_uint8=args.views_uint8,
        psnr_frames=args.psnr_frames,
    )
    n = raw["means"].shape[0]
    print(f"      fitted {n} gaussians")

    print(f"[4/5] FastSAM+CLIP distillation (every {args.distill_stride}th frame)")
    encoder = mapping.MaskClipEncoder(
        fastsam_weights=args.fastsam_weights, device=args.device
    )
    clip_feats, obs_count, bank = mapping.distill_clip_features(
        ds, cam_full, raw["means"], encoder, frame_stride=args.distill_stride,
    )
    torch.save(bank, out.parent / f"{args.scene.replace('/', '_')}_bank.pt")  # reused by train_head.py
    print(f"      observed {int((obs_count > 0).sum())}/{n} gaussians, "
          f"bank {bank.shape[0]} mask feats, dim {clip_feats.shape[1]}")

    print("[5/5] evidential head + opinions + export")
    C = len(mapping.INDOOR_VOCABULARY)
    head_path = Path(args.head_ckpt)
    if head_path.is_file():
        head = EvidentialHead(in_dim=clip_feats.shape[1], num_classes=C)
        head.load_state_dict(torch.load(head_path, map_location="cpu",
                                        weights_only=True))
        head = head.to(args.device).eval()
        print(f"      loaded shared EDL head from {head_path}")
    else:
        protos = encoder.text_prototypes(mapping.INDOOR_VOCABULARY)
        feats, labels = mapping.build_head_training_set(
            bank, clip_feats, obs_count, protos,
            margin=args.label_margin, seed=args.seed,
        )
        print(f"      training EDL head on {feats.shape[0]} confident feats "
              f"(mask + gaussian domains, C={C})")
        head = mapping.train_edl_head(
            feats, labels, num_classes=C, device=args.device, seed=args.seed,
            objective="ce" if args.head_mode == "softmax" else "edl",
        )
        head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), head_path)
        print(f"      saved shared EDL head to {head_path} "
              "(reused by later scenes and by --encoders)")

    beliefs, vacuities = mapping.gaussian_opinions(
        head, clip_feats, obs_count, full_evidence_obs=args.full_evidence_obs,
        mode=args.head_mode,
    )
    instance_ids = mapping.assign_instance_ids(
        raw["means"].cpu(), beliefs.argmax(dim=-1), voxel=args.instance_voxel,
    )
    mapping.export_map(out, raw, clip_feats, beliefs, vacuities, instance_ids,
                       obs_count=obs_count)
    print(f"saved {out}: {n} gaussians, {int(instance_ids.max()) + 1} instances, "
          f"median vacuity {float(vacuities.median()):.3f}, "
          f"{time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
