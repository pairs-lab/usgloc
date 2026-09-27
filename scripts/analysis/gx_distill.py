#!/usr/bin/env python3
"""Re-run the distillation stages (train_map.py [4/5] and [5/5]) on an EXISTING
map whose Gaussian set changed, and report what each stage costs.

A densified re-fit keeps the map's means but invalidates its distilled
``clip_feats`` / ``beliefs`` / ``vacuities`` / ``instance_ids`` / ``obs_count``
(and therefore ``ckpts/<scene>_sel_encoders.pt``).  train_map.py can only redo
those stages by redoing the photometric fit as well, which would throw the new
Gaussian set away; this script does stages 4-5 alone, in place, so the true
per-scene rollout cost of a Gaussian-set change can be measured:

    python3 scripts/analysis/gx_distill.py --scene chess \
        --map maps/chess_gxd1soft.pt --head-ckpt ckpts/edl_head_sel.pt

Afterwards ``scripts/train_encoders.py`` still has to be re-run for that scene
(the retrieval encoders are trained on the map's instances).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from usgloc import mapping                                   # noqa: E402
from usgloc.datasets import SevenScenes, TwelveScenes        # noqa: E402
from usgloc.evidential import EvidentialHead                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/7scenes_source")
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir",
                    default="data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes")
    ap.add_argument("--map", required=True, help="map to re-distill IN PLACE")
    ap.add_argument("--out", default=None, help="write here instead of --map")
    ap.add_argument("--head-ckpt", default="ckpts/edl_head_sel.pt",
                    help="shared EDL head, reused (never retrained here)")
    ap.add_argument("--distill-stride", type=int, default=20)
    ap.add_argument("--fastsam-weights", default="FastSAM-s.pt")
    ap.add_argument("--full-evidence-obs", type=float, default=4.0)
    ap.add_argument("--instance-voxel", type=float, default=0.4)
    ap.add_argument("--head-mode", choices=("edl", "softmax"), default="edl")
    ap.add_argument("--save-bank", action="store_true",
                    help="also write maps/<scene>_bank.pt (train_map.py does)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train", sfm_pose_dir=args.sfm_pose_dir)
    cam_full = ds.scene_camera(ds.color_width, ds.color_height)
    ckpt = torch.load(args.map, map_location="cpu", weights_only=False)
    means = ckpt["means"]
    n = int(means.shape[0])
    print(f"re-distilling {args.map}: {n} gaussians, {len(ds)} train frames, "
          f"every {args.distill_stride}th frame", flush=True)

    t_enc = time.time()
    encoder = mapping.MaskClipEncoder(fastsam_weights=args.fastsam_weights,
                                      device=args.device)
    t_enc = time.time() - t_enc
    t_dist = time.time()
    clip_feats, obs_count, bank = mapping.distill_clip_features(
        ds, cam_full, means.to(args.device), encoder,
        frame_stride=args.distill_stride,
    )
    t_dist = time.time() - t_dist
    print(f"  [4/5] distillation {t_dist:.0f}s (encoder load {t_enc:.0f}s): "
          f"observed {int((obs_count > 0).sum())}/{n}, bank {bank.shape[0]}",
          flush=True)
    if args.save_bank:
        torch.save(bank, Path(args.map).parent /
                   f"{args.scene.replace('/', '_')}_bank.pt")

    t_head = time.time()
    head_path = Path(args.head_ckpt)
    if not head_path.is_file():
        raise SystemExit(f"{head_path} missing: this script never trains the "
                         "shared head, it only re-applies it")
    head = EvidentialHead(in_dim=clip_feats.shape[1],
                          num_classes=len(mapping.INDOOR_VOCABULARY))
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head = head.to(args.device).eval()
    beliefs, vacuities = mapping.gaussian_opinions(
        head, clip_feats, obs_count, full_evidence_obs=args.full_evidence_obs,
        mode=args.head_mode)
    instance_ids = mapping.assign_instance_ids(
        means.cpu(), beliefs.argmax(dim=-1).cpu(), voxel=args.instance_voxel)
    t_head = time.time() - t_head
    print(f"  [5/5] opinions + instances {t_head:.0f}s: "
          f"{int(instance_ids.max()) + 1} instances, "
          f"median vacuity {float(vacuities.median()):.3f}", flush=True)

    ckpt["clip_feats"] = clip_feats.detach().cpu().float()
    ckpt["beliefs"] = beliefs.detach().cpu().float()
    ckpt["vacuities"] = vacuities.detach().cpu().float()
    ckpt["instance_ids"] = instance_ids.detach().cpu().long()
    ckpt["obs_count"] = obs_count.detach().cpu().float()
    ckpt.pop("semantics_nn_transferred", None)
    ckpt.pop("semantics_donor", None)
    ckpt.pop("semantics_invalid", None)
    ckpt["redistilled_seconds"] = time.time() - t0
    out = Path(args.out or args.map)
    torch.save(ckpt, out)
    print(f"saved {out}: stage4 {t_dist:.0f}s + stage5 {t_head:.0f}s, "
          f"total {time.time() - t0:.0f}s "
          f"(scripts/train_encoders.py still has to be re-run)", flush=True)


if __name__ == "__main__":
    main()
