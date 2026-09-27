#!/usr/bin/env python3
"""Re-apply an evidential head to already-built maps, without re-fitting.

A map stores the distilled per-Gaussian CLIP features and observation counts,
so swapping the evidential head only requires recomputing opinions and the
instance grouping (whose class term is the belief argmax). This is the cheap
repair path when a head turns out to be off-distribution for a map -- the
failure mode of Finding 2: a shared head must be trained on the same feature
distribution it will score, and a map re-fit at a different resolution moves
that distribution.

    python3 scripts/reapply_head.py --head ckpts/edl_head_hq.pt \
        --maps maps/chess_hq.pt maps/fire_hq.pt ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usgloc import mapping  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="evidential head checkpoint")
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--full-evidence-obs", type=float, default=4.0)
    ap.add_argument("--instance-voxel", type=float, default=0.4)
    ap.add_argument("--head-mode", choices=("edl", "softmax"), default="edl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    state = torch.load(args.head, map_location="cpu", weights_only=True)
    state = state["head"] if "head" in state else state
    in_dim = state["net.0.weight"].shape[1] if "net.0.weight" in state else 512
    head = mapping.EvidentialHead(
        in_dim=in_dim, num_classes=len(mapping.INDOOR_VOCABULARY))
    head.load_state_dict(state)
    head = head.to(args.device).eval()

    for mp in args.maps:
        p = Path(mp)
        if not p.is_file():
            print(f"  SKIP {mp} (missing)")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=True)
        if "obs_count" not in ck:
            print(f"  SKIP {mp} (no obs_count; re-run distillation)")
            continue
        feats = ck["clip_feats"].to(args.device)
        obs = ck["obs_count"].to(args.device)
        old_u = float(ck["vacuities"].median())

        beliefs, vac = mapping.gaussian_opinions(
            head, feats, obs, full_evidence_obs=args.full_evidence_obs,
            mode=args.head_mode)
        inst = mapping.assign_instance_ids(
            ck["means"].to(args.device),
            beliefs.argmax(dim=-1).to(args.device),
            voxel=args.instance_voxel)

        ck["beliefs"] = beliefs.detach().cpu().float()
        ck["vacuities"] = vac.detach().cpu().float()
        ck["instance_ids"] = inst.detach().cpu().long()
        torch.save(ck, p)
        n_inst = int(inst.unique().numel())
        print(f"  {p.name:24s} median vacuity {old_u:.3f} -> "
              f"{float(vac.median()):.3f} | {n_inst} instances | "
              f"observed {int((obs > 0).sum())}/{obs.numel()}")


if __name__ == "__main__":
    main()
