#!/usr/bin/env python3
"""Train the shared EDL head POOLED across scenes (plan step 2.5b).

A head trained on one scene's features does not transfer (measured: 8.9%
pseudo-label agreement on fire for a chess-only head, and the fire map's
evidence collapsed). This script pools BOTH head input domains over scenes:

* mask domain  -- FastSAM+CLIP crop embeddings of every --distill-stride-th
  train frame of EVERY listed scene (needs no map; banks are cached to
  maps/<scene>_bank.pt);
* gaussian domain -- mask-averaged per-Gaussian features from every listed
  scene whose map checkpoint already exists (unobserved Gaussians, u == 1
  exactly, are excluded; that proxy is head-independent).

Then trains once (mapping.train_edl_head), saves --out, and prints the
trained head's pseudo-label agreement per scene so the freeze decision
(plan 2.5b) is documented. After this, REBUILD any map/encoders that were
built with an older head.

Usage:
    python3 scripts/train_head.py --root data/7scenes_source \\
        --scenes chess fire heads office pumpkin redkitchen stairs \\
        --out ckpts/edl_head.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usgloc import mapping
from usgloc.datasets import SevenScenes
from usgloc.evidential import evidence_to_subjective_opinion


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--maps-dir", default="maps")
    ap.add_argument("--out", default="ckpts/edl_head.pt")
    ap.add_argument("--distill-stride", type=int, default=20)
    ap.add_argument("--label-margin", type=float, default=0.01)
    ap.add_argument("--max-gauss-per-scene", type=int, default=20_000)
    ap.add_argument("--fastsam-weights", default="FastSAM-s.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = mapping.MaskClipEncoder(fastsam_weights=args.fastsam_weights,
                                  device=args.device)
    protos = enc.text_prototypes(mapping.INDOOR_VOCABULARY)
    g = torch.Generator().manual_seed(args.seed)

    feats_all, labels_all, per_scene = [], [], {}
    for scene in args.scenes:
        # -- mask domain (cached) --
        bank_path = Path(args.maps_dir) / f"{scene}_bank.pt"
        if bank_path.is_file():
            bank = torch.load(bank_path, weights_only=True)
        else:
            ds = SevenScenes(args.root, scene, split="train")
            feats = []
            for i in range(0, len(ds), args.distill_stride):
                img = ds.load_image(i)
                bgr = (img.numpy() * 255).astype("uint8")[:, :, ::-1]
                feats.extend(f for _, f in enc.encode_frame(bgr))
            bank = torch.stack(feats)
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bank, bank_path)
        lab, keep = mapping.pseudo_labels(bank, protos, margin=args.label_margin)
        feats_all.append(bank[keep])
        labels_all.append(lab[keep])
        n_mask = int(keep.sum())

        # -- gaussian domain (only if this scene's map exists) --
        n_gauss = 0
        map_path = Path(args.maps_dir) / f"{scene}.pt"
        if map_path.is_file():
            ckpt = torch.load(map_path, weights_only=True)
            observed = ckpt["vacuities"] < 0.9999
            gf = ckpt["clip_feats"][observed]
            if gf.shape[0] > args.max_gauss_per_scene:
                sel = torch.randperm(gf.shape[0], generator=g)[: args.max_gauss_per_scene]
                gf = gf[sel]
            glab, gkeep = mapping.pseudo_labels(gf, protos, margin=args.label_margin)
            feats_all.append(gf[gkeep])
            labels_all.append(glab[gkeep])
            n_gauss = int(gkeep.sum())
        per_scene[scene] = (bank, lab, keep)
        print(f"  {scene}: {n_mask} mask + {n_gauss} gaussian confident feats"
              f"{'' if map_path.is_file() else ' (no map yet)'}")

    feats = torch.cat(feats_all)
    labels = torch.cat(labels_all)
    print(f"training pooled head on {feats.shape[0]} feats, "
          f"C={len(mapping.INDOOR_VOCABULARY)}")
    head = mapping.train_edl_head(feats, labels,
                                  num_classes=len(mapping.INDOOR_VOCABULARY),
                                  device=args.device, seed=args.seed)
    torch.save(head.state_dict(), out)
    print(f"saved {out}")

    print("per-scene agreement of the trained head (confident mask feats):")
    dev = args.device
    with torch.no_grad():
        for scene, (bank, lab, keep) in per_scene.items():
            e = head(bank[keep].to(dev)).cpu()
            agree = float((e.argmax(-1) == lab[keep]).float().mean())
            _, _, u_conf = evidence_to_subjective_opinion(e)
            e_a = head(bank[~keep].to(dev)).cpu() if int((~keep).sum()) else None
            if e_a is not None:
                _, _, u_amb = evidence_to_subjective_opinion(e_a)
                order = "holds" if float(u_amb.mean()) > float(u_conf.mean()) else "FAILS"
            else:
                order = "n/a"
            print(f"  {scene}: agreement {agree:.1%}, vacuity ordering {order}")


if __name__ == "__main__":
    main()
