#!/usr/bin/env python3
"""Train the retrieval encoders on real data (Sec. IV-C).

Produces the ``--encoders`` checkpoint of ``scripts/run_7scenes.py``:
``{"descriptor", "aggregator", "edl_head"}`` state_dicts.

Adapts ``usgloc.synthetic.train_retrieval`` to real frames: positive pairs
are (FastSAM+CLIP+EDL instances of frames near a submap's anchor, the
submap's reference primitives), with the neighboring frames acting as the
natural noise variants of the query side.  Anchors/submaps are built exactly
as in run_7scenes.py (same --anchor-stride / --submap-radius) so training
matches evaluation.

Usage:
    python3 scripts/train_encoders.py --root data/7scenes_source --scene chess \\
        --sfm-pose-dir data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes \\
        --map maps/chess.pt --head-ckpt ckpts/edl_head.pt \\
        --out ckpts/chess_encoders.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usgloc.datasets import SevenScenes, TwelveScenes, load_gaussian_map
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator, info_nce_loss
from usgloc.evidential import EvidentialHead
from usgloc.pipeline import submap_reference_inputs
from usgloc.primitives import make_submaps
from usgloc.query import RealQueryEncoder


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", choices=("7scenes", "12scenes"), default="7scenes")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir", default=None)
    ap.add_argument("--map", required=True, help="map from scripts/train_map.py")
    ap.add_argument("--head-ckpt", default="ckpts/edl_head.pt",
                    help="shared EDL head state_dict from train_map.py")
    ap.add_argument("--out", default=None,
                    help="output checkpoint (default ckpts/<scene>_encoders.pt)")
    ap.add_argument("--anchor-stride", type=int, default=20)
    ap.add_argument("--submap-radius", type=float, default=3.0)
    ap.add_argument("--angle-weight", type=float, default=0.5,
                    help="metres-per-radian weight of the view-angle term in the "
                         "frame-to-anchor assignment")
    ap.add_argument("--pos-radius", type=float, default=0.35,
                    help="positives: max camera-centre distance (m) from the anchor")
    ap.add_argument("--pos-angle", type=float, default=35.0,
                    help="positives: max viewing-direction angle (deg) from the anchor")
    ap.add_argument("--variants-per-anchor", type=int, default=6,
                    help="covisible frames sampled per anchor as query variants")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pairs-per-iter", type=int, default=24,
                    help="submaps sampled per InfoNCE iteration (memory bound)")
    ap.add_argument("--fastsam-weights", default="FastSAM-s.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out) if args.out else Path("ckpts") / f"{args.scene}_encoders.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/4] loading scene + map")
    DS = SevenScenes if args.dataset == "7scenes" else TwelveScenes
    ds = DS(args.root, args.scene, split="train",
            sfm_pose_dir=args.sfm_pose_dir)
    emap = load_gaussian_map(args.map)
    anchor_ids = list(range(0, len(ds), args.anchor_stride))
    make_submaps(emap, [ds[i].pose_cw for i in anchor_ids],
                 radius=args.submap_radius,
                 camera=ds.scene_camera(ds.color_width, ds.color_height))
    # make_submaps drops empty anchors, so re-associate anchor frame ids by
    # matching poses (submaps preserve anchor order).
    kept = []
    k = 0
    for i in anchor_ids:
        if k < len(emap.submaps) and torch.allclose(
            emap.submaps[k].anchor_pose, ds[i].pose_cw
        ):
            kept.append(i)
            k += 1
    assert len(kept) == len(emap.submaps)
    print(f"      {len(emap.primitives)} primitives, {len(emap.submaps)} submaps")

    print("[2/4] building the FastSAM+CLIP+EDL front end")
    clip_dim = emap.primitives[0].clip_feat.shape[0]
    C = emap.primitives[0].belief.shape[0]
    head = EvidentialHead(in_dim=clip_dim, num_classes=C)
    head.load_state_dict(torch.load(args.head_ckpt, map_location="cpu",
                                    weights_only=True))
    front = RealQueryEncoder(edl_head=head, fastsam_weights=args.fastsam_weights,
                             device=str(device))

    # Voronoi-assigned positives: every train frame belongs to exactly ONE
    # nearest anchor (pose distance + angle penalty), and an anchor's query
    # variants are frames sampled from its own cell. Two failure modes drove
    # this design: fixed +-offset variants (near-identical views) collapse on
    # held-out test sequences (val 1.9 cm vs test 61 cm), and radius-overlap
    # covisibility positives are near-positives of NEIGHBORING submaps too,
    # so in-batch InfoNCE sees contradictory negatives and collapses to
    # chance (measured: final loss ln 24). Unique assignment keeps the
    # multi-sequence viewpoint diversity with one positive per frame, and
    # matches the retrieval task exactly (map a view to its nearest anchor).
    import math

    poses = [ds[j].pose_cw for j in range(len(ds))]
    centres = torch.stack([-T[:3, :3].T @ T[:3, 3] for T in poses])
    dirs = torch.stack([T[2, :3] for T in poses])  # optical axis in world
    a_centres = centres[torch.tensor(kept)]
    a_dirs = dirs[torch.tensor(kept)]
    # distance + angle penalty (radians scaled to metres): frame -> anchor
    d_all = torch.cdist(centres, a_centres)
    ang = torch.arccos(((dirs @ a_dirs.T)).clamp(-1, 1))
    score = d_all + args.angle_weight * ang
    assign = score.argmin(dim=1)
    ok = (d_all.gather(1, assign[:, None]).squeeze(1) < args.pos_radius) & (
        ang.gather(1, assign[:, None]).squeeze(1) < math.radians(args.pos_angle))
    g_var = torch.Generator().manual_seed(args.seed)
    cache = {}

    def encode_frame(j):
        if j not in cache:
            img = ds.load_image(j)
            bgr = (img.numpy() * 255).astype("uint8")[:, :, ::-1]
            cache[j] = front.encode(bgr)
        return cache[j]

    print(f"[3/4] encoding Voronoi-cell query variants "
          f"(<= {args.variants_per_anchor} per anchor) -- the slow part")
    pairs = []
    for n_done, (k, i) in enumerate(zip(range(len(emap.submaps)), kept)):
        sm = emap.submaps[k]
        if sm.primitive_idx.numel() < 2:
            continue
        cell = torch.nonzero((assign == k) & ok, as_tuple=False).squeeze(-1).tolist()
        if i in cell:
            cell.remove(i)
        picks = [cell[int(torch.randint(len(cell), (1,), generator=g_var))]
                 for _ in range(min(args.variants_per_anchor - 1, len(cell)))]
        variants = []
        for j in dict.fromkeys([i] + picks):
            qi = encode_frame(j)
            if len(qi) >= 2:
                variants.append([q for q in qi])
        if variants:
            pairs.append((sm, variants))
        if n_done % 10 == 0:
            print(f"      anchor {n_done + 1}/{len(emap.submaps)}: "
                  f"{len(pairs)} trainable pairs, {len(cache)} frames encoded")
    if len(pairs) < 4:
        raise SystemExit(f"only {len(pairs)} trainable submap/query pairs; "
                         "lower --anchor-stride or check the front end")
    print(f"      {len(pairs)} trainable pairs")

    print(f"[4/4] InfoNCE training ({args.iters} iters)")
    # Move everything used in the loss loop to the training device once.
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        v = getattr(emap, f)
        if v is not None:
            setattr(emap, f, v.to(device))
    for gprim in emap.primitives:
        import dataclasses
        for fld in dataclasses.fields(gprim):
            setattr(gprim, fld.name, getattr(gprim, fld.name).to(device))
    for sm, variants in pairs:
        sm.anchor_pose = sm.anchor_pose.to(device)
        for qi in variants:
            for q in qi:
                q.clip_feat = q.clip_feat.to(device)
                q.centroid = q.centroid.to(device)
                q.area = q.area.to(device)
                q.belief = q.belief.to(device)
                q.vacuity = q.vacuity.to(device)

    desc = InstanceDescriptorNet(clip_dim=clip_dim).to(device)
    agg = SceneAggregator().to(device)
    params = list(desc.parameters()) + list(agg.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    for it in range(args.iters):
        sel = torch.randperm(len(pairs), generator=g)[: args.pairs_per_iter]
        f2d, f3d = [], []
        for si in sel.tolist():
            sm, variants = pairs[si]
            qi = variants[int(torch.randint(len(variants), (1,), generator=g))]
            clip = torch.stack([q.clip_feat for q in qi])
            cen = torch.stack([q.centroid for q in qi])
            area = torch.stack([q.area for q in qi])
            vac_q = torch.stack([q.vacuity for q in qi])
            d2 = desc.encode_query(clip, cen, area)
            f2d.append(agg(d2[None], vac_q[None])[0])

            c3, m3, S3, v3 = submap_reference_inputs(emap, sm)
            d3 = desc.encode_reference(c3, m3, S3)
            f3d.append(agg(d3[None], v3[None])[0])
        loss = info_nce_loss(torch.stack(f2d), torch.stack(f3d))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(f"      iter {it:4d}: InfoNCE {float(loss):.4f}")

    torch.save(
        {
            "descriptor": desc.cpu().state_dict(),
            "aggregator": agg.cpu().state_dict(),
            "edl_head": head.cpu().state_dict(),
        },
        out,
    )
    print(f"saved {out} ({time.time() - t0:.0f}s total, final loss {float(loss):.4f})")


if __name__ == "__main__":
    main()
