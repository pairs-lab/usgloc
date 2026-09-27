#!/usr/bin/env python3
"""Where does retrieval put the geometrically-correct anchor?

For every query frame, rank all submap anchors by (a) the CLIP global
similarity that forms the shortlist and (b) the evidential gated similarity
that reranks inside it, then report the rank of the nearest anchor (and of the
nearest anchor within `--near-cm`).  Pure retrieval: no rendering, no matcher,
so it costs one CLIP forward per frame.

    python3 scripts/analysis/diag_anchor_rank.py --scene stairs --exclude-seq
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator  # noqa: E402
from usgloc.matching import retrieve_hypotheses                 # noqa: E402
from usgloc.pipeline import PipelineConfig, USGLocPipeline      # noqa: E402
from usgloc.primitives import make_submaps                      # noqa: E402
from usgloc.rendering import TorchSplatRenderer                 # noqa: E402

_spec = importlib.util.spec_from_file_location("r7", str(ROOT / "scripts/run_7scenes.py"))
r7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r7)

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def centre(T):
    return (-T[:3, :3].T @ T[:3, 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stairs")
    ap.add_argument("--map", default=None)
    ap.add_argument("--encoders", default=None)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--val-stride", type=int, default=7)
    ap.add_argument("--val-offset", type=int, default=3)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--exclude-anchors", type=int, default=40)
    ap.add_argument("--exclude-seq", action="store_true")
    ap.add_argument("--anchor-stride", type=int, default=20)
    ap.add_argument("--near-cm", type=float, default=40.0)
    ap.add_argument("--out", default="results/stairs_tread")
    args = ap.parse_args()

    scene = args.scene
    mp = args.map or f"maps/{scene}_ft.pt"
    enc = args.encoders or f"ckpts/{scene}_sel_encoders.pt"
    dev = "cuda"

    tr = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
    qs = tr if args.split == "val" else SevenScenes(
        "data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
    emap = load_gaussian_map(mp)
    anchor_ids = list(range(0, len(tr), args.anchor_stride))
    make_submaps(emap, [tr[i].pose_cw for i in anchor_ids], radius=3.0,
                 camera=tr.scene_camera(tr.color_width, tr.color_height))
    emap = r7.move_map(emap, dev)

    ck = torch.load(enc, map_location="cpu", weights_only=True)
    clip_dim = emap.primitives[0].clip_feat.shape[0]
    desc = InstanceDescriptorNet(clip_dim=clip_dim)
    desc.load_state_dict(ck["descriptor"])
    agg = SceneAggregator()
    agg.load_state_dict(ck["aggregator"])
    desc, agg = desc.to(dev).eval(), agg.to(dev).eval()
    front = r7.build_front_end(emap, clip_dim, ck, dev)

    pipe = USGLocPipeline(emap, TorchSplatRenderer(), tr.scene_camera(160, 120),
                          desc, agg, PipelineConfig())
    pipe.encode_map()

    kept_frames, k = [], 0
    for i in anchor_ids:
        if k < len(emap.submaps) and torch.allclose(
                emap.submaps[k].anchor_pose, tr[i].pose_cw.to(dev)):
            kept_frames.append(i)
            k += 1
    A = torch.stack([centre(tr[i].pose_cw) for i in kept_frames]).numpy()
    aseq = [tr[i].seq for i in kept_frames]
    gfeat = []
    for i in kept_frames:
        bgr = (tr.load_image(i).numpy() * 255).astype("uint8")[:, :, ::-1]
        gfeat.append(front.encode_global(bgr))
    G = torch.stack(gfeat)
    print(f"{len(kept_frames)} anchors")

    if args.split == "val":
        idx = list(range(args.val_offset, len(qs), args.val_stride))
    else:
        idx = list(range(len(qs)))
    if args.queries > 0:
        sel = torch.linspace(0, len(idx) - 1, args.queries).round().long().unique()
        idx = [idx[k] for k in sel.tolist()]

    rows = []
    for i in idx:
        rec = qs[i]
        img_full = qs.load_image(i)
        bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
        inst = r7.move_instances(front.encode(bgr), dev)
        if len(inst) < 2:
            continue
        c = centre(rec.pose_cw).numpy()
        d = np.linalg.norm(A - c, axis=1)
        mask = np.ones(len(A), dtype=bool)
        if args.split == "val":
            mask &= np.array([abs(f - i) >= args.exclude_anchors for f in kept_frames])
            if args.exclude_seq:
                mask &= np.array([s != rec.seq for s in aseq])
        qg = front.encode_global(bgr)
        sims = (G @ qg).detach().cpu().numpy()
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)
        # rank of the nearest visible anchor, and of the best anchor within near_cm
        dv = np.where(mask, d, np.inf)
        j_near = int(np.argmin(dv))
        rank_near = int(np.where(order == j_near)[0][0])
        near_set = np.where(dv <= args.near_cm / 100.0)[0]
        rank_best_near = (int(min(np.where(order == j)[0][0] for j in near_set))
                          if len(near_set) else None)
        # evidential gated rerank inside a full-scene ranking
        _, scene_desc, _ = pipe.encode_query(inst)
        hyps = retrieve_hypotheses(emap, scene_desc, top_k=len(A),
                                   candidates=np.where(mask)[0].tolist())
        gorder = [int(h.submap_index) for h in hyps]
        grank_near = gorder.index(j_near)
        grank_best_near = (min(gorder.index(j) for j in near_set) if len(near_set) else None)
        rows.append(dict(frame=int(i), seq=rec.seq, d_near_cm=float(dv[j_near]) * 100,
                         n_near=int(len(near_set)),
                         clip_rank_near=rank_near, clip_rank_best_near=rank_best_near,
                         gate_rank_near=grank_near, gate_rank_best_near=grank_best_near,
                         top8_min_cm=float(min(d[j] for j in order[:8])) * 100))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.split}" + ("_seqout" if args.exclude_seq else "")
    with open(out / f"rank_{scene}{tag}.json", "w") as f:
        json.dump(dict(args=vars(args), rows=rows), f, indent=1)

    def stat(key):
        v = [r[key] for r in rows if r[key] is not None]
        v = np.array(v)
        return (f"med {np.median(v):5.1f}  p75 {np.percentile(v, 75):5.1f}  "
                f"p90 {np.percentile(v, 90):5.1f}  "
                f"<10 {100 * (v < 10).mean():3.0f}%  <24 {100 * (v < 24).mean():3.0f}%  "
                f"<40 {100 * (v < 40).mean():3.0f}%")

    print(f"\n{scene} {tag}  n={len(rows)}  anchors={len(A)}")
    print(f"  nearest anchor dist cm   med "
          f"{np.median([r['d_near_cm'] for r in rows]):.1f}")
    print(f"  anchors within {args.near_cm:g}cm  med "
          f"{np.median([r['n_near'] for r in rows]):.0f}")
    print(f"  CLIP rank of nearest        {stat('clip_rank_near')}")
    print(f"  CLIP rank of best <{args.near_cm:g}cm     {stat('clip_rank_best_near')}")
    print(f"  gated rank of nearest       {stat('gate_rank_near')}")
    print(f"  gated rank of best <{args.near_cm:g}cm    {stat('gate_rank_best_near')}")
    t8 = np.array([r["top8_min_cm"] for r in rows])
    print(f"  closest of CLIP top-8 (cm)  med {np.median(t8):.1f}  "
          f">60cm {100 * (t8 > 60).mean():.0f}%")
    print(f"saved {out}/rank_{scene}{tag}.json")


if __name__ == "__main__":
    main()
