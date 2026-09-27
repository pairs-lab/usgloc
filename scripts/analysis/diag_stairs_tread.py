#!/usr/bin/env python3
"""Where does the Stairs wrong-tread error come from?

Runs the FULL pipeline on TRAIN-side frames with the pipeline's debug capture
on, and for every query classifies the failure:

  (i)   never proposed    - no corrected candidate pose lands near the GT
  (ii)  out-weighted      - a good candidate existed but a worse one was
                            reported (pruned, deduped, or lost the weight)
  (iii) condensation loss - the top-weight component was good but the
                            condensed/reported pose is not

Anchor difficulty is controlled two ways, so the train-side protocol can be
made as hard as the test split (where the query sequence is unseen):
  --exclude-anchors N   hide anchors within N train frames of the query
                        (run_7scenes.py --val-exclude-anchors)
  --exclude-seq         hide every anchor from the query's OWN sequence
                        (leave-one-sequence-out; the test-split condition)

Writes <out>/diag_<scene><tag>.json and prints the breakdown.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from usgloc.correction import EvidentialMatchCorrector          # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator  # noqa: E402
from usgloc.pipeline import PipelineConfig, USGLocPipeline      # noqa: E402
from usgloc.primitives import make_submaps                      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402

_spec = importlib.util.spec_from_file_location("r7", str(ROOT / "scripts/run_7scenes.py"))
r7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r7)

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def t_err_cm(Ta, Tb):
    """Camera-centre distance between two camera-from-world poses, cm."""
    Ta, Tb = Ta.double(), Tb.double()
    ca = -Ta[:3, :3].T @ Ta[:3, 3]
    cb = -Tb[:3, :3].T @ Tb[:3, 3]
    return float((ca - cb).norm()) * 100


def wait_gpu(min_free_mb=4000, tries=240):
    import subprocess
    for _ in range(tries):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()
        if out and int(out[0]) >= min_free_mb:
            return
        time.sleep(20)
    raise RuntimeError("GPU never freed enough memory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stairs")
    ap.add_argument("--map", default=None, help="default maps/<scene>_ft.pt")
    ap.add_argument("--encoders", default=None, help="default ckpts/<scene>_sel_encoders.pt")
    ap.add_argument("--val-stride", type=int, default=7)
    ap.add_argument("--val-offset", type=int, default=3)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--exclude-anchors", type=int, default=40)
    ap.add_argument("--exclude-seq", action="store_true",
                    help="also hide every anchor from the query's own sequence")
    ap.add_argument("--anchor-stride", type=int, default=20)
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--corr-rounds", type=int, default=5)
    ap.add_argument("--support-power", type=float, default=1.0)
    ap.add_argument("--escalate-inliers", type=int, default=0)
    ap.add_argument("--escalate-ratio", type=float, default=0.0)
    ap.add_argument("--escalate-tiers", type=int, default=3)
    ap.add_argument("--escalate-batch", type=int, default=8)
    ap.add_argument("--escalate-order", default="shortlist",
                    choices=["gated", "shortlist", "spread"])
    ap.add_argument("--escalate-min-dist", type=float, default=0.0)
    ap.add_argument("--good-cm", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/stairs_tread")
    args = ap.parse_args()

    scene = args.scene
    mp = args.map or f"maps/{scene}_ft.pt"
    enc = args.encoders or f"ckpts/{scene}_sel_encoders.pt"
    dev = "cuda"
    wait_gpu()

    tr = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
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

    cam = tr.scene_camera(160, 120)
    corr_cam = tr.scene_camera(640, 480)
    rend = GsplatRenderer()
    corr = EvidentialMatchCorrector(emap, rend, corr_cam, gamma=3.0,
                                    rounds=args.corr_rounds)
    cfg = PipelineConfig(top_k=args.top_k, refine_iters=60, gamma=3.0,
                         corr_refine_iters=60, support_power=args.support_power,
                         escalate_inliers=args.escalate_inliers,
                         escalate_ratio=args.escalate_ratio,
                         escalate_tiers=args.escalate_tiers,
                         escalate_batch=args.escalate_batch,
                         escalate_order=args.escalate_order,
                         escalate_min_dist=args.escalate_min_dist)
    pipe = USGLocPipeline(emap, rend, cam, desc, agg, cfg, corrector=corr)
    pipe.debug = True
    pipe.encode_map()

    kept_frames, k = [], 0
    for i in anchor_ids:
        if k < len(emap.submaps) and torch.allclose(
                emap.submaps[k].anchor_pose, tr[i].pose_cw.to(dev)):
            kept_frames.append(i)
            k += 1
    assert len(kept_frames) == len(emap.submaps)
    gfeat = []
    for i in kept_frames:
        bgr = (tr.load_image(i).numpy() * 255).astype("uint8")[:, :, ::-1]
        gfeat.append(front.encode_global(bgr))
    anchor_globals = torch.stack(gfeat)
    anchor_seq = [tr[i].seq for i in kept_frames]

    idx = list(range(args.val_offset, len(tr), args.val_stride))
    if args.queries > 0:
        sel = torch.linspace(0, len(idx) - 1, args.queries).round().long().unique()
        idx = [idx[k] for k in sel.tolist()]

    rows = []
    for i in idx:
        rec = tr[i]
        img_full = tr.load_image(i)
        img = r7.resize_image(img_full, 160, 120).to(dev)
        gf = r7.resize_image(img_full, 640, 480).to(dev)
        gray = (0.299 * gf[..., 0] + 0.587 * gf[..., 1] + 0.114 * gf[..., 2]).clamp(0, 1)
        bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
        inst = r7.move_instances(front.encode(bgr), dev)
        if len(inst) < 2:
            continue
        qg = front.encode_global(bgr)
        sims = anchor_globals @ qg
        mask = torch.tensor([abs(f - i) >= args.exclude_anchors for f in kept_frames],
                            device=sims.device)
        if args.exclude_seq:
            mask &= torch.tensor([s != rec.seq for s in anchor_seq], device=sims.device)
        sims = torch.where(mask, sims, torch.full_like(sims, -float("inf")))
        cand = sims.topk(min(args.shortlist, int(mask.sum()))).indices.tolist()
        T_gt = rec.pose_cw.to(dev)
        for attempt in range(20):
            try:
                res = pipe.localize(img, inst, candidates=cand, gray_full=gray)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM on {i}, waiting", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError("persistent OOM")
        dbg = pipe.last_debug
        snaps = [dict(anchor_cm=t_err_cm(s["T0"], T_gt.cpu()),
                      snap_cm=t_err_cm(s["T"], T_gt.cpu()) if s["success"] else None,
                      ok=s["success"], inliers=s["inliers"],
                      n_matches=s["n_matches"],
                      psnr=s.get("psnr"), psnr_anchor=s.get("psnr_anchor"),
                      verify_failed=s.get("verify_failed", False))
                 for s in dbg["snaps"]]
        kept = [dict(cm=t_err_cm(c["T"], T_gt.cpu()), psnr=c["psnr"],
                     inliers=c["inliers"], support=c["support"],
                     weight=c["weight"], snapped=c["snapped"])
                for c in dbg["kept"]]
        rep_cm = t_err_cm(res.condensed.T_mode, T_gt)
        rows.append(dict(frame=int(i), seq=rec.seq, frame_id=int(rec.frame_id),
                         reported_cm=rep_cm, snaps=snaps, kept=kept,
                         n_cand=len(cand)))
        best_snap = min([s["snap_cm"] for s in snaps if s["ok"]], default=None)
        best_kept = min([c["cm"] for c in kept], default=None)
        print(f"  {rec.seq}/{rec.frame_id:06d} rep {rep_cm:7.2f}  "
              f"best_snap {('%.2f' % best_snap) if best_snap is not None else '  --':>7}  "
              f"best_kept {('%.2f' % best_kept) if best_kept is not None else '  --':>7}  "
              f"anchors {' '.join('%.0f' % s['anchor_cm'] for s in snaps)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("_seqout" if args.exclude_seq else f"_ex{args.exclude_anchors}")
    with open(out / f"diag_{scene}{tag}.json", "w") as f:
        json.dump(dict(args=vars(args), rows=rows), f, indent=1)

    report(rows, args.good_cm)
    print(f"saved {out}/diag_{scene}{tag}.json")


def report(rows, good_cm):
    rep = np.array([r["reported_cm"] for r in rows])
    print(f"\nn={len(rows)}  reported median {np.median(rep):.2f} cm  "
          f">{good_cm:g}cm {100 * (rep > good_cm).mean():.0f}%")
    bad = [r for r in rows if r["reported_cm"] > good_cm]
    if not bad:
        print("no failures at this threshold -- protocol is too easy to diagnose on")
        return
    cats = dict(never_proposed=0, anchor_never_close=0, outweighted_pruned=0,
                outweighted_kept=0, condensation=0)
    for r in bad:
        snap_ok = [s["snap_cm"] for s in r["snaps"] if s["ok"]]
        kept_cm = [c["cm"] for c in r["kept"]]
        anc = [s["anchor_cm"] for s in r["snaps"]]
        if not snap_ok or min(snap_ok) > good_cm:
            cats["never_proposed"] += 1
            if not anc or min(anc) > 40.0:
                cats["anchor_never_close"] += 1
        elif not kept_cm or min(kept_cm) > good_cm:
            cats["outweighted_pruned"] += 1
        else:
            top = max(range(len(r["kept"])), key=lambda j: r["kept"][j]["weight"])
            if r["kept"][top]["cm"] > good_cm:
                cats["outweighted_kept"] += 1
            else:
                cats["condensation"] += 1
    n = len(bad)
    print(f"failures n={n}")
    print(f"  (i)   never proposed       {cats['never_proposed']:3d}  "
          f"({100 * cats['never_proposed'] / n:3.0f}%)   "
          f"of which no anchor within 40 cm: {cats['anchor_never_close']}")
    print(f"  (ii)a proposed, then pruned {cats['outweighted_pruned']:3d}  "
          f"({100 * cats['outweighted_pruned'] / n:3.0f}%)")
    print(f"  (ii)b kept but out-weighted {cats['outweighted_kept']:3d}  "
          f"({100 * cats['outweighted_kept'] / n:3.0f}%)")
    print(f"  (iii) condensation loss     {cats['condensation']:3d}  "
          f"({100 * cats['condensation'] / n:3.0f}%)")
    orc_snap = np.array([min([s["snap_cm"] for s in r["snaps"] if s["ok"]], default=np.nan)
                         for r in rows])
    orc_kept = np.array([min([c["cm"] for c in r["kept"]], default=np.nan) for r in rows])
    print(f"  ORACLE over snaps  median {np.nanmedian(orc_snap):.2f} cm  "
          f">{good_cm:g}cm {100 * np.nanmean(orc_snap > good_cm):.0f}%")
    print(f"  ORACLE over kept   median {np.nanmedian(orc_kept):.2f} cm  "
          f">{good_cm:g}cm {100 * np.nanmean(orc_kept > good_cm):.0f}%")


if __name__ == "__main__":
    main()
