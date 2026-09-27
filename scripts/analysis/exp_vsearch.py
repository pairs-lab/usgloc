#!/usr/bin/env python3
"""Adaptive virtual-view search: trigger / radius / n / sampling-law study.

One GPU session instead of one full ``run_7scenes.py`` run per setting.

For every TRAIN-side validation frame (test-like anchor distance -- the
shortlist exclusion of the harness is replicated here) the script

  1. runs the REAL pipeline with ``pipe.debug`` on and the search OFF, which
     gives the baseline pose error and every anchor's corrected snap, i.e. the
     quantities a trigger could be built from (inlier count, inlier ratio,
     render fidelity, vacuity-weighted mass);
  2. on the frames whose support is weak, draws ``--views`` virtual views for
     each (law, sigma_t, sigma_r) in the grid and puts EVERY one through the
     existing correction block, recording its score and the pose error of the
     pose it produced.

Because every view of a configuration is recorded, one pass answers several
questions at once: the best-of-first-k curve is the n study, the best-by-score
of a different score column is the score study, and the minimum error over the
views is the ORACLE -- the separation between oracle and selected says whether
the sampling or the selection criterion is the limit.

Writes one JSON; nothing else. Train-side only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from run_7scenes import build_front_end, move_map, move_instances, resize_image  # noqa: E402

from usgloc.datasets import SevenScenes, load_gaussian_map  # noqa: E402
from usgloc.descriptors import InstanceDescriptorNet, SceneAggregator  # noqa: E402
from usgloc.metrics import pose_errors  # noqa: E402
from usgloc.pipeline import PipelineConfig, USGLocPipeline  # noqa: E402
from usgloc.primitives import make_submaps  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/7scenes_source")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sfm-pose-dir",
                    default="data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes")
    ap.add_argument("--map", required=True)
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--anchor-stride", type=int, default=20)
    ap.add_argument("--submap-radius", type=float, default=3.0)
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--corr-refine-iters", type=int, default=60)
    ap.add_argument("--corr-rounds", type=int, default=5)
    ap.add_argument("--support-power", type=float, default=1.0)
    ap.add_argument("--val-stride", type=int, default=7)
    ap.add_argument("--val-offset", type=int, default=3)
    ap.add_argument("--val-exclude-anchors", type=int, default=40)
    ap.add_argument("--val-exclude-seq", action="store_true")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--gamma", type=float, default=3.0)
    # --- the search grid -------------------------------------------------
    ap.add_argument("--views", type=int, default=64,
                    help="virtual views drawn per configuration (sample 0 is the "
                         "unperturbed pose); the n study is the best-of-first-k curve")
    ap.add_argument("--grid", default="shell:0.15:5,shell:0.30:5,shell:0.50:5,"
                                      "shell:0.80:5,normal:0.30:5,uniform:0.50:5",
                    help="law:sigma_t_m:sigma_r_deg, comma separated")
    ap.add_argument("--vsearch-rounds", type=int, default=1,
                    help="correction rounds spent per virtual view")
    ap.add_argument("--weak-inliers", type=int, default=1500,
                    help="frames whose best snap is below this get the search")
    ap.add_argument("--all-frames", action="store_true",
                    help="search on EVERY frame, not only the weak ones (measures "
                         "what the search would cost/break if it always fired)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)

    train_ds = SevenScenes(args.root, args.scene, split="train",
                           sfm_pose_dir=args.sfm_pose_dir)
    emap = load_gaussian_map(args.map)
    anchor_ids = list(range(0, len(train_ds), args.anchor_stride))
    make_submaps(emap, [train_ds[i].pose_cw for i in anchor_ids],
                 radius=args.submap_radius,
                 camera=train_ds.scene_camera(train_ds.color_width,
                                              train_ds.color_height))
    emap = move_map(emap, dev)
    clip_dim = emap.primitives[0].clip_feat.shape[0]
    desc, agg = InstanceDescriptorNet(clip_dim=clip_dim), SceneAggregator()
    ck = torch.load(args.encoders, map_location="cpu", weights_only=True)
    desc.load_state_dict(ck["descriptor"])
    agg.load_state_dict(ck["aggregator"])
    desc, agg = desc.to(dev).eval(), agg.to(dev).eval()
    front_end = build_front_end(emap, clip_dim, ck, dev)

    from usgloc.correction import EvidentialMatchCorrector
    from usgloc.rendering import GsplatRenderer

    renderer = GsplatRenderer()
    cam = train_ds.scene_camera(args.width, args.height)
    corrector = EvidentialMatchCorrector(
        emap, renderer, train_ds.scene_camera(640, 480), gamma=args.gamma,
        rounds=args.corr_rounds, min_inlier_ratio=0.0)
    cfg = PipelineConfig(top_k=args.top_k, refine_iters=args.iters,
                         gamma=args.gamma,
                         corr_refine_iters=args.corr_refine_iters,
                         support_power=args.support_power)
    pipe = USGLocPipeline(emap, renderer, cam, desc, agg, cfg, corrector=corrector)
    pipe.encode_map()
    pipe.debug = True

    # anchor globals for the CLIP shortlist (the harness's own front end)
    kept_frames, k = [], 0
    for i in anchor_ids:
        if k < len(emap.submaps) and torch.allclose(
                emap.submaps[k].anchor_pose, train_ds[i].pose_cw.to(dev)):
            kept_frames.append(i)
            k += 1
    gfeats = []
    for i in kept_frames:
        bgr = (train_ds.load_image(i).numpy() * 255).astype("uint8")[:, :, ::-1]
        gfeats.append(front_end.encode_global(bgr))
    anchor_globals = torch.stack(gfeats)
    anchor_seqs = [train_ds[i].seq for i in kept_frames]
    print(f"      {len(kept_frames)} anchors embedded", flush=True)

    idx = list(range(args.val_offset, len(train_ds), args.val_stride))
    if args.queries > 0:
        sel = torch.linspace(0, len(idx) - 1, args.queries).round().long().unique()
        idx = [idx[j] for j in sel.tolist()]

    grid = []
    for spec in args.grid.split(","):
        law, st, sr = spec.split(":")
        grid.append((law, float(st), float(sr)))

    out = {"args": vars(args), "grid": [list(g) for g in grid], "frames": []}
    for i in idx:
        rec = train_ds[i]
        img_full = train_ds.load_image(i)
        img = resize_image(img_full, args.width, args.height).to(dev)
        bgr = (img_full.numpy() * 255).astype("uint8")[:, :, ::-1]
        inst = move_instances(front_end.encode(bgr), dev)
        if len(inst) < 2:
            continue
        qg = front_end.encode_global(bgr)
        sims = anchor_globals @ qg
        far = torch.tensor([abs(f - i) >= args.val_exclude_anchors
                            for f in kept_frames], device=sims.device)
        if args.val_exclude_seq:
            far &= torch.tensor([s != rec.seq for s in anchor_seqs],
                                device=sims.device)
        sims = torch.where(far, sims, torch.full_like(sims, -float("inf")))
        cand = sims.topk(min(args.shortlist, int(far.sum()))).indices.tolist()
        gf = resize_image(img_full, 640, 480).to(dev)
        gray = (0.299 * gf[..., 0] + 0.587 * gf[..., 1] + 0.114 * gf[..., 2]).clamp(0, 1)
        T_gt = rec.pose_cw.to(dev)

        t0 = time.perf_counter()
        res = pipe.localize(img, inst, candidates=cand, gray_full=gray)
        t_base = time.perf_counter() - t0
        t_err, r_err = pose_errors(res.condensed.T_mode, T_gt)
        snaps = [s for s in pipe.last_debug["snaps"] if s.get("success")]
        best_inl = max([int(s["inliers"]) for s in snaps], default=0)
        best_psnr = max([float(s.get("psnr", 0.0)) for s in snaps], default=0.0)
        best_ratio = max([int(s["inliers"]) / max(int(s["n_matches"]), 1)
                          for s in snaps], default=0.0)
        f = dict(seq=rec.seq, frame=int(rec.frame_id), t_err_cm=100 * t_err,
                 r_err_deg=r_err, best_inliers=best_inl, best_psnr=best_psnr,
                 best_ratio=best_ratio, n_snaps=len(snaps), t_base_s=t_base,
                 configs={})
        print(f"   {rec.seq}/{rec.frame_id:06d}  base {100*t_err:7.2f} cm  "
              f"inl {best_inl:5d}  ratio {best_ratio:.2f}  psnr {best_psnr:5.1f}",
              flush=True)

        if args.all_frames or best_inl < args.weak_inliers:
            # centre the search on the best CORRECTED pose we have
            if snaps:
                sb = max(snaps, key=lambda s: int(s["inliers"]))
                T_c = sb["T"].to(dev).to(img.dtype)
            else:
                T_c = res.hypotheses[0].T0
            ec, _ = pose_errors(T_c, T_gt)
            f["centre_err_cm"] = 100 * ec
            for law, st, sr_deg in grid:
                key = f"{law}:{st}:{sr_deg}"
                dcs, phis = corrector._sample_xi(args.views, st, sr_deg, law,
                                                 args.seed)
                dcs, phis = dcs.to(dev), phis.to(dev)
                rows, t1 = [], time.perf_counter()
                for v in range(args.views):
                    T_v = T_c if v == 0 else corrector._perturb(T_c, dcs[v], phis[v])
                    c = corrector.correct(gray, T_v, rounds=args.vsearch_rounds)
                    if c.success:
                        e, rd = pose_errors(c.T.to(img.dtype), T_gt)
                    else:
                        e, rd = float("nan"), float("nan")
                    rows.append(dict(ok=bool(c.success), inl=int(c.inliers),
                                     nm=int(c.n_matches), wm=float(c.wmass),
                                     rms=float(c.reproj_px),
                                     err_cm=100 * e if c.success else None,
                                     err_deg=rd if c.success else None,
                                     view_cm=float(dcs[v].norm()) * 100))
                dt = time.perf_counter() - t1
                f["configs"][key] = dict(sec=dt, views=rows)
                ok = [r for r in rows if r["ok"]]
                bw = max(ok, key=lambda r: r["wm"]) if ok else None
                orc = min([r["err_cm"] for r in ok], default=None)
                print(f"        {key:<16} {dt:5.1f}s  sel(wmass) "
                      f"{bw['err_cm']:7.2f} cm  oracle {orc if orc is None else round(orc,2)} "
                      f"cm  base_view {rows[0]['err_cm']}", flush=True)
        out["frames"].append(f)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
