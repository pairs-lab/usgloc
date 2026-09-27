#!/usr/bin/env python3
"""Sanity evidence for the per-Gaussian descriptor field, BEFORE anyone
consumes it for localisation.

Measured on HELD-OUT TRAIN frames (the views ``train_locfeat.py`` excluded
with ``--holdout``; the test split is never touched).  Four questions:

1. Does the rasterised descriptor field look like what SuperPoint sees?
   ``cos_gt``  -- mean per-pixel cosine between the rendered descriptor map at
   the GROUND-TRUTH pose and the query's SuperPoint map.
   ``cos_pert`` -- the same from a perturbed render pose, and ``cos_shuf`` --
   against a spatially shuffled rendered map (the chance level).

2. Does the field MATCH better than LoFTR?  At the ground-truth pose the
   correspondence a render-side match implies is the IDENTITY (the render and
   the query share the pose, so lifting a render pixel through any depth and
   reprojecting returns that same pixel).  So the precision of any 2D-2D
   matcher against the render is the fraction of matches within ``tol`` px of
   the identity -- directly comparable between LoFTR and the field's
   coarse-to-fine dense stage.  ``field_dense`` vs ``loftr``.

3. Does the field's POSE-FREE 2D-3D stage beat that?  ``field_lm``: query
   SuperPoint keypoints matched (mutual-NN) against the whole Gaussian
   descriptor field, each match's 3D point being the Gaussian MEAN; the error
   is the reprojection of that mean at the ground-truth pose.  No render, no
   depth buffer, no pose hypothesis.

4. Away from the ground truth, where the depth buffer actually bites:
   ``*_pert`` variants render at a 20 cm / 10 deg perturbation, lift the
   render-side keypoint through the RENDERED depth, reproject at the
   ground-truth pose and measure the same precision.  ``field_lm`` is
   unchanged by the perturbation by construction.

    FLOOR_FRAMES=30 python3 scripts/analysis/exp_locfeat_sanity.py office chess \
        --map-suffix _r1 --feats-suffix _lf256 --loftr --perturb 0.2 10 \
        --out results/locfeat

Every precision is reported as ``p<tol>`` = the fraction of that frame's matches
whose error is at most ``tol`` full-resolution pixels, and ``med`` = the median
error in px; the table prints the across-frame MEDIAN of each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc import locfeat, se3                                # noqa: E402
from usgloc.correction import _to_gray                         # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402
from usgloc.superpoint import SuperPoint, upsample_descriptors   # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
N_FRAMES = int(os.environ.get("FLOOR_FRAMES", 30))
TOLS = (1.0, 2.0, 4.0, 8.0, 12.0)

# The gates EvidentialMatchCorrector applies to LoFTR matches by default, so
# the LoFTR row is the precision of the match set the 0.763 cm floor was
# actually measured with (see usgloc/correction.py's __init__ defaults).
GATE = dict(conf_min=0.20, min_alpha=0.5, min_depth=0.10, edge_rel=0.10, u_max=0.9)


def perturb(T, tm, rd, seed):
    """exp_floor_matcher.py's perturbation, verbatim, so the two scripts'
    'perturbed start' means the same thing."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.randn(3, generator=g); dt = dt / dt.norm() * tm
    dr = torch.randn(3, generator=g); dr = dr / dr.norm() * (rd * 3.14159265 / 180)
    return se3.apply_tangent(T.to(torch.float32),
                             torch.cat([dt, dr]).to(T.device, torch.float32))


def prec(err: torch.Tensor) -> dict:
    """Precision at each tolerance + the median error, from a (m,) px error."""
    if err.numel() == 0:
        return dict(n=0, med=float("nan"), **{f"p{t:g}": float("nan") for t in TOLS})
    e = err.double()
    return dict(n=int(e.numel()), med=float(e.median()),
                **{f"p{t:g}": float((e <= t).double().mean()) for t in TOLS})


def prec_top(err: torch.Tensor, score: torch.Tensor, frac: float = 0.25) -> dict:
    """:func:`prec` on the ``frac`` highest-scoring matches -- the subset a
    RANSAC effectively leans on, and the fairest LoFTR-vs-field comparison at
    equal match counts."""
    k = max(1, int(round(frac * err.numel()))) if err.numel() else 0
    if k == 0:
        return prec(err)
    keep = torch.topk(score.double(), k).indices
    return prec(err[keep])


def project(X_w: torch.Tensor, T_cw: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """(M, 3) world points -> (M, 2) pixel INDICES (x, y) under T_cw."""
    p = X_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = p[:, 2].clamp_min(1e-6)
    u = K[0, 0] * p[:, 0] / z + K[0, 2]
    v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return torch.stack([u, v], dim=-1) - 0.5        # back to index convention


def lift(uv: torch.Tensor, depth: torch.Tensor, T_cw: torch.Tensor,
         K: torch.Tensor) -> torch.Tensor:
    """(M, 2) render pixel indices + (H, W) depth -> (M, 3) world points.

    The same bilinear expected-depth lift ``correction.correct`` performs.
    """
    H, W = depth.shape
    gx = 2.0 * (uv[:, 0] + 0.5) / W - 1.0
    gy = 2.0 * (uv[:, 1] + 0.5) / H - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None, None].to(depth.dtype)
    d = F.grid_sample(depth[None, None], grid, mode="bilinear",
                      align_corners=False)[0, 0, 0]
    uv1 = torch.cat([uv.double() + 0.5, torch.ones_like(uv[:, :1]).double()], dim=1)
    x_cam = (torch.linalg.inv(K.double()) @ uv1.T).T * d.double()[:, None]
    T_wc = torch.linalg.inv(T_cw.double())
    return (T_wc[:3, :3] @ x_cam.T).T + T_wc[:3, 3]


def _render(rend, emap, T, cam, lf):
    return rend.render(emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
                       vacuity=emap.gaussian_vacuity, with_depth=True, loc_feats=lf)


def loftr_gates(out, kq, kr, conf, cam):
    """correction.correct's per-match gates on a render-side keypoint set."""
    H, W = cam.height, cam.width
    xi = kr[:, 0].round().long().clamp(0, W - 1)
    yi = kr[:, 1].round().long().clamp(0, H - 1)
    depth = out["depth"]
    d, a, u = depth[yi, xi], out["alpha"][yi, xi], out["vacuity"][yi, xi]
    dm = depth[None, None]
    spread = (F.max_pool2d(dm, 3, 1, 1) + F.max_pool2d(-dm, 3, 1, 1))[0, 0]
    keep = ((conf >= GATE["conf_min"]) & (a >= GATE["min_alpha"])
            & (d >= GATE["min_depth"]) & (spread[yi, xi] <= GATE["edge_rel"] * d)
            & ((u / a.clamp_min(1e-6)) <= GATE["u_max"]))
    return keep


def run_scene(scene, suffix, args, feats=None, dev="cuda"):
    ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
    cam = ds.scene_camera(640, 480)
    H, W = cam.height, cam.width
    K = cam.K(device=dev, dtype=torch.float32)
    path = f"maps/{scene}{suffix}.pt"
    emap = load_gaussian_map(path)
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    if feats:
        meta = locfeat.attach_loc_feats(emap, feats)
    else:
        meta = torch.load(path, map_location="cpu", weights_only=False)
    stride = int(meta.get("loc_feat_stride", 5))
    holdout = int(meta.get("loc_feat_holdout", 2))
    held = [i for i in range(len(ds)) if i % stride == holdout % stride]
    idx = torch.linspace(0, len(held) - 1, min(N_FRAMES, len(held))
                         ).round().long().unique().tolist()
    frames = [held[i] for i in idx]
    del meta
    if emap.loc_feats is None:
        raise SystemExit(f"{path} carries no loc_feats (pass --feats <sidecar.pt>)")
    D = int(emap.loc_feats.shape[1])
    lf = emap.loc_feats.to(dev).float()
    # Landmark set: the Gaussians the sparse 2D-3D stage may match against.
    # Never-observed Gaussians carry the zero descriptor and are always
    # excluded; --lm-min-obs additionally keeps only Gaussians seen in at least
    # that many training views, a training-free stand-in for STDLoc's learned
    # landmark detector (which we do NOT have).
    ocount = (emap.loc_feat_obs.to(dev).float() if emap.loc_feat_obs is not None
              else (lf.abs().sum(-1) > 0).float())
    obs = ocount > max(0.0, args.lm_min_obs - 0.5)
    lm_idx = torch.nonzero(obs, as_tuple=False).squeeze(-1)
    lm_feat, lm_xyz = lf[lm_idx], emap.means[lm_idx]
    pca = None
    if emap.loc_feat_pca_basis is not None:
        pca = (emap.loc_feat_pca_mean.to(dev), emap.loc_feat_pca_basis.to(dev))
    rend = GsplatRenderer()
    enc = SuperPoint().to(dev).eval()
    matcher = None
    if args.loftr:
        import copy

        from kornia.feature import LoFTR
        from kornia.feature.loftr.loftr import default_cfg as _cfg
        cfg = copy.deepcopy(_cfg)
        cfg["match_coarse"]["thr"] = GATE["conf_min"]
        matcher = LoFTR(pretrained="indoor_new", config=cfg).eval().to(dev)

    print(f"{scene}{suffix}: {emap.means.shape[0]} gaussians, D={D}, "
          f"{lm_idx.numel()} observed landmarks ({100*lm_idx.numel()/lf.shape[0]:.1f}%), "
          f"{len(frames)} held-out frames (i % {stride} == {holdout})", flush=True)

    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i in frames:
        r = dict(frame=int(i))
        img = ds.load_image(i).to(dev)
        T = ds[i].pose_cw.to(dev, torch.float32)
        with torch.no_grad():
            desc, scores = enc.dense_descriptors(img)              # (1,256,h,w)
            if pca is not None:
                desc_p = locfeat.project_query_map(desc, pca[0], pca[1])
            else:
                desc_p = desc
            q_coarse = F.normalize(desc_p[0], p=2, dim=0)           # (D, h, w)
            q_fine = upsample_descriptors(desc_p, H, W)[0]          # (D, H, W)

            t0 = time.perf_counter()
            out = _render(rend, emap, T, cam, lf)
            torch.cuda.synchronize()
            r["sec_render_feat"] = time.perf_counter() - t0
            fm = out["feats"].permute(2, 0, 1).contiguous()          # (D, H, W)
            m = out["alpha"] >= args.alpha_min

            # ---- 1. descriptor agreement -----------------------------------
            cos = (fm * q_fine).sum(dim=0)
            r["cos_gt"] = float(cos[m].mean()) if int(m.sum()) else float("nan")
            perm = torch.randperm(H * W, device=dev)
            shuf = fm.reshape(D, -1)[:, perm].reshape(D, H, W)
            r["cos_shuf"] = float((shuf * q_fine).sum(dim=0)[m].mean()) if int(m.sum()) \
                else float("nan")
            r["alpha_frac"] = float(m.double().mean())
            del perm, shuf, cos

            # ---- 2. dense field match at the GT pose -----------------------
            r_coarse = F.normalize(F.interpolate(
                fm[None], size=q_coarse.shape[-2:], mode="bilinear",
                align_corners=False)[0], p=2, dim=0)
            cq, cr, _ = locfeat.dense_feature_match(
                q_coarse, r_coarse, dual_softmax_temp=args.dsm_temp)
            wc = q_coarse.shape[-1]
            fq = cq[:, 1] * wc + cq[:, 0]
            fr = cr[:, 1] * wc + cr[:, 0]
            uq, ur, _ = locfeat.refine_match_windows(q_fine, fm, fq, fr,
                                                     temp=args.dsm_temp)
            r["field_dense"] = prec((uq - ur).float().norm(dim=-1))

            # ---- 3. pose-free landmark match -------------------------------
            t0 = time.perf_counter()
            kp = locfeat.detect_keypoints(scores[0], args.detect_num, args.nms)
            qd = locfeat.sample_query_descriptors(desc, kp, H, W, pca=pca)
            qi, gi, sc = locfeat.mnn_match(qd, lm_feat, chunk=args.chunk,
                                           thr=args.lm_thr, ratio=args.lm_ratio)
            torch.cuda.synchronize()
            r["sec_landmark"] = time.perf_counter() - t0
            uv_p = project(lm_xyz[gi].double(), T.double(), K.double())
            e_lm = (kp[qi].double() - uv_p).norm(dim=-1)
            r["field_lm"] = prec(e_lm)
            r["field_lm_top"] = prec_top(e_lm, sc, args.top_frac)
            r["n_keypoints"] = int(kp.shape[0])

            # ---- 4. LoFTR against the rendered RGB -------------------------
            if matcher is not None:
                gray_q = _to_gray(img)[None, None]
                t0 = time.perf_counter()
                c = matcher({"image0": gray_q,
                             "image1": _to_gray(out["rgb"])[None, None].float()})
                torch.cuda.synchronize()
                r["sec_loftr"] = time.perf_counter() - t0
                kq, kr, cf = c["keypoints0"], c["keypoints1"], c["confidence"]
                r["loftr_raw"] = prec((kq - kr).float().norm(dim=-1))
                keep = loftr_gates(out, kq, kr, cf, cam)
                e_lo = (kq[keep] - kr[keep]).float().norm(dim=-1)
                r["loftr"] = prec(e_lo)
                r["loftr_top"] = prec_top(e_lo, cf[keep], args.top_frac)
            del out, fm, r_coarse
            torch.cuda.empty_cache()

            # ---- 5. perturbed render: the depth buffer bites ----------------
            if args.perturb:
                Tp = perturb(T, args.perturb[0], args.perturb[1], 7 * i)
                out = _render(rend, emap, Tp, cam, lf)
                fm = out["feats"].permute(2, 0, 1).contiguous()
                mp = out["alpha"] >= args.alpha_min
                r["cos_pert"] = float((fm * q_fine).sum(dim=0)[mp].mean()) \
                    if int(mp.sum()) else float("nan")
                r_coarse = F.normalize(F.interpolate(
                    fm[None], size=q_coarse.shape[-2:], mode="bilinear",
                    align_corners=False)[0], p=2, dim=0)
                cq, cr, _ = locfeat.dense_feature_match(
                    q_coarse, r_coarse, dual_softmax_temp=args.dsm_temp)
                fq = cq[:, 1] * wc + cq[:, 0]
                fr = cr[:, 1] * wc + cr[:, 0]
                uq, ur, _ = locfeat.refine_match_windows(q_fine, fm, fq, fr,
                                                         temp=args.dsm_temp)
                Xw = lift(ur, out["depth"], Tp, K)
                r["field_dense_pert"] = prec(
                    (uq.double() - project(Xw, T.double(), K.double())).norm(dim=-1))
                if matcher is not None:
                    c = matcher({"image0": gray_q,
                                 "image1": _to_gray(out["rgb"])[None, None].float()})
                    kq, kr, cf = c["keypoints0"], c["keypoints1"], c["confidence"]
                    keep = loftr_gates(out, kq, kr, cf, cam)
                    Xw = lift(kr[keep], out["depth"], Tp, K)
                    r["loftr_pert"] = prec(
                        (kq[keep].double() - project(Xw, T.double(), K.double())
                         ).norm(dim=-1))
                del out, fm, r_coarse
                torch.cuda.empty_cache()
        rows.append(r)
        msg = (f"    {scene:<8}f{i:<6} cos {r['cos_gt']:.3f} (shuf {r['cos_shuf']:+.3f})"
               f"  lm n={r['field_lm']['n']:<5} p2={r['field_lm']['p2']:.2f}"
               f" med={r['field_lm']['med']:.2f}"
               f"  dense n={r['field_dense']['n']:<5} p2={r['field_dense']['p2']:.2f}")
        if "loftr" in r:
            msg += f"  loftr n={r['loftr']['n']:<5} p2={r['loftr']['p2']:.2f}"
        print(msg, flush=True)

    def agg(key):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            return None
        if isinstance(vals[0], dict):
            return {k: float(np.nanmedian([v[k] for v in vals])) for k in vals[0]}
        return float(np.nanmedian(vals))

    summ = dict(scene=scene, map=f"maps/{scene}{suffix}.pt", feats=feats, dim=D,
                n_gaussians=int(emap.means.shape[0]),
                n_landmarks=int(lm_idx.numel()), n_frames=len(rows),
                stride=stride, holdout=holdout, alpha_min=args.alpha_min,
                lm_min_obs=args.lm_min_obs,
                dsm_temp=args.dsm_temp, detect_num=args.detect_num,
                perturb=args.perturb, tols=list(TOLS),
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
    for k in ("cos_gt", "cos_shuf", "cos_pert", "alpha_frac", "n_keypoints",
              "sec_render_feat", "sec_landmark", "sec_loftr",
              "field_dense", "field_lm", "field_lm_top", "loftr", "loftr_top",
              "loftr_raw", "field_dense_pert", "loftr_pert"):
        v = agg(k)
        if v is not None:
            summ[k] = v
    summ["frames"] = rows
    del emap, lf, lm_feat, matcher
    torch.cuda.empty_cache()
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["office", "chess"])
    ap.add_argument("--map-suffix", default="_r1",
                    help="geometry map maps/<scene><suffix>.pt")
    ap.add_argument("--feats-suffix", default="",
                    help="read the descriptor field from the SIDECAR "
                         "maps/<scene><feats-suffix>.pt (train_locfeat.py "
                         "--feats-only); empty = the field is inside --map-suffix")
    ap.add_argument("--perturb", nargs=2, type=float, default=None,
                    metavar=("T_M", "R_DEG"))
    ap.add_argument("--loftr", action="store_true",
                    help="also measure LoFTR on the same frames (the baseline)")
    ap.add_argument("--alpha-min", type=float, default=0.2,
                    help="rendered-alpha floor for the cosine statistics")
    ap.add_argument("--dsm-temp", type=float, default=0.1,
                    help="dual-softmax temperature of the dense stage (STDLoc: 0.1)")
    ap.add_argument("--detect-num", type=int, default=2048)
    ap.add_argument("--nms", type=int, default=4)
    ap.add_argument("--lm-min-obs", type=float, default=1.0,
                    help="keep only Gaussians observed in >= this many training "
                         "views as sparse-stage landmarks (1 = every observed one)")
    ap.add_argument("--top-frac", type=float, default=0.25,
                    help="fraction of best-scoring matches for the *_top rows")
    ap.add_argument("--lm-thr", type=float, default=-1.0)
    ap.add_argument("--lm-ratio", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=16384,
                    help="gaussians per mutual-NN chunk (2048 x chunk score matrix)")
    ap.add_argument("--out", default="results/locfeat_sanity")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (args.feats_suffix or args.map_suffix).lstrip("_")
    if args.lm_min_obs > 1:
        tag += f"_obs{args.lm_min_obs:g}"
    if args.perturb:
        tag += f"_p{args.perturb[0]:g}m{args.perturb[1]:g}deg"
    summaries = [
        run_scene(s, args.map_suffix, args,
                  feats=(f"maps/{s}{args.feats_suffix}.pt" if args.feats_suffix
                         else None))
        for s in args.scenes
    ]
    for s in summaries:
        with open(out / f"sanity_{s['scene']}_{tag}.json", "w") as f:
            json.dump(s, f, indent=1)
    cols = ("field_lm", "field_lm_top", "field_dense", "loftr", "loftr_top",
            "field_dense_pert", "loftr_pert")
    hdr = f"{'scene':<9}{'D':>4}{'cosGT':>7}{'shuf':>7}{'cosP':>7}"
    for name in cols:
        hdr += f"{name:>26}"
    print("\n" + hdr)
    print(" " * 34 + ("      n   med    p2   p12" * len(cols)))
    for s in summaries:
        line = (f"{s['scene']:<9}{s['dim']:>4}{s.get('cos_gt', float('nan')):7.3f}"
                f"{s.get('cos_shuf', float('nan')):7.3f}"
                f"{s.get('cos_pert', float('nan')):7.3f}")
        for name in cols:
            d = s.get(name)
            line += ("         -     -     -     -" if d is None else
                     f"{d['n']:7.0f}{d['med']:6.2f}{d['p2']:6.2f}{d['p12']:6.2f}")
        print(line)
    # Scene names are part of the filename: two runs of the same field tag on
    # different scenes must not overwrite each other's summary.
    with open(out / f"summary_{'-'.join(args.scenes)}_{tag}.json", "w") as f:
        json.dump([{k: v for k, v in s.items() if k != "frames"} for s in summaries],
                  f, indent=1)
    print(f"\nprecision keys are p<tol px>; tolerances {list(TOLS)}\n"
          f"wrote {out}/summary_{'-'.join(args.scenes)}_{tag}.json")


if __name__ == "__main__":
    main()
