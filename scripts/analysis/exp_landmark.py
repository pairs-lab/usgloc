#!/usr/bin/env python3
"""Pose-free landmark association: accuracy, and its value as an INITIALISER.

Two questions, on TRAIN frames only (the test split is never opened):

(a) How good is the pose that comes out of matching query keypoints straight
    against the descriptors of a set of landmark Gaussians (no render, no depth
    buffer, no pose prior)?  Reported as the translation/rotation error of that
    absolute pose, plus the 2D-3D reprojection precision of the matches it was
    built from (the median reprojection error of the matched Gaussian MEANS at
    the ground-truth pose, and the fraction within 2 / 4 / 12 px).

(b) Is it a better STARTING POINT than a retrieval anchor?  Our hypotheses all
    come from a shortlist of anchor keyframes, so every one of them inherits the
    anchor-to-query gap.  For each query we take the CLOSEST admissible anchor
    (an oracle over the anchor set -- no retrieval front end can do better) and
    compare, and we run the existing LoFTR render-match correction from four
    starts: the ground truth (the floor), a 20 cm / 10 deg perturbation, the
    landmark pose, and the best anchor.

    FLOOR_FRAMES=30 python3 scripts/analysis/exp_landmark.py office stairs \
        --map-suffix _r1 --feats-suffix _lfinit --score-suffix _lmsc_init \
        --variants mo16384 top16384 random16384 --out results/landmark

Frames are the same evenly spaced train frames exp_floor_matcher.py uses, and
``--perturb`` uses its perturbation with the same per-frame seed, so the numbers
are directly comparable to the floors that script reports.

``--no-corr`` skips every LoFTR correction (and the renderer): that mode costs
~0.1 s/frame and is the one to sweep landmark configurations with.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc import landmarks as lmmod, locfeat, se3            # noqa: E402
from usgloc.correction import _to_gray                         # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
N_FRAMES = int(os.environ.get("FLOOR_FRAMES", 30))
TOLS = (2.0, 4.0, 12.0)


def perturb(T, tm, rd, seed):
    """exp_floor_matcher.py's perturbation, verbatim."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.randn(3, generator=g); dt = dt / dt.norm() * tm
    dr = torch.randn(3, generator=g); dr = dr / dr.norm() * (rd * 3.14159265 / 180)
    return se3.apply_tangent(T.to(torch.float32),
                             torch.cat([dt, dr]).to(T.device, torch.float32))


def pose_err(Ta, Tb):
    """exp_floor_matcher.py's error: |t| of Ta @ Tb^-1 (cm) and its angle."""
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    t_cm = float(d[:3, 3].norm()) * 100
    c = ((torch.trace(d[:3, :3]) - 1) / 2).clamp(-1, 1)
    return t_cm, float(torch.rad2deg(torch.acos(c)))


def centre_cm(Ta, Tb):
    """Camera-centre distance between two camera-from-world poses, cm."""
    Ta, Tb = Ta.double(), Tb.double()
    ca = -Ta[:3, :3].T @ Ta[:3, 3]
    cb = -Tb[:3, :3].T @ Tb[:3, 3]
    return float((ca - cb).norm()) * 100


def project(X_w, T_cw, K):
    p = X_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = p[:, 2].clamp_min(1e-6)
    u = K[0, 0] * p[:, 0] / z + K[0, 2]
    v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return torch.stack([u, v], dim=-1)


# Variant spec: "<sample><num>[:k=32][:vac=gate][:umax=0.5][:topk=1][:mnn=1]
#                [:thr=0][:ratio=0.9][:minobs=1][:detn=2048][:nms=4]
#                [:rthr=12][:pnp=poselib][:refine=none][:wvac=1][:seed=0]"
_VKEYS = {"k": ("landmark_k", int), "vac": ("landmark_vac", str),
          "umax": ("landmark_u_max", float), "topk": ("landmark_topk", int),
          "mnn": ("landmark_mnn", lambda v: bool(int(v))),
          "thr": ("landmark_thr", float), "ratio": ("landmark_ratio", float),
          "minobs": ("landmark_min_obs", float),
          "scmin": ("landmark_score_min", float),
          "detmin": ("landmark_det_min", float),
          "detn": ("landmark_detect", int), "nms": ("landmark_nms", int),
          "rthr": ("reproj_thresh_px", float), "pnp": ("solver", str),
          "refine": ("refine", str), "rsig": ("refine_sigma", float),
          "wvac": ("wvac", float), "seed": ("landmark_seed", int),
          "cull": ("landmark_cull", float),
          "mininl": ("min_inliers", int)}


def parse_variant(spec):
    parts = spec.split(":")
    head = parts[0]
    for s in ("mo", "top", "det", "random", "all"):
        if head.startswith(s):
            kw = {"landmark_sample": s}
            n = head[len(s):]
            if n:
                kw["landmark_num"] = int(n)
            break
    else:
        raise SystemExit(f"variant must start with mo|top|det|random|all: {spec!r}")
    for p in parts[1:]:
        if not p:
            continue
        k, v = p.split("=", 1)
        if k not in _VKEYS:
            raise SystemExit(f"unknown variant key {k!r} in {spec!r}")
        name, cast = _VKEYS[k]
        kw[name] = cast(v)
    return kw


def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def prec(err):
    if len(err) == 0:
        return dict(n=0, med=float("nan"), **{f"p{t:g}": float("nan") for t in TOLS})
    e = np.asarray(err, dtype=np.float64)
    return dict(n=int(e.size), med=float(np.median(e)),
                **{f"p{t:g}": float((e <= t).mean()) for t in TOLS})


def run_scene(scene, args, dev="cuda"):
    cuda = str(dev).startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
    cam = ds.scene_camera(640, 480)
    K = cam.K(device=dev, dtype=torch.float64)
    path = f"maps/{scene}{args.map_suffix}.pt"
    emap = load_gaussian_map(path)
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    if args.feats_suffix:
        locfeat.attach_loc_feats(emap, f"maps/{scene}{args.feats_suffix}.pt", device=dev)
    score_path = (f"maps/{scene}{args.score_suffix}.pt" if args.score_suffix else None)
    idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()

    # --- admissible retrieval anchors (diag_stairs_tread.py's protocol) -----
    anchor_ids = list(range(0, len(ds), args.anchor_stride))
    anchor_T = [ds[i].pose_cw.to(dev, torch.float32) for i in anchor_ids]
    anchor_seq = [ds[i].seq for i in anchor_ids]

    rend = corr = None
    if not args.no_corr:
        from usgloc.correction import EvidentialMatchCorrector
        rend = GsplatRenderer()
        # The correction matcher is a free choice: the landmark stage only
        # supplies the START pose.  'stage' polishes with the map's own
        # descriptor field once LoFTR has converged, which is where the
        # accuracy is (held-out Chess floor 0.449 cm LoFTR vs 0.128 cm field).
        corr = EvidentialMatchCorrector(emap, rend, cam, gamma=3.0,
                                        rounds=args.corr_rounds,
                                        matcher=args.corr_matcher,
                                        feat_stage=args.corr_feat_stage)

    out_rows = []
    for spec in args.variants:
        kw = parse_variant(spec)
        t0 = time.time()
        lm = lmmod.LandmarkCorrector(emap, None, cam, landmark_score=score_path,
                                     gamma=3.0, **kw)
        build_sec = time.time() - t0
        info = lm.landmarks.info
        print(f"{scene} {spec}: {info['n_landmarks']} landmarks of "
              f"{info['n_candidates']} candidates (D={info['dim']}), "
              f"built in {build_sec:.1f}s", flush=True)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        rows = []
        for i in idx:
            rec = ds[i]
            img = ds.load_image(i).to(dev)
            gray = _to_gray(img)
            T_gt = rec.pose_cw.to(dev, torch.float32)
            r = dict(frame=int(i), seq=rec.seq, frame_id=int(rec.frame_id))

            # ---- (a) the pose-free landmark solve ---------------------------
            sync()
            t0 = time.time()
            # NOTE: no pose is handed to match(), so the association here is
            # prior-free even if the variant sets `cull` (which needs a start
            # pose); measure culling with exp_floor_matcher.py, where T0 is a
            # real start rather than the answer.
            m = lm.match(gray)
            c = lm.correct(gray, T_gt, matches=m)
            sync()
            r["lm_sec"] = time.time() - t0
            r["lm_ok"] = bool(c.success)
            r["lm_inliers"] = int(c.inliers)
            r["lm_matches"] = int(c.n_matches)
            r["lm_reproj_px"] = float(c.reproj_px)
            T_lm = c.T if c.success else None
            if c.success:
                r["lm_cm"], r["lm_deg"] = pose_err(c.T, T_gt)
                r["lm_centre_cm"] = centre_cm(c.T, T_gt)
            else:
                r["lm_cm"] = r["lm_deg"] = r["lm_centre_cm"] = None
            # correspondence precision at the GT pose (no depth buffer involved)
            if len(m["obj"]):
                Xw = torch.as_tensor(m["obj"], device=dev, dtype=torch.float64)
                uv = project(Xw, T_gt.double(), K)
                e = (uv - torch.as_tensor(m["img"], device=dev, dtype=torch.float64)
                     ).norm(dim=-1).cpu().numpy()
                r["assoc"] = prec(e)
            else:
                r["assoc"] = prec([])

            # ---- (b) the best admissible retrieval anchor -------------------
            # Two difficulties, both an ORACLE over the anchor set (the closest
            # admissible anchor -- no retrieval front end can do better):
            #   'anchor'    anchors within --exclude-anchors frames hidden
            #   'anchor_so' additionally leave-one-sequence-out, the condition
            #               the test split puts the system in
            T_anc = {}
            for name, seq_out in (("anchor", False), ("anchor_so", True)):
                ok = [j for j, a in enumerate(anchor_ids)
                      if abs(a - i) >= args.exclude_anchors
                      and (not seq_out or anchor_seq[j] != rec.seq)]
                cms = [centre_cm(anchor_T[j], T_gt) for j in ok]
                # NOTE the criterion is the camera-centre distance, which for
                # camera-from-world poses is exactly the translation term of
                # pose_err (|t| of Ta Tb^-1 = |c_a - c_b|), so ranking by the
                # floor's own metric picks the SAME anchor -- verified.  It is
                # therefore an oracle in POSITION and agnostic about the
                # ORIENTATION gap, which is large (19-47 deg median).  The
                # 'pert' start (20 cm / 10 deg) is the complementary baseline:
                # the same order of translation gap with a small rotation gap.
                b = int(np.argmin(cms))
                T_anc[name] = anchor_T[ok[b]]
                r[f"{name}_centre_cm"] = cms[b]
                r[f"{name}_cm"], r[f"{name}_deg"] = pose_err(T_anc[name], T_gt)
                r[f"n_{name}"] = len(ok)

            if corr is not None:
                starts = {"gt": T_gt,
                          "pert": perturb(T_gt, args.perturb[0], args.perturb[1], 7 * i),
                          "anchor": T_anc["anchor"], "anchor_so": T_anc["anchor_so"]}
                if T_lm is not None:
                    starts["lm"] = T_lm.to(torch.float32)
                for name, T0 in starts.items():
                    for attempt in range(20):
                        try:
                            sync()
                            t0 = time.time()
                            cr = corr.correct(gray, T0)
                            sync()
                            dt = time.time() - t0
                            break
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"    OOM frame {i}/{name}; waiting 30 s", flush=True)
                            time.sleep(30)
                    else:
                        raise RuntimeError(f"frame {i}: persistent CUDA OOM")
                    tc, rd = pose_err(cr.T, T_gt) if cr.success else (None, None)
                    r[f"c_{name}"] = dict(ok=bool(cr.success), t_cm=tc, r_deg=rd,
                                          inliers=int(cr.inliers),
                                          n_matches=int(cr.n_matches), sec=dt)
                if "lm" not in starts:
                    r["c_lm"] = dict(ok=False, t_cm=None, r_deg=None, inliers=0,
                                     n_matches=0, sec=0.0)
            rows.append(r)
            msg = (f"    {scene:<8}f{i:<6} lm ok={int(r['lm_ok'])} "
                   f"{'%8.2f' % r['lm_cm'] if r['lm_cm'] is not None else '      --'}cm "
                   f"{'%6.2f' % r['lm_deg'] if r['lm_deg'] is not None else '    --'}deg "
                   f"inl={r['lm_inliers']:5d}/{r['lm_matches']:5d} "
                   f"assoc med={r['assoc']['med']:6.2f}px p12={r['assoc']['p12']:.2f}  "
                   f"anc {r['anchor_centre_cm']:5.1f}/{r['anchor_so_centre_cm']:5.1f}cm")
            if corr is not None:
                for name in ("gt", "pert", "lm", "anchor", "anchor_so"):
                    d = r.get(f"c_{name}")
                    v = ("  --  " if d is None or d["t_cm"] is None
                         else "%6.2f" % d["t_cm"])
                    msg += f"  {name}->{v}"
            print(msg, flush=True)

        summ = dict(scene=scene, variant=spec, map=path,
                    feats=(f"maps/{scene}{args.feats_suffix}.pt"
                           if args.feats_suffix else None),
                    score=score_path, n_frames=len(rows),
                    landmark_info=info, build_sec=build_sec,
                    perturb=list(args.perturb), corr_rounds=args.corr_rounds,
                    anchor_stride=args.anchor_stride,
                    exclude_anchors=args.exclude_anchors,
                    exclude_seq=bool(args.exclude_seq),
                    peak_gpu_gb=(torch.cuda.max_memory_allocated() / 1e9
                                 if cuda else 0.0))
        okr = [r for r in rows if r["lm_ok"]]
        summ["lm_reg_rate"] = len(okr) / max(len(rows), 1)
        summ["lm_cm"] = med([r["lm_cm"] for r in okr])
        summ["lm_deg"] = med([r["lm_deg"] for r in okr])
        summ["lm_centre_cm"] = med([r["lm_centre_cm"] for r in okr])
        summ["lm_inliers"] = med([r["lm_inliers"] for r in okr])
        summ["lm_matches"] = med([r["lm_matches"] for r in rows])
        summ["lm_sec"] = med([r["lm_sec"] for r in rows])
        for key in ("med", "p2", "p4", "p12", "n"):
            summ[f"assoc_{key}"] = med([r["assoc"][key] for r in rows])
        for a in ("anchor", "anchor_so"):
            summ[f"{a}_centre_cm"] = med([r[f"{a}_centre_cm"] for r in rows])
            summ[f"{a}_cm"] = med([r[f"{a}_cm"] for r in rows])
            summ[f"{a}_deg"] = med([r[f"{a}_deg"] for r in rows])
            # how often is the prior-free pose closer than the best anchor?
            cmp_c = [(r["lm_centre_cm"], r[f"{a}_centre_cm"]) for r in rows
                     if r["lm_centre_cm"] is not None]
            summ[f"lm_beats_{a}_centre"] = (
                float(np.mean([x < y for x, y in cmp_c])) if cmp_c else float("nan"))
            summ[f"lm_beats_{a}_n"] = len(cmp_c)
        summ["lm_bad5"] = float(np.mean([
            (not r["lm_ok"]) or r["lm_cm"] > 5.0 for r in rows]))
        summ["lm_worst_cm"] = (max([r["lm_cm"] for r in okr]) if okr
                               else float("nan"))
        if corr is not None:
            for name in ("gt", "pert", "lm", "anchor", "anchor_so"):
                got = [r[f"c_{name}"] for r in rows if r.get(f"c_{name}")]
                good = [d for d in got if d["ok"]]
                summ[f"floor_{name}_cm"] = med([d["t_cm"] for d in good])
                summ[f"floor_{name}_deg"] = med([d["r_deg"] for d in good])
                summ[f"reg_{name}"] = len(good) / max(len(rows), 1)
                summ[f"inl_{name}"] = med([d["inliers"] for d in good])
                summ[f"sec_{name}"] = med([d["sec"] for d in got])
                # a frame that did not register, or registered further than
                # 5 cm away, is a gross failure (on stairs: the wrong tread)
                summ[f"bad5_{name}"] = (
                    float(np.mean([not d["ok"] or d["t_cm"] > 5.0 for d in got]))
                    if got else float("nan"))
                summ[f"worst_{name}_cm"] = (max([d["t_cm"] for d in good])
                                            if good else float("nan"))
            # better/worse frame split against the GT start (the floor) and
            # against the anchor start (what the pipeline does today)
            for ref in ("gt", "pert", "anchor", "anchor_so"):
                bw = [(r["c_lm"], r[f"c_{ref}"]) for r in rows
                      if r.get("c_lm") and r.get(f"c_{ref}")]
                both = [(a["t_cm"], b["t_cm"]) for a, b in bw
                        if a["ok"] and b["ok"]]
                summ[f"vs_{ref}_better"] = int(sum(1 for a, b in both if a < b))
                summ[f"vs_{ref}_worse"] = int(sum(1 for a, b in both if a > b))
                summ[f"vs_{ref}_both_ok"] = len(both)
                summ[f"vs_{ref}_only_lm_ok"] = int(sum(
                    1 for a, b in bw if a["ok"] and not b["ok"]))
                summ[f"vs_{ref}_only_ref_ok"] = int(sum(
                    1 for a, b in bw if b["ok"] and not a["ok"]))
        summ["frames"] = rows
        out_rows.append(summ)
        del lm
        if cuda:
            torch.cuda.empty_cache()
    del emap, corr
    if cuda:
        torch.cuda.empty_cache()
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["office"])
    ap.add_argument("--map-suffix", default="_r1")
    ap.add_argument("--feats-suffix", default="_lfinit",
                    help="descriptor-field sidecar maps/<scene><suffix>.pt "
                         "(empty = the field is inside the map)")
    ap.add_argument("--score-suffix", default="_lmsc_init",
                    help="match-score sidecar maps/<scene><suffix>.pt from "
                         "train_landmarks.py (empty = no score field, which "
                         "forces sample=random|all)")
    ap.add_argument("--variants", nargs="+", default=["mo16384"])
    ap.add_argument("--perturb", nargs=2, type=float, default=[0.2, 10.0],
                    metavar=("T_M", "R_DEG"))
    ap.add_argument("--corr-rounds", type=int, default=5)
    ap.add_argument("--corr-matcher", default="loftr",
                    choices=["loftr", "feat", "pooled", "stage"],
                    help="matcher for the render-match correction run from each "
                         "start. The field paths reuse the SAME --feats-suffix "
                         "sidecar the landmark stage matches against.")
    ap.add_argument("--corr-feat-stage", type=int, default=5,
                    help="--corr-matcher stage: leading LoFTR rounds; must be "
                         "long enough for LoFTR to converge (see run_7scenes.py)")
    ap.add_argument("--anchor-stride", type=int, default=20)
    ap.add_argument("--exclude-anchors", type=int, default=40,
                    help="hide anchors within this many train frames of the query")
    ap.add_argument("--exclude-seq", action="store_true",
                    help="also hide every anchor from the query's own sequence "
                         "(leave-one-sequence-out: the test-split condition)")
    ap.add_argument("--no-corr", action="store_true",
                    help="skip every LoFTR correction (and the renderer)")
    ap.add_argument("--device", default="cuda",
                    help="cpu is usable for --no-corr sweeps (slow, but needs "
                         "no GPU slot)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/landmark")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (args.score_suffix or args.feats_suffix).lstrip("_")
    if args.exclude_seq:
        tag += "_seqout"
    summaries = []
    for s in args.scenes:
        summaries += run_scene(s, args, dev=args.device)
    for s in summaries:
        v = s["variant"].replace(":", "_").replace("=", "")
        with open(out / f"lm_{s['scene']}_{v}_{tag}.json", "w") as f:
            json.dump(s, f, indent=1)
    hdr = (f"\n{'scene':<9}{'variant':<26}{'#lm':>7}{'reg':>6}{'cm':>8}{'deg':>7}"
           f"{'assoc med':>10}{'p12':>6}{'anc cm':>8}{'beat':>6}")
    if not args.no_corr:
        hdr += f"{'gt':>8}{'pert':>8}{'lm':>8}{'anchor':>8}{'anc_so':>8}"
    print(hdr)
    for s in summaries:
        line = (f"{s['scene']:<9}{s['variant']:<26}"
                f"{s['landmark_info']['n_landmarks']:>7}"
                f"{100*s['lm_reg_rate']:5.0f}%{s['lm_cm']:8.2f}{s['lm_deg']:7.2f}"
                f"{s['assoc_med']:10.2f}{s['assoc_p12']:6.2f}"
                f"{s['anchor_centre_cm']:8.1f}"
                f"{100*s['lm_beats_anchor_centre']:5.0f}%")
        if not args.no_corr:
            for name in ("gt", "pert", "lm", "anchor", "anchor_so"):
                line += f"{s[f'floor_{name}_cm']:8.3f}"
        print(line)
    with open(out / f"summary_{'-'.join(args.scenes)}_{tag}.json", "w") as f:
        json.dump([{k: v for k, v in s.items() if k != "frames"} for s in summaries],
                  f, indent=1)
    print(f"\nwrote {out}/summary_{'-'.join(args.scenes)}_{tag}.json")


if __name__ == "__main__":
    main()
