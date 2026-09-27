#!/usr/bin/env python3
"""Train-side correction floor per matcher (LoFTR vs RoMa).

Same protocol as select_map_ft.py's score(): 20 train frames spread evenly
over the split, correction started from the ground-truth pose with rounds=5
on maps/<scene>_ft.pt; report the median translation error of the frames
that register (the "floor"), the registration rate, the median PnP inlier
count and the wall time per correct() call.

    python3 scripts/analysis/exp_floor_matcher.py chess office stairs \
        --matchers loftr roma:indoor roma:outdoor --out results/matcher_roma

Matcher specs: ``loftr`` | ``roma:indoor`` | ``roma:outdoor`` (optionally
``roma:indoor:num=2000``) | ``mast3r`` | ``feat`` | ``pooled``.  Runs are
sequential; the corrector of each spec is freed before the next is built.

``lm`` is the pose-free LANDMARK association (usgloc.landmarks): query keypoints
matched straight against the descriptors of a sampled set of Gaussians, whose
MEANS are the 3D points, with no render and no dependence on the start pose --
so its GT and perturbed runs must come out identical.  See _LM_KEYS below.

``feat`` matches the query's dense SuperPoint descriptors against the map's own
RASTERISED per-Gaussian descriptor field (STDLoc's mechanism) instead of the
rendered RGB, and ``pooled`` concatenates LoFTR's and the field's
correspondences before the vacuity gate.  Both need the field on the map:

    FLOOR_FRAMES=60 python3 scripts/analysis/exp_floor_matcher.py office \
        --matchers loftr feat pooled --map-suffix _r1 --feats-suffix _lf64 \
        --perturb 0.2 10 --out results/locfeat_floor
"""
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

from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402
from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402
from usgloc import se3                                          # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
N_FRAMES = int(os.environ.get("FLOOR_FRAMES", 20))


def perturb(T, tm, rd, seed):
    """exp_floor.py's perturbation: |dt| = tm metres, |dr| = rd degrees,
    random direction from a per-frame seed."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.randn(3, generator=g); dt = dt / dt.norm() * tm
    dr = torch.randn(3, generator=g); dr = dr / dr.norm() * (rd * 3.14159265 / 180)
    return se3.apply_tangent(T.to(torch.float32), torch.cat([dt, dr]).to(T.device, torch.float32))


def pose_err(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    t_cm = float(d[:3, 3].norm()) * 100
    c = ((torch.trace(d[:3, :3]) - 1) / 2).clamp(-1, 1)
    r_deg = float(torch.rad2deg(torch.acos(c)))
    return t_cm, r_deg


# Multi-round accumulation knobs, usable on any matcher spec as k=v tails, e.g.
#   loftr:acc=union                 pool every round's inliers, mean-dedup, LM solve
#   loftr:acc=union:dd=all          keep every instance of a repeated query pixel
#   loftr:acc=weighted:k=1.5        drop rounds whose inlier RMS > 1.5 * the best round's
#   loftr:acc=union:solve=ransac    re-run RANSAC over the pooled set
#   loftr:acc=union:jit=0.02:jdeg=1 jitter the render pose of rounds >= 1
#   loftr:acc=union:eq=0.7          keep the best-70% pooled correspondences by their
#                                   own round's reprojection error
#   loftr:acc=union:athr=1.0        gate the pooled set at 1 px (pooling buys the
#                                   correspondences a stricter gate costs)
#   loftr:acc=union:aref=huber      IRLS (Huber) instead of plain LM on the pool
#   loftr:acc=union:seen=3          keep only query pixels re-matched in >= 3 rounds
#   loftr:acc=union:seen=2:sq=0.7   ... and, of those, the 70% whose world points
#                                   agree best across rounds
#   loftr:r=7                       override --rounds for this spec only
_ACC_KEYS = {"acc": ("accumulate", str), "dd": ("acc_dedup", str),
             "solve": ("acc_solve", str), "k": ("acc_rms_k", float),
             "guard": ("acc_guard", lambda v: bool(int(v))),
             "jit": ("acc_jitter", float), "jdeg": ("acc_jitter_deg", float),
             "jseed": ("acc_jitter_seed", int), "r": ("rounds", int),
             "eq": ("acc_err_q", float), "athr": ("acc_thresh_px", float),
             "aref": ("acc_refine", str), "seen": ("acc_min_seen", int),
             "sq": ("acc_spread_q", float)}

# Estimator knobs (2D-3D association + solver), same k=v tail syntax, e.g.
#   loftr:lift=maxw                 lift through the max-weight Gaussian's depth
#   loftr:lift=median               ... the median-weight (surface) depth
#   loftr:lift=band:band=0.02       ... the weighted mean inside a +-2% depth band
#   loftr:lift=maxw_d               the (mode - mean) correction on the bilinear depth
#   loftr:thr=1.0                   RANSAC reprojection threshold, px
#   loftr:refine=lm+huber:rsig=1.0  IRLS after the LM, Huber scale 1 px
#   loftr:refine=cauchy:rall=1      IRLS over every fed match, no plain LM
#   loftr:pnp=poselib:refine=none    poselib LO-RANSAC whose Cauchy bundle is KEPT
#                                   (refine=lm, the default, overwrites it with a
#                                   plain L2 solvePnPRefineLM)
#   loftr:pnp=poselib:pli=100000    poselib RANSAC max_iterations (STDLoc's budget)
#   loftr:plloss=TRUNCATED:plsc=2   poselib bundle loss_type / loss_scale
#   loftr:refine=plb                poselib's robust bundle as the final refinement
#   loftr:refine=cauchy:wvac=1      weight each correspondence by its rendered
#                                   vacuity weight exp(-gamma*U/alpha) in the IRLS
#   loftr:wthr=1:wclip=3            per-correspondence reprojection gate scaled by
#                                   that weight (soft u_max)
_EST_KEYS = {"lift": ("lift", str), "band": ("lift_band", float),
             "thr": ("reproj_thresh_px", float), "pnp": ("solver", str),
             "refine": ("refine", str), "rsig": ("refine_sigma", float),
             "riter": ("refine_iters", int),
             "rall": ("refine_all", lambda v: bool(int(v))),
             "ri": ("ransac_iters", int), "edge": ("edge_rel", float),
             "guess": ("use_guess", lambda v: bool(int(v))),
             "dir": ("loftr_dir", str),
             "pli": ("pl_max_iters", int), "plmin": ("pl_min_iters", int),
             "plloss": ("pl_loss", str), "plsc": ("pl_loss_scale", float),
             "plbi": ("pl_bundle_iters", int),
             "wvac": ("wvac", float), "wthr": ("wvac_thr", float),
             "wclip": ("wvac_clip", float),
             # descriptor-field matcher ('feat' / 'pooled'), see correction.py
             "lft": ("feat_temp", float), "lfw": ("feat_win", int),
             "lfc": ("feat_conf_min", float), "lfct": ("feat_coarse_thr", float),
             "lfsp": ("feat_subpix", float), "lftop": ("feat_top", float),
             "lffm": ("feat_fine", str), "lffthr": ("feat_fine_thr", float),
             "lfmax": ("feat_max", int), "lfst": ("feat_stage", int),
             # photometric arbitration of the returned pose
             "arb": ("arbiter", str), "arbm": ("arb_margin", float),
             "arbc": ("arb_min_cover", float),
             # convergence early stop
             "sdt": ("stop_dt", float), "sdr": ("stop_dr", float),
             # over-relaxation of the field rounds
             "lfg": ("feat_gain", float), "lfgm": ("feat_gain_max", float)}


# Pose-free LANDMARK association (usgloc.landmarks.LandmarkCorrector): query
# keypoints matched straight against the descriptors of a sampled set of
# Gaussians, the 3D point being the Gaussian mean.  Nothing is rendered and the
# result does not depend on the start pose at all, so the GT and the perturbed
# run differ only in which frames the harness calls it on.  Needs a descriptor
# field (--feats-suffix or a baked map) and, for the matching-oriented sampling,
# a match-score sidecar from scripts/analysis/train_landmarks.py:
#   lm:sc=_lmsc_init                STDLoc's recipe: 16384 matching-oriented
#                                   landmarks, 2048 keypoints, one-way NN,
#                                   poselib, 12 px
#   lm:sc=_lmsc_init:s=top:n=4096   the 4096 globally best-scoring landmarks
#   lm:sc=_lmsc_init:vac=gate:umax=0.5   drop vacuous landmarks (the map's own
#                                   per-Gaussian vacuity, no render involved)
#   lm:sc=_lmsc_init:refine=cauchy:wvac=1   weight the IRLS by exp(-gamma*u_i)
_LM_KEYS = {"sc": ("_lm_score_suffix", str), "s": ("landmark_sample", str),
            "n": ("landmark_num", int), "k": ("landmark_k", int),
            "seed": ("landmark_seed", int),
            "minobs": ("landmark_min_obs", float),
            "scmin": ("landmark_score_min", float),
            "detmin": ("landmark_det_min", float),
            "vac": ("landmark_vac", str), "umax": ("landmark_u_max", float),
            "detn": ("landmark_detect", int), "nms": ("landmark_nms", int),
            "topk": ("landmark_topk", int),
            "mnn": ("landmark_mnn", lambda v: bool(int(v))),
            "cthr": ("landmark_thr", float),
            "cull": ("landmark_cull", float),
            "ratio": ("landmark_ratio", float),
            "rthr": ("reproj_thresh_px", float), "pnp": ("solver", str),
            "ri": ("ransac_iters", int), "plmin": ("pl_min_iters", int),
            "plloss": ("pl_loss", str), "plsc": ("pl_loss_scale", float),
            "plbi": ("pl_bundle_iters", int),
            "refine": ("refine", str), "rsig": ("refine_sigma", float),
            "riter": ("refine_iters", int), "wvac": ("wvac", float),
            "mininl": ("min_inliers", int),
            "mratio": ("min_inlier_ratio", float)}


def _acc_kv(kw, parts):
    """Consume the accumulation / estimator k=v tails; return the unknown ones."""
    rest = []
    for p in parts:
        if not p or "=" not in p:
            rest.append(p)
            continue
        k, v = p.split("=", 1)
        if k in _ACC_KEYS:
            name, cast = _ACC_KEYS[k]
            kw[name] = cast(v)
        elif k in _EST_KEYS:
            name, cast = _EST_KEYS[k]
            kw[name] = cast(v)
        else:
            rest.append(p)
    return rest


def parse_spec(spec):
    parts = spec.split(":")
    kw = {}
    if parts[0] == "lm":
        # Pose-free landmark association; built by run() as a
        # usgloc.landmarks.LandmarkCorrector, not an EvidentialMatchCorrector.
        kw = {"_landmark": True, "rounds": 1}
        for p in parts[1:]:
            if not p:
                continue
            if "=" not in p:
                raise SystemExit(f"lm options are k=v, got {p!r} in {spec!r}")
            k, v = p.split("=", 1)
            if k not in _LM_KEYS:
                raise SystemExit(f"unknown lm option {k!r} in {spec!r}")
            name, cast = _LM_KEYS[k]
            kw[name] = cast(v)
        return kw
    if parts[0] == "loftr":
        left = _acc_kv(kw, parts[1:])
        if any(left):
            raise SystemExit(f"unknown loftr option(s) {left} in {spec!r}")
        return kw
    parts = [parts[0]] + _acc_kv(kw, parts[1:])
    if parts[0] in ("feat", "pooled", "stage"):
        # STDLoc-style dense matching against the rasterised per-Gaussian
        # descriptor field ('feat'), or that pooled with LoFTR ('pooled').
        # Needs the field on the map: --feats-suffix, or a baked map.
        kw["matcher"] = parts[0]
        if any(parts[1:]):
            raise SystemExit(f"unknown {parts[0]} option(s) {parts[1:]} in {spec!r}")
        return kw
    if parts[0] == "roma":
        kw["matcher"] = "roma"
        kw["roma_model"] = parts[1] if len(parts) > 1 and parts[1] else "indoor"
        for p in parts[2:]:
            k, v = p.split("=")
            kw[{"num": "roma_num", "seed": "roma_seed", "up": "roma_upsample_res",
                "coarse": "roma_coarse_res"}[k]] = int(v)
    elif parts[0] == "mast3r":
        kw["matcher"] = "mast3r"
        for p in parts[1:]:
            if not p:
                continue
            k, v = p.split("=")
            if k == "ratio":
                kw["min_inlier_ratio"] = float(v)
            else:
                kw[{"size": "mast3r_size", "sub": "mast3r_subsample"}[k]] = int(v)
    else:
        raise SystemExit(f"unknown matcher spec {spec!r}")
    return kw


# Single-entry map cache for --cache-map.  GsplatRenderer keys its eigenvalue
# cache on covs.data_ptr(), so re-loading the same map for every spec throws
# that decomposition away and pays ~1 min of CPU eigh per spec.  Keeping the
# tensors alive across the specs of one scene keeps the cache warm; the map is
# read-only here, so the results are identical (off by default).
_MAP_CACHE = {}


def load_map(scene, suffix, dev, cache=False, feats=None):
    key = (scene, suffix, feats)
    if cache and key in _MAP_CACHE:
        return _MAP_CACHE[key]
    if cache:
        _MAP_CACHE.clear()          # one scene resident at a time
        torch.cuda.empty_cache()
    emap = load_gaussian_map(f"maps/{scene}{suffix}.pt")
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    if feats:
        # Sidecar per-Gaussian descriptor field (train_locfeat.py --feats-only),
        # merged into the geometry map in place; only matcher='feat'/'pooled'
        # reads it, every other path is unaffected.
        from usgloc import locfeat
        locfeat.attach_loc_feats(emap, feats, device=dev)
    if cache:
        _MAP_CACHE[key] = emap
    return emap


def run(scene, spec, ds, idx, cam, rend, dev, rounds, pert=None, suffix="_ft", psnr=False,
        cache_map=False, feats=None):
    emap = load_map(scene, suffix, dev, cache=cache_map, feats=feats)
    kw = parse_spec(spec)
    rounds = int(kw.pop("rounds", rounds))     # `r=N` in the spec overrides --rounds
    build = EvidentialMatchCorrector
    if kw.pop("_landmark", False):
        # Pose-free landmark association (the 'lm' spec): a different
        # estimator class with the same correct(query, T0) contract.
        from usgloc.landmarks import LandmarkCorrector
        build = LandmarkCorrector
        sfx = kw.pop("_lm_score_suffix", None)
        if sfx is not None:
            kw["landmark_score"] = f"maps/{scene}{sfx}.pt"
    torch.cuda.reset_peak_memory_stats()
    # Moving the matcher's weights onto a shared, momentarily full card is the
    # one OOM the per-frame retry below cannot catch: retry it here too.
    for attempt in range(20):
        try:
            corr = build(emap, rend, cam, rounds=rounds, **kw)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    OOM building {spec} (attempt {attempt + 1}); waiting 30 s",
                  flush=True)
            time.sleep(30)
    else:
        raise RuntimeError(f"{scene}/{spec}: persistent CUDA OOM building the matcher")
    use_rgb = kw.get("matcher", "loftr") != "loftr"
    rows = []
    psnrs = []
    for i in idx:
        img = ds.load_image(i).to(dev)
        T = ds[i].pose_cw.to(dev, torch.float32)
        if psnr:
            with torch.no_grad():
                o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                T, cam, vacuity=emap.gaussian_vacuity, with_depth=True)
                psnrs.append(float(10 * torch.log10(
                    1.0 / torch.mean((o["rgb"] - img) ** 2).clamp_min(1e-12))))
                del o
        T0 = T if pert is None else perturb(T, pert[0], pert[1], 7 * i)
        q = img if use_rgb else _to_gray(img)
        # The GPU is shared with a long benchmark; on a transient OOM wait
        # and retry the frame rather than lose the run.
        for attempt in range(20):
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = corr.correct(q, T0)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    OOM on frame {i} (attempt {attempt + 1}); waiting 30 s", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"frame {i}: persistent CUDA OOM")
        t_cm, r_deg = pose_err(r.T, T) if r.success else (float("nan"), float("nan"))
        # Arbiter audit: the true error of EVERY candidate it ranked, so the
        # choice can be scored against the oracle (best candidate) and against
        # the inlier rule it replaced.  Diagnostic only; nothing feeds back.
        arb = None
        if getattr(r, "arb", None):
            errs = [pose_err(torch.tensor(P, dtype=torch.float64,
                                          device=T.device), T.double())[0]
                    for P in r.arb["poses"]]
            arb = dict(scores=r.arb["scores"], errs=errs, pick=r.arb["pick"],
                       pick_inl=r.arb["pick_inl"],
                       oracle=int(np.argmin(errs)))
        rows.append(dict(frame=int(i), success=bool(r.success), t_cm=t_cm, r_deg=r_deg,
                         arb=arb,
                         inliers=int(r.inliers), n_matches=int(r.n_matches),
                         reproj_px=float(r.reproj_px), rounds=int(r.rounds), sec=dt,
                         acc_used=bool(getattr(r, "acc_used", False)),
                         acc_points=int(getattr(r, "acc_points", 0)),
                         acc_rounds=int(getattr(r, "acc_rounds", 0)),
                         acc_inl=int(getattr(r, "acc_inl", 0)),
                         acc_win=bool(getattr(r, "acc_win", False))))
        print(f"    {scene:<8}{spec:<26}frame {i:5d}  ok={int(r.success)} "
              f"t={t_cm:7.3f}cm r={r_deg:6.3f}deg inl={r.inliers:5d}/{r.n_matches:5d} "
              f"rounds={r.rounds} acc={int(getattr(r, 'acc_used', False))}"
              f"/{getattr(r, 'acc_points', 0):5d} {dt:5.2f}s", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    n_gauss = int(emap.means.shape[0])
    del corr
    if not cache_map:
        del emap
    torch.cuda.empty_cache()
    ok = [r for r in rows if r["success"]]
    summ = dict(
        scene=scene, matcher=spec, rounds=rounds, n_frames=len(idx),
        map=f"maps/{scene}{suffix}.pt", feats=feats, n_gaussians=n_gauss,
        psnr_gt=float(np.mean(psnrs)) if psnrs else None,
        perturb=None if pert is None else dict(t_m=pert[0], r_deg=pert[1]),
        registered=len(ok), reg_rate=len(ok) / len(idx),
        floor_cm=float(np.median([r["t_cm"] for r in ok])) if ok else float("inf"),
        floor_deg=float(np.median([r["r_deg"] for r in ok])) if ok else float("inf"),
        mean_cm=float(np.mean([r["t_cm"] for r in ok])) if ok else float("inf"),
        median_inliers=float(np.median([r["inliers"] for r in ok])) if ok else 0.0,
        median_matches=float(np.median([r["n_matches"] for r in ok])) if ok else 0.0,
        sec_median=float(np.median([r["sec"] for r in rows])),
        sec_mean=float(np.mean([r["sec"] for r in rows])),
        acc_used_frac=float(np.mean([r["acc_used"] for r in ok])) if ok else 0.0,
        acc_win_frac=float(np.mean([r["acc_win"] for r in ok])) if ok else 0.0,
        acc_points_median=float(np.median([r["acc_points"] for r in ok])) if ok else 0.0,
        peak_gpu_gb=peak_gb, frames=rows,
    )
    # Arbiter audit, over the frames where it actually had a choice: how often
    # it picked the most accurate candidate, and the median error under the
    # arbiter, under the inlier rule, and under an oracle that always picks the
    # best candidate (the ceiling any selection rule can reach).
    A = [r["arb"] for r in ok if r.get("arb")]
    if A:
        summ["arb_n"] = len(A)
        summ["arb_agree_oracle"] = float(np.mean([a["pick"] == a["oracle"] for a in A]))
        summ["arb_flip_frac"] = float(np.mean([a["pick"] != a["pick_inl"] for a in A]))
        summ["arb_med_cm"] = float(np.median([a["errs"][a["pick"]] for a in A]))
        summ["arb_med_inl_cm"] = float(np.median(
            [a["errs"][a["pick_inl"]] for a in A if a["pick_inl"] >= 0]))
        summ["arb_med_oracle_cm"] = float(np.median([min(a["errs"]) for a in A]))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["chess", "office", "stairs"])
    ap.add_argument("--matchers", nargs="+", default=["loftr", "roma:indoor"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--perturb", nargs=2, type=float, default=None, metavar=("T_M", "R_DEG"),
                    help="start from the ground truth perturbed by T_M metres / "
                         "R_DEG degrees (exp_floor.py protocol) instead of from "
                         "the ground truth: measures the basin, not the floor")
    ap.add_argument("--map-suffix", default="_ft",
                    help="score maps/<scene><suffix>.pt instead of the default maps/<scene>_ft.pt")
    ap.add_argument("--feats-suffix", default="",
                    help="attach the SIDECAR per-Gaussian descriptor field "
                         "maps/<scene><feats-suffix>.pt (train_locfeat.py "
                         "--feats-only) to the geometry map; required by the "
                         "'feat' / 'pooled' matcher specs unless the field is "
                         "already baked into --map-suffix")
    ap.add_argument("--psnr", action="store_true",
                    help="also render at the ground-truth pose and report the mean PSNR")
    ap.add_argument("--cache-map", action="store_true",
                    help="keep one scene's map resident across the specs of that "
                         "scene, so the renderer's eigendecomposition cache stays "
                         "warm (identical results, ~1 min/spec faster)")
    ap.add_argument("--frame-mod", default=None, metavar="M:R",
                    help="draw the measured frames only from those with i %% M == R "
                         "instead of from the whole train split.  '5:2' is the set "
                         "train_locfeat.py held out (--stride 5 --holdout 2), i.e. "
                         "views the descriptor FIELD never saw -- while the RGB map "
                         "itself was fit on every train view, so this is the "
                         "conservative comparison for the field")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="which split the floor is measured on. 'train' (default) "
                         "is the map-selection protocol. 'test' measures the "
                         "ceiling on the frames Table I actually reports, i.e. on "
                         "views neither the map nor the descriptor field was fit "
                         "on -- the honest ceiling, and the one that predicts the "
                         "reported median.")
    ap.add_argument("--out", default="results/matcher_roma")
    args = ap.parse_args()
    ptag = "" if args.perturb is None else f"_p{args.perturb[0]:g}m{args.perturb[1]:g}deg"
    if args.frame_mod:
        ptag = f"_fm{args.frame_mod.replace(':', '-')}" + ptag
    if args.feats_suffix:
        ptag = f"_{args.feats_suffix.lstrip('_')}" + ptag
    if args.map_suffix != "_ft":
        ptag = f"_{args.map_suffix.lstrip('_')}" + ptag
    if args.split != "train":
        ptag = f"_{args.split}" + ptag
    import os
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rend = GsplatRenderer()
    summaries = []
    for s in args.scenes:
        ds = SevenScenes("data/7scenes_source", s, split=args.split, sfm_pose_dir=PGT)
        pool = list(range(len(ds)))
        if args.frame_mod:
            m, r = (int(x) for x in args.frame_mod.split(":"))
            pool = [i for i in pool if i % m == r % m]
            if not pool:
                raise SystemExit(f"--frame-mod {args.frame_mod} selects no frame")
        sel = torch.linspace(0, len(pool) - 1, min(N_FRAMES, len(pool))
                             ).round().long().unique().tolist()
        idx = [pool[j] for j in sel]
        cam = ds.scene_camera(640, 480)
        for spec in args.matchers:
            summ = run(s, spec, ds, idx, cam, rend, dev, args.rounds, args.perturb,
                       suffix=args.map_suffix, psnr=args.psnr,
                       cache_map=args.cache_map,
                       feats=(f"maps/{s}{args.feats_suffix}.pt"
                              if args.feats_suffix else None))
            summaries.append(summ)
            tag = spec.replace(":", "_").replace("=", "") + ptag
            with open(out / f"floor_{s}_{tag}.json", "w") as f:
                json.dump(summ, f, indent=1)
            print(f"==> {s:<8}{spec:<30} floor {summ['floor_cm']:.3f} cm / "
                  f"{summ['floor_deg']:.3f} deg  reg {100*summ['reg_rate']:.0f}%  "
                  f"med inl {summ['median_inliers']:.0f}  "
                  f"{summ['sec_median']:.2f} s/corr  peak {summ['peak_gpu_gb']:.2f} GB"
                  + ("" if summ["psnr_gt"] is None else f"  PSNR {summ['psnr_gt']:.2f}")
                  + ("" if "arb_n" not in summ else
                     f"\n    arbiter: n={summ['arb_n']} picks-oracle "
                     f"{100*summ['arb_agree_oracle']:.0f}% flips "
                     f"{100*summ['arb_flip_frac']:.0f}% | median arb "
                     f"{summ['arb_med_cm']:.3f} vs inlier-rule "
                     f"{summ['arb_med_inl_cm']:.3f} vs oracle "
                     f"{summ['arb_med_oracle_cm']:.3f} cm"),
                  flush=True)
    print(f"\n{'scene':<9}{'matcher':<30}{'floor cm':>9}{'deg':>7}{'reg':>6}"
          f"{'med inl':>9}{'s/corr':>8}{'GB':>6}{'acc%':>6}{'win%':>6}")
    for m in summaries:
        print(f"{m['scene']:<9}{m['matcher']:<30}{m['floor_cm']:9.3f}{m['floor_deg']:7.3f}"
              f"{100*m['reg_rate']:5.0f}%{m['median_inliers']:9.0f}{m['sec_median']:8.2f}"
              f"{m['peak_gpu_gb']:6.2f}{100*m.get('acc_used_frac',0):6.0f}"
              f"{100*m.get('acc_win_frac',0):6.0f}")
    with open(out / f"summary{ptag}.json", "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "frames"} for m in summaries],
                  f, indent=1)


if __name__ == "__main__":
    main()
