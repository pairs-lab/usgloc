#!/usr/bin/env python3
"""Estimator sweep on the correction floor, with the map loaded ONCE per scene.

Identical protocol and identical output files to
``scripts/analysis/exp_floor_matcher.py`` (20 train frames from
``torch.linspace`` over the train split, ``rounds=5``, ``maps/<scene>_ft.pt``,
correction started from the ground truth and/or from a perturbation), and it
reuses that script's ``parse_spec`` so every spec keyword is the same.

The only difference is bookkeeping: ``exp_floor_matcher.py`` re-reads the map
for every (scene, spec) pair, and the covariance -> quat/scale decomposition
that follows costs more than the 20 corrections themselves on a loaded
machine.  Here each scene's map is read once and every spec, in both start
conditions, is scored against it.

    python3 scripts/analysis/exp_estimator.py chess office stairs \
        --matchers loftr loftr:lift=maxw loftr:thr=1.0 \
        --both --out results/estimator_lift/floor
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
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from exp_floor_matcher import PGT, N_FRAMES, parse_spec, perturb, pose_err  # noqa: E402

from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map        # noqa: E402
from usgloc.rendering import GsplatRenderer                       # noqa: E402


def score(scene, spec, emap, ds, idx, cam, rend, rounds, pert, suffix):
    kw = parse_spec(spec)
    rounds = int(kw.pop("rounds", rounds))
    dev = emap.means.device
    torch.cuda.reset_peak_memory_stats()
    corr = EvidentialMatchCorrector(emap, rend, cam, rounds=rounds, **kw)
    use_rgb = kw.get("matcher", "loftr") != "loftr"
    rows = []
    for i in idx:
        img = ds.load_image(i).to(dev)
        T = ds[i].pose_cw.to(dev, torch.float32)
        T0 = T if pert is None else perturb(T, pert[0], pert[1], 7 * i)
        q = img if use_rgb else _to_gray(img)
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
                print(f"    OOM on frame {i} (attempt {attempt + 1}); waiting 30 s",
                      flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"frame {i}: persistent CUDA OOM")
        t_cm, r_deg = pose_err(r.T, T) if r.success else (float("nan"), float("nan"))
        rows.append(dict(frame=int(i), success=bool(r.success), t_cm=t_cm, r_deg=r_deg,
                         inliers=int(r.inliers), n_matches=int(r.n_matches),
                         reproj_px=float(r.reproj_px), rounds=int(r.rounds), sec=dt))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del corr
    torch.cuda.empty_cache()
    ok = [r for r in rows if r["success"]]
    return dict(
        scene=scene, matcher=spec, rounds=rounds, n_frames=len(idx),
        map=f"maps/{scene}{suffix}.pt", n_gaussians=int(emap.means.shape[0]),
        psnr_gt=None,
        perturb=None if pert is None else dict(t_m=pert[0], r_deg=pert[1]),
        registered=len(ok), reg_rate=len(ok) / len(idx),
        floor_cm=float(np.median([r["t_cm"] for r in ok])) if ok else float("inf"),
        floor_deg=float(np.median([r["r_deg"] for r in ok])) if ok else float("inf"),
        mean_cm=float(np.mean([r["t_cm"] for r in ok])) if ok else float("inf"),
        median_inliers=float(np.median([r["inliers"] for r in ok])) if ok else 0.0,
        median_matches=float(np.median([r["n_matches"] for r in ok])) if ok else 0.0,
        sec_median=float(np.median([r["sec"] for r in rows])),
        sec_mean=float(np.mean([r["sec"] for r in rows])),
        peak_gpu_gb=peak_gb, frames=rows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["chess", "office", "stairs"])
    ap.add_argument("--matchers", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--perturb", nargs=2, type=float, default=[0.2, 10.0],
                    metavar=("T_M", "R_DEG"))
    ap.add_argument("--both", action="store_true", default=True,
                    help="score both the from-GT and the perturbed start (default)")
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--pert-only", action="store_true")
    ap.add_argument("--map-suffix", default="_ft")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES,
                    help="smoke-test escape hatch; the reported protocol is 20")
    ap.add_argument("--out", default="results/estimator_lift/floor")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    starts = []
    if not args.pert_only:
        starts.append(None)
    if not args.gt_only:
        starts.append(tuple(args.perturb))
    dev = "cuda"
    rend = GsplatRenderer()
    summaries = []
    for scene in args.scenes:
        print(f"[{scene}] loading map", flush=True)
        emap = load_gaussian_map(f"maps/{scene}{args.map_suffix}.pt")
        for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
            setattr(emap, f, getattr(emap, f).to(dev))
        ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, args.n_frames).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        # warm the renderer's covariance cache once (the expensive part)
        t0 = time.perf_counter()
        with torch.no_grad():
            rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                        ds[idx[0]].pose_cw.to(dev, torch.float32), cam,
                        vacuity=emap.gaussian_vacuity, with_depth=True)
        print(f"[{scene}] map ready ({time.perf_counter() - t0:.0f} s)", flush=True)
        for pert in starts:
            ptag = "" if pert is None else f"_p{pert[0]:g}m{pert[1]:g}deg"
            if args.map_suffix != "_ft":
                ptag = f"_{args.map_suffix.lstrip('_')}" + ptag
            for spec in args.matchers:
                t0 = time.perf_counter()
                s = score(scene, spec, emap, ds, idx, cam, rend, args.rounds,
                          pert, args.map_suffix)
                summaries.append(s)
                tag = spec.replace(":", "_").replace("=", "") + ptag
                with open(out / f"floor_{scene}_{tag}.json", "w") as f:
                    json.dump(s, f, indent=1)
                print(f"==> {scene:<8}{spec:<28}{'GT ' if pert is None else 'pert'}"
                      f" floor {s['floor_cm']:7.3f} cm / {s['floor_deg']:6.3f} deg"
                      f"  mean {s['mean_cm']:7.3f}  reg {100*s['reg_rate']:3.0f}%"
                      f"  inl {s['median_inliers']:4.0f}  {s['sec_median']:5.2f} s/corr"
                      f"  peak {s['peak_gpu_gb']:.2f} GB  [{time.perf_counter()-t0:.0f} s]",
                      flush=True)
        del emap
        torch.cuda.empty_cache()
    with open(out / "summary_estimator.json", "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "frames"} for m in summaries],
                  f, indent=1)


if __name__ == "__main__":
    main()
