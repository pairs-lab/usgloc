#!/usr/bin/env python3
"""Correction floor + render PSNR of an arbitrary map file, LoFTR only.

Self-contained copy of exp_floor_matcher.py's protocol (20 train frames from
torch.linspace over the train split, rounds=5, LoFTR) that takes the map
SUFFIX as an argument, so depth-supervised variants (maps/<scene>_d1.pt ...)
can be scored against maps/<scene>_ft.pt without touching the map files or
the existing script.  It also reports the render PSNR at the ground-truth
pose on the same frames.

    python3 scripts/analysis/exp_floor_depthsup.py chess \
        --suffixes _ft _dw0.2 --out results/depthsup/floor
    python3 scripts/analysis/exp_floor_depthsup.py chess \
        --suffixes _ft _dw0.2 --perturb 0.2 10 --out results/depthsup/floor
"""
from __future__ import annotations

import argparse
import json
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
N_FRAMES = 20


def perturb(T, tm, rd, seed):
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


def wait_for_gpu(need_mib=4500, tries=240):
    import subprocess
    for _ in range(tries):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines()
        if out and int(out[0]) >= need_mib:
            return
        print(f"    waiting for GPU memory ({out[0] if out else '?'} MiB free)", flush=True)
        time.sleep(30)


def run(scene, suffix, ds, idx, cam, rend, dev, rounds, pert=None):
    emap = load_gaussian_map(f"maps/{scene}{suffix}.pt")
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    torch.cuda.reset_peak_memory_stats()
    corr = EvidentialMatchCorrector(emap, rend, cam, rounds=rounds)
    rows, psnrs = [], []
    for i in idx:
        img = ds.load_image(i).to(dev)
        T = ds[i].pose_cw.to(dev, torch.float32)
        with torch.no_grad():
            o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities, T, cam)
            mse = float(((o["rgb"].clamp(0, 1) - img) ** 2).mean())
        psnrs.append(-10.0 * np.log10(max(mse, 1e-12)))
        T0 = T if pert is None else perturb(T, pert[0], pert[1], 7 * i)
        q = _to_gray(img)
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
        rows.append(dict(frame=int(i), success=bool(r.success), t_cm=t_cm, r_deg=r_deg,
                         inliers=int(r.inliers), n_matches=int(r.n_matches),
                         reproj_px=float(r.reproj_px), rounds=int(r.rounds), sec=dt,
                         psnr=float(psnrs[-1])))
        print(f"    {scene:<8}{suffix:<10}frame {i:5d}  ok={int(r.success)} "
              f"t={t_cm:7.3f}cm r={r_deg:6.3f}deg inl={r.inliers:5d}/{r.n_matches:5d} "
              f"psnr={psnrs[-1]:5.2f} {dt:5.2f}s", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del corr, emap
    torch.cuda.empty_cache()
    ok = [r for r in rows if r["success"]]
    return dict(
        scene=scene, suffix=suffix, rounds=rounds, n_frames=len(idx),
        perturb=None if pert is None else dict(t_m=pert[0], r_deg=pert[1]),
        registered=len(ok), reg_rate=len(ok) / len(idx),
        floor_cm=float(np.median([r["t_cm"] for r in ok])) if ok else float("inf"),
        floor_deg=float(np.median([r["r_deg"] for r in ok])) if ok else float("inf"),
        mean_cm=float(np.mean([r["t_cm"] for r in ok])) if ok else float("inf"),
        median_inliers=float(np.median([r["inliers"] for r in ok])) if ok else 0.0,
        psnr_mean=float(np.mean(psnrs)), psnr_median=float(np.median(psnrs)),
        sec_median=float(np.median([r["sec"] for r in rows])),
        peak_gpu_gb=peak_gb, frames=rows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--suffixes", nargs="+", default=["_ft"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--perturb", nargs=2, type=float, default=None, metavar=("T_M", "R_DEG"))
    ap.add_argument("--out", default="results/depthsup/floor")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each (scene, map) this many times. LoFTR + PnP-RANSAC on a "
                         "contended GPU is not bit-reproducible, so repeats measure the "
                         "noise floor of the metric itself")
    ap.add_argument("--min-free-mib", type=int, default=4500,
                    help="wait until the shared GPU has this much free memory before "
                         "starting (measured peak of this script: ~1.3 GB)")
    args = ap.parse_args()
    ptag = "" if args.perturb is None else f"_p{args.perturb[0]:g}m{args.perturb[1]:g}deg"
    import os
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    wait_for_gpu(args.min_free_mib)
    rend = GsplatRenderer()
    summaries = []
    for s in args.scenes:
        ds = SevenScenes("data/7scenes_source", s, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        for suf in args.suffixes:
            if not (ROOT / "maps" / f"{s}{suf}.pt").exists():
                print(f"!! maps/{s}{suf}.pt missing, skipped", flush=True)
                continue
            for rep in range(args.repeat):
                # The GPU is shared and heavily contended: a run can die
                # anywhere (even uploading the map) when another job spikes.
                # Retry the whole scene/map run after waiting for memory.
                for attempt in range(12):
                    try:
                        wait_for_gpu(args.min_free_mib)
                        summ = run(s, suf, ds, idx, cam, rend, dev, args.rounds, args.perturb)
                        break
                    except Exception as e:
                        if "out of memory" not in str(e).lower() and \
                           "launch timed out" not in str(e).lower():
                            raise
                        torch.cuda.empty_cache()
                        print(f"  !! {s}{suf} attempt {attempt + 1} died ({type(e).__name__}); "
                              f"waiting 120 s", flush=True)
                        time.sleep(120)
                else:
                    raise RuntimeError(f"{s}{suf}: persistent CUDA OOM")
                summ["repeat"] = rep
                summaries.append(summ)
                rtag = "" if args.repeat == 1 else f"_r{rep}"
                with open(out / f"floor_{s}{suf}{ptag}{rtag}.json", "w") as f:
                    json.dump(summ, f, indent=1)
                print(f"==> {s:<8}{suf:<10}r{rep} floor {summ['floor_cm']:.3f} cm / "
                      f"{summ['floor_deg']:.3f} deg  reg {100*summ['reg_rate']:.0f}%  "
                      f"med inl {summ['median_inliers']:.0f}  psnr {summ['psnr_mean']:.2f}  "
                      f"{summ['sec_median']:.2f} s/corr  peak {summ['peak_gpu_gb']:.2f} GB",
                      flush=True)
    print(f"\n{'scene':<9}{'map':<12}{'floor cm':>9}{'deg':>7}{'reg':>6}{'med inl':>9}"
          f"{'PSNR':>7}{'s/corr':>8}{'GB':>6}")
    for m in summaries:
        print(f"{m['scene']:<9}{m['suffix']:<12}{m['floor_cm']:9.3f}{m['floor_deg']:7.3f}"
              f"{100*m['reg_rate']:5.0f}%{m['median_inliers']:9.0f}{m['psnr_mean']:7.2f}"
              f"{m['sec_median']:8.2f}{m['peak_gpu_gb']:6.2f}")
    with open(out / f"summary{ptag}.json", "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "frames"} for m in summaries],
                  f, indent=1)


if __name__ == "__main__":
    main()
