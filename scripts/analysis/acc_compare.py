#!/usr/bin/env python3
"""Paired per-frame comparison of exp_floor_matcher.py runs.

Reads every floor_<scene>_<tag>.json under the given directories and, for
each scene, compares every variant against the baseline spec frame by frame
(the frame indices and the perturbation seeds are identical across specs, so
the comparison is paired).  Reports the floor (median translation over the
registered frames), the median rotation, the registration rate, and how many
of the 20 frames got better / worse -- with a two-sided sign test, because a
0.05 cm median shift on 20 frames is not a result on its own.

    python3 scripts/analysis/acc_compare.py results/acc_rounds/gt [...] \
        --baseline loftr
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def sign_p(better, worse):
    """Two-sided sign test on the frames that moved."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    c = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * c / 2.0 ** n)


def load(dirs):
    runs = {}
    for d in dirs:
        for p in sorted(Path(d).glob("floor_*.json")):
            j = json.load(open(p))
            runs[(j["scene"], j["matcher"], str(j.get("perturb")))] = j
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--baseline", default="loftr")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="ignore per-frame changes smaller than this (cm)")
    ap.add_argument("--json", default=None,
                    help="also archive the per-(condition, scene, spec) table here")
    args = ap.parse_args()
    arch = []
    runs = load(args.dirs)
    conds = sorted({k[2] for k in runs})
    for cond in conds:
        print(f"\n### start = {'ground truth' if cond == 'None' else cond}")
        hdr = (f"{'scene':<9}{'spec':<30}{'floor cm':>9}{'deg':>7}{'reg':>6}"
               f"{'dt_cm':>8}{'btr':>5}{'wrs':>5}{'p':>7}{'inl':>7}{'acc%':>6}"
               f"{'win%':>6}{'s/corr':>8}")
        print(hdr)
        for scene in sorted({k[0] for k in runs if k[2] == cond}):
            base = runs.get((scene, args.baseline, cond))
            specs = sorted({k[1] for k in runs if k[0] == scene and k[2] == cond},
                           key=lambda s: (s != args.baseline, s))
            for spec in specs:
                j = runs[(scene, spec, cond)]
                ok = [r for r in j["frames"] if r["success"]]
                fl = np.median([r["t_cm"] for r in ok]) if ok else float("nan")
                fd = np.median([r["r_deg"] for r in ok]) if ok else float("nan")
                b = w = 0
                dmed = float("nan")
                if base is not None and spec != args.baseline:
                    bf = {r["frame"]: r for r in base["frames"]}
                    d = []
                    for r in j["frames"]:
                        o = bf.get(r["frame"])
                        if o is None or not o["success"] or not r["success"]:
                            continue
                        d.append(r["t_cm"] - o["t_cm"])
                    d = np.asarray(d)
                    if len(d):
                        b = int((d < -args.eps).sum())
                        w = int((d > args.eps).sum())
                        dmed = float(np.median(d))
                print(f"{scene:<9}{spec:<30}{fl:9.3f}{fd:7.3f}"
                      f"{100 * j['reg_rate']:5.0f}%{dmed:8.3f}{b:5d}{w:5d}"
                      f"{sign_p(b, w):7.3f}{j['median_inliers']:7.0f}"
                      f"{100 * j.get('acc_used_frac', 0):6.0f}"
                      f"{100 * j.get('acc_win_frac', 0):6.0f}{j['sec_median']:8.2f}")
                arch.append(dict(
                    start="gt" if cond == "None" else cond, scene=scene, spec=spec,
                    rounds=j.get("rounds"), floor_cm=None if ok == [] else float(fl),
                    floor_deg=None if ok == [] else float(fd),
                    reg_rate=j["reg_rate"], median_inliers=j["median_inliers"],
                    d_median_cm=None if b + w == 0 else float(dmed),
                    frames_better=b, frames_worse=w, sign_p=sign_p(b, w),
                    acc_used_frac=j.get("acc_used_frac", 0.0),
                    acc_win_frac=j.get("acc_win_frac", 0.0),
                    acc_points_median=j.get("acc_points_median", 0.0),
                    sec_median=j["sec_median"], peak_gpu_gb=j.get("peak_gpu_gb")))
        # mean over scenes
        print(f"{'-' * len(hdr)}")
        for spec in sorted({k[1] for k in runs if k[2] == cond},
                           key=lambda s: (s != args.baseline, s)):
            js = [runs[k] for k in runs if k[1] == spec and k[2] == cond]
            if not js:
                continue
            fl, fd, bb, ww, dd = [], [], 0, 0, []
            for j in js:
                ok = [r for r in j["frames"] if r["success"]]
                if ok:
                    fl.append(np.median([r["t_cm"] for r in ok]))
                    fd.append(np.median([r["r_deg"] for r in ok]))
                base = runs.get((j["scene"], args.baseline, cond))
                if base is not None and spec != args.baseline:
                    bf = {r["frame"]: r for r in base["frames"]}
                    for r in j["frames"]:
                        o = bf.get(r["frame"])
                        if o is None or not o["success"] or not r["success"]:
                            continue
                        dv = r["t_cm"] - o["t_cm"]
                        dd.append(dv)
                        bb += dv < -args.eps
                        ww += dv > args.eps
            print(f"{'MEAN':<9}{spec:<30}{np.mean(fl):9.3f}{np.mean(fd):7.3f}"
                  f"{'':6}{(np.median(dd) if dd else float('nan')):8.3f}"
                  f"{bb:5d}{ww:5d}{sign_p(bb, ww):7.3f}")
            arch.append(dict(start="gt" if cond == "None" else cond, scene="MEAN",
                             spec=spec, floor_cm=float(np.mean(fl)),
                             floor_deg=float(np.mean(fd)),
                             d_median_cm=float(np.median(dd)) if dd else None,
                             frames_better=int(bb), frames_worse=int(ww),
                             sign_p=sign_p(bb, ww)))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(arch, f, indent=1)
        print(f"\narchived {len(arch)} rows -> {args.json}")


if __name__ == "__main__":
    main()
