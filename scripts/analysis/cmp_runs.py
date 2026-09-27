#!/usr/bin/env python3
"""Compare run_7scenes.py result JSONs frame by frame.

Reports, per arm, the median translation error over ALL recorded frames (not
the metric-table subset), the fraction beyond a threshold, the correction
budget actually spent, and -- against the first arm -- the per-frame
better/worse split on the frames both arms localized.  A median shift with a
better/worse split near 50/50 is noise; the split is the number that matters.

    python3 scripts/analysis/cmp_runs.py results/x/base results/x/variant \
        --scene stairs --thresh 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(d, scene, dataset="7scenes"):
    p = Path(d) / f"{dataset}_{scene}.json"
    if not p.is_file():
        return None
    j = json.load(open(p))
    return {(r["seq"], r["frame_id"]): r for r in j["records"]}, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset", default="7scenes")
    ap.add_argument("--thresh", type=float, default=5.0, help="cm")
    args = ap.parse_args()

    arms = []
    for d in args.dirs:
        r = load(d, args.scene, args.dataset)
        if r is None:
            print(f"missing: {d}/{args.dataset}_{args.scene}.json")
            continue
        arms.append((Path(d).name, r[0], r[1]))
    if not arms:
        return

    print(f"{args.scene}   threshold {args.thresh:g} cm")
    print(f"{'arm':<16}{'n':>4}{'med cm':>9}{'mean':>9}{'p75':>8}{'>thr':>7}"
          f"{'med deg':>9}{'corr/q':>8}{'esc/q':>7}{'s/q':>7}")
    base = None
    for name, recs, j in arms:
        e = np.array([r["t_err"] for r in recs.values()]) * 100
        rr = np.array([r["r_err"] for r in recs.values()])
        nc = np.array([r.get("n_corrections", 0) for r in recs.values()], dtype=float)
        ne = np.array([r.get("n_escalations", 0) for r in recs.values()], dtype=float)
        t = np.array([r["t_localize_s"] for r in recs.values()])
        print(f"{name:<16}{len(e):>4}{np.median(e):9.2f}{e.mean():9.2f}"
              f"{np.percentile(e, 75):8.2f}{100 * (e > args.thresh).mean():6.0f}%"
              f"{np.median(rr):9.3f}{nc.mean():8.1f}{ne.mean():7.2f}{np.median(t):7.1f}")
        if base is None:
            base = (name, recs)
    bname, brecs = base
    print()
    for name, recs, _ in arms[1:]:
        common = sorted(set(brecs) & set(recs))
        a = np.array([brecs[k]["t_err"] for k in common]) * 100
        b = np.array([recs[k]["t_err"] for k in common]) * 100
        d = b - a
        tol = 0.05   # cm; below this a frame is unchanged
        better = int((d < -tol).sum())
        worse = int((d > tol).sum())
        same = len(d) - better - worse
        fa = int((a > args.thresh).sum())
        fb = int((b > args.thresh).sum())
        print(f"{name} vs {bname}  (n={len(common)})")
        print(f"   better {better}  worse {worse}  unchanged {same}   "
              f"median delta {np.median(d):+.2f} cm   mean delta {d.mean():+.2f} cm")
        print(f"   frames >{args.thresh:g}cm: {fa} -> {fb}   "
              f"fixed {int(((a > args.thresh) & (b <= args.thresh)).sum())}  "
              f"broken {int(((a <= args.thresh) & (b > args.thresh)).sum())}")
        big = [(k, a[i], b[i]) for i, k in enumerate(common) if abs(d[i]) > 1.0]
        for k, x, y in sorted(big, key=lambda z: z[1] - z[2])[:8]:
            print(f"      {k[0]}/{k[1]:06d}  {x:8.2f} -> {y:8.2f}")
        print()


if __name__ == "__main__":
    main()
