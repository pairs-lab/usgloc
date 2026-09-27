#!/usr/bin/env python3
"""Compare run_7scenes.py result JSONs for the adaptive virtual-view search.

Same contract as ``cmp_runs.py`` (frame-by-frame, first directory is the
baseline) plus the three numbers this experiment is judged on: the SR@5cm/5deg
over ALL recorded frames, the fraction beyond 5 cm, and how often the search
actually fired / was accepted.

    python3 scripts/analysis/vsearch_table.py results/vsearch/base \
        results/vsearch/vs_a --scene stairs
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


def col(recs, key, default=0.0):
    return np.array([r.get(key, default) for r in recs.values()], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset", default="7scenes")
    ap.add_argument("--thresh", type=float, default=5.0, help="cm")
    ap.add_argument("--deg", type=float, default=5.0)
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

    print(f"{args.scene}   success @ {args.thresh:g} cm / {args.deg:g} deg")
    print(f"{'arm':<22}{'n':>4}{'med cm':>8}{'p75':>7}{'>5cm':>7}{'SR':>7}"
          f"{'fire':>7}{'used':>7}{'views':>7}{'s/q':>7}")
    for name, recs, j in arms:
        e = col(recs, "t_err") * 100
        rr = col(recs, "r_err")
        sr = float(((e <= args.thresh) & (rr <= args.deg)).mean())
        fired = col(recs, "vsearch_fired")
        used = col(recs, "vsearch_used")
        views = col(recs, "n_vviews")
        t = col(recs, "t_localize_s")
        print(f"{name:<22}{len(e):>4}{np.median(e):8.2f}{np.percentile(e, 75):7.2f}"
              f"{100 * (e > args.thresh).mean():6.0f}%{100 * sr:6.0f}%"
              f"{100 * fired.mean():6.0f}%{100 * used.mean():6.0f}%"
              f"{views.mean():7.1f}{np.median(t):7.1f}")

    bname, brecs = arms[0][0], arms[0][1]
    print()
    for name, recs, _ in arms[1:]:
        common = sorted(set(brecs) & set(recs))
        a = np.array([brecs[k]["t_err"] for k in common]) * 100
        b = np.array([recs[k]["t_err"] for k in common]) * 100
        d = b - a
        tol = 0.05
        print(f"{name} vs {bname}  (n={len(common)})")
        print(f"   better {int((d < -tol).sum())}  worse {int((d > tol).sum())}  "
              f"unchanged {int((np.abs(d) <= tol).sum())}   "
              f"median delta {np.median(d):+.2f} cm")
        fa, fb = a > args.thresh, b > args.thresh
        print(f"   frames >{args.thresh:g}cm: {int(fa.sum())} -> {int(fb.sum())}   "
              f"fixed {int((fa & ~fb).sum())}  broken {int((~fa & fb).sum())}")
        big = [(k, a[i], b[i]) for i, k in enumerate(common) if abs(d[i]) > 1.0]
        for k, x, y in sorted(big, key=lambda z: z[1] - z[2])[:10]:
            print(f"      {k[0]}/{k[1]:06d}  {x:8.2f} -> {y:8.2f}")
        print()


if __name__ == "__main__":
    main()
