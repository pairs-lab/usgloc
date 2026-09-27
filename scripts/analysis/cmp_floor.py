#!/usr/bin/env python3
"""Per-scene comparison table for exp_floor_matcher.py runs.

20 frames is a small sample, so a median shift of 0.05 cm means nothing on its
own.  This reads the per-spec ``floor_<scene>_<tag>.json`` files a run writes
(each carries the per-frame rows) and prints, for every variant against a
chosen baseline spec and per scene:

    floor cm / floor deg / mean cm / reg rate / median inliers / s per corr
    better  worse  (frames whose translation error improved / worsened)
    median per-frame delta and a two-sided sign test p-value

    python3 scripts/analysis/cmp_floor.py results/estimator_lift/floor \
        --base loftr --scenes chess office stairs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def sign_p(b, w):
    """Two-sided sign-test p-value for b better out of b + w."""
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    c = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * c / 2.0 ** n)


def load(d: Path, ptag: str):
    runs = {}
    for f in sorted(d.glob("floor_*.json")):
        j = json.load(open(f))
        if not isinstance(j, dict) or "frames" not in j:
            continue
        pt = "" if j.get("perturb") is None else "p"
        if pt != ptag:
            continue
        runs[(j["scene"], j["matcher"])] = j
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base", default="loftr")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--perturbed", action="store_true",
                    help="compare the --perturb runs instead of the from-GT ones")
    args = ap.parse_args()
    ptag = "p" if args.perturbed else ""
    runs = {}
    for d in args.dirs:
        runs.update(load(Path(d), ptag))
    scenes = args.scenes or sorted({s for s, _ in runs})
    specs = []
    for _, m in runs:
        if m not in specs:
            specs.append(m)
    specs = [args.base] + [s for s in specs if s != args.base]
    head = ("start = " + ("perturbed" if args.perturbed else "ground truth"))
    print(f"\n=== {head} ===")
    print(f"{'scene':<9}{'spec':<26}{'floor cm':>9}{'deg':>7}{'mean cm':>9}{'reg':>6}"
          f"{'inl':>6}{'s':>6}{'better':>8}{'worse':>7}{'d_med cm':>10}{'p':>8}")
    for sc in scenes:
        base = runs.get((sc, args.base))
        for sp in specs:
            r = runs.get((sc, sp))
            if r is None:
                continue
            b = w = 0
            deltas = []
            if base is not None and sp != args.base:
                bf = {x["frame"]: x for x in base["frames"]}
                for x in r["frames"]:
                    y = bf.get(x["frame"])
                    if y is None:
                        continue
                    if x["success"] and y["success"]:
                        dd = x["t_cm"] - y["t_cm"]
                        deltas.append(dd)
                        if dd < 0:
                            b += 1
                        elif dd > 0:
                            w += 1
                    elif x["success"] != y["success"]:
                        (b, w) = (b + 1, w) if x["success"] else (b, w + 1)
            deltas.sort()
            dm = deltas[len(deltas) // 2] if deltas else float("nan")
            p = sign_p(b, w) if (b or w) else float("nan")
            print(f"{sc:<9}{sp:<26}{r['floor_cm']:9.3f}{r['floor_deg']:7.3f}"
                  f"{r['mean_cm']:9.3f}{100*r['reg_rate']:5.0f}%"
                  f"{r['median_inliers']:6.0f}{r['sec_median']:6.2f}"
                  + ("" if sp == args.base else
                     f"{b:8d}{w:7d}{dm:10.3f}{p:8.3f}"))
        print()
    # pooled over scenes
    print(f"{'MEAN':<9}{'spec':<26}{'floor cm':>9}{'deg':>7}{'mean cm':>9}{'reg':>6}"
          f"{'':>6}{'':>6}{'better':>8}{'worse':>7}")
    for sp in specs:
        rs = [runs[(sc, sp)] for sc in scenes if (sc, sp) in runs]
        if len(rs) != len(scenes):
            continue
        b = w = 0
        if sp != args.base:
            for sc in scenes:
                base = runs.get((sc, args.base))
                r = runs[(sc, sp)]
                bf = {x["frame"]: x for x in base["frames"]}
                for x in r["frames"]:
                    y = bf.get(x["frame"])
                    if y is None:
                        continue
                    if x["success"] and y["success"]:
                        if x["t_cm"] < y["t_cm"]:
                            b += 1
                        elif x["t_cm"] > y["t_cm"]:
                            w += 1
                    elif x["success"] != y["success"]:
                        (b, w) = (b + 1, w) if x["success"] else (b, w + 1)
        f = sum(r["floor_cm"] for r in rs) / len(rs)
        g = sum(r["floor_deg"] for r in rs) / len(rs)
        mc = sum(r["mean_cm"] for r in rs) / len(rs)
        rr = sum(r["reg_rate"] for r in rs) / len(rs)
        print(f"{'':<9}{sp:<26}{f:9.3f}{g:7.3f}{mc:9.3f}{100*rr:5.0f}%{'':6}{'':6}"
              + ("" if sp == args.base else f"{b:8d}{w:7d}"
                 + f"   p={sign_p(b, w):.4f}"))


if __name__ == "__main__":
    main()
