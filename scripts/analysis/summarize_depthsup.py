#!/usr/bin/env python3
"""Collect the depth-supervision floor runs into one table.

    python3 scripts/analysis/summarize_depthsup.py --dir results/depthsup/floor
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/depthsup/floor")
    args = ap.parse_args()
    d = Path(args.dir)
    rows = {}
    for f in sorted(d.glob("floor_*.json")):
        j = json.load(open(f))
        key = (j["scene"], j["suffix"])
        slot = "pert" if j.get("perturb") else "gt"
        rows.setdefault(key, {})[slot] = j
    print(f"{'scene':<9}{'map':<10}{'floorGT':>9}{'degGT':>8}{'regGT':>7}{'inl':>7}"
          f"{'PSNR':>7}{'floorP':>9}{'degP':>8}{'regP':>7}")
    for (scene, suf), v in sorted(rows.items()):
        g, p = v.get("gt"), v.get("pert")
        def f(x, k, fmt="{:9.3f}"):
            return fmt.format(x[k]) if x else " " * 9
        print(f"{scene:<9}{suf:<10}"
              f"{f(g,'floor_cm')}{f(g,'floor_deg','{:8.3f}')}"
              f"{(100*g['reg_rate'] if g else 0):6.0f}%{(g['median_inliers'] if g else 0):7.0f}"
              f"{(g['psnr_mean'] if g else 0):7.2f}"
              f"{f(p,'floor_cm')}{f(p,'floor_deg','{:8.3f}')}"
              f"{(100*p['reg_rate'] if p else 0):6.0f}%")


if __name__ == "__main__":
    main()
