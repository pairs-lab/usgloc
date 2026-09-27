#!/usr/bin/env python3
"""Compare map-fit variants against a baseline, per scene AND per frame.

``exp_floor_matcher.py`` writes one JSON per (scene, matcher spec, perturbation)
holding both the summary and the per-frame rows.  A 0.05 cm median shift on 20
frames is noise, so every claimed win has to come with the better/worse split
over the frames that BOTH runs registered; this script prints it.

    python3 scripts/analysis/gsfit_table.py \
        --baseline results/ft_recipe/baseline --runs results/gsfit \
        --scenes chess office stairs

Rows are (scene, variant) x (from GT, from a 0.2 m / 10 deg perturbation):
PSNR at the fit/GT pose, median floor cm/deg, registration rate, median
inliers, seconds per correct(), and against the baseline the paired
better/worse frame counts and the median of the per-frame differences.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(d: Path):
    """dir -> {(scene, variant, pert_tag): summary dict with frames}."""
    out = {}
    for f in sorted(glob.glob(str(d / "floor_*.json"))):
        m = json.load(open(f))
        pert = "" if m.get("perturb") is None else "p"
        variant = Path(m["map"]).stem            # e.g. chess_ft, chess_gsS
        scene = m["scene"]
        tag = variant[len(scene) + 1:] if variant.startswith(scene + "_") else variant
        out[(scene, tag, pert)] = m
    return out


def paired(a, b):
    """(better, worse, median delta cm) over the frames both runs registered.

    ``a`` is the variant and ``b`` the baseline; a NEGATIVE delta means the
    variant is closer to the ground truth on that frame.
    """
    fb = {r["frame"]: r for r in b["frames"] if r["success"]}
    d = [r["t_cm"] - fb[r["frame"]]["t_cm"]
         for r in a["frames"] if r["success"] and r["frame"] in fb]
    if not d:
        return 0, 0, float("nan"), 0
    d = np.asarray(d)
    return int((d < 0).sum()), int((d > 0).sum()), float(np.median(d)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/ft_recipe/baseline")
    ap.add_argument("--runs", nargs="+", default=["results/gsfit"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--base-tag", default="ft", help="variant tag of the baseline map")
    args = ap.parse_args()
    os.chdir(ROOT)

    base = load(Path(args.baseline))
    runs = {}
    for d in args.runs:
        runs.update(load(Path(d)))
    scenes = args.scenes or sorted({k[0] for k in list(base) + list(runs)})

    hdr = (f"{'scene':<11}{'variant':<12}{'start':<6}{'PSNR':>7}{'floor cm':>10}"
           f"{'deg':>7}{'reg':>6}{'inl':>7}{'s':>6}{'vs base':>10}{'btr/wrs':>9}{'n':>4}")
    print(hdr)
    print("-" * len(hdr))
    for s in scenes:
        tags = [args.base_tag] + sorted({k[1] for k in runs if k[0] == s})
        for tag in tags:
            for pert, name in (("", "GT"), ("p", "+20cm")):
                m = base.get((s, tag, pert)) or runs.get((s, tag, pert))
                if m is None:
                    continue
                bm = base.get((s, args.base_tag, pert))
                cmp_s, bw, n = "", "", ""
                if bm is not None and m is not bm:
                    btr, wrs, dmed, n_ = paired(m, bm)
                    cmp_s = f"{dmed:+10.3f}"
                    bw = f"{btr}/{wrs}"
                    n = str(n_)
                psnr = m.get("psnr_gt")
                print(f"{s:<11}{tag:<12}{name:<6}"
                      f"{('n/a' if psnr is None else f'{psnr:.2f}'):>7}"
                      f"{m['floor_cm']:10.3f}{m['floor_deg']:7.3f}"
                      f"{100 * m['reg_rate']:5.0f}%{m['median_inliers']:7.0f}"
                      f"{m['sec_median']:6.2f}{cmp_s:>10}{bw:>9}{n:>4}")
        print()


if __name__ == "__main__":
    main()
