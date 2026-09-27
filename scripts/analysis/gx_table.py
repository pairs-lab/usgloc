#!/usr/bin/env python3
"""Tabulate the densification experiment (results/densify/<scene>_<variant>.json).

    python3 scripts/analysis/gx_table.py --dir results/densify [--baseline r1]

Prints, per scene and variant: the Gaussian count, the fit PSNR, the PSNR at
the ground-truth pose, the floor from GT and from the 20 cm / 10 deg start
(cm and deg), the registration rates, the median inlier count, the build time,
and -- against ``--baseline`` -- how many of the shared frames each variant
improves and worsens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def paired(a: dict, b: dict, key: str = "gt_frames"):
    """Paired comparison of ``a`` against baseline ``b`` over the frames that
    register in BOTH.

    The plain floor is a median over whichever frames registered, so a variant
    that drops the hard frames looks better while being worse: always read it
    next to the registration rate and this paired median.
    Returns (median_base, median_var, n_common, better, worse).
    """
    fb = {f["frame"]: f for f in b.get(key, []) if f["success"]}
    fa = {f["frame"]: f for f in a.get(key, []) if f["success"]}
    common = sorted(set(fa) & set(fb))
    if not common:
        return (float("nan"), float("nan"), 0, 0, 0)
    import statistics
    mb = statistics.median(fb[k]["t_cm"] for k in common)
    mv = statistics.median(fa[k]["t_cm"] for k in common)
    better = sum(1 for k in common if fa[k]["t_cm"] < fb[k]["t_cm"])
    worse = sum(1 for k in common if fa[k]["t_cm"] > fb[k]["t_cm"])
    return (mb, mv, len(common), better, worse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/densify")
    ap.add_argument("--baseline", default="r1")
    args = ap.parse_args()
    d = Path(args.dir)
    recs = []
    for p in sorted(d.glob("*_*.json")):
        if p.name.startswith(("floor_", "score_", "summary")):
            continue
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "variant" in r:
            recs.append(r)
    scenes = sorted({r["scene"] for r in recs})
    hdr = (f"{'scene':<8}{'variant':<9}{'splats':>9}{'PSNR@GT':>8}"
           f"{'flGT':>7}{'regGT':>6}{'flP':>7}{'regP':>6}{'inl':>6}{'build':>7}"
           f"  {'pairedGT ' + args.baseline + '->var':>24}{'b/w':>8}"
           f"  {'pairedPert':>22}{'b/w':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in scenes:
        rows = [r for r in recs if r["scene"] == s]
        base = next((r for r in rows if r["variant"] == args.baseline), None)
        for r in sorted(rows, key=lambda x: x["gt"]["floor_cm"]):
            g, p = r["gt"], r["pert"]
            cells = []
            for key in ("gt_frames", "pert_frames"):
                if base is None:
                    cells += ["-", "-"]
                    continue
                mb, mv, n, bet, wor = paired(r, base, key)
                cells += [f"{mb:.3f} -> {mv:.3f} (n={n})", f"{bet}/{wor}"]
            print(f"{s:<8}{r['variant']:<9}{r['n_gaussians']:>9}"
                  f"{(g.get('psnr_gt') or 0):8.2f}"
                  f"{g['floor_cm']:7.3f}{100 * g['reg_rate']:5.0f}%"
                  f"{p['floor_cm']:7.3f}{100 * p['reg_rate']:5.0f}%"
                  f"{g['median_inliers']:6.0f}{(r.get('build_s') or 0):7.0f}"
                  f"  {cells[0]:>24}{cells[1]:>8}  {cells[2]:>22}{cells[3]:>8}")
        print()


if __name__ == "__main__":
    main()
