#!/usr/bin/env python3
"""Tabulate the fine-tune recipe sweep written by exp_ft_recipe.py.

    python3 scripts/analysis/ft_recipe_table.py [--out results/ft_recipe]
"""
import argparse
import json
from pathlib import Path

ORDER = ["base", "i15k", "i30k", "s2", "s1", "lrm3x", "lrm3rd",
         "geo3x", "geo10x", "col0", "dens", "best"]
SCENES = ["office", "stairs", "pumpkin", "chess"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ft_recipe")
    args = ap.parse_args()
    out = Path(args.out)

    base = {}
    for f in sorted((out / "baseline").glob("floor_*_loftr*.json")):
        d = json.loads(f.read_text())
        pert = d["perturb"] is not None
        key = (d["scene"], "sel" if "_sel" in f.name else "ft")
        base.setdefault(key, {})["pert" if pert else "gt"] = d

    print(f"{'scene':<10}{'variant':<9}{'floor GT':>9}{'floor pert':>11}{'regGT':>7}"
          f"{'regP':>6}{'PSNR':>7}{'splats':>9}{'build s':>9}")
    for s in SCENES:
        for kind in ("ft", "sel"):
            d = base.get((s, kind))
            if not d or "gt" not in d:
                continue
            g, p = d["gt"], d.get("pert")
            print(f"{s:<10}{'BASE ' + kind:<9}{g['floor_cm']:9.3f}"
                  f"{(p['floor_cm'] if p else float('nan')):11.3f}"
                  f"{100*g['reg_rate']:6.0f}%{100*(p['reg_rate'] if p else 0):5.0f}%"
                  f"{(g['psnr_gt'] or 0):7.2f}{g['n_gaussians']:9d}{'-':>9}")
        rows = []
        for v in ORDER:
            f = out / f"{s}_{v}.json"
            if f.is_file():
                rows.append(json.loads(f.read_text()))
        for r in rows:
            g, p = r["gt"], r["pert"]
            print(f"{s:<10}{r['variant']:<9}{g['floor_cm']:9.3f}{p['floor_cm']:11.3f}"
                  f"{100*g['reg_rate']:6.0f}%{100*p['reg_rate']:5.0f}%"
                  f"{(g['psnr_gt'] or 0):7.2f}{g['n_gaussians']:9d}{r['build_s']:9.0f}")
        print()


if __name__ == "__main__":
    main()
