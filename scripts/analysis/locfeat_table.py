#!/usr/bin/env python3
"""Tabulate every exp_locfeat_sanity.py run in a results directory.

    python3 scripts/analysis/locfeat_table.py results/locfeat
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROWS = ("field_lm", "field_lm_top", "field_dense", "loftr", "loftr_top",
        "loftr_raw", "field_dense_pert", "loftr_pert")


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/locfeat")
    runs = []
    for p in sorted(d.glob("sanity_*.json")):
        with open(p) as f:
            runs.append((p.name, json.load(f)))
    if not runs:
        raise SystemExit(f"no sanity_*.json under {d}")
    print(f"{'run':<34}{'D':>4}{'N_lm':>9}{'cosGT':>7}{'shuf':>7}{'cosP':>7}"
          f"{'a%':>5}{'s/rend':>8}{'s/lm':>7}")
    for name, s in runs:
        print(f"{name[:34]:<34}{s['dim']:>4}{s['n_landmarks']:>9}"
              f"{s.get('cos_gt', float('nan')):7.3f}{s.get('cos_shuf', float('nan')):7.3f}"
              f"{s.get('cos_pert', float('nan')):7.3f}"
              f"{100*s.get('alpha_frac', float('nan')):5.0f}"
              f"{s.get('sec_render_feat', float('nan')):8.3f}"
              f"{s.get('sec_landmark', float('nan')):7.3f}")
    for key in ROWS:
        present = [(n, s) for n, s in runs if key in s]
        if not present:
            continue
        print(f"\n--- {key} ---")
        print(f"{'run':<34}{'n':>7}{'med px':>8}" +
              "".join(f"{k:>7}" for k in ("p1", "p2", "p4", "p8", "p12")))
        for n, s in present:
            v = s[key]
            print(f"{n[:34]:<34}{v['n']:7.0f}{v['med']:8.2f}" +
                  "".join(f"{v[k]:7.3f}" for k in ("p1", "p2", "p4", "p8", "p12")))


if __name__ == "__main__":
    main()
