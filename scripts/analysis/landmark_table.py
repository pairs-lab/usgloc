#!/usr/bin/env python3
"""Read the landmark experiments back as tables.

    python3 scripts/analysis/landmark_table.py            # everything it finds
    python3 scripts/analysis/landmark_table.py --dir results/landmark

Three tables:

1. LANDMARK SET SWEEP (exp_landmark.py --no-corr): the pose-free absolute pose
   per landmark configuration.
2. FLOOR (exp_floor_matcher.py, spec 'lm'): the same estimator measured by the
   canonical tool, from the ground truth and from the 20 cm / 10 deg start, with
   the per-frame better/worse split against the LoFTR run on the SAME frames
   (the reference JSON is found automatically under results/).
3. INITIALISER (exp_landmark.py with corrections): what the LoFTR correction
   reaches when started from the landmark pose, versus from the ground truth,
   from the perturbation and from the closest admissible retrieval anchor.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(p):
    with open(p) as f:
        return json.load(f)


def ref_loftr(scene, pert, n_frames):
    """The LoFTR floor run on the same map / frames, wherever it was written."""
    tail = "_p0.2m10deg" if pert else ""
    best = None
    for p in glob.glob(f"results/*/floor_{scene}_loftr_r1{tail}.json"):
        d = load(p)
        if d.get("map") != f"maps/{scene}_r1.pt" or d.get("n_frames") != n_frames:
            continue
        if bool(d.get("perturb")) != bool(pert):
            continue
        if best is None or os.path.getmtime(p) > os.path.getmtime(best[0]):
            best = (p, d)
    return best


def split(a_rows, b_rows):
    """Per-frame better/worse of run a against run b (translation error)."""
    A = {r["frame"]: r for r in a_rows}
    B = {r["frame"]: r for r in b_rows}
    keys = sorted(set(A) & set(B))
    better = worse = only_a = only_b = neither = 0
    for k in keys:
        x, y = A[k], B[k]
        if x["success"] and y["success"]:
            if x["t_cm"] < y["t_cm"]:
                better += 1
            elif x["t_cm"] > y["t_cm"]:
                worse += 1
        elif x["success"]:
            only_a += 1
        elif y["success"]:
            only_b += 1
        else:
            neither += 1
    return dict(n=len(keys), better=better, worse=worse, only_a=only_a,
                only_b=only_b, neither=neither)


def split_lm(rows, ref_rows):
    """Per-frame better/worse of the PRIOR-FREE landmark pose (exp_landmark.py
    rows) against a LoFTR correction run on the same frames."""
    A = {r["frame"]: r for r in rows}
    B = {r["frame"]: r for r in ref_rows}
    keys = sorted(set(A) & set(B))
    better = worse = only_a = only_b = 0
    for k in keys:
        x, y = A[k], B[k]
        if x["lm_ok"] and y["success"]:
            if x["lm_cm"] < y["t_cm"]:
                better += 1
            else:
                worse += 1
        elif x["lm_ok"]:
            only_a += 1
        elif y["success"]:
            only_b += 1
    return dict(n=len(keys), better=better, worse=worse, only_a=only_a,
                only_b=only_b)


def field_view_split(rows, stride=5):
    """The descriptor field was built from every ``stride``-th train frame, so
    some evaluation frames contributed to it.  Median error on each group."""
    seen = [r["lm_cm"] for r in rows if r["lm_ok"] and r["frame"] % stride == 0]
    unseen = [r["lm_cm"] for r in rows if r["lm_ok"] and r["frame"] % stride != 0]
    return (float(np.median(seen)) if seen else float("nan"), len(seen),
            float(np.median(unseen)) if unseen else float("nan"), len(unseen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/landmark")
    args = ap.parse_args()
    os.chdir(ROOT)
    d = Path(args.dir)

    for sub in ("cpu60", "sweep"):
        runs = sorted(glob.glob(str(d / sub / "lm_*.json")))
        runs = [p for p in runs if load(p)["n_frames"] in (20, 60)]
        if not runs:
            continue
        print(f"\n== 1b. prior-free landmark pose vs the LoFTR correction on the "
              f"SAME frames ({sub})")
        print(f"{'scene':<8}{'variant':<22}{'lm cm':>7}{'loftr cm':>9}{'reg':>6}"
              f"  better/worse/only-lm/only-loftr   field-view cm (n) / unseen cm (n)")
        for p in runs:
            s = load(p)
            r = ref_loftr(s["scene"], False, s["n_frames"])
            sp = split_lm(s["frames"], r[1]["frames"]) if r else None
            a, na, b, nb = field_view_split(s["frames"])
            print(f"{s['scene']:<8}{s['variant']:<22}{s['lm_cm']:7.2f}"
                  f"{(r[1]['floor_cm'] if r else float('nan')):9.3f}"
                  f"{100*s['lm_reg_rate']:5.0f}%  "
                  + (f"{sp['better']:3d}/{sp['worse']:3d}/{sp['only_a']:3d}/"
                     f"{sp['only_b']:3d}" if sp else "   (no reference)")
                  + f"            {a:6.2f} ({na:2d}) / {b:6.2f} ({nb:2d})")

    sweeps = sorted(glob.glob(str(d / "sweep" / "lm_*.json")))
    if sweeps:
        print("\n== 1. landmark set sweep (pose-free absolute pose, no correction)")
        print(f"{'scene':<8}{'feats':<10}{'variant':<24}{'#lm':>8}{'reg':>6}"
              f"{'cm':>8}{'deg':>7}{'inl':>7}{'matches':>8}"
              f"{'assoc med':>10}{'p4':>6}{'p12':>6}{'bad5':>6}{'worst':>8}{'s':>6}")
        for p in sweeps:
            s = load(p)
            f = (s.get("feats") or "-").split("/")[-1].replace(".pt", "")
            print(f"{s['scene']:<8}{f.split('_')[-1]:<10}{s['variant']:<24}"
                  f"{s['landmark_info']['n_landmarks']:>8}"
                  f"{100*s['lm_reg_rate']:5.0f}%{s['lm_cm']:8.2f}{s['lm_deg']:7.2f}"
                  f"{s['lm_inliers']:7.0f}{s['lm_matches']:8.0f}"
                  f"{s['assoc_med']:10.2f}{s['assoc_p4']:6.2f}{s['assoc_p12']:6.2f}"
                  f"{100*s['lm_bad5']:5.0f}%{s['lm_worst_cm']:8.1f}{s['lm_sec']:6.2f}")

    floors = sorted(glob.glob(str(d / "floor" / "floor_*_lm_*.json")))
    if floors:
        print("\n== 2. floor by the canonical tool (exp_floor_matcher.py, spec 'lm')")
        print(f"{'scene':<8}{'start':<7}{'floor cm':>9}{'deg':>7}{'reg':>6}{'inl':>7}"
              f"{'s/corr':>8}{'GB':>6}   vs LoFTR on the same frames "
              f"(better/worse/only-lm/only-loftr)")
        for p in floors:
            s = load(p)
            pert = bool(s.get("perturb"))
            r = ref_loftr(s["scene"], pert, s["n_frames"])
            sp = split(s["frames"], r[1]["frames"]) if r else None
            txt = (f"{sp['better']:3d}/{sp['worse']:3d}/{sp['only_a']:3d}/{sp['only_b']:3d}"
                   f"  (LoFTR {r[1]['floor_cm']:.3f} cm, reg {100*r[1]['reg_rate']:.0f}%,"
                   f" {r[0]})" if sp else "  (no LoFTR reference found)")
            print(f"{s['scene']:<8}{'pert' if pert else 'gt':<7}{s['floor_cm']:9.3f}"
                  f"{s['floor_deg']:7.3f}{100*s['reg_rate']:5.0f}%"
                  f"{s['median_inliers']:7.0f}{s['sec_median']:8.2f}"
                  f"{s['peak_gpu_gb']:6.2f}   {txt}")

    corrs = sorted(glob.glob(str(d / "corr" / "lm_*.json")))
    if corrs:
        print("\n== 3. the landmark pose as an INITIALISER (LoFTR correction chained on)")
        print(f"{'scene':<8}{'variant':<20}{'lm cm':>7}{'anc cm':>8}{'ancSO':>7}"
              f"{'beat':>6}{'beatSO':>7}  | correction floor from each start (cm), "
              f"reg%, >5cm%")
        for p in corrs:
            s = load(p)
            print(f"{s['scene']:<8}{s['variant']:<20}{s['lm_centre_cm']:7.2f}"
                  f"{s['anchor_centre_cm']:8.1f}{s['anchor_so_centre_cm']:7.1f}"
                  f"{100*s['lm_beats_anchor_centre']:5.0f}%"
                  f"{100*s['lm_beats_anchor_so_centre']:6.0f}%")
            for name in ("gt", "pert", "lm", "anchor", "anchor_so"):
                print(f"         {name:<12}{s[f'floor_{name}_cm']:8.3f} cm "
                      f"{s[f'floor_{name}_deg']:6.3f} deg  reg "
                      f"{100*s[f'reg_{name}']:3.0f}%  >5cm "
                      f"{100*s[f'bad5_{name}']:3.0f}%  worst "
                      f"{s[f'worst_{name}_cm']:8.2f}  inl {s[f'inl_{name}']:5.0f}"
                      f"  {s[f'sec_{name}']:5.2f} s")
            for ref in ("gt", "pert", "anchor", "anchor_so"):
                print(f"         lm-start vs {ref:<9} better {s[f'vs_{ref}_better']:3d}"
                      f"  worse {s[f'vs_{ref}_worse']:3d}"
                      f"  (both ok {s[f'vs_{ref}_both_ok']:3d},"
                      f" only-lm {s[f'vs_{ref}_only_lm_ok']:2d},"
                      f" only-{ref} {s[f'vs_{ref}_only_ref_ok']:2d})")
    print()


if __name__ == "__main__":
    main()
