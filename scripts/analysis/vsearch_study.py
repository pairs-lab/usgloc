#!/usr/bin/env python3
"""Read exp_vsearch.py's JSON and answer the four design questions.

  trigger  -- which weak-support statistic separates the failures, and where
              the threshold goes (fire rate vs failures captured)
  n        -- best-of-first-k over the recorded views (the views of one
              configuration were all evaluated, so the n curve is free)
  radius   -- the (law, sigma_t, sigma_r) grid, each at its best n
  score    -- wmass vs inliers vs ratio, plus the ORACLE (min error over the
              sampled views), which says whether sampling or selection is the limit

    python3 scripts/analysis/vsearch_study.py results/vsearch/diag_stairs_los.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def sel_err(rows, k, score, centre_cm):
    """Error the pipeline would report using the first ``k`` views.

    Sample 0 is the unperturbed pose; when it wins, the search is a no-op and
    the pipeline keeps the pose it already had (``centre_cm``).
    """
    ok = [(j, r) for j, r in enumerate(rows[:k]) if r["ok"] and r["err_cm"] is not None]
    if not ok:
        return centre_cm, 0
    key = {"wmass": lambda r: r["wm"], "inliers": lambda r: r["inl"],
           "ratio": lambda r: r["inl"] / max(r["nm"], 1),
           "rms": lambda r: -r["rms"]}[score]
    j, r = max(ok, key=lambda jr: key(jr[1]))
    return (r["err_cm"], j) if j > 0 else (centre_cm, 0)


def oracle(rows, k):
    e = [r["err_cm"] for r in rows[:k] if r["ok"] and r["err_cm"] is not None]
    return min(e) if e else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--thresh", type=float, default=5.0)
    ap.add_argument("--ks", default="2,4,8,16,32,64")
    args = ap.parse_args()
    j = json.load(open(args.json))
    F = j["frames"]
    ks = [int(x) for x in args.ks.split(",")]
    thr = args.thresh

    e = np.array([f["t_err_cm"] for f in F])
    print(f"{len(F)} frames   baseline median {np.median(e):.2f} cm   "
          f">{thr:g}cm {100*(e>thr).mean():.0f}%   "
          f"SR {100*np.mean((e<=thr) & (np.array([f['r_err_deg'] for f in F])<=5)):.0f}%")

    # ---- 1. trigger ---------------------------------------------------
    print("\n--- trigger: which weak-support statistic separates the failures")
    bad = e > thr
    for stat, lo_is_weak, cands in (
            ("best_inliers", True, [500, 800, 1000, 1200, 1500, 1800, 2000, 2500]),
            ("best_ratio", True, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]),
            ("best_psnr", True, [14, 15, 16, 17, 18, 19, 20])):
        v = np.array([f.get(stat, 0.0) for f in F], dtype=float)
        print(f"  {stat:<13} failures {np.round(v[bad], 2).tolist()[:12]}")
        print(f"  {'':<13} ok med {np.median(v[~bad]):.2f}  ok p05 {np.percentile(v[~bad], 5):.2f}")
        for t in cands:
            fire = v < t if lo_is_weak else v > t
            if fire.sum() == 0:
                continue
            print(f"      thr {t:<6} fires {100*fire.mean():3.0f}%  "
                  f"captures {int((fire & bad).sum())}/{int(bad.sum())} failures  "
                  f"wasted {int((fire & ~bad).sum())}")

    # ---- 2/3/4. the grid ----------------------------------------------
    keys = sorted({k for f in F for k in f.get("configs", {})})
    if not keys:
        return
    searched = [f for f in F if f.get("configs")]
    print(f"\n--- {len(searched)} frames searched "
          f"(baseline median {np.median([f['t_err_cm'] for f in searched]):.2f} cm, "
          f"{int(sum(f['t_err_cm'] > thr for f in searched))} of them beyond {thr:g} cm)")
    print(f"\n{'config':<18}{'n':>4}{'score':>9}{'med cm':>9}{'>5cm':>7}"
          f"{'fixed':>7}{'broke':>7}{'orcl':>8}{'fire%':>7}{'sec':>7}")
    for key in keys:
        secs = np.array([f["configs"][key]["sec"] for f in searched
                         if key in f["configs"]])
        for k in ks:
            for score in ("wmass", "inliers", "ratio"):
                fin, base, nfire = [], [], 0
                for f in searched:
                    if key not in f["configs"]:
                        continue
                    c = f["configs"][key]["sec"], f["configs"][key]["views"]
                    centre = f.get("centre_err_cm", f["t_err_cm"])
                    ef, jj = sel_err(c[1], k, score, centre)
                    # the pipeline only replaces its answer when the search
                    # actually produced one; otherwise the baseline stands
                    fin.append(ef if jj > 0 else f["t_err_cm"])
                    base.append(f["t_err_cm"])
                    nfire += int(jj > 0)
                fin, base = np.array(fin), np.array(base)
                orc = np.array([oracle(f["configs"][key]["views"], k) or np.nan
                                for f in searched if key in f["configs"]])
                if score == "wmass" or k == max(ks):
                    print(f"{key if (k==ks[0] and score=='wmass') else '':<18}"
                          f"{k:>4}{score:>9}{np.median(fin):9.2f}"
                          f"{100*(fin>thr).mean():6.0f}%"
                          f"{int(((base>thr)&(fin<=thr)).sum()):>7}"
                          f"{int(((base<=thr)&(fin>thr)).sum()):>7}"
                          f"{np.nanmedian(orc):8.2f}{100*nfire/max(len(fin),1):6.0f}%"
                          f"{secs.mean()*k/max(len(f['configs'][key]['views']),1):7.1f}")
        print()


if __name__ == "__main__":
    main()
