#!/usr/bin/env python3
"""Emit LaTeX table rows from run_7scenes.py result JSONs.

    python3 scripts/make_tables.py results/final_seed0 [results/final_seed1 ...]

With several directories the per-scene value is the MEDIAN over seeds, which
is the protocol both papers state. Prints the Table I row, the per-scene
calibration rows, and a diagnostic table (snap rate, inliers, runtime).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


def load(d, scene, ds="7scenes"):
    p = Path(d) / f"{ds}_{scene}.json"
    return json.load(open(p)) if p.is_file() else None


def temper(recs, tau, dof=6):
    return [dict(r, d2=r["d2"] / tau,
                 sigma_t=r["sigma_t"] * tau ** 0.5) for r in recs]


def fix_record(r):
    """Repair a record whose stored covariance was indefinite (negative or
    non-finite d2 / logdet): recompute from the saved pose, covariance and
    ground truth after projecting the covariance to positive-definite.
    Returns None when the record cannot be repaired (no saved covariance)."""
    ok = np.isfinite(r.get("d2", np.nan)) and np.isfinite(r.get("logdet", np.nan)) and r["d2"] >= 0
    if ok:
        return r
    if not all(k in r for k in ("T", "Sigma", "T_gt")):
        return None
    import torch
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from usgloc import se3
    from usgloc.posterior import _project_pd
    T, S, G = (torch.tensor(r[k], dtype=torch.float64) for k in ("T", "Sigma", "T_gt"))
    S = _project_pd(S)
    e = se3.relative_tangent(T, G)
    d2 = float(e @ torch.linalg.solve(S, e)); logdet = float(torch.logdet(S))
    r = dict(r, d2=d2, logdet=logdet, nll=0.5 * (d2 + logdet + 6 * np.log(2 * np.pi)),
             sigma_t=float(torch.sqrt(torch.trace(S[:3, :3]) / 3)))
    return r


def stats(d):
    """Metrics from one run, applying the fitted temperature."""
    tau = d.get("tau", 1.0)
    fixed = [fix_record(r) for r in d["records"]]
    n_drop = sum(f is None for f in fixed)
    if n_drop:
        print(f"% note: {n_drop} record(s) without a repairable covariance dropped from calibration stats", file=sys.stderr)
    recs = temper([f for f in fixed if f is not None], tau)
    t = np.array([100 * r["t_err"] for r in recs])
    rot = np.array([r["r_err"] for r in recs])
    d2 = np.array([r["d2"] for r in recs])
    from scipy.stats import chi2
    cov = {a: float((d2 <= chi2.ppf(a, 6)).mean()) for a in (0.683, 0.954, 0.997)}
    sn = np.array([r.get("snapped", False) for r in recs])
    inl = np.array([r.get("inliers", 0) for r in recs])
    tl = np.array([r.get("t_localize_s", np.nan) for r in recs])
    # sparsification AUSE on translation
    order_p = np.argsort(-np.array([r["sigma_t"] for r in recs]))
    order_o = np.argsort(-t)
    # sparsification grid: remove 0-90% of queries (the convention of every
    # AUSE number in the papers; the last 10% is dominated by a handful of
    # queries and is not informative)
    fr = np.linspace(0, 0.9, 91)
    n = len(t)
    cp = np.array([t[order_p[int(f * n):]].mean() for f in fr])
    co = np.array([t[order_o[int(f * n):]].mean() for f in fr])
    ause = float(np.trapz(cp / cp[0] - co / co[0], fr))
    return dict(
        med_t=float(np.median(t)), med_r=float(np.median(rot)),
        sr55=float(((t <= 5) & (rot <= 5)).mean()),
        sr22=float(((t <= 2) & (rot <= 2)).mean()),
        cov=cov, ause=ause, snap=float(sn.mean()),
        inl=float(np.median(inl[sn])) if sn.any() else 0.0,
        t_loc=float(np.median(tl)), n=n,
        le1=float((t <= 1).mean()),
    )


def main():
    dirs = sys.argv[1:] or ["results/final_seed0"]
    per = {}
    for s in SCENES:
        runs = [load(d, s) for d in dirs]
        runs = [r for r in runs if r]
        if not runs:
            continue
        st = [stats(r) for r in runs]
        per[s] = {k: (float(np.median([x[k] for x in st]))
                      if not isinstance(st[0][k], dict) else st[0][k])
                  for k in st[0]}
        per[s]["seeds"] = len(st)

    if not per:
        print("no results found in", dirs); return

    done = [s for s in SCENES if s in per]
    print(f"% scenes: {', '.join(done)}  (median over {per[done[0]]['seeds']} seed(s))\n")

    # ---- Table I row
    cells = []
    for s in SCENES:
        cells.append(f"{per[s]['med_t']:.2f}/{per[s]['med_r']:.2f}"
                     if s in per else r"\res")
    avg_t = np.mean([per[s]["med_t"] for s in done])
    avg_r = np.mean([per[s]["med_r"] for s in done])
    avg = f"{avg_t:.2f}/{avg_r:.2f}" if len(done) == 7 else r"\res"
    print("%% ---- Table I (RA-L) Ours row")
    print(r"\textbf{\method{} (Ours)}  & " + " & ".join(cells) + f" & {avg}" + r"\\")
    print()

    # ---- per-scene calibration rows (full.tex)
    print("%% ---- per-scene calibration (full.tex)")
    for s in SCENES:
        if s not in per:
            print(f"{s} & " + " & ".join([r"\res"] * 5) + r" \\"); continue
        p = per[s]
        print(f"{s} & {100*p['cov'][0.683]:.0f} & {100*p['cov'][0.954]:.0f} & "
              f"{100*p['cov'][0.997]:.0f} & {p['ause']:.3f} & --- \\\\")
    print()

    # ---- diagnostics (not a paper table; for us)
    print("%% ---- diagnostics")
    print(f"%% {'scene':<11}{'med':>7}{'rot':>7}{'SR55':>7}{'SR22':>7}"
          f"{'<=1cm':>7}{'snap':>7}{'inl':>7}{'s/q':>6}")
    for s in done:
        p = per[s]
        print(f"%% {s:<11}{p['med_t']:7.2f}{p['med_r']:7.2f}{100*p['sr55']:7.0f}"
              f"{100*p['sr22']:7.0f}{100*p['le1']:7.0f}{100*p['snap']:7.0f}"
              f"{p['inl']:7.0f}{p['t_loc']:6.1f}")
    if len(done) == 7:
        print(f"%% {'AVG':<11}{avg_t:7.2f}{avg_r:7.2f}"
              f"{100*np.mean([per[s]['sr55'] for s in done]):7.0f}"
              f"{100*np.mean([per[s]['sr22'] for s in done]):7.0f}")


if __name__ == "__main__":
    main()
