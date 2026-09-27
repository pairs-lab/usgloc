#!/usr/bin/env python3
"""LaTeX rows for the prior-refinement (multi-network) table.

Every USGLoc cell is np.median over the 100 paired frames of the scene, the
convention of every other table. GS-CPR rows are as reported in its Table 1
(SfM pseudo-GT), whose own prior rows agree with our re-runs to <0.1 cm.

    python3 scripts/analysis/make_prior_table.py
"""
import json
import os

import numpy as np

SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
GSCPR = {  # Liu et al., GS-CPR Table 1 (SfM GT): prior + GS-CPR
    "ace": [(0.5, 0.15), (0.6, 0.25), (0.4, 0.28), (0.9, 0.26), (1.0, 0.23), (0.7, 0.17), (1.4, 0.42)],
    "marepo": [(0.6, 0.18), (0.7, 0.28), (0.5, 0.32), (1.1, 0.29), (1.0, 0.26), (0.8, 0.21), (1.5, 0.44)],
}
# correction-only and full-pipeline refinements of each prior, on the
# SELECTED (fine-tuned) maps; the *_v2 / *_seed0 directories hold the same
# experiment on the earlier maps and are kept for the map comparison.
RUNS = {
    "ace": ("results/ace_dsacstar_100", "results/prior_final/ace_dsacstar_100", "results/prior_seed0"),
    "marepo": ("results/marepo_100", "results/prior_final/marepo_100", "results/prior_marepo_ft"),
}


def med(path):
    if not os.path.exists(path):
        return None
    R = json.load(open(path))["records"]
    return (float(np.median([100 * r["t_err"] for r in R])), float(np.median([r["r_err"] for r in R])))


def cell(x, bold=False):
    if x is None:
        return "\\res"
    s = f"{x[0]:.2f}/{x[1]:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def avg(vals):
    v = [x for x in vals if x is not None]
    return (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) if len(v) == 7 else None


def main():
    for prior, (p_dir, c_dir, f_dir) in RUNS.items():
        P = [med(f"{p_dir}/7scenes_{s}.json") for s in SC]
        C = [med(f"{c_dir}/7scenes_{s}.json") for s in SC]
        F = [med(f"{f_dir}/7scenes_{s}.json") for s in SC]
        G = GSCPR[prior]
        name = "ACE~\\cite{ace}" if prior == "ace" else "Marepo~\\cite{marepo}"
        rows = [
            (f"{name} prior$^\\ddagger$", P + [avg(P)]),
            ("\\quad $+$ GS-CPR~\\cite{gscpr}$^\\dagger$", G + [GSCPR_AVG[prior]]),
            ("\\quad $+$ \\method{} correction only", C + [avg(C)]),
            ("\\quad $+$ \\method{} full", F + [avg(F)]),
        ]
        for label, vals in rows:
            print(label + " & " + " & ".join(cell(v) for v in vals) + "\\\\")
        print("\\midrule")
        # improvement summary
        if avg(C) and avg(P):
            print(f"% {prior}: prior avg {avg(P)[0]:.2f} -> corr {avg(C)[0]:.2f} ({100*(1-avg(C)[0]/avg(P)[0]):+.0f}%)"
                  + (f" -> full {avg(F)[0]:.2f}" if avg(F) else ""))


if __name__ == "__main__":
    main()
