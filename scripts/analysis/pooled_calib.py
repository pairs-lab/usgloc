#!/usr/bin/env python3
"""Pooled calibration row (coverage at 68.3/95.4/99.7, AUSE on the 0-0.9 grid,
ECE on the 120-level grid) over the seven 7Scenes runs of one result layout.

    python3 scripts/analysis/pooled_calib.py results/ev_seed0            # <dir>/7scenes_<scene>.json
    python3 scripts/analysis/pooled_calib.py results/ablation_nocorr_%s  # per-scene dirs
Add --raw to ignore the fitted temperature.
"""
import json, sys
import numpy as np
from scipy.stats import chi2

SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


def ause(sig, t):
    fr = np.linspace(0, 0.9, 91); n = len(t); op = np.argsort(-sig); oo = np.argsort(-t)
    cp = np.array([t[op[int(f * n):]].mean() for f in fr]); co = np.array([t[oo[int(f * n):]].mean() for f in fr])
    return float(np.trapz(cp / cp[0] - co / co[0], fr))


def main():
    lay = sys.argv[1]; raw = "--raw" in sys.argv
    S, T, D = [], [], []
    for s in SC:
        p = (lay % s if "%s" in lay else lay) + f"/7scenes_{s}.json"
        d = json.load(open(p)); tau = 1.0 if raw else d["tau"]
        for r in d["records"]:
            S.append(r["sigma_t"] * tau ** 0.5); T.append(100 * r["t_err"]); D.append(r["d2"] / tau)
    S, T, D = map(np.array, (S, T, D))
    cov = [100 * (D <= chi2.ppf(a, 6)).mean() for a in (0.683, 0.954, 0.997)]
    lv = np.linspace(0.01, 0.999, 120)
    ece = float(np.mean([abs((D <= chi2.ppf(a, 6)).mean() - a) for a in lv]))
    print(f"{lay} {'raw' if raw else 'tempered'} n={len(T)} | median {np.median(T):.2f} cm | cov {cov[0]:.1f}/{cov[1]:.1f}/{cov[2]:.1f} | AUSE {ause(S, T):.3f} | ECE {ece:.3f}")


if __name__ == "__main__":
    main()
