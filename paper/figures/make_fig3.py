#!/usr/bin/env python3
"""Fig. 3 (calibration diagnostics) from MEASURED run_7scenes.py results.

Usage:
    python3 make_fig3.py out.pdf results/preview_seed0/7scenes_chess.json [more.json ...]

Panel (a): histogram of the tempered normalized squared error
d^2 = e^T (tau Sigma)^{-1} e over all pooled queries vs. the ideal chi^2_6
density, with the raw (untempered) histogram as the overconfidence contrast.
Panel (b): translation sparsification -- remove queries worst-first by
PREDICTED uncertainty (tau^0.5 sigma_t) vs. by TRUE error (oracle); AUSE is
the area between the normalized curves.
"""
import json
import sys
from math import gamma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def chi2_pdf(x, k):
    return x ** (k / 2 - 1) * np.exp(-x / 2) / (2 ** (k / 2) * gamma(k / 2))


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    d2_raw, d2_temp, err, unc = [], [], [], []
    for p in paths:
        d = json.load(open(p))
        tau = d["tau"]
        for r in d["records"]:
            d2_raw.append(r["d2"])
            d2_temp.append(r["d2"] / tau)
            err.append(100 * r["t_err"])                # cm
            unc.append((tau ** 0.5) * r["sigma_t"])     # tempered sigma_t
    d2_raw, d2_temp = np.array(d2_raw), np.array(d2_temp)
    err, unc = np.array(err), np.array(unc)
    n = len(err)
    print(f"pooled {n} queries from {len(paths)} scene runs")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    # (a) coverage reliability: empirical coverage vs credibility level.
    from scipy.stats import chi2 as chi2_dist

    levels = np.linspace(0.01, 0.999, 120)
    q = chi2_dist.ppf(levels, 6)
    cov_temp = [(d2_temp <= t).mean() for t in q]
    cov_raw = [(d2_raw <= t).mean() for t in q]
    ax_a.plot(levels, levels, "k--", lw=1.2, label="ideal")
    ax_a.plot(levels, cov_temp, color="#1B9E77", lw=1.8, label="Ours (tempered)")
    ax_a.plot(levels, cov_raw, color="#D95F02", lw=1.4, label="raw Laplace")
    for a in (0.683, 0.954, 0.997):
        ax_a.axvline(a, color="0.75", lw=0.6, zorder=0)
    ax_a.set_xlabel("credibility level")
    ax_a.set_ylabel("empirical coverage")
    ax_a.set_title("(a) coverage reliability", fontsize=9)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.legend(fontsize=7, frameon=False, loc="upper left")

    # (b) sparsification
    frac = np.linspace(0, 0.9, 91)   # same grid as make_fig5 / make_tables
    order_pred = np.argsort(-unc)   # remove most-uncertain first
    order_orac = np.argsort(-err)
    curve_pred = np.array([err[order_pred[int(f * n):]].mean() for f in frac])
    curve_orac = np.array([err[order_orac[int(f * n):]].mean() for f in frac])
    curve_pred /= curve_pred[0]
    curve_orac /= curve_orac[0]
    ause = float(np.trapz(curve_pred - curve_orac, frac))
    ax_b.plot(frac, curve_pred, color="#1B9E77", lw=1.6,
              label=f"Ours (AUSE {ause:.3f})")
    ax_b.plot(frac, curve_orac, "k--", lw=1.2, label="oracle")
    ax_b.set_xlabel("fraction of queries removed")
    ax_b.set_ylabel("normalized mean error")
    ax_b.set_title("(b) sparsification (translation)", fontsize=9)
    ax_b.legend(fontsize=7, frameon=False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out} | tempered d2 median {np.median(d2_temp):.2f} "
          f"(ideal {5.35:.2f}) | AUSE {ause:.3f}")


if __name__ == "__main__":
    main()
