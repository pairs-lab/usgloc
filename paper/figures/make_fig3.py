#!/usr/bin/env python3
"""Fig. 3: calibration of the reported covariance, from MEASURED results.

Run behind Table IV and Sec. V-D (results/final_seed0), 7 scenes x 100 queries.
  (a) coverage reliability: empirical coverage of the 6-DoF Mahalanobis error
      against the chi^2_6 credibility level, tempered vs raw Laplace.
  (b) the distribution behind (a): d^2 = e^T Sigma^{-1} e on a log axis against
      the chi^2_6 density, tempered (d^2/tau) and raw.
Drawn at IEEE column width (3.5 in) so it prints at scale 1.0.
    python3 make_fig3.py fig3_calibration.pdf
    python3 make_fig3.py fig3_calibration_acepnp.pdf --ace-pnp   # + ACE with PnP covariance
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
RUN = "results/final_seed0"
LEVELS = (0.683, 0.954, 0.997)
ACE_PNP = "results/ace_pnpcov_100"     # scripts/analysis/ace_pnp_covariance.py, 2-fold tau
C_OURS, C_RAW, C_REF, C_ACE = "#1B9E77", "#D95F02", "black", "#7570B3"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.22, "grid.linewidth": 0.5, "pdf.fonttype": 42,
})


def load():
    d2r, d2t = [], []
    for s in SCENES:
        d = json.load(open(ROOT / f"{RUN}/7scenes_{s}.json")); tau = d["tau"]
        for r in d["records"]:
            d2r.append(r["d2"]); d2t.append(r["d2"] / tau)
    return np.array(d2r), np.array(d2t)


def load_ace_pnp():
    return np.array([r["d2"] / r["tau"] for s in SCENES
                     for r in json.load(open(ROOT / f"{ACE_PNP}/7scenes_{s}.json"))["records"]])


def main():
    out = next((a for a in sys.argv[1:] if not a.startswith("--")), "fig3_calibration.pdf")
    d2r, d2t = load()
    d2a = load_ace_pnp() if "--ace-pnp" in sys.argv else None
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.62))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.215, top=0.885, wspace=0.40)

    # (a) coverage reliability ------------------------------------------------
    lv = np.linspace(0.005, 0.999, 200); q = chi2.ppf(lv, 6)
    ct = np.array([(d2t <= t).mean() for t in q]); cr = np.array([(d2r <= t).mean() for t in q])
    a.fill_between(lv, lv, 0, color=C_RAW, alpha=0.06, lw=0)          # below: over-confident
    a.plot([0, 1], [0, 1], "--", color=C_REF, lw=0.9, label="ideal")
    a.plot(lv, ct, color=C_OURS, lw=1.5, label="tempered (ours)")
    if d2a is not None:
        a.plot(lv, [(d2a <= t).mean() for t in q], color=C_ACE, lw=1.5, label="ACE + PnP cov.")
    a.plot(lv, cr, color=C_RAW, lw=1.5, label="raw Laplace", clip_on=False)
    for lev in LEVELS:                                                  # the levels of Table IV
        a.plot(lev, (d2t <= chi2.ppf(lev, 6)).mean(), "o", ms=3.2, color=C_OURS,
               mec="white", mew=0.5, zorder=5)
    a.text(0.97, 0.07, "over-confident", ha="right", fontsize=6.2, color=C_RAW, style="italic")
    a.set_xlim(0, 1); a.set_ylim(-0.035, 1.0)
    a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    a.set_xlabel("credibility level", labelpad=1.5)
    a.set_ylabel("empirical coverage", labelpad=1.5)
    a.set_title("(a) coverage", pad=3)
    a.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.1)

    # (b) the d^2 distribution behind it -------------------------------------
    bins = np.linspace(-1, 7, 49)
    for v, c, al in ((d2t, C_OURS, 0.30), (d2r, C_RAW, 0.22)):
        b.hist(np.log10(v), bins=bins, density=True, histtype="stepfilled", color=c, alpha=al, lw=0)
        b.hist(np.log10(v), bins=bins, density=True, histtype="step", color=c, lw=1.0)
    if d2a is not None:
        b.hist(np.log10(d2a), bins=bins, density=True, histtype="step", color=C_ACE, lw=1.0)
    xs = np.linspace(-1, 2.2, 300); x = 10**xs
    b.plot(xs, chi2.pdf(x, 6) * x * np.log(10), "--", color=C_REF, lw=0.9)  # density of log10 d^2
    b.text(0.42, 1.45, r"$\chi^2_6$", ha="right", va="center", fontsize=7)
    ym = 1.12
    b.annotate("", xy=(1.35, ym), xytext=(np.log10(np.median(d2r)), ym),
               arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.25", mutation_scale=6))
    b.text(0.5 * (1.35 + np.log10(np.median(d2r))), ym + 0.05, r"$\div\,\tau$", ha="center",
           va="bottom", fontsize=7, color="0.25")
    b.set_xlim(-1, 7); b.set_ylim(0, 1.68)
    b.set_xticks([0, 2, 4, 6]); b.set_xticklabels([r"$1$", r"$10^{2}$", r"$10^{4}$", r"$10^{6}$"])
    b.set_yticks([0, 0.5, 1, 1.5])
    b.set_xlabel(r"$d^2=\mathbf{e}^\top\boldsymbol{\Sigma}^{-1}\mathbf{e}$", labelpad=1.0)
    b.set_ylabel("density", labelpad=1.5)
    b.set_title(r"(b) $d^2$ distribution", pad=3)

    fig.savefig(out)
    cov = " / ".join(f"{100*(d2t<=chi2.ppf(l,6)).mean():.1f}" for l in LEVELS)
    print(f"saved {out} | {RUN} n={len(d2t)} | coverage 68.3/95.4/99.7: {cov}"
          f" | raw {100*(d2r<=chi2.ppf(0.997,6)).mean():.1f}% at 99.7"
          f" | median d2 tempered {np.median(d2t):.2f} (chi2_6 {chi2.median(6):.2f}), raw {np.median(d2r):.0f}")


if __name__ == "__main__":
    main()
