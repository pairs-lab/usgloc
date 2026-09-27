#!/usr/bin/env python3
"""Fig. 5: uncertainty quality vs. the ACE baseline, on identical queries.

(a) Sparsification: remove queries worst-first by each method's OWN confidence
    signal (ours: tempered sigma_t; ACE: negative PnP inlier count) against the
    oracle ordering by true error. Each curve is normalized by its own initial
    error, so the comparison is of ORDERING quality, not of accuracy.
(b) Median true error within each decile of each method's own confidence
    signal. A useful signal should be monotone and span a wide range; the
    units differ between methods, so the decile index is the shared axis.

Usage: python3 make_fig5.py out.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
RUN = "results/ev_seed0"
OURS = "#1B9E77"
ACEC = "#7570B3"


def load():
    ae, ai, oe, os_ = [], [], [], []
    for s in SCENES:
        a = json.load(open(ROOT / f"results/ace_dsacstar_100/7scenes_{s}.json"))
        o = json.load(open(ROOT / f"{RUN}/7scenes_{s}.json"))
        tau = o["tau"]
        A = {(r["seq"], r["frame_id"]): r for r in a["records"]}
        O = {(r["seq"], r["frame_id"]): r for r in o["records"]}
        for k in sorted(set(A) & set(O)):
            ae.append(100 * A[k]["t_err"]); ai.append(float(A[k]["inliers"]))
            oe.append(100 * O[k]["t_err"])
            os_.append(O[k]["sigma_t"] * tau ** 0.5)
    return map(np.array, (ae, ai, oe, os_))


def sparsify(sig, err):
    """Mean remaining error as the most-uncertain fraction is removed."""
    n = len(err)
    order = np.argsort(-sig)
    fr = np.linspace(0, 0.9, 91)
    c = np.array([err[order[int(f * n):]].mean() for f in fr])
    return fr, c / c[0]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fig5_uncertainty.pdf"
    global RUN
    RUN = sys.argv[2] if len(sys.argv) > 2 else "results/ev_seed0"
    ae, ai, oe, os_ = load()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # ---- (a) sparsification by each method's own signal
    fr, c_ours = sparsify(os_, oe)
    _, c_ace = sparsify(-ai, ae)
    _, c_orac_o = sparsify(oe, oe)
    ax.plot(fr, c_ours, color=OURS, lw=1.9, label="Ours ($\\sigma_t$)")
    ax.plot(fr, c_ace, color=ACEC, lw=1.6, label="ACE (inlier count)")
    ax.plot(fr, c_orac_o, "k--", lw=1.1, label="oracle (true error)")
    ax.set_xlabel("fraction of queries removed")
    ax.set_ylabel("normalized mean error")
    ax.set_title("(a) sparsification by own confidence", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(0, 0.9)

    # ---- (b) is the confidence signal informative across its whole range?
    # Median true error within each decile of each method's own confidence.
    def deciles(sig, err):
        """sig: higher = less confident. Returns median error per decile."""
        order = np.argsort(sig)
        parts = np.array_split(order, 10)
        return [float(np.median(err[p])) for p in parts]

    xs = np.arange(1, 11)
    bx.semilogy(xs, deciles(os_, oe), "o-", color=OURS, lw=1.8, ms=4,
                label="Ours ($\\sigma_t$)")
    bx.semilogy(xs, deciles(-ai, ae), "s-", color=ACEC, lw=1.6, ms=4,
                label="ACE (inlier count)")
    bx.set_xlabel("decile of reported uncertainty (1 = most confident)")
    bx.set_ylabel("median error (cm)")
    bx.set_title("(b) error vs. reported confidence", fontsize=9)
    bx.set_xticks(xs)
    bx.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    from scipy.stats import spearmanr
    print(f"saved {out}")
    print(f"  n={len(ae)}  AUSE ours {np.trapz(c_ours - c_orac_o, fr):.3f} "
          f"vs ACE {np.trapz(c_ace - sparsify(ae, ae)[1], fr):.3f}")
    print(f"  rank-corr ours {spearmanr(os_, oe).correlation:.3f} "
          f"ACE {spearmanr(-ai, ae).correlation:.3f}")
    def dec(sig, err):
        o = np.argsort(sig)
        return [float(np.median(err[p])) for p in np.array_split(o, 10)]
    print(f"  ours deciles  {[round(x,2) for x in dec(os_, oe)]}")
    print(f"  ACE  deciles  {[round(x,2) for x in dec(-ai, ae)]}")


if __name__ == "__main__":
    main()
