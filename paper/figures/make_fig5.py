#!/usr/bin/env python3
"""Fig. 4: uncertainty quality vs. the ACE baseline, on identical queries.

Run behind Sec. V-D (results/final_seed0) and ACE re-run with its own dsacstar
solver on the same 100 frames per scene (results/ace_dsacstar_100).
(a) Sparsification: remove queries worst-first by each method's OWN confidence
    signal (ours: tempered sigma_t; ACE: PnP inlier count). Each curve and its
    oracle (rank by true error) are normalized by that method's initial error,
    so only ORDERING quality is compared. AUSE = shaded area, first 90%.
(b) Median true error within each decile of each method's own confidence.
Drawn at IEEE column width (3.5 in) so it prints at scale 1.0.
    python3 make_fig5.py fig5_uncertainty.pdf
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
RUN = "results/final_seed0"
ACE = "results/ace_dsacstar_100"
C_OURS, C_ACE = "#1B9E77", "#7570B3"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.22, "grid.linewidth": 0.5, "pdf.fonttype": 42,
})


def load():
    ae, ai, oe, os_ = [], [], [], []
    for s in SCENES:
        a = json.load(open(ROOT / f"{ACE}/7scenes_{s}.json"))
        o = json.load(open(ROOT / f"{RUN}/7scenes_{s}.json")); tau = o["tau"]
        A = {(r["seq"], r["frame_id"]): r for r in a["records"]}
        O = {(r["seq"], r["frame_id"]): r for r in o["records"]}
        for k in sorted(set(A) & set(O)):
            ae.append(100 * A[k]["t_err"]); ai.append(float(A[k]["inliers"]))
            oe.append(100 * O[k]["t_err"]); os_.append(O[k]["sigma_t"] * tau ** 0.5)
    return map(np.array, (ae, ai, oe, os_))


def sparsify(sig, err):
    """Mean remaining error as the most-uncertain fraction is removed (sig: high = uncertain)."""
    n = len(err); order = np.argsort(-sig); fr = np.linspace(0, 0.9, 91)
    c = np.array([err[order[int(f * n):]].mean() for f in fr])
    return fr, c / c[0]


def deciles(sig, err):
    order = np.argsort(sig)
    return np.array([np.median(err[p]) for p in np.array_split(order, 10)])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fig5_uncertainty.pdf"
    ae, ai, oe, os_ = load()
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.62))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.215, top=0.885, wspace=0.40)

    # (a) sparsification by each method's own signal ---------------------------
    ause = {}
    for name, sig, err, c in (("ours", os_, oe, C_OURS), ("ACE", -ai, ae, C_ACE)):
        fr, cur = sparsify(sig, err); _, orc = sparsify(err, err)
        ause[name] = np.trapz(cur - orc, fr)
        a.fill_between(fr, orc, cur, color=c, alpha=0.16, lw=0)
        a.plot(fr, orc, "--", color=c, lw=0.8)                  # oracle: rank by true error
        a.plot(fr, cur, color=c, lw=1.5)
    a.legend(handles=[Line2D([], [], color=C_OURS, lw=1.5, label=rf"ours, $\sigma_t$ ({ause['ours']:.3f})"),
                      Line2D([], [], color=C_ACE, lw=1.5, label=f"ACE, inliers ({ause['ACE']:.3f})")],
             loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.1)
    a.set_xlim(0, 0.9); a.set_ylim(0, 1.04)
    a.set_xticks([0, 0.3, 0.6, 0.9]); a.set_yticks([0, 0.5, 1])
    a.set_xlabel("fraction of queries removed", labelpad=1.5)
    a.set_ylabel("normalized mean error", labelpad=1.5)
    a.set_title("(a) sparsification (AUSE)", pad=3)

    # (b) median error per decile of reported confidence -----------------------
    xs = np.arange(1, 11)
    do, da = deciles(os_, oe), deciles(-ai, ae)
    b.semilogy(xs, do, "o-", color=C_OURS, lw=1.4, ms=3.0, mec="white", mew=0.4, label=r"ours, $\sigma_t$")
    b.semilogy(xs, da, "s-", color=C_ACE, lw=1.4, ms=2.8, mec="white", mew=0.4, label="ACE, inliers")
    for x, y, dy, va in ((1, do[0], -4, "top"), (1, da[0], 4, "bottom"),
                         (10, do[-1], 4, "bottom"), (10, da[-1], -4, "top")):
        b.annotate(f"{y:.2f}" if y < 1 else f"{y:.1f}", (x, y), xytext=(0, dy), textcoords="offset points",
                   ha="center", va=va, fontsize=6.0, color=C_OURS if y in (do[0], do[-1]) else C_ACE)
    b.set_xlim(0.4, 10.6); b.set_ylim(0.15, 20)
    b.set_xticks([1, 4, 7, 10])
    b.set_yticks([0.3, 1, 3, 10]); b.set_yticklabels(["0.3", "1", "3", "10"])
    b.set_xlabel("decile (1 = most confident)", labelpad=1.5)
    b.set_ylabel("median error (cm)", labelpad=1.5)
    b.set_title("(b) error per decile", pad=3)
    b.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.1)

    fig.savefig(out)
    print(f"saved {out} | n={len(ae)} | AUSE ours {ause['ours']:.4f} ACE {ause['ACE']:.4f}"
          f" | rho ours {spearmanr(os_, oe).correlation:.3f} ACE {spearmanr(-ai, ae).correlation:.3f}")
    print(f"  ours deciles {np.round(do, 2).tolist()} (x{do[-1]/do[0]:.0f})")
    print(f"  ACE  deciles {np.round(da, 2).tolist()} (x{da[-1]/da[0]:.1f})")


if __name__ == "__main__":
    main()
