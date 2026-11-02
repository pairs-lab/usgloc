#!/usr/bin/env python3
"""Fig. 4: uncertainty quality vs. the ACE baseline, on identical queries.

Run behind Sec. V-D (results/final_seed0) and ACE re-run with its own dsacstar
solver on the same 100 frames per scene (results/ace_dsacstar_100).
(a) Sparsification: remove queries worst-first by each method's OWN confidence
    signal (ours: tempered sigma_t; ACE: PnP inlier count). Each curve and its
    oracle (rank by true error) are normalized by that method's initial error,
    so only ORDERING quality is compared. AUSE = shaded area, first 90%.
(b) Median true error within each decile of each method's own confidence.
    DSAC* (the authors' published SfM-pGT estimates) and Marepo report no
    confidence, so every decile of a random order has their overall median:
    they are drawn flat.
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
MAREPO = "results/marepo_100"
DSAC = "data/visloc_pseudo_gt_limitations/estimates/sfm/7scenes"
C_OURS, C_ACE, C_DSAC, C_MP = "#1B9E77", "#7570B3", "#A6761D", "#6E6E6E"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.22, "grid.linewidth": 0.5, "pdf.fonttype": 42,
})


def centre(T):
    T = np.asarray(T, float); return -T[:3, :3].T @ T[:3, 3]


def dsac_poses(s):
    """Published DSAC* (RGB) estimates: name qw qx qy qz tx ty tz (world->camera)."""
    from scipy.spatial.transform import Rotation
    P = {}
    for line in open(ROOT / f"{DSAC}/{s}_dsac_rgb.txt"):
        p = line.split(); q = [float(x) for x in p[1:5]]
        T = np.eye(4); T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = [float(x) for x in p[5:8]]; P[p[0]] = T
    return P


def load():
    ae, ai, oe, os_, de, me = [], [], [], [], [], []
    for s in SCENES:
        a = json.load(open(ROOT / f"{ACE}/7scenes_{s}.json"))
        o = json.load(open(ROOT / f"{RUN}/7scenes_{s}.json")); tau = o["tau"]
        m = json.load(open(ROOT / f"{MAREPO}/7scenes_{s}.json"))
        A = {(r["seq"], r["frame_id"]): r for r in a["records"]}
        O = {(r["seq"], r["frame_id"]): r for r in o["records"]}
        M = {(r["seq"], r["frame_id"]): r for r in m["records"]}
        D = dsac_poses(s)
        for k in sorted(set(A) & set(O) & set(M)):
            ae.append(100 * A[k]["t_err"]); ai.append(float(A[k]["inliers"]))
            oe.append(100 * O[k]["t_err"]); os_.append(O[k]["sigma_t"] * tau ** 0.5)
            me.append(100 * M[k]["t_err"])
            Td = D[f"{k[0]}/frame-{k[1]:06d}.color.png"]
            de.append(100 * np.linalg.norm(centre(Td) - centre(O[k]["T_gt"])))
    return map(np.array, (ae, ai, oe, os_, de, me))


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
    ae, ai, oe, os_, de, me = load()
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
    for e, c, lab, x in ((me, C_MP, "Marepo (no conf.)", 4.6), (de, C_DSAC, "DSAC* (no conf.)", 3.4)):
        b.plot([0.7, 10.3], [np.median(e)] * 2, ls=(0, (4, 2)), color=c, lw=1.0)
        b.annotate(lab, (x, np.median(e)), xytext=(0, 1.5), textcoords="offset points",
                   ha="center", va="bottom", fontsize=6.0, color=c)
    for x, y, c in ((1, do[0], C_OURS), (1, da[0], C_ACE), (10, do[-1], C_OURS), (10, da[-1], C_ACE)):
        b.annotate(f"{y:.2f}" if y < 1 else f"{y:.1f}", (x, y), xytext=(-4 if x == 1 else 4, 0),
                   textcoords="offset points", ha="right" if x == 1 else "left", va="center",
                   fontsize=6.0, color=c)
    b.set_xlim(-0.6, 11.4); b.set_ylim(0.15, 20)
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
    print(f"  no-confidence medians: DSAC* {np.median(de):.2f} cm, Marepo {np.median(me):.2f} cm"
          f" | pooled medians ours {np.median(oe):.2f} ACE {np.median(ae):.2f}")


if __name__ == "__main__":
    main()
