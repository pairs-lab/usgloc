#!/usr/bin/env python3
"""Fig. 1 teaser.

LEFT  2x2: the query and the 3DGS map rendered at the pose each method returned,
           with that method's translation error printed on the panel.
RIGHT    : the fire 3D Gaussian map from a free camera. Every test query is a dot
           coloured by the uncertainty USGLoc reported for it; the selected query
           is opened up with its estimated pose, the true pose, and the 2-sigma
           covariance projected into this view.

Selection: among queries where USGLoc is under 1 cm, this is the one with the
largest margin over BOTH baselines whose reported sigma is still sensible
(sigma 1.17 cm for a 0.19 cm error). It is a favourable example chosen on
purpose; Tables I-II carry the distribution.

Sources rendered by scratchpad/render_fire.py into paper/figures/fig1_src/.
    CUDA_VISIBLE_DEVICES="" python paper/figures/make_fig1_teaser.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Polygon, Ellipse

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
SRC = ROOT/"paper/figures/fig1_src"
FW, FH = 7.16, 2.78
C_EST, C_GT, C_UNC = "#ff3b30", "#19c37d", "#4da3ff"

M = json.load(open(SRC/"meta.json")); tau = M["tau"]
q = np.load(SRC/"query.npy")
V = {k: np.load(SRC/f"view_{k}.npy") for k in ("gt", "ours", "ace", "marepo")}
z = np.load(SRC/"bg.npz")
bg, cam = z["rgb"], dict(eye=z["eye"], R=z["R"], fx=float(z["fx"]),
                         cx=float(z["cx"]), cy=float(z["cy"]))
def centre(T):
    T = np.asarray(T, float); return -T[:3, :3].T @ T[:3, 3]
def project(P, c):
    pc = (np.atleast_2d(P) - c["eye"]) @ c["R"].T
    zz = np.maximum(pc[:, 2], 1e-6)
    return c["fx"]*pc[:, 0]/zz + c["cx"], c["fx"]*pc[:, 1]/zz + c["cy"]

fig = plt.figure(figsize=(FW, FH))
PW, PH, X0, Y0, GAP = 1.575, 1.218, 0.020, 0.018, 0.030
CELLS = [("query", None, q), ("ours", "USGLoc (ours)", V["ours"]),
         ("ace", "ACE", V["ace"]), ("marepo", "Marepo", V["marepo"])]
for i, (key, lab, im) in enumerate(CELLS):
    r, c = divmod(i, 2)
    a = fig.add_axes([(X0 + c*(PW+GAP))/FW, (Y0 + (1-r)*(PH+0.175))/FH, PW/FW, PH/FH])
    a.imshow(im); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_color("#b0b0b0"); s.set_linewidth(0.5)
    if key == "query":
        a.set_title("query image", fontsize=7.8, pad=2.5)
    else:
        a.set_title(lab, fontsize=7.8, pad=2.5,
                    fontweight=("bold" if key == "ours" else "normal"))
        col = C_EST if key == "ours" else "black"
        a.text(0.035, 0.05, f"{M[key]:.2f} cm", transform=a.transAxes, fontsize=7.6,
               color="white", va="bottom", fontweight=("bold" if key=="ours" else "normal"),
               bbox=dict(boxstyle="round,pad=0.22", fc=col, alpha=0.88, lw=0))


RX, RW = 3.265, 3.875
TOP = Y0 + 2*PH + 0.175          # top of the left image grid
ax = fig.add_axes([RX/FW, Y0/FH, RW/FW, (TOP-Y0)/FH])
ax.imshow(bg, extent=[0, int(z["W"]), int(z["H"]), 0], interpolation="bilinear")
ax.set_xlim(0, int(z["W"])); ax.set_ylim(int(z["H"]), 0)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)

o = json.load(open("results/m3u8_seed0/7scenes_fire.json"))
sig = np.array([100*r["sigma_t"]*tau**0.5 for r in o["records"]])
C = np.array([centre(r["T"]) for r in o["records"]])
u, v = project(C, cam)
cmap, norm = plt.cm.plasma, LogNorm(vmin=0.2, vmax=300.0)
fr = np.clip((np.log10(sig)-np.log10(0.2))/(np.log10(300)-np.log10(0.2)), 0, 1)
ax.scatter(u, v, s=8+140*fr**1.8, c=sig, cmap=cmap, norm=norm, lw=0.35,
           ec="white", alpha=0.95, zorder=4)

C_e, C_t = centre(M["T"]), centre(M["T_gt"])
Rw = np.asarray(M["T"], float)[:3, :3].T
F = 0.26*(Rw @ np.array([[0,0,0],[-.6,-.45,1],[.6,-.45,1],[.6,.45,1],[-.6,.45,1]]).T).T + C_e
fu, fv = project(F, cam)
ax.add_patch(Polygon(np.c_[fu[1:], fv[1:]], closed=True, fill=False, ec=C_EST, lw=1.3, zorder=6))
for j in range(1, 5):
    ax.plot([fu[0], fu[j]], [fv[0], fv[j]], color=C_EST, lw=1.3, zorder=6)
Sig = np.array(M["Sigma"])[:3, :3]*tau
pc = (C_e - cam["eye"]) @ cam["R"].T; zz = float(pc[2])
J = (cam["fx"]/zz)*np.array([[1, 0, -pc[0]/zz], [0, 1, -pc[1]/zz]])
ev, evec = np.linalg.eigh(J @ (cam["R"] @ Sig @ cam["R"].T) @ J.T)
w, h = 2*2*np.sqrt(np.clip(ev, 0, None))
ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
ax.add_patch(Ellipse((fu[0], fv[0]), w, h, angle=ang, fc=C_UNC, alpha=0.32, ec="none", zorder=5))
for ec, lw in (("white", 2.6), (C_UNC, 1.3)):
    ax.add_patch(Ellipse((fu[0], fv[0]), w, h, angle=ang, fc="none", ec=ec, lw=lw, zorder=7))
tu, tv = project(C_t, cam)
ax.plot(tu, tv, "o", mfc="none", mec=C_GT, mew=2.0, ms=10, zorder=9)
ax.plot(fu[0], fv[0], "^", color=C_EST, ms=8, mec="white", mew=0.8, zorder=10)
ax.annotate(f"$\\sigma_t$ {M['sigma']:.1f} cm\nerror {M['ours']:.2f} cm",
            xy=(fu[0], fv[0]), xytext=(0.055, 0.70), textcoords="axes fraction",
            fontsize=7.2, color="white", zorder=21,
            bbox=dict(boxstyle="round,pad=0.26", fc="black", alpha=0.62, ec="none"),
            arrowprops=dict(arrowstyle="-", color="white", lw=0.9,
                            shrinkA=2, shrinkB=9))

ax.text(0.012, 0.975,
        "USGLoc: every query returns a pose $\\it{and}$ its covariance",
        transform=ax.transAxes, fontsize=8.6, fontweight="bold", color="white",
        va="top", ha="left", zorder=22,
        bbox=dict(boxstyle="square,pad=0.34", fc="black", alpha=0.62, ec="none"))

ax.plot([], [], "^", color=C_EST, ms=7, label="estimated pose")
ax.plot([], [], "o", mfc="none", mec=C_GT, mew=1.8, ms=8, label="true pose")
ax.add_patch(Ellipse((-99, -99), 1, 1, fc=C_UNC, alpha=0.6, ec=C_UNC,
                     label=r"covariance ($2\sigma$)"))
lg = ax.legend(loc="lower left", fontsize=7.4, framealpha=0.62, facecolor="black",
               edgecolor="none", labelcolor="white", handletextpad=0.4,
               borderpad=0.33, labelspacing=0.30)
lg.set_zorder(20)
cbb = fig.add_axes([(RX+RW-0.86)/FW, (Y0+0.07)/FH, 0.76/FW, 1.00/FH])
cbb.set_facecolor("black"); cbb.patch.set_alpha(0.55); cbb.set_xticks([]); cbb.set_yticks([])
for s in cbb.spines.values(): s.set_visible(False)
cbb.set_zorder(18)
cax = fig.add_axes([(RX+RW-0.78)/FW, (Y0+0.14)/FH, 0.075/FW, 0.86/FH]); cax.set_zorder(19)
cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
cb.set_label(r"reported $\sigma_t$ [cm]", fontsize=7.0, color="white", labelpad=2)
cb.ax.tick_params(labelsize=6.5, colors="white", width=0.5, length=2)
cb.outline.set_edgecolor("white"); cb.outline.set_linewidth(0.5)

fig.savefig("paper/figures/fig1_teaser.pdf", dpi=300)
print(f"wrote fig1_teaser.pdf | fire {M['seq']}/{M['frame']}: ours {M['ours']:.2f} cm, "
      f"ACE {M['ace']:.2f}, Marepo {M['marepo']:.2f}, sigma {M['sigma']:.2f} cm")
