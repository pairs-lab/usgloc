#!/usr/bin/env python3
"""Fig. 1 teaser: the 3DGS map, every query coloured by its reported uncertainty,
and one query opened up to show estimated pose, true pose and the covariance.

Built on the cand_mapfrusta view (same cached gsplat background render of
maps/chess_r1.pt) with the prose stripped out: no title bar, no statistics
block, no call-out paragraph. What is left is the picture plus a legend.

    CUDA_VISIBLE_DEVICES="" python paper/figures/make_fig1_teaser.py
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"paper/figures"))
from make_cand_mapfrusta import (centre, render_background, project,
                                 load_queries, PANEL, FW, FH, UP)  # noqa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Polygon, Ellipse

C_EST, C_GT, C_UNC = "#ff3b30", "#19c37d", "#4da3ff"

rows, tau = load_queries()
bg, cam = render_background(rows)
d = json.load(open("results/m3u8_seed0/7scenes_chess.json"))
by = {(r["seq"], r["frame_id"]): r for r in d["records"]}

sig = np.array([r["sig"] for r in rows]); err = np.array([r["err"] for r in rows])
u, v, _ = project(np.array([r["C"] for r in rows]), cam)
hi = int(np.argmax(err))                      # largest TRUE error, not cherry-picked
rec = by[(rows[hi]["seq"], rows[hi]["frame"])]
C_e, C_t = centre(rec["T"]), centre(rec["T_gt"])
Sig = np.array(rec["Sigma"])[:3, :3] * tau

fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([PANEL[0]/FW, PANEL[1]/FH, PANEL[2]/FW, PANEL[3]/FH])
ax.imshow(bg, extent=[0, cam["W"], cam["H"], 0], interpolation="bilinear")
ax.set_xlim(0, cam["W"]); ax.set_ylim(cam["H"], 0); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)

cmap, norm = plt.cm.plasma, LogNorm(vmin=0.2, vmax=300.0)
frac = np.clip((np.log10(sig) - np.log10(0.2)) / (np.log10(300) - np.log10(0.2)), 0, 1)
ax.scatter(u, v, s=8 + 150*frac**1.8, c=sig, cmap=cmap, norm=norm,
           lw=0.35, ec="white", alpha=0.95, zorder=4)

# ---- the highlighted query: estimated pose, true pose, covariance
Rw = np.asarray(rec["T"], float)[:3, :3].T
fr = 0.30*(Rw @ np.array([[0,0,0],[-.6,-.45,1],[.6,-.45,1],[.6,.45,1],[-.6,.45,1]]).T).T + C_e
fu, fv, _ = project(fr, cam)
ax.add_patch(Polygon(np.c_[fu[1:], fv[1:]], closed=True, fill=False,
                     ec=C_EST, lw=1.3, zorder=6))
for j in range(1, 5):
    ax.plot([fu[0], fu[j]], [fv[0], fv[j]], color=C_EST, lw=1.3, zorder=6)

pc = (C_e - cam["eye"]) @ cam["R"].T; z = float(pc[2])
J = (cam["fx"]/z) * np.array([[1, 0, -pc[0]/z], [0, 1, -pc[1]/z]])
Sp = J @ (cam["R"] @ Sig @ cam["R"].T) @ J.T          # covariance in view pixels
ev, evec = np.linalg.eigh(Sp)
w, h = 2*2*np.sqrt(np.clip(ev, 0, None))
ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
ax.add_patch(Ellipse((fu[0], fv[0]), w, h, angle=ang, fc=C_UNC, alpha=0.30,
                     ec="none", zorder=5))
for ec, lw in (("white", 2.6), (C_UNC, 1.3)):
    ax.add_patch(Ellipse((fu[0], fv[0]), w, h, angle=ang, fc="none", ec=ec,
                         lw=lw, zorder=7))
tu, tv, _ = project(C_t, cam)
ax.plot([fu[0], tu[0]], [fv[0], tv[0]], "-", color="white", lw=2.4, zorder=7)
ax.plot([fu[0], tu[0]], [fv[0], tv[0]], "-", color=C_GT, lw=1.1, zorder=8)
ax.plot(tu, tv, "o", mfc="none", mec=C_GT, mew=2.0, ms=10, zorder=9)
ax.plot(fu[0], fv[0], "^", color=C_EST, ms=8, mec="white", mew=0.8, zorder=10)

ax.plot([], [], "^", color=C_EST, ms=7, label="estimated pose")
ax.plot([], [], "o", mfc="none", mec=C_GT, mew=1.8, ms=8, label="true pose")
ax.add_patch(Ellipse((-99, -99), 1, 1, fc=C_UNC, alpha=0.5, ec=C_UNC,
                     label=r"covariance ($2\sigma$)"))
lg = ax.legend(loc="lower left", fontsize=7.6, framealpha=0.60, facecolor="black",
               edgecolor="none", labelcolor="white", handletextpad=0.4,
               borderpad=0.35, labelspacing=0.32)
lg.set_zorder(20)

cbbg = fig.add_axes([(PANEL[0]+PANEL[2]-0.78)/FW, (PANEL[1]+0.08)/FH, 0.62/FW, 1.11/FH])
cbbg.set_facecolor("black"); cbbg.patch.set_alpha(0.55)
cbbg.set_xticks([]); cbbg.set_yticks([])
for sp in cbbg.spines.values(): sp.set_visible(False)
cbbg.set_zorder(18)
cax = fig.add_axes([(PANEL[0]+PANEL[2]-0.60)/FW, (PANEL[1]+0.16)/FH, 0.085/FW, 0.95/FH])
cax.set_zorder(19)
cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
cb.set_label(r"reported $\sigma_t$  [cm]", fontsize=7.4, color="white", labelpad=3)
cb.ax.tick_params(labelsize=6.8, colors="white", width=0.6, length=2.2)
cb.outline.set_edgecolor("white"); cb.outline.set_linewidth(0.5)

# ---- right block: the query and our render at the highlighted pose
sel = json.load(open("paper/figures/fig_qual3/selection.json"))
for i, (tag, lab) in enumerate([("query", "query"), ("T_ours", "render at our pose")]):
    a = fig.add_axes([(4.44 + i*1.37)/FW, 1.30/FH, 1.30/FW, 1.14/FH])
    a.imshow(plt.imread(f"paper/figures/fig_qual3/0_{tag}.png"))
    a.set_xticks([]); a.set_yticks([]); a.set_title(lab, fontsize=7.6, pad=2.5)
    for s in a.spines.values(): s.set_color("#bbbbbb"); s.set_linewidth(0.5)
fig.text(4.44/FW, 1.06/FH,
         f"above:  $\\sigma_t$ {100*by[('seq-05',878)]['sigma_t']*tau**0.5:.2f} cm "
         f"$\\rightarrow$ error {sel[0]['e_ours']:.2f} cm", fontsize=7.4, va="baseline")
fig.text(4.44/FW, 0.76/FH,
         f"highlighted in the map:  $\\sigma_t$ {rows[hi]['sig']:.0f} cm "
         f"$\\rightarrow$ error {err[hi]:.1f} cm", fontsize=7.4, va="baseline", color=C_EST)
fig.text(4.44/FW, 0.34/FH, "Every query returns a pose and its covariance;\n"
         "no prior 3DGS localizer reports one.", fontsize=7.6, va="baseline")
fig.savefig("paper/figures/fig1_teaser.pdf", dpi=300)
print(f"wrote fig1_teaser.pdf | highlighted chess {rows[hi]['seq']}/{rows[hi]['frame']}: "
      f"sigma {rows[hi]['sig']:.1f} cm, err {err[hi]:.2f} cm, ellipse {w:.0f}x{h:.0f} px")
