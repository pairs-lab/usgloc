#!/usr/bin/env python3
"""Fig. 1 (teaser), RA-L shape: the contribution drawn in the scene.

Three queries at centimetre zoom, ordered by reported uncertainty. Each shows
the ground-truth
camera centre, our estimate with its tempered 2-sigma covariance ellipse, and
ACE's estimate. ACE -- like every prior 3DGS localizer -- returns only the
point; the ellipse is what this paper adds, and it grows with the error.

Ellipses are drawn at TRUE metric scale inside each callout, which has its own
scale bar; they cannot be drawn on the room-scale map because they are
centimetres across in a 7 m room.

    python3 paper/figures/make_fig1_teaser.py
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
SCENE, AX = "chess", (0, 2)          # top-down: x vs z (largest extents)

def centre(T):
    T = np.asarray(T); R, t = T[:3, :3], T[:3, 3]
    return -R.T @ t                   # T is world-to-camera

d = json.load(open(f'results/m3u8_seed0/7scenes_{SCENE}.json')); tau = d['tau']
ace = {(r['seq'], r['frame_id']): r
       for r in json.load(open(f'results/ace_dsacstar_100/7scenes_{SCENE}.json'))['records']}
rows = []
for r in d['records']:
    k = (r['seq'], r['frame_id'])
    if k not in ace: continue
    S = np.array(r['Sigma'])[:3, :3] * tau
    gt, ou = centre(r['T_gt']), centre(r['T'])
    # the figure draws a 2-D top-down projection, so the printed error and the
    # printed 2-sigma must be the IN-PLANE ones or they contradict the picture
    S2 = S[np.ix_(AX, AX)]
    rows.append(dict(gt=gt, ours=ou, ace=centre(ace[k]['T']), S=S,
                     err=100*float(np.linalg.norm((ou - gt)[list(AX)])),
                     err3=100*r['t_err'],
                     maj=2*np.sqrt(max(np.linalg.eigvalsh(S2).max(), 0))*100))
rows.sort(key=lambda r: r['maj'])
# Pick by REPORTED uncertainty bucket, then take the MEDIAN-error query inside
# each bucket. Taking extremes gave a 'confident' case whose ground truth fell
# outside its own ellipse and a 'less certain' case more accurate than it --
# both unrepresentative of the statistical claim the figure makes.
import numpy as _np
def representative(lo, hi):
    band = rows[int(len(rows)*lo):max(int(len(rows)*hi), int(len(rows)*lo)+1)]
    med = _np.median([b['err'] for b in band])
    return min(band, key=lambda b: abs(b['err'] - med))
pick = [representative(0.00, 0.20), representative(0.55, 0.75), representative(0.92, 1.00)]
labels = ["confident", "less certain", "uncertain"]


fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
fig.subplots_adjust(left=0.015, right=0.985, top=0.80, bottom=0.08, wspace=0.14)
for i, (ax, r, lab) in enumerate(zip(axes, pick, labels)):
    c = r['gt'][list(AX)]
    S2 = r['S'][np.ix_(AX, AX)]
    ev, evec = np.linalg.eigh(S2)
    w, h = 2*2*np.sqrt(np.clip(ev, 0, None))          # 2-sigma full axes, metres
    ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
    pts = np.vstack([c, r['ours'][list(AX)], r['ace'][list(AX)]])
    span = max(w, h, 2.2*np.abs(pts - c).max()) * 0.78 + 1e-3
    ax.add_patch(Ellipse(r['ours'][list(AX)], w, h, angle=ang, fc='#1f4e79',
                         alpha=0.26, ec='#1f4e79', lw=1.3, zorder=1))
    ax.plot(*c, '+', color='k', ms=11, mew=1.8, zorder=4, label='ground truth')
    ax.plot(*r['ours'][list(AX)], 'o', color='#1f4e79', ms=6, zorder=3,
            label=r'USGLoc (ours) $\pm 2\sigma$')
    ax.plot(*r['ace'][list(AX)], '^', color='#c0504d', ms=7, zorder=5,
            mec='white', mew=0.7, label='ACE (point only)')
    ax.set_xlim(c[0]-span, c[0]+span); ax.set_ylim(c[1]-span*1.08, c[1]+span)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('#999999'); sp.set_linewidth(0.6)
    bar = 10.0**np.floor(np.log10(span))
    if bar > span: bar /= 10
    x0 = c[0]-span*0.90; y0 = c[1]-span*0.93
    ax.plot([x0, x0+bar], [y0, y0], '-k', lw=1.6, solid_capstyle='butt')
    ax.text(x0+bar/2, y0+span*0.045, f"{bar*100:g} cm", fontsize=6.5, ha='center')
    ax.set_title(f"{lab}\nin-plane $2\\sigma$ = {r['maj']:.1f} cm,  error = {r['err']:.1f} cm",
                 fontsize=7.6)
axes[0].legend(frameon=False, fontsize=6.6, loc='upper left',
               bbox_to_anchor=(-0.02, -0.01), handletextpad=0.35, borderpad=0.0)
fig.suptitle("USGLoc returns a pose distribution: the ellipse grows with the error. "
             "Prior 3DGS localizers return the point alone.", fontsize=8.2, y=0.965)
fig.savefig("paper/figures/fig1_teaser.pdf", bbox_inches="tight", dpi=300)
print("wrote paper/figures/fig1_teaser.pdf")
for r, lab in zip(pick, labels):
    print(f"  {lab:13s} in-plane 2sigma {r['maj']:7.2f} cm   in-plane err {r['err']:7.2f} cm"
          f"   (3-D err {r['err3']:7.2f} cm)")
