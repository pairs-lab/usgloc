#!/usr/bin/env python3
"""Real visuals for the overview figure, CPU only (the GPU is busy).

panel_map.png   the evidential primitives G_i = (mu_i, Sigma_i, ..., u_i) of the
                chess map, drawn as shaded 2-sigma ellipsoids coloured by fused
                vacuity, over a faint render of the map. Software-splatted from
                the real primitives built by usgloc.datasets.load_gaussian_map.
panel_post.png  the same free camera, with the estimated camera of query
                seq-03/909 and an inset at centimetre scale showing its
                2-sigma covariance and the true pose.
Both reuse the cached free-camera render of maps/chess_r1.pt
(cand_mapfrusta_bg.npz), so the two panels share one viewpoint.
"""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image

OUT = ROOT/"paper/figures/fig1_blocks"; OUT.mkdir(exist_ok=True)
z = np.load(ROOT/"paper/figures/cand_mapfrusta_bg.npz")
bg = np.clip(z["rgb"], 0, 1); eye, R = z["eye"], z["R"]
fx, cx, cy = float(z["fx"]), float(z["cx"]), float(z["cy"])
H, W = bg.shape[:2]
# vacuity palette shared with the rest of the figure: teal -> gold -> orange
UCM = LinearSegmentedColormap.from_list("u", ["#1B9E77", "#E6AB02", "#D95F02"])

def cam_of(P):
    return (np.atleast_2d(P) - eye) @ R.T

def proj(P):
    pc = cam_of(P); zz = np.maximum(pc[:, 2], 1e-6)
    return fx*pc[:, 0]/zz + cx, fx*pc[:, 1]/zz + cy, pc[:, 2]

def cov2d(mu, S):
    pc = cam_of(mu)[0]; zz = pc[2]
    J = (fx/zz)*np.array([[1, 0, -pc[0]/zz], [0, 1, -pc[1]/zz]])
    return J @ (R @ S @ R.T) @ J.T

# crop window shared by both panels: the room around the chess table
X0, X1, Y0, Y1 = int(0.20*W), int(0.80*W), int(0.16*H), int(0.86*H)

# ---------------------------------------------------------------- panel_map
from usgloc.datasets import load_gaussian_map
emap = load_gaussian_map("maps/chess_r1.pt")
# Legibility: keep primitives with >= 10 member Gaussians (a single-Gaussian
# "primitive" has no meaningful shape), and draw a seeded random subset so the
# room stays visible. Vacuity is the real fused value; nothing is recoloured.
allP = [(g.mu.numpy(), g.Sigma.numpy(), float(g.effective_vacuity), len(g.member_idx))
        for g in emap.primitives]
P = [p for p in allP if p[3] >= 10]
rng = np.random.default_rng(0)
P = [P[i] for i in sorted(rng.choice(len(P), size=min(380, len(P)), replace=False))]
print(f"{len(allP)} primitives, {sum(p[3]>=10 for p in allP)} with >=10 members, drawing {len(P)}")
buf = 0.35*bg + 0.65*bg.mean(2, keepdims=True)          # faint, desaturated map
buf = 0.70*buf + 0.30                                    # lift toward white
order = sorted(range(len(P)), key=lambda i: -cam_of(P[i][0])[0, 2])   # far -> near
drawn = 0
for i in order:
    mu, S, u, _ = P[i]
    if cam_of(mu)[0, 2] <= 0.15: continue
    C = cov2d(mu, S)
    if not np.all(np.isfinite(C)) or np.linalg.det(C) <= 1e-6: continue
    pu, pv, _ = proj(mu); pu, pv = pu[0], pv[0]
    r = 1.0*np.sqrt(np.linalg.eigvalsh(C).max())          # 1-sigma extent
    if r < 2.0 or r > 0.10*W: continue
    a0, a1 = int(max(pu-r, 0)), int(min(pu+r+1, W)); b0, b1 = int(max(pv-r, 0)), int(min(pv+r+1, H))
    if a1 <= a0 or b1 <= b0: continue
    xs, ys = np.meshgrid(np.arange(a0, a1)-pu, np.arange(b0, b1)-pv)
    Ci = np.linalg.inv(C)
    d2 = Ci[0, 0]*xs*xs + 2*Ci[0, 1]*xs*ys + Ci[1, 1]*ys*ys
    m = d2 < 1.0
    if not m.any(): continue
    h = np.sqrt(np.clip(1 - d2, 0, 1))                     # ellipsoid height
    ox, oy = -0.35*np.sqrt(C[0, 0]), -0.40*np.sqrt(C[1, 1])  # light from upper-left
    d2s = Ci[0, 0]*(xs-ox)**2 + 2*Ci[0, 1]*(xs-ox)*(ys-oy) + Ci[1, 1]*(ys-oy)**2
    base = np.array(UCM(u)[:3])
    shade = 0.45 + 0.55*h
    spec = 0.55*np.exp(-9.0*d2s)
    col = np.clip(base[None, None, :]*shade[..., None] + spec[..., None], 0, 1)
    alpha = (0.92*m*np.clip(1.2*h + 0.25, 0, 1))[..., None]
    buf[b0:b1, a0:a1] = buf[b0:b1, a0:a1]*(1-alpha) + col*alpha
    drawn += 1
print(f"drew {drawn} primitive ellipsoids")
Image.fromarray((np.clip(buf[Y0:Y1, X0:X1], 0, 1)*255).astype("uint8")).save(OUT/"panel_map.png")

# --------------------------------------------------------------- panel_post
d = json.load(open("results/ablation_k1_chess/7scenes_chess.json")); tau = d["tau"]
rec = next(r for r in d["records"] if r["seq"] == "seq-03" and r["frame_id"] == 909)
def centre(T):
    T = np.asarray(T, float); return -T[:3, :3].T @ T[:3, 3]
Ce, Ct = centre(rec["T"]), centre(rec["T_gt"])
Sig = np.array(rec["Sigma"])[:3, :3]*tau
err = 100*rec["t_err"]; sig = 100*rec["sigma_t"]*tau**0.5

crop = bg[Y0:Y1, X0:X1]; ch, cw = crop.shape[:2]
fig = plt.figure(figsize=(cw/200, ch/200), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(crop); ax.set_axis_off()
ax.set_xlim(0, cw); ax.set_ylim(ch, 0)
Rw = np.asarray(rec["T"], float)[:3, :3].T
F = 0.30*(Rw @ np.array([[0,0,0],[-.6,-.45,1],[.6,-.45,1],[.6,.45,1],[-.6,.45,1]]).T).T + Ce
fu, fv, _ = proj(F); fu -= X0; fv -= Y0
ax.add_patch(Polygon(np.c_[fu[1:], fv[1:]], closed=True, fc="#ff3b30", alpha=0.18, ec="#ff3b30", lw=2.0))
for j in range(1, 5):
    ax.plot([fu[0], fu[j]], [fv[0], fv[j]], color="#ff3b30", lw=2.0)
ax.plot(fu[0], fv[0], "o", color="#ff3b30", ms=6, mec="white", mew=1.2)
# inset at centimetre scale, anchored to the camera centre
Cp = cov2d(Ce, Sig); ev, evec = np.linalg.eigh(Cp)
w2, h2 = 2*2*np.sqrt(np.clip(ev, 0, None)); ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
tu, tv, _ = proj(Ct); tu, tv = tu[0]-X0, tv[0]-Y0
span = max(w2, h2, 2*np.hypot(tu-fu[0], tv-fv[0]))*0.85 + 1e-3
ia = ax.inset_axes([0.48, 0.03, 0.50, 0.56])
ia.imshow(crop, interpolation="bicubic"); ia.set_xlim(fu[0]-span, fu[0]+span); ia.set_ylim(fv[0]+span, fv[0]-span)
ia.set_xticks([]); ia.set_yticks([])
for sp in ia.spines.values(): sp.set_edgecolor("white"); sp.set_linewidth(2.2)
ia.add_patch(Ellipse((fu[0], fv[0]), w2, h2, angle=ang, fc="#4da3ff", alpha=0.40, ec="none"))
for ec, lw in (("white", 3.0), ("#1f4e79", 1.6)):
    ia.add_patch(Ellipse((fu[0], fv[0]), w2, h2, angle=ang, fc="none", ec=ec, lw=lw))
ia.plot(tu, tv, "o", mfc="none", mec="#19c37d", mew=3.0, ms=17)
ia.plot(fu[0], fv[0], "^", color="#ff3b30", ms=14, mec="white", mew=1.3)
px_per_cm = fx/float(cam_of(Ce)[0, 2])/100.0
bar = 0.5 if span/px_per_cm < 2 else 1.0
ia.plot([fu[0]-span*0.85, fu[0]-span*0.85+bar*px_per_cm], [fv[0]+span*0.80]*2, "-w", lw=3.2, solid_capstyle="butt")
ia.text(fu[0]-span*0.85+bar*px_per_cm/2, fv[0]+span*0.66, f"{bar:g} cm", color="white",
        ha="center", fontsize=13, fontweight="bold")
ax.add_patch(Rectangle((fu[0]-span, fv[0]-span), 2*span, 2*span, fill=False, ec="white", lw=1.4))
fig.savefig(OUT/"panel_post.png", dpi=200)
print(f"seq-03/909: error {err:.2f} cm, reported 1-sigma {sig:.2f} cm")
