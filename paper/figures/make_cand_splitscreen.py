#!/usr/bin/env python3
"""Fig. 1 teaser candidate "cand_splitscreen" -- what you get from them vs. from us.

DESIGN
------
One real query, one real map, two outputs side by side.  The two output panels
are the same picture twice -- same scene, same crop, same scale bar, same
estimated pose, same ground truth -- and differ in exactly one thing: the right
one also carries the posterior.  That one difference is the paper's
contribution, so it is the only difference the eye can find.

  left column  : the RGB frame the localizer is given (real 7-Scenes image).
  middle panel : "PRIOR 3DGS LOCALIZERS RETURN A POSE".  The real Gaussian map
                 seen from above with the single returned camera.  The true
                 pose is 8.8 cm away and nothing in the output says so.
  right panel  : "USGLoc RETURNS A POSE + COVARIANCE".  Same picture plus 2500
                 camera centres and 22 camera frusta drawn from the tempered
                 posterior N(0, tau*Sigma) in the chart the code itself uses,
                 xi = log(T T_mode^-1), T = exp(xi) T_mode (usgloc/posterior.py
                 lines 4 and 54), plus the reported scalar sigma_t as a dashed
                 circle.  The cloud is banana-shaped, not elliptical, because
                 translation and rotation are correlated in Sigma; that
                 curvature is the model's real output, not a drawn ellipse.

WHY THIS QUERY (disclosed rule, not hand-picked).  Of the 700 pooled queries
(7 scenes x 100 frames) in results/m3u8_seed0 that ACE also ran, keep those
where
  (a) our translation error > 5 cm, so a bare point estimate would really
      mislead a robot;
  (b) the ground truth sits at a 6-DoF tempered Mahalanobis distance BELOW the
      pooled median of 2.35, so the frame is a representative draw from the
      posterior rather than a flattering one;
  (c) ACE -- the only Table I baseline that emits any per-query confidence --
      is also wrong (> 5 cm), so the figure never rests on us winning on
      accuracy.
Exactly one frame survives: redkitchen seq-04 / 464.

PROJECTION.  Bird's-eye orthographic: world x across the page, world z up the
page, world y (the room's vertical) is the view direction.  7.9 cm of the
8.8 cm 3-D error and 0.99 of the posterior cloud's principal axis lie in this
plane, so the drawing does not hide what it prints.  The backdrop is the real
map (maps/redkitchen_r1.pt) splatted orthographically with a nearest-surface
z-buffer, with everything above the camera (the ceiling) cut away, then
desaturated and lightened so it reads as context and the overlays stay on top.

NUMBERS.  Everything printed is recomputed here from
results/m3u8_seed0/*.json and results/ace_dsacstar_100/*.json at draw time, so
the figure cannot drift from the data.  The decile ratio sorts the 500 queries
of the 5 scenes by each method's own confidence (our sigma_t, ACE's inlier
count) and divides the median error of the bottom 50 by that of the top 50.

No GPU: the backdrop is a numpy splat of the stored means / colors.

    CUDA_VISIBLE_DEVICES="" ~/venvs/uags/bin/python paper/figures/make_cand_splitscreen.py
    pdftoppm -png -r 150 paper/figures/cand_splitscreen.pdf /tmp/cand_splitscreen
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.patheffects as pe

ROOT = "/home/thanhnc/gauss_ws/src/usgloc"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from usgloc import se3  # noqa: E402

SCENE, SEQ, FRAME = "redkitchen", "seq-04", 464
NAME = "cand_splitscreen"

INK = "#1A1A1A"
GREY = "#5E666E"
OURS = "#C8371F"          # the returned pose
POST = "#1C5FA8"          # the posterior
TRUE = "#0E7A38"          # ground truth
BAR_L = "#868E96"         # "prior work" header bar
BAR_R = "#1C5FA8"         # "ours" header bar

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


HALO = [pe.withStroke(linewidth=2.1, foreground="white")]


def centre(T):
    """Camera centre of a world-to-camera matrix."""
    T = np.asarray(T, float)
    return -T[:3, :3].T @ T[:3, 3]


# -------------------------------------------------------------- measured
res = json.load(open(f"results/m3u8_seed0/7scenes_{SCENE}.json"))
tau = res["tau"]
rec = next(r for r in res["records"] if r["seq"] == SEQ and r["frame_id"] == FRAME)
Sigma = np.array(rec["Sigma"], float) * tau            # tempered 6x6
T_map = np.array(rec["T"], float)
c_map, c_gt = centre(rec["T"]), centre(rec["T_gt"])
err_cm = 100.0 * rec["t_err"]
sigma_t_cm = 100.0 * rec["sigma_t"] * tau ** 0.5

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
pool = []
for s in SCENES:
    d = json.load(open(f"results/m3u8_seed0/7scenes_{s}.json"))
    pool += [100.0 * r["sigma_t"] * d["tau"] ** 0.5 for r in d["records"]]
sig_pct = 100.0 * float(np.mean(np.array(pool) <= sigma_t_cm))
print(f"{SCENE} {SEQ}/{FRAME}: t_err {err_cm:.2f} cm, r_err {rec['r_err']:.2f} deg, "
      f"sigma_t {sigma_t_cm:.2f} cm ({sig_pct:.1f} pct of {len(pool)} queries)")

# ------------------------------------------------- pooled statistics
# Recomputed here from the same files so the strip under the figure cannot
# drift from the data.  Deciles: sort the 500 queries of the 5 scenes by the
# method's OWN confidence, take the median error of the top and bottom 50.
S5 = ["chess", "fire", "office", "redkitchen", "stairs"]
eo, so, ea, ia = [], [], [], []
for s in S5:
    d = json.load(open(f"results/m3u8_seed0/7scenes_{s}.json"))
    ac = {(r["seq"], r["frame_id"]): r for r in
          json.load(open(f"results/ace_dsacstar_100/7scenes_{s}.json"))["records"]}
    for r in d["records"]:
        k = (r["seq"], r["frame_id"])
        if k not in ac:
            continue
        eo.append(100 * r["t_err"]); so.append(r["sigma_t"] * d["tau"] ** 0.5)
        ea.append(100 * ac[k]["t_err"]); ia.append(float(ac[k]["inliers"]))
eo, so, ea, ia = map(np.array, (eo, so, ea, ia))
NQ = len(eo)
K = NQ // 10


def decile_ratio(err, conf, high_conf_is_small):
    o = np.argsort(conf if high_conf_is_small else -conf)
    return float(np.median(err[o[-K:]]) / np.median(err[o[:K]]))


R_OURS = decile_ratio(eo, so, True)
R_ACE = decile_ratio(ea, ia, False)

med_o, med_a = [], []
for s in SCENES:
    d = json.load(open(f"results/m3u8_seed0/7scenes_{s}.json"))
    ac = {(r["seq"], r["frame_id"]): r for r in
          json.load(open(f"results/ace_dsacstar_100/7scenes_{s}.json"))["records"]}
    pr = [(r, ac[(r["seq"], r["frame_id"])]) for r in d["records"]
          if (r["seq"], r["frame_id"]) in ac]
    med_o.append(100 * np.median([x[0]["t_err"] for x in pr]))
    med_a.append(100 * np.median([x[1]["t_err"] for x in pr]))
AVG_O, AVG_A = float(np.mean(med_o)), float(np.mean(med_a))
print(f"pooled: N={NQ}, decile ratio ours {R_OURS:.1f}x  ACE {R_ACE:.1f}x; "
      f"7-scene avg median err ours {AVG_O:.2f} cm  ACE {AVG_A:.2f} cm")

# --------------------------------------------------- posterior samples
gen = torch.Generator().manual_seed(11)
xi = torch.randn(2500, 6, generator=gen, dtype=torch.float64).numpy() \
    @ np.linalg.cholesky(Sigma).T
T_s = (se3.se3_exp(torch.tensor(xi)) @ torch.tensor(T_map)).numpy()
C_s = np.stack([centre(t) for t in T_s])
F_s = np.stack([t[:3, :3].T @ np.array([0.0, 0.0, 1.0]) for t in T_s])

# -------------------------------------------------------------- backdrop
H_AX, V_AX, D_AX = 0, 2, 1            # x across, z up the page, y = view dir


def splat(extent, W, H, near, opmin=0.35, delta=0.16):
    """Orthographic nearest-surface splat of the Gaussian map (CPU, numpy)."""
    m = torch.load(f"maps/{SCENE}_r1.pt", map_location="cpu", weights_only=False)
    mu = m["means"].numpy().astype(np.float32)
    col = m["colors"].numpy()
    op = m["opacities"].numpy()
    rad = np.sqrt(np.clip(np.diagonal(m["covs"].numpy(), axis1=1, axis2=2),
                          0, None)).mean(1)
    del m

    h, v, d = mu[:, H_AX], mu[:, V_AX], mu[:, D_AX]
    x0, x1, y0, y1 = extent
    mpp = (x1 - x0) / W
    keep = (op > opmin) & (d >= near)
    h, v, d, col, rad, op = h[keep], v[keep], d[keep], col[keep], rad[keep], op[keep]
    px = ((h - x0) / (x1 - x0) * W).astype(np.int32)
    py = ((1.0 - (v - y0) / (y1 - y0)) * H).astype(np.int32)
    rp = np.clip(rad / mpp, 0.8, 4.0)

    I, D, C, O = [], [], [], []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if np.hypot(a, b) > 4.0:
                continue
            qx, qy = px + a, py + b
            g = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H) & (np.hypot(a, b) <= rp)
            if g.any():
                I.append(qy[g] * W + qx[g]); D.append(d[g])
                C.append(col[g]); O.append(op[g])
    I = np.concatenate(I); D = np.concatenate(D)
    C = np.concatenate(C); O = np.concatenate(O)

    z = np.full(H * W, np.inf, np.float32)
    np.minimum.at(z, I, D)
    s = D <= z[I] + delta
    I, D, C, O = I[s], D[s], C[s], O[s]
    w = O * np.exp(-(D - z[I]) / delta * 1.5)
    acc = np.zeros((H * W, 3), np.float32)
    wsum = np.zeros(H * W, np.float32)
    np.add.at(acc, I, C * w[:, None])
    np.add.at(wsum, I, w)
    img = np.ones((H * W, 3), np.float32)
    nz = wsum > 0.50
    img[nz] = acc[nz] / wsum[nz, None]
    img = img.reshape(H, W, 3)
    img = 0.78 * img + 0.22 * img.mean(2, keepdims=True)    # desaturate
    return np.clip(0.74 * img + 0.26, 0, 1)                 # lighten

# ---------------------------------------------------------------- layout
FW, FH = 7.16, 2.74                       # inches; IEEE two-column figure*
M = 0.05
BAR_H = 0.235
BAR_Y = FH - M - BAR_H
PAN_Y, PAN_T = 0.465, BAR_Y - 0.025
PAN_H = PAN_T - PAN_Y
QW = 1.85
GAP1, GAP2 = 0.105, 0.115
PAN_W = (FW - M - (M + QW + GAP1) - GAP2) / 2.0
AX_L = M + QW + GAP1
AX_R = AX_L + PAN_W + GAP2
ASPECT = PAN_H / PAN_W

HALF_W = 1.30                             # metres, half the panel width
ch = float(np.median(C_s[:, H_AX]))
cv = float(np.median(C_s[:, V_AX]))
EXT = (ch - HALF_W, ch + HALF_W, cv - HALF_W * ASPECT, cv + HALF_W * ASPECT)
BG = splat(EXT, 1200, int(1200 * ASPECT), near=float(c_map[D_AX]) - 0.35)

fig = plt.figure(figsize=(FW, FH))


def rect(x, y, w, h):
    return [x / FW, y / FH, w / FW, h / FH]


# ---- query image -----------------------------------------------------
img_q = plt.imread(f"data/7scenes_source/{SCENE}/{SEQ}/frame-{FRAME:06d}.color.png")
QH = QW * img_q.shape[0] / img_q.shape[1]
QTOP = FH - M - 0.20
axq = fig.add_axes(rect(M, QTOP - QH, QW, QH))
axq.imshow(img_q)
axq.set_xticks([]); axq.set_yticks([])
for sp in axq.spines.values():
    sp.set_color(INK); sp.set_linewidth(0.9)

fig.text(M / FW, (FH - M - 0.09) / FH, "QUERY IMAGE", fontsize=8.8,
         weight="bold", color=INK, va="center", ha="left")

fig.text(M / FW, (QTOP - QH - 0.085) / FH,
         f"7-Scenes {SCENE}   {SEQ}/{FRAME}",
         fontsize=7.0, color=GREY, va="top", ha="left")
fig.text(M / FW, (QTOP - QH - 0.235) / FH,
         "One RGB frame in; the map is a\n"
         "semantic 3D Gaussian-splatting\nreconstruction of the room.",
         fontsize=7.4, color=INK, va="top", ha="left", linespacing=1.45)


# ---- panels ----------------------------------------------------------
def frustum(ax, c, f, length, half=31.0, **kw):
    """Plan-view camera wedge: apex at the centre, opening along the heading."""
    th = np.arctan2(f[H_AX], f[V_AX])
    a = np.radians(half)
    pts = [(c[H_AX], c[V_AX])]
    for s in (-a, a):
        k = length / np.cos(a)
        pts.append((c[H_AX] + k * np.sin(th + s), c[V_AX] + k * np.cos(th + s)))
    ax.add_patch(Polygon(pts, closed=True, **kw))


def panel(x, lead, punch, barcol, show_post):
    ax = fig.add_axes(rect(x, PAN_Y, PAN_W, PAN_H))
    ax.imshow(BG, extent=list(EXT), origin="upper", interpolation="bilinear",
              zorder=0, aspect="auto")
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#AEB4BA"); sp.set_linewidth(0.7)

    bar = fig.add_axes(rect(x, BAR_Y, PAN_W, BAR_H))
    bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
    bar.add_patch(Rectangle((0, 0), 1, 1, fc=barcol, ec="none", clip_on=False))
    bar.text(0.5, 0.47, lead + "  " + punch, fontsize=7.2, color="white",
             va="center", ha="center", weight="bold")

    if show_post:
        ax.scatter(C_s[:, H_AX], C_s[:, V_AX], s=2.0, c=POST, alpha=0.10,
                   lw=0, zorder=2, rasterized=True)
        order = np.argsort(C_s[:, H_AX])
        for i in order[np.linspace(12, len(order) - 13, 22).astype(int)]:
            frustum(ax, C_s[i], F_s[i], 0.145, fc=POST, ec=POST, alpha=0.30,
                    lw=0.4, zorder=3)
        ax.add_patch(Circle((c_map[H_AX], c_map[V_AX]), sigma_t_cm / 100.0,
                            fc="none", ec=POST, lw=0.9, ls=(0, (3.0, 2.2)),
                            zorder=4))

    frustum(ax, c_map, T_map[:3, :3].T @ np.array([0.0, 0.0, 1.0]), 0.26,
            fc=OURS, ec="white", alpha=0.95, lw=0.8, zorder=6)
    ax.plot(c_map[H_AX], c_map[V_AX], "o", ms=4.4, mfc=OURS, mec="white",
            mew=1.0, zorder=8)
    ax.plot(c_gt[H_AX], c_gt[V_AX], "o", ms=6.0, mfc="none", mec="white",
            mew=2.4, zorder=8)
    ax.plot(c_gt[H_AX], c_gt[V_AX], "o", ms=6.0, mfc="none", mec=TRUE,
            mew=1.3, zorder=9)

    # shared scale bar, identical in both panels
    bx, by = EXT[1] - 0.10 - 0.5, EXT[3] - 0.125
    ax.plot([bx, bx + 0.5], [by, by], lw=2.2, color=INK,
            solid_capstyle="butt", zorder=11)
    ax.text(bx + 0.25, by - 0.055, "50 cm", fontsize=7.0, color=INK,
            ha="center", va="top", zorder=11, path_effects=HALO)
    return ax


axA = panel(AX_L, "PRIOR 3DGS LOCALIZERS RETURN", "A POSE", BAR_L, False)
axB = panel(AX_R, "USGLoc RETURNS", "A POSE + COVARIANCE", BAR_R, True)

# ---- annotation ------------------------------------------------------
axA.annotate(f"true pose,\n{err_cm:.1f} cm away",
             xy=(c_gt[H_AX] + 0.012, c_gt[V_AX] - 0.012),
             xytext=(c_gt[H_AX] + 0.26, c_gt[V_AX] - 0.52),
             fontsize=7.4, color=TRUE, weight="bold", ha="left", va="center",
             linespacing=1.35,
             arrowprops=dict(arrowstyle="-", lw=0.8, color=TRUE,
                             shrinkA=2, shrinkB=3), zorder=12,
             path_effects=HALO)
axA.annotate("the estimated pose",
             xy=(c_map[H_AX] - 0.055, c_map[V_AX] + 0.055),
             xytext=(EXT[0] + 0.07, EXT[3] - 0.085),
             fontsize=7.4, color=OURS, weight="bold", ha="left", va="top",
             arrowprops=dict(arrowstyle="-", lw=0.8, color=OURS,
                             shrinkA=2, shrinkB=3), zorder=12,
             path_effects=HALO)

axB.annotate("2500 camera poses drawn\nfrom the returned posterior",
             xy=(float(np.percentile(C_s[:, H_AX], 10)),
                 float(np.percentile(C_s[:, V_AX], 88))),
             xytext=(EXT[0] + 0.07, EXT[3] - 0.085),
             fontsize=7.4, color=POST, weight="bold", ha="left", va="top",
             linespacing=1.35,
             arrowprops=dict(arrowstyle="-", lw=0.8, color=POST,
                             shrinkA=2, shrinkB=3), zorder=12)
axB.text(c_map[H_AX] + sigma_t_cm / 100.0 * 0.76 + 0.025,
         c_map[V_AX] - sigma_t_cm / 100.0 * 0.76 - 0.025,
         f"$\\sigma_t$ = {sigma_t_cm:.0f} cm", fontsize=7.6, color=POST,
         weight="bold", ha="left", va="top", zorder=12, path_effects=HALO)

for ax, txt, col in (
        (axA, f"no confidence returned:\nthe {err_cm:.1f} cm error is invisible",
         "#7A2516"),
        (axB, f"{sig_pct:.0f}th percentile of our own uncertainty:\n"
              "flag this fix, re-localize", "#133F6E")):
    ax.text(0.015, 0.018, txt, transform=ax.transAxes, fontsize=7.4, color=col,
            ha="left", va="bottom", zorder=13, linespacing=1.35,
            bbox=dict(boxstyle="square,pad=0.26", fc="white", ec="none",
                      alpha=0.88))

fig.add_artist(plt.Line2D([(AX_R - GAP2 / 2) / FW] * 2,
                          [(PAN_Y - 0.055) / FH, (FH - M) / FH],
                          color="#C3C8CD", lw=0.8))

# ---- bottom strip: the pooled claim behind the anecdote ---------------
fig.text(M / FW, (PAN_Y - 0.075) / FH,
         f"Pooled over {NQ} queries (5 scenes): binning each method by its own "
         f"confidence, the median error of the least-confident decile is\n"
         f"{R_OURS:.0f}$\\times$ that of the most-confident decile for our "
         f"$\\sigma_t$, but only {R_ACE:.0f}$\\times$ for ACE's inlier count, "
         f"the sole baseline confidence in Table I.\n"
         f"7-scene avg. median error: ours {AVG_O:.2f} cm, ACE {AVG_A:.2f} cm "
         f"(identical frames).  Both panels draw the same pose; only the "
         f"posterior differs.",
         fontsize=7.0, color=GREY, va="top", ha="left", linespacing=1.42)

out = f"paper/figures/{NAME}.pdf"
fig.savefig(out, dpi=600)
print("wrote", out)
