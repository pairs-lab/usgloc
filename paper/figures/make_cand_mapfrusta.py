#!/usr/bin/env python3
"""Fig. 1 teaser candidate "cand_mapfrusta" -- CONCEPT C, "the posterior lives in the map".

DESIGN
------
The contribution of this paper is not accuracy, it is that the pose comes out
of the 3DGS map with a *calibrated SE(3) covariance* attached.  A teaser must
therefore show (i) a real map, (ii) real queries localized inside it, and
(iii) the fact that the reported uncertainty tracks the real error.  Two earlier
attempts failed by being abstract (boxes with dots) or by being an analysis plot
with unreadable thumbnails, so this one is built from exactly three large
elements:

  (a) ONE big photorealistic view of the chess 3D Gaussian map, rendered with
      gsplat from a free virtual camera (maps/chess_r1.pt -- the same map the
      reported run used).  All 100 test queries are drawn where USGLoc placed
      them, each as a dot whose COLOUR and AREA encode the reported sigma_t.
      Two queries are drawn as explicit camera frusta so the dots read as
      cameras.  Most of the loop is dark and small (confident); a handful blow
      up bright and large -- and those are the ones that are actually wrong,
      which the overlaid measured line states in numbers.

  (b),(c) The two ends of that scale as real imagery: the query photo next to
      the map rendered at the pose we returned.  In (b) the reported sigma is the
      smallest in chess and the two pictures are the same scene; in (c) the
      reported sigma is the largest of the 500 pooled queries and the render is
      visibly the wrong part of the room.  This is what the ellipse buys you.

THE SCALE PROBLEM, SOLVED HONESTLY
    Reported sigma_t spans 0.2 cm to 3 m inside a 5 m room, so an ellipse drawn at
    metric scale is either invisible or swallows the map.  Nothing here is drawn
    at metric scale: sigma_t is encoded by colour and marker area on a LOG axis and
    both the colour-bar label and the panel note say so.

WHAT IS DRAWN WHERE
    white frustum "1" = the query with the SMALLEST reported sigma_t in chess.
    yellow frustum    = the query with the LARGEST TRUE error in chess, picked
                        on error (not on sigma) so it cannot be cherry-picked;
                        the figure then states where its sigma_t ranks.
    vertical colour bar sits over the blown-out right edge of the render.

APPROXIMATIONS / CAVEATS
  * the big panel renders maps/chess_r1.pt (the map the reported run used);
    the four small images were pre-rendered from maps/chess_sel.pt and
    maps/stairs_sel.pt by make_fig_qual3.py, so the two are not byte-identical
    maps of the same scene.
  * the background render culls 3DGS floaters (opacity <= 0.08, longest axis
    >= 0.10 m, or outside the room box) purely so the free virtual view is
    legible; the stated map size, 688,973, is the real one.
  * the background is desaturated and darkened so the markers read on top.
  * "calibrated" is deliberately NOT claimed in the figure: the tempered
    covariance is near nominal at 68% and over-confident above it.

MEASURED INPUTS (nothing in this figure is invented)
    results/m3u8_seed0/7scenes_chess.json   poses T, T_gt, sigma_t, t_err, tau
    results/ace_dsacstar_100/7scenes_chess.json   ACE on the same 100 frames
    paper/figures/fig_qual3/{0,1}_{query,T_ours}.png + selection.json
    maps/chess_r1.pt                         the Gaussian map itself

RUN
    # background render (GPU, once; cached to cand_mapfrusta_bg.npz afterwards)
    export CUDA_HOME=/home/thanhnc/.local/cuda-12.8
    export PATH=/home/thanhnc/venvs/uags/bin:$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    python paper/figures/make_cand_mapfrusta.py
    # every later run (no GPU needed) reuses the cache
    CUDA_VISIBLE_DEVICES="" python paper/figures/make_cand_mapfrusta.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
FIG = ROOT / "paper/figures"
CACHE = FIG / "cand_mapfrusta_bg.npz"
NAME = "cand_mapfrusta"

# ---------------------------------------------------------------- figure size
FW, FH = 7.16, 2.75                 # in; IEEE full width (figure*), <= 2.8 in tall
PANEL = (0.030, 0.055, 4.300, 2.510)        # (a) left, bottom, width, height [in]
RIGHT_L, RIGHT_R = 4.405, 7.135             # right block [in]
UP = np.array([0.0, -1.0, 0.0])             # 7-Scenes/SfM world: +y is down
VIEW = dict(az=295.0, el=46.0, dist=3.10)   # virtual camera around the trajectory
BG_W = 1500                                 # background render width [px]


# ------------------------------------------------------------------- geometry
def centre(T):
    T = np.asarray(T, float)
    return -T[:3, :3].T @ T[:3, 3]           # T is world-to-camera


def look_at(eye, target, up):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(up, f); r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), f])  # rows: right, image-down, forward


def load_queries():
    d = json.load(open("results/m3u8_seed0/7scenes_chess.json"))
    tau = d["tau"]
    ace = {(r["seq"], r["frame_id"]): r for r in
           json.load(open("results/ace_dsacstar_100/7scenes_chess.json"))["records"]}
    rows = []
    for r in d["records"]:
        k = (r["seq"], r["frame_id"])
        rows.append(dict(seq=k[0], frame=k[1],
                         C=centre(r["T"]), R=np.asarray(r["T"], float)[:3, :3],
                         sig=100.0 * r["sigma_t"] * tau ** 0.5,     # tempered, cm
                         err=100.0 * r["t_err"],                    # cm
                         ace=100.0 * ace[k]["t_err"] if k in ace else np.nan))
    return rows, tau


# ------------------------------------------------- background 3DGS map render
def render_background(rows):
    """gsplat render of maps/chess_r1.pt from a free virtual camera, framed so
    the query loop fills the middle of the panel.  Cached to NPZ."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z["rgb"], dict(eye=z["eye"], R=z["R"], fx=float(z["fx"]),
                              cx=float(z["cx"]), cy=float(z["cy"]),
                              W=int(z["W"]), H=int(z["H"]), n=int(z["n"]), ntot=int(z["ntot"]))
    import torch
    from usgloc.rendering import Camera, GsplatRenderer

    C = np.array([r["C"] for r in rows])
    tgt = C.mean(0)
    a, e = np.radians(VIEW["az"]), np.radians(VIEW["el"])
    eye = tgt + VIEW["dist"] * np.array([np.cos(a) * np.cos(e), -np.sin(e),
                                         np.sin(a) * np.cos(e)])
    R = look_at(eye, tgt, UP)

    W = BG_W
    H = int(round(W * PANEL[3] / PANEL[2]))
    pc = (C - eye) @ R.T
    xn, yn = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    fx = 0.600 * W / (xn.max() - xn.min())           # loop spans 62 % of the width
    cx = 0.440 * W - fx * 0.5 * (xn.min() + xn.max())
    cy = 0.500 * H - fx * 0.5 * (yn.min() + yn.max())

    m = torch.load("maps/chess_r1.pt", map_location="cpu", weights_only=False)
    mu = m["means"].numpy()
    smax = np.sqrt(np.clip(np.linalg.eigvalsh(m["covs"].numpy())[:, -1], 0, None))
    lo, hi = C.min(0), C.max(0)
    box = (np.all(mu > lo - np.array([1.8, 1.3, 1.8]), 1) &
           np.all(mu < hi + np.array([1.8, 1.3, 1.8]), 1))
    # drop near-transparent splats, over-long streaks and anything far outside
    # the room the trajectory lives in; these are 3DGS floaters, not geometry
    keep = torch.from_numpy(box & (m["opacities"].numpy() > 0.08) & (smax < 0.10))
    print(f"[bg] rendering {int(keep.sum())}/{len(mu)} Gaussians at {W}x{H}")

    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ eye
    with torch.no_grad():
        out = GsplatRenderer().render(
            m["means"][keep].cuda(), m["covs"][keep].cuda(),
            m["colors"][keep].cuda(), m["opacities"][keep].cuda(),
            torch.tensor(T, dtype=torch.float32, device="cuda"),
            Camera(fx=fx, fy=fx, cx=cx, cy=cy, width=W, height=H))
    rgb = out["rgb"].clamp(0, 1).cpu().numpy().astype(np.float32)
    cam = dict(eye=eye, R=R, fx=fx, cx=cx, cy=cy, W=W, H=H,
               n=int(keep.sum()), ntot=len(mu))
    np.savez_compressed(CACHE, rgb=rgb, **cam)
    return rgb, cam


def project(P, cam):
    pc = (np.atleast_2d(P) - cam["eye"]) @ cam["R"].T
    z = np.maximum(pc[:, 2], 1e-6)
    return cam["fx"] * pc[:, 0] / z + cam["cx"], cam["fx"] * pc[:, 1] / z + cam["cy"], z


# ----------------------------------------------------------------------- main
def ordinal(x):
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon
    from PIL import Image

    rows, tau = load_queries()
    sig = np.array([r["sig"] for r in rows])
    err = np.array([r["err"] for r in rows])
    ace = np.array([r["ace"] for r in rows])
    rgb, cam = render_background(rows)
    N_MAP = cam["ntot"]            # Gaussians in maps/chess_r1.pt

    # every number printed in the figure is computed here, from those files
    o = np.argsort(sig)
    med_conf = float(np.median(err[o[:10]]))        # 10 most-confident queries
    med_unc = float(np.median(err[o[-10:]]))        # 10 least-confident queries
    med_ace = float(np.median(ace))                 # ACE on the same 100 frames
    i_conf = int(o[0])                              # smallest reported sigma_t
    i_worst = int(np.argmax(err))                   # largest TRUE error, not cherry-picked
    pct_worst = 100.0 * float(np.mean(sig <= sig[i_worst]))

    # background treatment: desaturate + darken so the colour-coded cameras
    # read as the foreground layer.  Cosmetic only; no geometry is altered.
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    bg = np.clip(0.88 * (0.62 * rgb + 0.38 * lum[..., None]), 0, 1)

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    def axes_in(x, y, w, h):
        a = fig.add_axes([x / FW, y / FH, w / FW, h / FH])
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set(color="0.30", linewidth=0.7)
        return a

    def ftext(x, y, t, **kw):
        return fig.text(x / FW, y / FH, t, zorder=20, va="baseline", **kw)

    def frect(x, y, w, h, **kw):
        fig.patches.append(plt.Rectangle((x / FW, y / FH), w / FW, h / FH,
                                         transform=fig.transFigure, ec="none",
                                         zorder=15, **kw))

    # =================================================== (a) the map panel ===
    ax = axes_in(*PANEL)
    ax.imshow(bg, extent=[0, cam["W"], cam["H"], 0], interpolation="bilinear")
    ax.set_xlim(0, cam["W"]); ax.set_ylim(cam["H"], 0)

    C = np.array([r["C"] for r in rows])
    u, v, _ = project(C, cam)
    cmap, norm = plt.cm.plasma, LogNorm(vmin=0.2, vmax=300.0)
    # marker AREA uses the same log axis as the colour: a 1000x change in
    # sigma_t is ~10x in marker diameter.  Neither is a metric size.
    frac = np.log10(sig / 0.2) / np.log10(300.0 / 0.2)
    ax.scatter(u, v, s=10.0 + 170.0 * frac ** 1.8, c=sig, cmap=cmap, norm=norm,
               linewidths=0.4, edgecolors="white", zorder=3)

    def frustum(i, color, lw, depth=0.26):
        """Draw query i as a real camera frustum sitting in the map."""
        r = rows[i]
        Rw = r["R"].T                                  # world axes of the camera
        w, h = depth * 0.75, depth * 0.56
        cor = np.array([[-w, -h, depth], [w, -h, depth],
                        [w, h, depth], [-w, h, depth]]) @ Rw.T + r["C"]
        pu, pv, _ = project(cor, cam)
        au, av, _ = project(r["C"][None], cam)
        ax.add_patch(Polygon(np.c_[pu, pv], closed=True, fill=False,
                             ec=color, lw=lw, zorder=5))
        for j in range(4):
            ax.plot([au[0], pu[j]], [av[0], pv[j]], color=color, lw=lw, zorder=5)
        return au[0], av[0]

    cu, cv = frustum(i_conf, "white", 1.0)
    wu, wv = frustum(i_worst, "#fde047", 1.2)

    ax.annotate(f"worst of the 100: {err[i_worst]:.1f} cm error\n"
                f"its $\\sigma_t$ is this scene's {ordinal(pct_worst)} percentile",
                xy=(wu, wv), xytext=(0.030, 0.355), textcoords="axes fraction",
                color="#fde047", fontsize=7.3, ha="left", va="bottom", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.30", fc="black", alpha=0.62, ec="none"),
                arrowprops=dict(arrowstyle="-", color="#fde047", lw=0.8,
                                shrinkA=4, shrinkB=8), zorder=6)
    ax.annotate("1", xy=(cu, cv), xytext=(0.645, 0.530), textcoords="axes fraction",
                color="white", fontsize=8.0, fontweight="bold", ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.24", fc="black", ec="white", lw=0.9),
                arrowprops=dict(arrowstyle="-", color="white", lw=0.8,
                                shrinkA=6, shrinkB=8))

    # ---- title bar (two lines, full panel width)
    frect(PANEL[0], PANEL[1] + PANEL[3] - 0.315, PANEL[2], 0.315, fc="black", alpha=0.64)
    ftext(PANEL[0] + 0.075, PANEL[1] + PANEL[3] - 0.145,
          "USGLoc: every query returns a pose $and$ its covariance",
          fontsize=8.8, color="white", fontweight="bold", ha="left")
    ftext(PANEL[0] + 0.075, PANEL[1] + PANEL[3] - 0.262,
          f"chess 3D Gaussian map  |  100 test queries  |  {N_MAP:,} Gaussians",
          fontsize=7.0, color="0.80", ha="left")

    # ---- measured numbers, bottom LEFT only, so the map stays visible at right
    frect(PANEL[0] + 0.020, 0.072, 2.520, 0.600, fc="black", alpha=0.80)
    ftext(0.120, 0.545,
          f"median error, 10 most-confident queries:   {med_conf:.2f} cm",
          fontsize=7.2, color="white", ha="left")
    ftext(0.120, 0.405,
          f"median error, 10 least-confident queries:  {med_unc:.2f} cm",
          fontsize=7.2, color="white", ha="left")
    ftext(0.120, 0.265,
          f"all 100 frames: ours {np.median(err):.2f} cm, ACE {med_ace:.2f} cm "
          f"(no covariance)",
          fontsize=7.0, color="0.80", ha="left")
    ftext(0.120, 0.120,
          "colour $+$ marker area encode $\\sigma_t$; never metric scale",
          fontsize=7.1, color="0.80", ha="left")

    # ---- vertical colour bar, overlaid on the right edge of the panel
    cbx = PANEL[0] + PANEL[2] - 0.575
    frect(cbx - 0.065, 0.780, 0.600, 1.300, fc="black", alpha=0.78)
    ftext(cbx + 0.235, 1.945, "reported", fontsize=7.1, color="white", ha="center")
    ftext(cbx + 0.235, 1.822, "$\\sigma_t$  [cm, log]", fontsize=7.1,
          color="white", ha="center")
    cb_ax = fig.add_axes([(cbx + 0.355) / FW, 0.845 / FH, 0.055 / FW, 0.890 / FH],
                         zorder=16)
    cb = ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks([0.3, 1, 10, 100]); cb.ax.set_yticklabels(["0.3", "1", "10", "100"])
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.tick_params(labelsize=7.0, colors="white", length=1.6, pad=1.4)
    cb.outline.set(color="white", linewidth=0.5)

    # ============================= (b),(c) the two ends of that scale ========
    sel = json.load(open(FIG / "fig_qual3/selection.json"))
    iw, gap = 1.310, 0.050
    ih = iw * 480.0 / 640.0
    for j, t in enumerate(["query image", "map rendered at our pose"]):
        ftext(RIGHT_L + j * (iw + gap) + iw / 2, 2.620, t, fontsize=7.6,
              ha="center", color="0.12")
    cases = [dict(i=0, y=1.605, tag="1",
                  head=f"chess, query {sel[0]['frame']}:   $\\sigma_t$ = {sel[0]['sig']:.2f} cm"
                       f"   $\\rightarrow$   error {sel[0]['e_ours']:.2f} cm"),
             dict(i=1, y=0.455, tag="2",
                  head=f"stairs, query {sel[1]['frame']}:   $\\sigma_t$ = {sel[1]['sig']:.0f} cm"
                       f"   $\\rightarrow$   error {sel[1]['e_ours']:.0f} cm")]
    for c in cases:
        for j, key in enumerate(["query", "T_ours"]):
            a = axes_in(RIGHT_L + j * (iw + gap), c["y"], iw, ih)
            a.imshow(np.asarray(Image.open(FIG / f"fig_qual3/{c['i']}_{key}.png")))
        ftext(RIGHT_L + 0.010, c["y"] - 0.135, c["head"], fontsize=7.4,
              ha="left", color="0.08")
        fig.text((RIGHT_L + 0.105) / FW, (c["y"] + ih - 0.100) / FH, c["tag"],
                 fontsize=7.8, color="white", fontweight="bold", ha="center",
                 va="center", zorder=19,
                 bbox=dict(boxstyle="circle,pad=0.26", fc="black", ec="white", lw=0.9))
    ftext(RIGHT_L + 0.010, 0.185,
          "Pose 2 is unusable $-$ and USGLoc says so first.",
          fontsize=7.4, ha="left", color="0.08", fontweight="bold")
    ftext(RIGHT_L + 0.010, 0.055,
          "No prior 3DGS localizer reports a pose covariance.",
          fontsize=7.0, ha="left", color="0.35")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{NAME}.pdf", dpi=600, facecolor="white")
    print(f"saved {FIG / (NAME + '.pdf')}")
    print(f"  tau={tau:.2f}  sigma_t {sig.min():.2f}-{sig.max():.1f} cm  "
          f"err {err.min():.2f}-{err.max():.2f} cm  (chess, 100 queries)")
    print(f"  median err: ours {np.median(err):.2f} cm   ACE {med_ace:.2f} cm")
    print(f"  10 most-confident median {med_conf:.2f} cm | "
          f"10 least-confident {med_unc:.2f} cm  ({med_unc / med_conf:.1f}x)")
    print(f"  frustum 1  {rows[i_conf]['seq']}/{rows[i_conf]['frame']}  "
          f"sigma {sig[i_conf]:.2f} cm  err {err[i_conf]:.2f} cm")
    print(f"  frustum 2  {rows[i_worst]['seq']}/{rows[i_worst]['frame']}  "
          f"err {err[i_worst]:.2f} cm  sigma {sig[i_worst]:.1f} cm = {pct_worst:.0f}th pct")


if __name__ == "__main__":
    main()
