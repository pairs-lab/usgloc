#!/usr/bin/env python3
"""Figure 1 candidate "cand_filterstory" -- CONCEPT D: a measurement a filter can use.

DESIGN
------
USGLoc is not the most accurate localizer in the paper's Table I; it is the first that
returns a *calibrated SE(3) covariance* with every pose. The teaser must therefore
answer one question at a glance: "what do I get that I do not get elsewhere?"
Answer: a per-frame noise model a Bayesian filter can consume directly.

Layout (IEEE figure*, 7.16 in x 2.78 in; no \\resizebox, every label >= 7 pt):

  LEFT COLUMN (2.22 in) -- what the system is
    * a real query photo beside the semantic 3DGS map rendered at our estimated pose,
      so the reader sees "camera image vs. Gaussian map" instead of an abstract box;
    * two typographic rows: USGLoc emits (pose, Sigma); the baselines emit a bare pose,
      so the filter has to assume one hand-tuned noise;
    * a two-bar chart showing the Sigma is informative: ranking each method by its OWN
      confidence, the median error of the least-confident over the most-confident decile
      is 44x for our sigma and only 7x for ACE's inlier count (500 queries, 5 scenes).

  RIGHT PANEL (dominant, 4.10 in) -- the measured proof on a real sequence
    150 consecutive queries of chess seq-03. Filled band = the reported 1-sigma of the
    calibrated covariance, dark line = the true error, dashed line = the single fixed
    noise a filter must guess without Sigma. The band sits at ~0.5 cm through the
    texture-rich stretch and rises ~180x where the camera faces a near-blank whiteboard.
    The two thumbnails above the panel are the actual query frames at those two moments,
    so the reader sees *why* the covariance moves.

Deliberately avoided (two earlier candidates failed on exactly these): six small
thumbnails plus an abstract decile plot, and abstract boxes holding only dots and
ellipses at scales 1000x apart. Here: 4 images, one dominant measured panel, direct
labels instead of legend boxes.

Every printed number is recomputed from the result JSONs at run time. Run with
  CUDA_VISIBLE_DEVICES="" ~/venvs/uags/bin/python paper/figures/make_cand_filterstory.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
NAME = "cand_filterstory"

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.minor.size": 0.0, "ytick.minor.size": 0.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600,
})

C_FILL = "#D7E4F4"      # reported 1-sigma fill
C_BLUE = "#2F6DB5"      # "ours" accent
C_EDGE = "#6E9BD1"      # band edge
C_ERR = "#15202B"       # true error
C_FIX = "#C0562F"       # the fixed-noise alternative
C_GREY = "#7A7A7A"
C_INK = "#15202B"
C_SUB = "#4A4A4A"

# ------------------------------------------------------------------------------ data
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
DECILE_SCENES = ["chess", "fire", "heads", "office", "stairs"]   # the 500-query pool


def load(p):
    with open(ROOT / p) as fh:
        return json.load(fh)


def trace(scene="chess"):
    """Consecutive-frame trace: frame id, error (cm), calibrated 1-sigma (cm)."""
    d = load(f"results/traj_{scene}/7scenes_{scene}.json")
    tau = d["tau"]
    r = d["records"]
    f = np.array([x["frame_id"] for x in r], float)
    e = np.array([x["t_err"] for x in r]) * 100.0
    # sigma_t in the record is sqrt(trace(Sigma_tt)) of the RAW covariance; the
    # calibrated covariance is tau * Sigma (usgloc/calibration.py:136), so the reported
    # 1-sigma scale is sqrt(tau) * sigma_t  (same as scripts/make_tables.py:27).
    s = np.array([x["sigma_t"] for x in r]) * np.sqrt(tau) * 100.0
    return f, e, s, tau, r[0]["seq"]


def decile_ratio(conf_worse_is_larger, err):
    """Median error of the least- over the most-confident decile."""
    order = np.argsort(np.asarray(conf_worse_is_larger))
    err = np.asarray(err)[order]
    med = [float(np.median(err[b])) for b in np.array_split(np.arange(len(err)), 10)]
    return med[0], med[-1], med[-1] / med[0]


def pooled_confidence():
    E, S, EA, IA = [], [], [], []
    for s in DECILE_SCENES:
        d = load(f"results/m3u8_seed0/7scenes_{s}.json")
        tau = d["tau"]
        ours = {(x["seq"], x["frame_id"]): x for x in d["records"]}
        ace = {(x["seq"], x["frame_id"]): x
               for x in load(f"results/ace_dsacstar_100/7scenes_{s}.json")["records"]}
        for k in ours:
            if k not in ace:
                continue
            E.append(ours[k]["t_err"] * 100.0)
            S.append(ours[k]["sigma_t"] * np.sqrt(tau))
            EA.append(ace[k]["t_err"] * 100.0)
            IA.append(-ace[k]["inliers"])        # more inliers = more confident
    return decile_ratio(S, E), decile_ratio(IA, EA), len(E)


def accuracy_parity():
    """Table-I protocol: per-scene median over seeds, then the 7-scene average."""
    om, am = [], []
    for s in SCENES:
        per_seed = []
        for sd in (0, 1, 2):
            p = ROOT / f"results/m3u8_seed{sd}/7scenes_{s}.json"
            if p.is_file():
                with open(p) as fh:
                    per_seed.append(np.median([100 * r["t_err"]
                                               for r in json.load(fh)["records"]]))
        om.append(float(np.median(per_seed)))
        am.append(float(np.median([100 * r["t_err"] for r in
                                   load(f"results/ace_dsacstar_100/7scenes_{s}.json")["records"]])))
    return float(np.mean(om)), float(np.mean(am))


F, E, S, TAU, SEQ = trace("chess")
(OUR_TOP, OUR_BOT, OUR_R), (ACE_TOP, ACE_BOT, ACE_R), N_POOL = pooled_confidence()
MED_OURS, MED_ACE = accuracy_parity()

RMS = float(np.sqrt((E ** 2).mean()))            # the one fixed noise a filter must pick
S_P5, S_P95 = float(np.percentile(S, 5)), float(np.percentile(S, 95))
COV1, COV2 = float((E < S).mean()), float((E < 2 * S).mean())
TOO_LOOSE = float((S < RMS / 3).mean())          # fixed noise >= 3x too large
TOO_TIGHT = float((S > 3 * RMS).mean())          # fixed noise >= 3x too small

F_EASY, F_HARD = 60, 344
i_easy, i_hard = int(np.where(F == F_EASY)[0][0]), int(np.where(F == F_HARD)[0][0])

print(f"tau={TAU:.2f}  frames {F.min():.0f}-{F.max():.0f} (n={len(F)}, chess {SEQ})")
print(f"sigma p5-p95 {S_P5:.2f}-{S_P95:.1f} cm (x{S_P95/S_P5:.0f})   RMS err {RMS:.2f} cm")
print(f"coverage 1s {COV1:.3f} 2s {COV2:.3f}  fixed >=3x loose {TOO_LOOSE:.2f} tight {TOO_TIGHT:.2f}")
print(f"frame {F_EASY}: err {E[i_easy]:.2f} sig {S[i_easy]:.2f} | "
      f"frame {F_HARD}: err {E[i_hard]:.2f} sig {S[i_hard]:.1f}")
print(f"decile n={N_POOL}: ours {OUR_TOP:.2f}->{OUR_BOT:.2f} ({OUR_R:.1f}x)  "
      f"ace {ACE_TOP:.2f}->{ACE_BOT:.2f} ({ACE_R:.1f}x)")
print(f"7-scene median: ours {MED_OURS:.2f} cm   ace {MED_ACE:.2f} cm")

# ---------------------------------------------------------------------------- canvas
FW, FH = 7.16, 2.78
fig = plt.figure(figsize=(FW, FH))
fig.patch.set_facecolor("white")


def ax_in(x, y, w, h, **kw):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH], **kw)


def txt(x, y, s, **kw):
    kw.setdefault("color", C_INK)
    kw.setdefault("va", "baseline")
    return fig.text(x / FW, y / FH, s, **kw)


def rule(x0, x1, y, c="#D2D2D2"):
    fig.add_artist(plt.Line2D([x0 / FW, x1 / FW], [y / FH, y / FH], color=c, lw=0.6))


def imshow_png(ax, path, top=None, bot=None):
    ax.imshow(plt.imread(ROOT / path))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#3A3A3A")
    box = dict(boxstyle="square,pad=0.17", fc="#15202B", ec="none", alpha=0.80)
    if top:
        ax.text(0.035, 0.96, top, transform=ax.transAxes, va="top", ha="left",
                fontsize=7, color="white", zorder=5, bbox=box)
    if bot:
        ax.text(0.5, 0.04, bot, transform=ax.transAxes, va="bottom", ha="center",
                fontsize=7, color="white", zorder=5, bbox=box)


# ============================================================ LEFT: what you get
LX, LW = 0.07, 2.22
IW = (LW - 0.07) / 2.0
IH = IW * 0.75
IY = FH - 0.065 - IH

imshow_png(ax_in(LX, IY, IW, IH), "paper/figures/fig_qual3/0_query.png", "query",
           "seq-05 · frame 878")
imshow_png(ax_in(LX + IW + 0.07, IY, IW, IH), "paper/figures/fig_qual3/0_T_ours.png",
           "3DGS map")
txt(LX, IY - 0.145, "a camera image localized in a semantic 3DGS map", fontsize=7)
rule(LX, LX + LW, IY - 0.235)

# --- the two output contracts -------------------------------------------------
COL2 = LX + 0.72
y1 = IY - 0.385
fig.patches.append(Rectangle((LX / FW, (y1 - 0.145) / FH), 0.030 / FW, 0.265 / FH,
                             transform=fig.transFigure, fc=C_BLUE, ec="none"))
txt(LX + 0.085, y1, "USGLoc", fontsize=8.5, weight="bold", color=C_BLUE)
txt(COL2, y1, "→  pose  +  covariance Σ", fontsize=8.5)
txt(LX + 0.085, y1 - 0.125, "a filter can weight each frame by its own Σ",
    fontsize=7, color=C_SUB)

y2 = y1 - 0.305
fig.patches.append(Rectangle((LX / FW, (y2 - 0.145) / FH), 0.030 / FW, 0.265 / FH,
                             transform=fig.transFigure, fc="#C4C4C4", ec="none"))
txt(LX + 0.085, y2, "baselines", fontsize=8.5, weight="bold", color=C_GREY)
txt(COL2, y2, "→  pose only", fontsize=8.5, color=C_GREY)
txt(LX + 0.085, y2 - 0.125, "the filter must assume one hand-tuned noise",
    fontsize=7, color="#6E6E6E")

# --- is that confidence informative? ------------------------------------------
rule(LX, LX + LW, y2 - 0.235)
txt(LX, y2 - 0.385, "error spread by each method's own confidence",
    fontsize=7.5, weight="bold")
txt(LX, y2 - 0.495,
    f"least- vs. most-confident decile, {N_POOL} queries", fontsize=7.0, color=C_SUB)

BX0, BW = LX + 0.62, LW - 0.62 - 0.02
bax = ax_in(BX0, 0.235, BW, 0.40)
bax.barh([1, 0], [OUR_R, ACE_R], height=0.60, color=[C_BLUE, C_FIX], ec="none")
bax.set_xlim(0, OUR_R * 1.32)
bax.set_ylim(-0.60, 1.60)
bax.set_yticks([1, 0])
bax.set_yticklabels(["USGLoc σ", "ACE inliers"], fontsize=7.5)
bax.set_xticks([])
bax.tick_params(axis="y", length=0, pad=1.5)
for sp in bax.spines.values():
    sp.set_visible(False)
bax.text(OUR_R * 0.975, 1, f"{OUR_TOP:.2f} → {OUR_BOT:.1f} cm", va="center",
         ha="right", fontsize=7, color="white")
bax.text(OUR_R * 1.035, 1, f"{OUR_R:.0f}×", va="center", ha="left", fontsize=9.5,
         weight="bold", color=C_BLUE)
bax.text(ACE_R * 1.12 + OUR_R * 0.022, 0, f"{ACE_R:.0f}×", va="center", ha="left", fontsize=9.5,
         weight="bold", color=C_FIX)
bax.text(ACE_R * 1.12 + OUR_R * 0.200, 0, f"{ACE_TOP:.2f} → {ACE_BOT:.1f} cm",
         va="center", ha="left", fontsize=7, color=C_SUB)

txt(LX, 0.140, f"accuracy is on par — 7-scene median {MED_OURS:.2f} cm vs. ACE "
               f"{MED_ACE:.2f} cm;", fontsize=7.0, color=C_SUB)
txt(LX, 0.048, "the calibrated covariance is what is new.", fontsize=7.0, color=C_SUB)

# ================================================== RIGHT: the measured proof
RX0, RW = 3.00, 4.08
TH_W, TH_H = 0.78, 0.585
TH_Y = FH - 0.045 - TH_H
AX_Y0 = 0.525
AX_H = TH_Y - 0.075 - AX_Y0

ax = ax_in(RX0, AX_Y0, RW, AX_H)
ax.set_yscale("log")
ax.set_xlim(F.min() - 8, F.max() + 8)
ax.set_ylim(0.055, 1500)

ax.fill_between(F, 0.055, S, color=C_FILL, lw=0, zorder=1.5)
ax.plot(F, S, color=C_EDGE, lw=0.6, zorder=2.0)
ax.axhline(RMS, color=C_FIX, lw=0.9, ls=(0, (4.5, 2.2)), zorder=2.4)
ax.plot(F, E, color=C_ERR, lw=1.15, zorder=3, solid_joinstyle="round",
        path_effects=[pe.Stroke(linewidth=2.1, foreground="white"), pe.Normal()])

ax.set_xlabel(f"frame index   —   chess {SEQ}, {len(F)} consecutive queries",
              fontsize=7.5, labelpad=1.5)
ax.set_ylabel("error  /  reported 1σ   (cm)", fontsize=7.5, labelpad=3.0)
ax.set_yticks([0.1, 1, 10, 100, 1000])
ax.set_yticklabels(["0.1", "1", "10", "100", "1000"], fontsize=7.5)
ax.tick_params(axis="x", labelsize=7.5, pad=1.5)
ax.grid(axis="y", color="#EBEBEB", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# direct labels instead of a legend box
ax.annotate("reported 1σ  (calibrated Σ)", xy=(148, 14.7), xytext=(84, 330),
            fontsize=7.5, color=C_BLUE, ha="left", va="center", zorder=6,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C_BLUE, shrinkA=2, shrinkB=2))
ax.annotate("actual error", xy=(228, 0.74), xytext=(248, 0.093),
            fontsize=7.5, color=C_ERR, ha="left", va="center", zorder=6,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C_ERR, shrinkA=2, shrinkB=2))
ax.text(24, 2.15, f"fixed noise {RMS:.1f} cm", fontsize=7.5, color=C_FIX,
        ha="left", va="center", zorder=6)


def fmt(v):
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


for fr, idx, desc in ((F_EASY, i_easy, "texture-rich"),
                      (F_HARD, i_hard, "near-blank wall")):
    ax.plot([fr, fr], [S[idx] * 1.4, 1500], color="#A7AFB6", lw=0.5, ls=(0, (1.6, 1.6)),
            zorder=2.2)
    ax.plot([fr], [S[idx]], marker="o", ms=3.2, mfc="white", mec=C_BLUE, mew=0.9,
            zorder=6)
    xf = ax.transData.transform((fr, 1))[0] / fig.dpi            # frame -> inches
    axt = ax_in(xf - TH_W / 2, TH_Y, TH_W, TH_H)
    imshow_png(axt, f"data/7scenes_source/chess/{SEQ}/frame-{fr:06d}.color.png",
               desc, f"err {fmt(E[idx])} · σ {fmt(S[idx])} cm")
    fig.add_artist(plt.Line2D([xf / FW, xf / FW], [TH_Y / FH, (AX_Y0 + AX_H) / FH],
                              color="#A7AFB6", lw=0.5, ls=(0, (1.6, 1.6))))

fig.text((RX0 + RW / 2) / FW, 0.158 / FH,
         f"σ spans {S_P5:.2f}–{S_P95:.0f} cm (p5–p95, ×{S_P95/S_P5:.0f}); "
         f"{COV1*100:.0f}% of frames lie inside 1σ, {COV2*100:.0f}% inside 2σ.",
         fontsize=7, color=C_SUB, ha="center", va="baseline")
fig.text((RX0 + RW / 2) / FW, 0.055 / FH,
         f"Without Σ, one fixed noise is ≥3× too large on {TOO_LOOSE*100:.0f}% of "
         f"frames, ≥3× too small on {TOO_TIGHT*100:.0f}%.",
         fontsize=7, color=C_SUB, ha="center", va="baseline")

fig.add_artist(plt.Line2D([2.42 / FW, 2.42 / FW], [0.025 / FH, (FH - 0.03) / FH],
                          color="#DEDEDE", lw=0.7))

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{NAME}.pdf")
print("wrote", OUT / f"{NAME}.pdf")
