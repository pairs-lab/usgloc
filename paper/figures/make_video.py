#!/usr/bin/env python3
"""Supplementary video: per-method wipe strips + third-person 3DGS comparison.

Each frame (1280x720):
  top    three wipe strips, one per method (USGLoc, ACE, DSAC*): the query photo and
         the render at THAT method's pose, split by one line that sweeps back and forth
         in all three at once -- an accurate pose makes the seam vanish;
  bottom left   third-person view of the 3DGS map (slow orbit) with the trails and
         current frusta of ground truth and every method; inset: the estimates around
         the true camera centre at cm scale, with our tempered 2-sigma ellipse;
  bottom right  translation error of every method and our reported 1-sigma.
ACE: authors' models + dsacstar on the same frames (results/ace_traj). DSAC*: the
authors' published SfM-pGT estimates. Marepo has no continuous-sequence run.

Clips are our best cases, chosen by a disclosed rule: the whole Stairs sequence (we beat both
baselines on it) and, per other sequence, the 50-frame window that maximizes
min(median ACE, median DSAC*) / median USGLoc (logs: video_windows.txt). Chess's best window is
a tie (0.66 vs 0.70 cm), so it is left out.
    python3 paper/figures/make_video.py stairs redkitchen:91:141    # -> video_usgloc.mp4
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.rendering import GsplatRenderer, Camera          # noqa: E402

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
DSAC = "data/visloc_pseudo_gt_limitations/estimates/sfm/7scenes"
SCENES = sys.argv[1:] or ["stairs", "redkitchen:91:141"]
OUT = "paper/figures/video_usgloc.mp4"
C = dict(ours="#1B9E77", ace="#7570B3", dsac="#A6761D", gt="white")
LAB = dict(ours="USGLoc (ours)", ace="ACE", dsac="DSAC*")
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Liberation Serif", "STIXGeneral"],
                     "mathtext.fontset": "stix"})
W, H, FPS, PERIOD = 640, 480, 10, 50          # wipe: one sweep cycle every 5 s


def centre(T):
    T = np.asarray(T, float); return -T[:3, :3].T @ T[:3, 3]


def look_at(eye, tgt, up):
    f = tgt - eye; f /= np.linalg.norm(f)
    r = np.cross(up, f); r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), f])


def proj(P, cam):
    pc = (np.atleast_2d(P) - cam["eye"]) @ cam["R"].T
    z = np.maximum(pc[:, 2], 1e-6)
    return cam["fx"] * pc[:, 0] / z + W / 2, cam["fx"] * pc[:, 1] / z + H / 2


def frustum(T, d=0.20):
    """Camera-centre + 4 image-corner rays at depth d, in world coordinates."""
    T = np.asarray(T, float); R, t = T[:3, :3], T[:3, 3]
    k = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]) / 525.0
    pc = np.c_[k * d, np.full(4, d)]
    return np.vstack([centre(T), (pc - t) @ R])          # R^T (pc - t)


def dsac_poses(s):
    P = {}
    for line in open(f"{DSAC}/{s}_dsac_rgb.txt"):
        p = line.split(); q = [float(x) for x in p[1:5]]
        T = np.eye(4); T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = [float(x) for x in p[5:8]]; P[p[0]] = T
    return P


def clip(spec, writer):
    scene, *win = spec.split(":")
    d = json.load(open(f"results/traj_{scene}/7scenes_{scene}.json")); tau = d["tau"]
    ace = {(r["seq"], r["frame_id"]): r for r in
           json.load(open(f"results/ace_traj/7scenes_{scene}.json"))["records"] if "T" in r}
    dsac = dsac_poses(scene)
    recs = sorted([r for r in d["records"] if "T" in r and (r["seq"], r["frame_id"]) in ace],
                  key=lambda r: (r["seq"], r["frame_id"]))
    if win:
        recs = recs[int(win[0]):int(win[1])]
    recs = recs[:int(os.environ.get("VIDEO_TEST", 10**6))]
    Tm = {"ours": [np.array(r["T"]) for r in recs],
          "ace": [np.array(ace[(r["seq"], r["frame_id"])]["T"]) for r in recs],
          "dsac": [dsac[f"{r['seq']}/frame-{r['frame_id']:06d}.color.png"] for r in recs]}
    Tg = [np.array(r["T_gt"]) for r in recs]
    cg = np.array([centre(T) for T in Tg])
    err = {k: np.array([100 * np.linalg.norm(centre(T) - c) for T, c in zip(v, cg)]) for k, v in Tm.items()}
    sig = np.array([100 * r["sigma_t"] * tau ** 0.5 for r in recs])
    print(f"{scene}: {len(recs)} frames | median error ours {np.median(err['ours']):.2f}"
          f" ACE {np.median(err['ace']):.2f} DSAC* {np.median(err['dsac']):.2f} cm", flush=True)

    ds = SevenScenes("data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
    idx = {(ds[i].seq, ds[i].frame_id): i for i in range(len(ds))}
    emap = load_gaussian_map(d["args"]["map"])
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to("cuda"))
    rend, qcam = GsplatRenderer(), ds.scene_camera(W, H)
    mu = emap.means.detach().cpu().numpy(); op = emap.opacities.detach().cpu().numpy()
    UP = np.array([0., -1., 0.])                          # world up is -y in the SfM frame
    lo, hi = cg.min(0) - 1.5, cg.max(0) + 1.5
    top = (cg @ UP).max() + 0.4                            # drop the ceiling: nothing above the cameras
    keep = torch.from_numpy(np.all(mu > lo, 1) & np.all(mu < hi, 1) & (mu @ UP < top) & (op > 0.12)).cuda()
    tgt = cg.mean(0); span = np.linalg.norm(cg.max(0) - cg.min(0))
    fw = np.mean([T[2, :3] for T in Tg], 0); fw -= np.dot(fw, UP) * UP; fw /= np.linalg.norm(fw)

    for i, r in enumerate(recs):
        img = ds.load_image(idx[(r["seq"], r["frame_id"])])
        if img.shape[0] != H:
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(H, W),
                                                  mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        q = img.numpy()
        # tilted top-down orbit (60 deg) from behind the sequence's mean viewing direction
        a = np.radians(20 * np.sin(2 * np.pi * i / len(recs))); e = np.radians(60)
        f = np.cos(a) * fw + np.sin(a) * np.cross(UP, fw)
        eye = tgt + (0.8 * span + 1.4) * (np.sin(e) * UP - np.cos(e) * f)
        cam = dict(eye=eye, R=look_at(eye, tgt, f), fx=0.62 * W)
        Tv = np.eye(4); Tv[:3, :3] = cam["R"]; Tv[:3, 3] = -cam["R"] @ eye
        with torch.no_grad():
            ren = {k: rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                  torch.tensor(Tm[k][i], dtype=torch.float32, device="cuda"), qcam,
                                  vacuity=emap.gaussian_vacuity)["rgb"].clamp(0, 1).cpu().numpy()
                   for k in ("ours", "ace", "dsac")}
            mv = rend.render(emap.means[keep], emap.covs[keep], emap.colors[keep], emap.opacities[keep],
                             torch.tensor(Tv, dtype=torch.float32, device="cuda"),
                             Camera(fx=cam["fx"], fy=cam["fx"], cx=W / 2, cy=H / 2, width=W, height=H)
                             )["rgb"].clamp(0, 1).cpu().numpy()

        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        # ---- top: one wipe strip per method, one shared sweeping line -----------
        xl = int(W * (0.5 + 0.40 * np.sin(2 * np.pi * i / PERIOD)))
        for c, k in enumerate(("ours", "ace", "dsac")):
            ax = fig.add_axes([c / 3, 400 / 720, 1 / 3, 320 / 720]); ax.set_axis_off()
            wipe = ren[k].copy(); wipe[:, :xl] = q[:, :xl]
            ax.imshow(wipe); ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.axvline(xl, color="white", lw=1.8)
            ax.plot(xl, H / 2, "o", ms=10, mfc="white", mec="0.2", mew=1)
            ax.text(xl, H / 2, "◂▸", ha="center", va="center", fontsize=6.5, color="0.2", family="DejaVu Sans")
            for x, ha, t in ((10, "left", "query"), (W - 10, "right", "render")):
                ax.text(x, 12, t, ha=ha, va="top", fontsize=10, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
            txt = f"{LAB[k]}  {err[k][i]:.2f} cm"
            if k == "ours":
                txt += "   " + r"$1\sigma$" + f" {sig[i]:.2f} cm"
            ax.text(10, H - 12, txt, ha="left", va="bottom", fontsize=11, color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc=C[k], alpha=0.92, lw=0))
        # ---- bottom left: third-person map view ---------------------------------
        bx = fig.add_axes([0, 0, 533 / 1280, 400 / 720]); bx.set_axis_off()
        bx.imshow(mv); bx.set_xlim(0, W); bx.set_ylim(H, 0)
        u, v = proj(cg, cam); bx.plot(u, v, "-", color="white", lw=0.8, alpha=0.45)
        for k in ("dsac", "ace", "ours"):
            cs = np.array([centre(T) for T in Tm[k][:i + 1]])
            u, v = proj(cs, cam); bx.plot(u, v, "-", color=C[k], lw=1.2, alpha=0.9)
        for k, T in (("gt", Tg[i]), ("dsac", Tm["dsac"][i]), ("ace", Tm["ace"][i]), ("ours", Tm["ours"][i])):
            u, v = proj(frustum(T), cam)
            for j in range(1, 5):
                bx.plot([u[0], u[j]], [v[0], v[j]], color=C[k], lw=1.3 if k != "gt" else 1.6)
            bx.plot(np.r_[u[1:], u[1]], np.r_[v[1:], v[1]], color=C[k], lw=1.3 if k != "gt" else 1.6)
        bx.text(10, 14, f"{scene}  |  {r['seq']}  frame {i + 1}/{len(recs)}", ha="left", va="top",
                fontsize=11, color="white", bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, lw=0))
        # inset: estimates around the true centre, in the true camera frame (x right, y up), cm
        ix = fig.add_axes([(533 - 152) / 1280, 8 / 720, 144 / 1280, 144 / 720])
        ix.set_facecolor((0, 0, 0, 0.6))
        Rg = Tg[i][:3, :3]; L = 3.0
        # centre displacement in the camera frame is -rho, so its covariance is Sigma_rho,rho
        ev, evec = np.linalg.eigh(1e4 * tau * np.array(r["Sigma"])[:2, :2])   # tempered, cm^2
        ix.plot(0, 0, "+", color="white", ms=10, mew=1.6, zorder=6)
        for k, mk in (("dsac", "D"), ("ace", "s"), ("ours", "^")):
            dlt = 100 * Rg @ (centre(Tm[k][i]) - cg[i]); x, y = dlt[0], -dlt[1]   # y up
            s = max(abs(x), abs(y)) / (0.92 * L)
            if k == "ours" and s <= 1:                          # our posterior, centred on our estimate
                ix.add_patch(Ellipse((x, y), 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]),
                                     angle=-np.degrees(np.arctan2(evec[1, 1], evec[0, 1])),
                                     fc=C["ours"], alpha=0.25, ec=C["ours"], lw=1.0))
            if s > 1: x, y = x / s, y / s                       # off-scale: pin to the border
            ix.plot(x, y, mk, color=C[k], ms=6, mec="white", mew=0.6, alpha=1.0 if s <= 1 else 0.55,
                    zorder=7)
        ix.set_xlim(-L, L); ix.set_ylim(-L, L); ix.set_xticks([]); ix.set_yticks([])
        for sp in ix.spines.values(): sp.set_color("white"); sp.set_linewidth(0.6)
        ix.plot([-L + 0.3, -L + 1.3], [-L + 0.35] * 2, color="white", lw=1.5)
        ix.text(-L + 0.8, -L + 0.5, "1 cm", color="white", fontsize=8, ha="center", va="bottom")
        ix.text(0, L - 0.2, r"+ truth, USGLoc 2$\sigma$", color="white", fontsize=8, ha="center", va="top")
        # ---- bottom: error traces --------------------------------------------
        cx = fig.add_axes([(533 + 70) / 1280, 0.085, (1280 - 533 - 90) / 1280, 0.39])
        n = len(recs); xs = np.arange(n)
        for k in ("dsac", "ace", "ours"):
            cx.semilogy(xs[:i + 1], np.maximum(err[k][:i + 1], 1e-2), color=C[k], lw=1.6 if k == "ours" else 1.2,
                        label=f"{LAB[k]} error  {err[k][i]:.2f} cm")
        cx.semilogy(xs[:i + 1], sig[:i + 1], "--", color=C["ours"], lw=1.2, label=r"USGLoc reported $1\sigma$")
        cx.axvline(i, color="k", lw=0.7, alpha=0.4)
        allv = np.r_[err["ours"], err["ace"], err["dsac"], sig]
        cx.set_xlim(0, n - 1); cx.set_ylim(max(1e-2, allv.min() * 0.6), allv.max() * 1.8)
        cx.set_ylabel("translation (cm)", fontsize=11); cx.set_xlabel("test frame", fontsize=11, labelpad=1)
        cx.tick_params(labelsize=10); cx.grid(alpha=0.25); cx.spines[["top", "right"]].set_visible(False)
        cx.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02))
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(recs)}", flush=True)
    del emap; torch.cuda.empty_cache()


def main():
    out = os.environ.get("VIDEO_OUT", OUT)
    wr = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8,
                            ffmpeg_log_level="error")
    for s in SCENES:
        clip(s, wr)
    wr.close()
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
