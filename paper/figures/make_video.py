#!/usr/bin/env python3
"""Supplementary video: the pose posterior over a continuous test sequence.

Each frame shows the query, our render at the predicted pose, ACE's render at
ITS predicted pose, and the render at ground truth, above running traces of
both methods' actual error and of our reported 1-sigma uncertainty. ACE is the
Table I baseline with per-frame poses on a continuous sequence; Marepo has only
the 100-frame subsample, so it cannot appear here without a new run.

The point is the relationship between our two traces: when our render drifts
off the query, sigma_t rises with it. ACE has no comparable trace to draw --
it reports no covariance, which is the gap the paper is about.

    python3 make_video.py stairs            # writes paper/figures/video_stairs.mp4
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.rendering import GsplatRenderer                  # noqa: E402

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
scene = sys.argv[1] if len(sys.argv) > 1 else "stairs"
out = f"paper/figures/video_{scene}.mp4"

d = json.load(open(f"results/traj_{scene}/7scenes_{scene}.json"))
tau = d["tau"]
ace = {(r["seq"], r["frame_id"]): r
       for r in json.load(open(f"results/ace_traj/7scenes_{scene}.json"))["records"]
       if "T" in r}
recs = [r for r in d["records"] if "T" in r and (r["seq"], r["frame_id"]) in ace]
recs.sort(key=lambda r: (r["seq"], r["frame_id"]))
err = np.array([100 * r["t_err"] for r in recs])
sig = np.array([100 * r["sigma_t"] * tau ** 0.5 for r in recs])
err_a = np.array([100 * ace[(r["seq"], r["frame_id"])]["t_err"] for r in recs])
print(f"{scene}: {len(recs)} frames, error {err.min():.2f}-{err.max():.2f} cm")

ds = SevenScenes("data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
idx = {(ds[i].seq, ds[i].frame_id): i for i in range(len(ds))}
emap = load_gaussian_map(f"maps/{scene}_sel.pt")
for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
    setattr(emap, f, getattr(emap, f).to("cuda"))
rend, cam = GsplatRenderer(), ds.scene_camera(640, 480)

W = imageio.get_writer(out, fps=10, codec="libx264", quality=8,
                       macro_block_size=8, ffmpeg_log_level="error")
for i, r in enumerate(recs):
    img = ds.load_image(idx[(r["seq"], r["frame_id"])])
    if img.shape[0] != 480:
        img = torch.nn.functional.interpolate(img.permute(2,0,1)[None], size=(480,640),
                 mode="bilinear", align_corners=False)[0].permute(1,2,0)
    with torch.no_grad():
        a = ace[(r["seq"], r["frame_id"])]
        pan = [img.numpy()]
        for T_ in (r["T"], a["T"], r["T_gt"]):
            T = torch.tensor(T_, dtype=torch.float32, device="cuda")
            pan.append(rend.render(emap.means, emap.covs, emap.colors, emap.opacities,
                                   T, cam, vacuity=emap.gaussian_vacuity)["rgb"]
                       .clamp(0,1).cpu().numpy())
    fig = plt.figure(figsize=(12.8, 4.6), dpi=100)
    gs = fig.add_gridspec(2, 4, height_ratios=[2.6, 1.05], hspace=0.30, wspace=0.03,
                          left=0.045, right=0.988, top=0.90, bottom=0.13)
    for j, (p, t) in enumerate(zip(pan, ["query", "USGLoc (ours)", "ACE", "ground truth"])):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(p); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(err[:i+1], color="#c0504d", lw=1.3, label="USGLoc error (cm)")
    ax.plot(sig[:i+1], color="#1f4e79", lw=1.3, ls="--",
            label=r"USGLoc reported $1\sigma$ (cm)")
    ax.plot(err_a[:i+1], color="#7f7f7f", lw=1.2, ls=":",
            label="ACE error (cm) -- no uncertainty reported")
    ax.axvline(i, color="k", lw=0.6, alpha=0.4)
    ax.set_xlim(0, len(recs)-1); ax.set_yscale("log")
    ax.set_ylim(max(1e-2, min(err.min(), sig.min(), err_a.min())*0.6),
                max(err.max(), sig.max(), err_a.max())*1.6)
    ax.set_xlabel("test frame"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.grid(alpha=0.25); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"{scene}  |  frame {i+1}/{len(recs)}    "
                 f"USGLoc {err[i]:.2f} cm (reported $1\\sigma$ {sig[i]:.2f} cm)    "
                 f"ACE {err_a[i]:.2f} cm (no covariance)", fontsize=10)
    fig.canvas.draw()
    W.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    if (i+1) % 25 == 0: print(f"  {i+1}/{len(recs)}")
W.close()
print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
