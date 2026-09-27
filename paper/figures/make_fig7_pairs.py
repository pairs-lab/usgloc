#!/usr/bin/env python3
"""Qualitative paired comparison with ACE on identical queries.

For queries where the two methods disagree most, render the map at our pose,
at ACE's pose and at ground truth, next to the query. Our reported sigma_t
is printed on each panel: the point is that where we fail, we say so.

    python3 make_fig7_pairs.py            # writes fig7_pairs.pdf + fig7/*.png
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from usgloc.datasets import SevenScenes, load_gaussian_map  # noqa: E402
from usgloc.rendering import GsplatRenderer                # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
# the sequence-mode runs are the main-configuration runs that saved poses;
# ACE was run on their exact frames (scratchpad run_ace_frames.py)
SC = ["chess", "redkitchen", "stairs"]
OURS, ACE = "results/traj_{s}/7scenes_{s}.json", "results/ace_traj/7scenes_{s}.json"
OUT = Path("paper/figures/fig7")


def pick():
    """Two queries where we beat ACE by the most, two where ACE beats us."""
    rows = []
    for s in SC:
        o = json.load(open(OURS.format(s=s)))
        a = json.load(open(ACE.format(s=s)))
        tau = o["tau"]
        A = {(r["seq"], r["frame_id"]): r for r in a["records"]}
        for r in o["records"]:
            k = (r["seq"], r["frame_id"])
            if k in A and "T" in A[k]:
                rows.append(dict(scene=s, seq=k[0], frame=k[1], ours=100 * r["t_err"],
                                 ace=100 * A[k]["t_err"], sig=100 * r["sigma_t"] * tau ** 0.5,
                                 T_ours=r["T"], T_ace=A[k]["T"], T_gt=r["T_gt"]))
    # sigma percentile within the scene's run: the announcement signal
    for s in SC:
        sig = np.array([r["sig"] for r in rows if r["scene"] == s])
        for r in rows:
            if r["scene"] == s:
                r["pct"] = 100 * float(np.mean(sig <= r["sig"]))
    # Fixed, disclosed selection (largest disagreements, both directions):
    # a confident correct answer where ACE lands on the wrong tread; a
    # map-coverage failure and a wrong-registration failure that our sigma
    # announces; and one silent failure, which the limitations name.
    want = [("stairs", "seq-01", 320), ("chess", "seq-03", 144),
            ("redkitchen", "seq-03", 108), ("stairs", "seq-01", 36)]
    by = {(r["scene"], r["seq"], r["frame"]): r for r in rows}
    return [by[k] for k in want]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rend = GsplatRenderer()
    sel = pick()
    panels = []
    for r in sel:
        s = r["scene"]
        ds = SevenScenes("data/7scenes_source", s, split="test", sfm_pose_dir=PGT)
        idx = {(ds[i].seq, ds[i].frame_id): i for i in range(len(ds))}[(r["seq"], r["frame"])]
        emap = load_gaussian_map(f"maps/{s}_sel.pt")
        for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
            setattr(emap, f, getattr(emap, f).to(dev))
        cam = ds.scene_camera(640, 480)
        img = ds.load_image(idx)
        if img.shape[0] != 480:
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(480, 640),
                                                  mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        outs = {"query": img.numpy()}
        with torch.no_grad():
            for key in ("T_ours", "T_ace", "T_gt"):
                T = torch.tensor(r[key], dtype=torch.float32, device=dev)
                o = rend.render(emap.means, emap.covs, emap.colors, emap.opacities, T, cam,
                                vacuity=emap.gaussian_vacuity)
                outs[key] = o["rgb"].clamp(0, 1).cpu().numpy()
        panels.append((r, outs))
        print(f"{s:11s} {r['seq']}/{r['frame']:06d}  ours {r['ours']:6.2f} cm (sigma {r['sig']:5.2f})  ACE {r['ace']:6.2f} cm")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    cols = [("query", "query"), ("T_ours", "render at our pose"), ("T_ace", "render at ACE pose"), ("T_gt", "render at GT")]
    fig, axes = plt.subplots(len(panels), len(cols), figsize=(7.0, 1.45 * len(panels)))
    for i, (r, outs) in enumerate(panels):
        for j, (key, title) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(outs[key]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            Image.fromarray((outs[key] * 255).astype("uint8")).save(OUT / f"{i}_{key}.png")
            if i == 0:
                ax.set_title(title, fontsize=8)
        axes[i, 0].set_ylabel(f"{r['scene']}", fontsize=8)
        axes[i, 1].set_xlabel(f"{r['ours']:.2f} cm, $\\sigma_t$ at {r['pct']:.0f}th pct.", fontsize=7)
        axes[i, 2].set_xlabel(f"{r['ace']:.2f} cm (no covariance)", fontsize=7)
    fig.subplots_adjust(wspace=0.02, hspace=0.32, left=0.03, right=0.99, top=0.94, bottom=0.05)
    fig.savefig("paper/figures/fig7_pairs.pdf", dpi=200)
    json.dump([{k: v for k, v in r.items() if not k.startswith("T_")} for r, _ in panels],
              open(OUT / "selection.json", "w"), indent=1)
    print("saved paper/figures/fig7_pairs.pdf")


if __name__ == "__main__":
    main()
