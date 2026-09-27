#!/usr/bin/env python3
"""How much of our remaining error is the MAP, and how much is the PIPELINE?

For every scene we measure, on the SAME test frames the benchmark reports:
  floor    -- correction started from ground truth: the best this map allows;
  achieved -- the reported end-to-end median.
The difference is the cost of imperfect initialization and hypothesis
selection, i.e. the part still addressable without touching the map. This
separates "we are map-limited" from "we are leaving accuracy on the table",
which the per-scene numbers alone cannot distinguish.

    python3 floor_vs_achieved.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map        # noqa: E402
from usgloc.rendering import GsplatRenderer                       # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
# ACE with its own solver, on the identical frames
ACE = {"chess": 0.56, "fire": 0.77, "heads": 0.51, "office": 1.10,
       "pumpkin": 1.03, "redkitchen": 0.75, "stairs": 2.64}


def terr(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    return float(d[:3, 3].norm()) * 100


def main():
    dev = "cuda"
    rend = GsplatRenderer()
    rows = []
    print(f"{'scene':<11}{'achieved':>9}{'floor':>8}{'gap':>7}{'reg':>6}"
          f"{'ACE':>7}   floor beats ACE?")
    for s in SCENES:
        ds = SevenScenes("data/7scenes_source", s, split="test", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, 100).round().long().unique().tolist()
        emap = load_gaussian_map(f"maps/{s}_sel.pt")
        for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
            setattr(emap, f, getattr(emap, f).to(dev))
        cam = ds.scene_camera(640, 480)
        corr = EvidentialMatchCorrector(emap, rend, cam, rounds=5)
        fl = []
        for i in idx:
            img = ds.load_image(i).to(dev)
            T = ds[i].pose_cw.to(dev, torch.float32)
            r = corr.correct(_to_gray(img), T)
            if r.success:
                fl.append(terr(r.T, T))
        floor = float(np.median(fl)) if fl else float("nan")
        reg = 100.0 * len(fl) / len(idx)
        j = json.load(open(f"results/ev_seed0/7scenes_{s}.json"))
        ach = float(np.median([100 * r["t_err"] for r in j["records"]]))
        rows.append((s, ach, floor, reg))
        print(f"{s:<11}{ach:9.2f}{floor:8.2f}{ach-floor:7.2f}{reg:5.0f}%"
              f"{ACE[s]:7.2f}   {'YES' if floor < ACE[s] else 'no'}")
    ach = np.array([r[1] for r in rows]); fl = np.array([r[2] for r in rows])
    print(f"\n{'AVERAGE':<11}{ach.mean():9.2f}{fl.mean():8.2f}{(ach-fl).mean():7.2f}"
          f"{'':6}{np.mean(list(ACE.values())):7.2f}")
    print(f"\nIf every scene reached its own map floor, the average would be "
          f"{fl.mean():.2f} cm (ACE: {np.mean(list(ACE.values())):.2f}).")
    print(f"Map-limited share of current error: {100*fl.mean()/ach.mean():.0f}%")


if __name__ == "__main__":
    main()
