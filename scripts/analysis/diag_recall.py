#!/usr/bin/env python3
"""Is the right answer reachable? Retrieval recall vs. selection failure.

For each test query we ask two separate questions:
  (a) RECALL  -- does the top-k shortlist contain an anchor close enough to
      ground truth for the correction to converge (its basin is ~60 cm)?
  (b) SELECTION -- if yes, did the pipeline actually pick it?
A scene that fails (a) needs better retrieval or denser anchors; a scene
that fails only (b) needs a better hypothesis-selection criterion, which is
far cheaper to fix. Stairs is the scene under suspicion.

    python3 diag_recall.py stairs [office ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from usgloc.datasets import SevenScenes  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def centres(poses):
    """Camera centres from camera-from-world poses."""
    return np.stack([(-p[:3, :3].T @ p[:3, 3]).numpy() for p in poses])


def main():
    for scene in sys.argv[1:] or ["stairs"]:
        tr = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
        te = SevenScenes("data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
        for stride in (20, 10, 5):
            anc = centres([tr[i].pose_cw for i in range(0, len(tr), stride)])
            idx = torch.linspace(0, len(te) - 1, 100).round().long().unique().tolist()
            q = centres([te[i].pose_cw for i in idx])
            d = np.linalg.norm(q[:, None, :] - anc[None, :, :], axis=-1) * 100  # cm
            nn = d.min(axis=1)
            print(f"{scene:<9} anchor stride {stride:2d} ({len(anc):4d} anchors): "
                  f"nearest-anchor distance med {np.median(nn):6.1f}cm "
                  f"p90 {np.percentile(nn, 90):6.1f} | "
                  f"queries with an anchor within 60cm: {100*(nn<=60).mean():3.0f}% "
                  f"| within 30cm: {100*(nn<=30).mean():3.0f}%")
        print()


if __name__ == "__main__":
    main()
