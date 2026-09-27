#!/usr/bin/env python3
"""ACE (official encoder + released 12Scenes pgt heads, dsacstar solver) on
OUR 12Scenes test frames, identical frame rule to scripts/run_7scenes.py.

    python3 run_ace_12scenes.py apt1/kitchen [n_queries]

Writes results/ace_dsacstar_12scenes_100/12scenes_<building>_<scene>.json
with one record per query, keyed by (seq, frame_id).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path("/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "ace"))
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from ace_network import Regressor                       # noqa: E402
from run_ace_baseline import preprocess, solve_pose_dsacstar  # noqa: E402
from usgloc.datasets import TwelveScenes                # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/12scenes"
MODELS = SP / "ace_models" / "12Scenes_pgt"


def main():
    scene = sys.argv[1]
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    f = scene.replace("/", "_")
    dev = "cuda"
    ds = TwelveScenes("data/12scenes_source", scene, split="test", sfm_pose_dir=PGT)
    enc = torch.load(SP / "ace" / "ace_encoder_pretrained.pt", map_location="cpu", weights_only=False)
    head = torch.load(MODELS / f"pgt_12scenes_{f}.pt", map_location="cpu", weights_only=False)
    net = Regressor.create_from_split_state_dict(enc, head).to(dev).eval()

    n = len(ds) if n_q <= 0 else min(n_q, len(ds))
    idx = torch.linspace(0, len(ds) - 1, n).round().long().unique().tolist()
    recs = []
    for i in idx:
        fr = ds[i]
        rgb = ds.load_image(i).numpy()
        img_b, scale = preprocess(rgb, dev)
        cam = ds.scene_camera(rgb.shape[1], rgb.shape[0])
        T, n_inl = solve_pose_dsacstar(net, img_b, cam.fx * scale, cam.cx * scale, cam.cy * scale)
        T_gt = fr.pose_cw.double().numpy()
        d = T @ np.linalg.inv(T_gt)
        t_err = float(np.linalg.norm(d[:3, 3]))
        c = np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)
        r_err = float(np.degrees(np.arccos(c)))
        recs.append({"seq": fr.seq, "frame_id": fr.frame_id, "t_err": t_err,
                     "r_err": r_err, "inliers": n_inl, "failed": False, "T": T.tolist()})
    t = np.array([100 * r["t_err"] for r in recs]); rr = np.array([r["r_err"] for r in recs])
    out = Path("results/ace_dsacstar_12scenes_100"); out.mkdir(parents=True, exist_ok=True)
    json.dump({"scene": scene, "n": len(recs), "records": recs}, open(out / f"12scenes_{f}.json", "w"))
    print(f"ACE {scene:18s} n={len(recs)} | med {np.median(t):6.2f}cm/{np.median(rr):5.2f}deg | "
          f"SR@2/2 {100*np.mean((t<=2)&(rr<=2)):5.1f}% | SR@5/5 {100*np.mean((t<=5)&(rr<=5)):5.1f}%")


if __name__ == "__main__":
    main()
