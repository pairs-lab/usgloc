#!/usr/bin/env python3
"""Evaluate the pretrained ACE baseline on OUR test frames and OUR protocol.

Uses the official ACE encoder + the released per-scene pgt heads (trained on
the same Brachmann SfM pseudo-GT we evaluate against). The dsacstar C++ RANSAC
is replaced by cv2.solvePnPRansac on the same 2D-3D correspondences, which
avoids building the extension; the substitution is documented in the paper.

    python3 run_ace_baseline.py chess [n_queries]

Writes results/ace_baseline/7scenes_<scene>.json with one record per query,
keyed by (seq, frame_id) so it joins directly against our runs.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

ACE = Path("/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad/ace")
sys.path.insert(0, str(ACE))
sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from ace_network import Regressor                      # noqa: E402
from usgloc.datasets import SevenScenes                # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
MODELS = ACE.parent / "ace_models" / "7Scenes_pgt"
IMAGE_HEIGHT = 480          # ACE's test-time resolution
import os as _os
USE_DSACSTAR = _os.environ.get("USE_DSACSTAR", "1") == "1"


def load_net(scene, device):
    enc = torch.load(ACE / "ace_encoder_pretrained.pt", map_location="cpu",
                     weights_only=False)
    head = torch.load(MODELS / f"pgt_7scenes_{scene}.pt", map_location="cpu",
                      weights_only=False)
    net = Regressor.create_from_split_state_dict(enc, head)
    return net.to(device).eval()


def preprocess(rgb_hw3, device):
    """ACE test transform: grayscale, resize to height 480, normalize."""
    x = torch.from_numpy(rgb_hw3).permute(2, 0, 1)            # (3,H,W) in [0,1]
    h, w = x.shape[1], x.shape[2]
    if h != IMAGE_HEIGHT:
        new_w = int(round(w * IMAGE_HEIGHT / h))
        x = TF.resize(x, [IMAGE_HEIGHT, new_w], antialias=True)
    x = TF.rgb_to_grayscale(x)
    x = TF.normalize(x, mean=[0.4], std=[0.25])
    return x[None].to(device), x.shape[1] / h    # scale applied to the image


@torch.no_grad()
def solve_pose_dsacstar(net, img_b1hw, focal, ppx, ppy, sub=8):
    """Official ACE pose solver (dsacstar RANSAC). Returns T_cw."""
    import dsacstar
    with torch.autocast("cuda", enabled=True):
        sc = net(img_b1hw)
    sc = sc.float().cpu()
    out_pose = torch.zeros((4, 4))
    inl = dsacstar.forward_rgb(
        sc, out_pose, 64, 10, float(focal), float(ppx), float(ppy),
        100, 100, sub)
    # dsacstar returns camera-TO-world; our convention is camera-from-world
    T = torch.linalg.inv(out_pose.double()).numpy()
    return T, int(inl)


@torch.no_grad()
def solve_pose(net, img_b1hw, K, sub=8):
    """Scene coordinates -> pose (camera-from-world) via PnP-RANSAC."""
    with torch.autocast("cuda", enabled=True):
        sc = net(img_b1hw)                                     # (1,3,h/8,w/8)
    sc = sc.float().cpu()[0]                                   # (3,h8,w8)
    _, h8, w8 = sc.shape
    ys, xs = torch.meshgrid(torch.arange(h8), torch.arange(w8), indexing="ij")
    # pixel centre of each subsampled cell, in the resized image frame
    px = (xs.flatten().numpy() * sub + sub / 2).astype(np.float64)
    py = (ys.flatten().numpy() * sub + sub / 2).astype(np.float64)
    obj = sc.reshape(3, -1).T.numpy().astype(np.float64)
    img = np.stack([px, py], axis=1)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, None, reprojectionError=10.0,
        iterationsCount=1000, confidence=0.9999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return None, 0
    inl = inl.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T, int(len(inl))


def main():
    scene = sys.argv[1]
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    dev = "cuda"
    ds = SevenScenes("data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
    net = load_net(scene, dev)
    # EXACTLY the subsampling rule of scripts/run_7scenes.py (line ~488)
    # so the two methods are evaluated on identical frames.
    n = len(ds) if n_q <= 0 else min(n_q, len(ds))
    idx = torch.linspace(0, len(ds) - 1, n).round().long().unique().tolist()

    recs = []
    for i in idx:
        fr = ds[i]
        rgb = ds.load_image(i).numpy()
        img_b, scale = preprocess(rgb, dev)
        cam = ds.scene_camera(rgb.shape[1], rgb.shape[0])
        K = np.array([[cam.fx * scale, 0, cam.cx * scale],
                      [0, cam.fy * scale, cam.cy * scale],
                      [0, 0, 1]], dtype=np.float64)
        if USE_DSACSTAR:
            T, n_inl = solve_pose_dsacstar(net, img_b, K[0, 0], K[0, 2], K[1, 2])
        else:
            T, n_inl = solve_pose(net, img_b, K)
        T_gt = fr.pose_cw.double().numpy()
        if T is None:
            t_err, r_err = float("nan"), float("nan")
        else:
            d = T @ np.linalg.inv(T_gt)
            t_err = float(np.linalg.norm(d[:3, 3]))
            c = np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)
            r_err = float(np.degrees(np.arccos(c)))
        recs.append({"seq": fr.seq, "frame_id": fr.frame_id,
                     "t_err": t_err, "r_err": r_err, "inliers": n_inl,
                     "failed": T is None})
        if T is not None:
            recs[-1]["T"] = T.tolist()

    good = [r for r in recs if not r["failed"]]
    t = np.array([100 * r["t_err"] for r in good])
    rr = np.array([r["r_err"] for r in good])
    out = Path("results/ace_baseline")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"scene": scene, "n": len(recs), "records": recs},
              open(out / f"7scenes_{scene}.json", "w"))
    print(f"ACE {scene:11s} n={len(recs)} failed={len(recs)-len(good)} | "
          f"med {np.median(t):6.2f}cm/{np.median(rr):5.2f}deg | "
          f"SR@5/5 {100*np.mean((t<=5)&(rr<=5)):5.1f}% | "
          f"SR@2/2 {100*np.mean((t<=2)&(rr<=2)):5.1f}%")


if __name__ == "__main__":
    main()
