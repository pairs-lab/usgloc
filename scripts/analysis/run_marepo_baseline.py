#!/usr/bin/env python3
"""Evaluate the released Marepo (Chen et al., CVPR 2024) models on OUR test
frames and OUR protocol, saving per-frame poses.

Mirrors what marepo/test_marepo.py does internally (same CamLocDataset, same
Regressor.load_marepo_from_state_dict, same autocast pattern, same
load_scheme2_sc_map mean shift) but writes one JSON record per frame in the
schema of run_ace_baseline.py (seq, frame_id, t_err [m], r_err [deg],
T = 4x4 camera-from-world, failed).

    python3 run_marepo_baseline.py chess --regressor marepo_s --head ace_pgt --gt pgt

The whole test split is always evaluated; the 100-frame subset is picked with
EXACTLY the rule of run_ace_baseline.py / scripts/run_7scenes.py:
    torch.linspace(0, len(test_ds)-1, 100).round().long().unique()
over the SfM-pGT SevenScenes test dataset (unregistered frames dropped).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SP = Path("/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad")
MAREPO = SP / "marepo"
USGLOC = Path("/home/thanhnc/gauss_ws/src/usgloc")
sys.path.insert(0, str(SP / "pyshim"))      # pure-torch pytorch3d.transforms shim
sys.path.insert(0, str(MAREPO))
sys.path.insert(0, str(USGLOC))
os.chdir(MAREPO)                            # transformer json paths are relative

from marepo.marepo_network import Regressor      # noqa: E402
from dataset import CamLocDataset                # noqa: E402
from usgloc.datasets import SevenScenes          # noqa: E402

PGT = USGLOC / "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
DATA = {"pgt": SP / "ace_data", "dslam": SP / "ace_data_dslam"}
HEADS = {
    # heads shipped with marepo (pretrain.zip); byte-identical to ACE's dSLAM heads
    "marepo_dslam": lambda s: MAREPO / f"logs/pretrain/ace_models/7Scenes/7scenes_{s}.pt",
    # official ACE heads trained on the SfM pseudo-GT (ACE release, 7Scenes_pgt)
    "ace_pgt": lambda s: SP / "ace_models" / "7Scenes_pgt" / f"pgt_7scenes_{s}.pt",
}
REGRESSORS = {
    "marepo": lambda s: MAREPO / "logs/paper_model/marepo/marepo.pt",
    "marepo_s": lambda s: MAREPO / f"logs/paper_model/marepo_s_7scenes_{s}/marepo_s_7scenes_{s}.pt",
}
TRANSFORMER_JSON = MAREPO / "transformer/config/nerf_focal_12T1R_256_homo.json"
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


def load_network(scene, regressor, head, default_hw, device):
    enc = torch.load(MAREPO / "ace_encoder_pretrained.pt", map_location="cpu", weights_only=False)
    head_sd = torch.load(HEADS[head](scene), map_location="cpu", weights_only=False)
    tr_sd = torch.load(REGRESSORS[regressor](scene), map_location="cpu", weights_only=False)
    config = json.load(open(TRANSFORMER_JSON))
    config["transformer_pose_mean"] = torch.Tensor([0., 0., 0.])   # placeholder, as in test_marepo.py
    config["default_img_HW"] = list(default_hw)
    net = Regressor.load_marepo_from_state_dict(enc, head_sd, tr_sd, config)
    # --load_scheme2_sc_map True: shift by the ACE head's mean at test time
    net.transformer_head.transformer_pose_mean = net.heads.mean
    return net.to(device).eval()


def pose_errors(T_cw, T_gt_cw):
    """Same formula as run_ace_baseline.py (camera-from-world inputs)."""
    d = T_cw @ np.linalg.inv(T_gt_cw)
    t_err = float(np.linalg.norm(d[:3, 3]))
    c = np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)
    r_err = float(np.degrees(np.arccos(c)))
    return t_err, r_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=SCENES)
    ap.add_argument("--regressor", choices=list(REGRESSORS), default="marepo_s")
    ap.add_argument("--head", choices=list(HEADS), default="ace_pgt")
    ap.add_argument("--gt", choices=list(DATA), default="pgt",
                    help="which converted dataset (=which ground truth) to evaluate against")
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--tag", default="", help="suffix for output file names")
    opt = ap.parse_args()

    torch.manual_seed(2089); np.random.seed(2089); random.seed(2089)   # as test_marepo.py
    device = torch.device("cuda")

    test_root = DATA[opt.gt] / f"7scenes_{opt.scene}" / "test"
    ds = CamLocDataset(root_dir=test_root, mode=0, image_height=480, center_crop=False, load_rgb=False)
    stems = [Path(f).name.replace(".color.png", "") for f in ds.rgb_files]
    stem2idx = {s: i for i, s in enumerate(stems)}

    # Our query-frame rule, defined on the SfM-pGT SevenScenes test set.
    ref = SevenScenes(USGLOC / "data/7scenes_source", opt.scene, split="test", sfm_pose_dir=PGT)
    n = len(ref) if opt.n_queries <= 0 else min(opt.n_queries, len(ref))
    sel = torch.linspace(0, len(ref) - 1, n).round().long().unique().tolist()
    sel_stems = [f"{ref[i].seq}-frame-{ref[i].frame_id:06d}" for i in sel]
    missing = [s for s in sel_stems if s not in stem2idx]
    assert not missing, f"{len(missing)} selected frames absent from {test_root}: {missing[:3]}"

    dl = DataLoader(ds, shuffle=False, batch_size=opt.batch_size, num_workers=opt.workers)
    net = load_network(opt.scene, opt.regressor, opt.head, (ds.default_img_H, ds.default_img_W), device)
    print(f"[{opt.scene}] regressor={opt.regressor} head={opt.head} gt={opt.gt} "
          f"frames={len(ds)} sel={len(sel_stems)} img={ds.default_img_H}x{ds.default_img_W} "
          f"head.mean={net.heads.mean.flatten().tolist()}", flush=True)

    recs = {}
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device, non_blocking=True)
            K = batch["intrinsics"].to(device)
            with torch.autocast("cuda", enabled=True):
                feats = net.get_features(img)
                sc = net.get_scene_coordinates(feats).float()
            with torch.autocast("cuda", enabled=False):
                pred = net.get_pose(sc, K).float().cpu()          # [B,4,4] camera-to-world
            gt = batch["pose"]                                    # [B,4,4] camera-to-world
            for k, f in enumerate(batch["rgb_files"]):
                stem = Path(f).name.replace(".color.png", "")
                seq, fid = stem.split("-frame-")
                P = pred[k].double().numpy()
                ok = np.all(np.isfinite(P))
                if ok:
                    T_cw = np.linalg.inv(P)
                    T_gt = np.linalg.inv(gt[k].double().numpy())
                    t_err, r_err = pose_errors(T_cw, T_gt)
                else:
                    t_err, r_err = float("nan"), float("nan")
                rec = {"seq": seq, "frame_id": int(fid), "t_err": t_err, "r_err": r_err,
                       "failed": not bool(ok)}
                if ok:
                    rec["T"] = T_cw.tolist()
                recs[stem] = rec
    dt = time.time() - t0
    assert len(recs) == len(ds)

    config = {"method": "marepo", "regressor": opt.regressor,
              "regressor_file": str(REGRESSORS[opt.regressor](opt.scene)),
              "head": opt.head, "head_file": str(HEADS[opt.head](opt.scene)),
              "gt": opt.gt, "test_root": str(test_root),
              "transformer_json": str(TRANSFORMER_JSON), "load_scheme2_sc_map": True,
              "image_height": 480, "batch_size": opt.batch_size}

    def summary(rs, label):
        good = [r for r in rs if not r["failed"]]
        t = np.array([100 * r["t_err"] for r in good]); rr = np.array([r["r_err"] for r in good])
        print(f"MAREPO {opt.scene:11s} {label:5s} n={len(rs)} failed={len(rs)-len(good)} | "
              f"med {np.median(t):6.2f}cm/{np.median(rr):5.2f}deg | "
              f"SR@5/5 {100*np.mean((t<=5)&(rr<=5)):5.1f}% | SR@2/2 {100*np.mean((t<=2)&(rr<=2)):5.1f}%"
              + (f" | {1000*dt/len(rs):.1f} ms/frame incl. IO" if label == "full" else ""), flush=True)
        return {"n": len(rs), "failed": len(rs) - len(good),
                "median_t_cm": float(np.median(t)), "median_r_deg": float(np.median(rr)),
                "sr_5cm5deg": float(np.mean((t <= 5) & (rr <= 5))),
                "sr_2cm2deg": float(np.mean((t <= 2) & (rr <= 2)))}

    opt.out_dir.mkdir(parents=True, exist_ok=True)
    tag = opt.tag
    full = [recs[s] for s in stems]
    sub = [recs[s] for s in sel_stems]
    full_stats = summary(full, "full")
    sub_stats = summary(sub, f"n{len(sub)}")
    json.dump({"scene": opt.scene, "n": len(sub), "records": sub, "config": config,
               "stats": sub_stats, "full_split_stats": full_stats},
              open(opt.out_dir / f"7scenes_{opt.scene}{tag}.json", "w"))
    json.dump({"scene": opt.scene, "n": len(full), "records": full, "config": config,
               "stats": full_stats},
              open(opt.out_dir / f"7scenes_{opt.scene}{tag}.full.json", "w"))


if __name__ == "__main__":
    main()
