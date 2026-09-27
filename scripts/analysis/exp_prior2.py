#!/usr/bin/env python3
"""USGLoc in the PRIOR-REFINEMENT setting (the GS-CPR / UGS-Loc / STDLoc setting).

Those methods do not localize globally: they refine a scene-coordinate prior
(typically ACE) against a 3DGS map. This script runs OUR evidential matching
correction from the ACE poses we already computed, on the identical frames,
and reports the before/after. It isolates the correction's contribution from
our retrieval front end.

    python3 exp_prior2.py <scene> <map_suffix> <prior_dir> <out_dir>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/thanhnc/gauss_ws/src/usgloc")
os.chdir("/home/thanhnc/gauss_ws/src/usgloc")

from usgloc.datasets import SevenScenes, load_gaussian_map   # noqa: E402
from usgloc.rendering import GsplatRenderer                  # noqa: E402
from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SP = "/tmp/claude-1005/-home-thanhnc-gauss-ws-src/887d461d-31ca-4e38-98ab-35cd2cbd7c14/scratchpad"


def errs(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    c = ((d[0, 0] + d[1, 1] + d[2, 2] - 1) / 2).clamp(-1, 1)
    return float(d[:3, 3].norm()) * 100, float(torch.rad2deg(torch.arccos(c)))


def main():
    scene = sys.argv[1]
    suf = sys.argv[2]
    prior_dir, out_dir = sys.argv[3], sys.argv[4]
    # "sel" keeps the historical <scene>_sel.pt; "final" uses the map the
    # train-side criterion selected per scene (fine-tuned where it won).
    if suf == "final":
        sel = {l.split()[0]: l.split()[1] for l in open(f"{SP}/final_maps.txt")}

    dev = "cuda"
    mp = sel[scene] if suf == "final" else f"maps/{scene}{suf}.pt"
    if not Path(mp).is_file():
        print(f"SKIP {scene}: {mp} missing"); return

    ds = SevenScenes("data/7scenes_source", scene, split="test", sfm_pose_dir=PGT)
    emap = load_gaussian_map(mp)
    for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
        setattr(emap, f, getattr(emap, f).to(dev))
    cam = ds.scene_camera(640, 480)
    corr = EvidentialMatchCorrector(emap, GsplatRenderer(), cam, rounds=5)

    ace = json.load(open(f"{prior_dir}/7scenes_{scene}.json"))
    by_key = {(r["seq"], r["frame_id"]): r for r in ace["records"]}
    index = {(ds[i].seq, ds[i].frame_id): i for i in range(len(ds))}

    out, a_t, o_t, a_r, o_r = [], [], [], [], []
    for k, rec in by_key.items():
        if rec.get("failed") or "T" not in rec or k not in index:
            continue
        i = index[k]
        gray = _to_gray(ds.load_image(i).to(dev))
        T_gt = ds[i].pose_cw.to(dev, torch.float32)
        T0 = torch.tensor(rec["T"], dtype=torch.float32, device=dev)
        res = corr.correct(gray, T0)
        e0 = errs(T0, T_gt)
        e1 = errs(res.T, T_gt) if res.success else e0
        a_t.append(e0[0]); o_t.append(e1[0]); a_r.append(e0[1]); o_r.append(e1[1])
        out.append({"seq": k[0], "frame_id": k[1],
                    "t_err": e1[0] / 100, "r_err": e1[1],
                    "t_err_prior": e0[0] / 100, "r_err_prior": e0[1],
                    "snapped": bool(res.success), "inliers": res.inliers})

    a_t, o_t = np.array(a_t), np.array(o_t)
    a_r, o_r = np.array(a_r), np.array(o_r)
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    json.dump({"scene": scene, "map": mp, "prior": prior_dir, "records": out},
              open(d / f"7scenes_{scene}.json", "w"))
    imp = 100 * (1 - np.median(o_t) / np.median(a_t))
    print(f"{scene:11s} n={len(o_t):3d} | {prior_dir.split(chr(47))[-1]} prior {np.median(a_t):6.2f}cm/{np.median(a_r):5.2f} "
          f"-> +USGLoc corr {np.median(o_t):6.2f}cm/{np.median(o_r):5.2f}  ({imp:+.0f}%) | "
          f"snap {100*np.mean([r['snapped'] for r in out]):3.0f}% | "
          f"SR@2/2 {100*np.mean((o_t<=2)&(o_r<=2)):3.0f}% SR@1/1 {100*np.mean((o_t<=1)&(o_r<=1)):3.0f}%")


if __name__ == "__main__":
    main()
