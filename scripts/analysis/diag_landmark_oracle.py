#!/usr/bin/env python3
"""How good could landmark PnP be if the MATCHING were perfect?

The pose-free landmark stage (:mod:`usgloc.landmarks`) replaces the rendered
depth buffer with a map parameter: a correspondence's 3D point is a Gaussian
MEAN.  That removes the depth-buffer bias, but it introduces a different error
-- the Gaussian centre is not the surface point the query pixel actually sees,
it is the centre of a finite blob a few centimetres from its neighbours.

This diagnostic isolates that term.  For every SuperPoint keypoint of a train
frame it takes the ORACLE association: among the landmark Gaussians whose
centres project within ``--radius`` pixels of the keypoint at the GROUND-TRUTH
pose, the front-most one.  Then it solves the same PnP.

READ IT AS AN ASSOCIATION CEILING, NOT AS AN UNBIASED BOUND: the oracle selects
correspondences BY their reprojection at the answer, so it hands PnP a set whose
reprojection error at the ground truth is below ``--radius`` by construction, and
the pose it recovers is necessarily close to it.  What the number does say is
that such a set EXISTS and is large (how many keypoints have a landmark centre
within a couple of pixels), i.e. that the Gaussian-centre 3D points are not what
caps the mechanism at the pixel scale the real matcher operates at -- the
matching is.  ``ray_off_cm``, the perpendicular distance from the chosen
Gaussian centre to the keypoint's true ray, is the surface-quantisation scale
that the centres themselves carry.

No rendering and no GPU are needed (``--device cpu`` is the default), so this
runs without a GPU slot.

    python3 scripts/analysis/diag_landmark_oracle.py office chess \
        --map-suffix _r1 --feats-suffix _lfinit --frames 20 \
        --out results/landmark/oracle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc import locfeat                                     # noqa: E402
from usgloc.correction import _to_gray                          # noqa: E402
from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.superpoint import SuperPoint                        # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"


def pose_err(Ta, Tb):
    d = Ta.double() @ torch.linalg.inv(Tb.double())
    c = ((torch.trace(d[:3, :3]) - 1) / 2).clamp(-1, 1)
    return float(d[:3, 3].norm()) * 100, float(torch.rad2deg(torch.acos(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["office"])
    ap.add_argument("--map-suffix", default="_r1")
    ap.add_argument("--feats-suffix", default="_lfinit",
                    help="only used for its loc_feat_obs mask (which Gaussians "
                         "the field ever observed); no descriptor is matched here")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--radius", type=float, default=2.0,
                    help="a landmark counts as the oracle match when its centre "
                         "projects within this many pixels of the keypoint")
    ap.add_argument("--detect-num", type=int, default=2048)
    ap.add_argument("--nms", type=int, default=4)
    ap.add_argument("--reproj", type=float, default=12.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/landmark/oracle")
    args = ap.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    import poselib
    from scipy.spatial import cKDTree

    summaries = []
    for scene in args.scenes:
        ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
        cam = ds.scene_camera(640, 480)
        K = cam.K(dtype=torch.float64)
        emap = load_gaussian_map(f"maps/{scene}{args.map_suffix}.pt")
        if args.feats_suffix:
            locfeat.attach_loc_feats(emap, f"maps/{scene}{args.feats_suffix}.pt")
        obs = (emap.loc_feat_obs.float() if emap.loc_feat_obs is not None
               else torch.ones(emap.means.shape[0]))
        cand = torch.nonzero(obs > 0.5, as_tuple=False).squeeze(-1)
        xyz = emap.means[cand].double()
        enc = SuperPoint().to(dev).eval()
        pl_cam = {"model": "PINHOLE", "width": cam.width, "height": cam.height,
                  "params": [cam.fx, cam.fy, cam.cx, cam.cy]}
        idx = torch.linspace(0, len(ds) - 1, args.frames).round().long().unique().tolist()
        rows = []
        for i in idx:
            img = ds.load_image(i)
            T = ds[i].pose_cw.double()
            p = xyz @ T[:3, :3].T + T[:3, 3]
            z = p[:, 2]
            u = cam.fx * p[:, 0] / z.clamp_min(1e-6) + cam.cx - 0.5
            v = cam.fy * p[:, 1] / z.clamp_min(1e-6) + cam.cy - 0.5
            vis = ((z > 0.05) & (u >= -1) & (u < cam.width) & (v >= -1)
                   & (v < cam.height))
            j = torch.nonzero(vis, as_tuple=False).squeeze(-1)
            uv = torch.stack([u[j], v[j]], dim=-1).numpy()
            zz = z[j].numpy()
            _, scores = enc.dense_descriptors(_to_gray(img).to(dev))
            kp = locfeat.detect_keypoints(scores[0].cpu(), args.detect_num, args.nms)
            kpn = kp.double().numpy()
            tree = cKDTree(uv)
            # every landmark within radius; keep the front-most one
            near = tree.query_ball_point(kpn, r=args.radius, workers=-1)
            obj, im, off = [], [], []
            for q, candidates in enumerate(near):
                if not candidates:
                    continue
                c = np.asarray(candidates)
                pick = c[int(np.argmin(zz[c]))]
                g = int(j[pick])
                obj.append(xyz[g].numpy())
                im.append(kpn[q] + 0.5)
                # perpendicular distance from the Gaussian centre to the true ray
                # of that keypoint (the "quantisation" of the surface)
                d = np.array([(kpn[q][0] + 0.5 - cam.cx) / cam.fx,
                              (kpn[q][1] + 0.5 - cam.cy) / cam.fy, 1.0])
                d /= np.linalg.norm(d)
                pc = (xyz[g].numpy() @ T[:3, :3].numpy().T + T[:3, 3].numpy())
                off.append(float(np.linalg.norm(pc - d * float(pc @ d))))
            r = dict(frame=int(i), n_keypoints=int(kp.shape[0]), n_matches=len(obj),
                     ray_off_cm=float(np.median(off)) * 100 if off else float("nan"))
            if len(obj) >= 8:
                pose, info = poselib.estimate_absolute_pose(
                    np.ascontiguousarray(np.stack(im), dtype=np.float64),
                    np.ascontiguousarray(np.stack(obj), dtype=np.float64),
                    pl_cam, {"max_iterations": 100000, "min_iterations": 1000,
                             "max_reproj_error": args.reproj,
                             "success_prob": 0.99999},
                    {"verbose": False})
                Te = torch.eye(4, dtype=torch.float64)
                Te[:3, :3] = torch.as_tensor(np.asarray(pose.R, dtype=np.float64))
                Te[:3, 3] = torch.as_tensor(np.asarray(pose.t, dtype=np.float64))
                r["inliers"] = int(info["num_inliers"])
                r["t_cm"], r["r_deg"] = pose_err(Te, T)
            else:
                r["inliers"], r["t_cm"], r["r_deg"] = 0, None, None
            rows.append(r)
            print(f"    {scene:<8}f{i:<6} oracle matches {r['n_matches']:5d}/"
                  f"{r['n_keypoints']:5d}  ray offset {r['ray_off_cm']:5.2f} cm  "
                  f"inl {r['inliers']:5d}  err "
                  f"{'%.3f' % r['t_cm'] if r['t_cm'] is not None else ' -- '} cm",
                  flush=True)
        ok = [r for r in rows if r["t_cm"] is not None]
        s = dict(scene=scene, map=f"maps/{scene}{args.map_suffix}.pt",
                 radius=args.radius, reproj=args.reproj, n_frames=len(rows),
                 n_candidates=int(cand.numel()),
                 oracle_cm=float(np.median([r["t_cm"] for r in ok])) if ok else None,
                 oracle_deg=float(np.median([r["r_deg"] for r in ok])) if ok else None,
                 matches=float(np.median([r["n_matches"] for r in rows])),
                 inliers=float(np.median([r["inliers"] for r in rows])),
                 ray_off_cm=float(np.nanmedian([r["ray_off_cm"] for r in rows])),
                 frames=rows)
        summaries.append(s)
        print(f"==> {scene}: ORACLE landmark PnP {s['oracle_cm']:.3f} cm / "
              f"{s['oracle_deg']:.3f} deg from {s['matches']:.0f} oracle matches, "
              f"median ray offset {s['ray_off_cm']:.2f} cm", flush=True)
        with open(out / f"oracle_{scene}{args.map_suffix}_r{args.radius:g}.json",
                  "w") as f:
            json.dump(s, f, indent=1)
        del emap
    print(f"\n{'scene':<9}{'oracle cm':>10}{'deg':>8}{'matches':>9}{'inl':>7}"
          f"{'ray off cm':>11}")
    for s in summaries:
        print(f"{s['scene']:<9}{s['oracle_cm']:10.3f}{s['oracle_deg']:8.3f}"
              f"{s['matches']:9.0f}{s['inliers']:7.0f}{s['ray_off_cm']:11.2f}")


if __name__ == "__main__":
    main()
