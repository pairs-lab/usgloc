#!/usr/bin/env python3
"""Are the correction's rounds independent measurements of the same pose?

The 'pool every round' idea rests on the rounds being near-independent: each
renders from a slightly different pose, so its depth-blending error and its
matcher noise should differ, and averaging should beat any single round.  This
script measures that assumption directly.  For every train frame it runs the
correction with ``acc_debug=True`` (which returns every round's pose) from the
ground truth and reports, in centimetres:

  err(best)    the error of the round the estimator keeps (current behaviour)
  err(rnd k)   the error of each round on its own
  spread       the RMS distance of the rounds about their own mean pose
               -- how much the rounds actually disagree
  err(mean)    the error of the (chordal) mean of the round poses

If ``spread`` is far smaller than ``err(best)``, the rounds are re-measuring
the same systematic error and no amount of pooling can remove it; the floor is
bias, not variance.  If ``spread`` is comparable to ``err(best)``, pooling has
something to average and ``err(mean)`` should fall below ``err(best)``.

    python3 scripts/analysis/diag_round_indep.py chess office stairs \
        --rounds 5 --out results/acc_rounds/diag
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from usgloc.datasets import SevenScenes, load_gaussian_map      # noqa: E402
from usgloc.rendering import GsplatRenderer                     # noqa: E402
from usgloc.correction import EvidentialMatchCorrector, _to_gray  # noqa: E402
from usgloc import se3                                          # noqa: E402

PGT = "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
N_FRAMES = 20


def t_cm(A, B):
    d = np.asarray(A, np.float64) @ np.linalg.inv(np.asarray(B, np.float64))
    return float(np.linalg.norm(d[:3, 3])) * 100


def r_deg(A, B):
    d = np.asarray(A, np.float64) @ np.linalg.inv(np.asarray(B, np.float64))
    c = np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def mean_pose(Ts, w=None):
    """Chordal mean: average the rotation matrices and project back with SVD,
    average the camera centres (translation averaging in the world frame)."""
    Ts = [np.asarray(T, np.float64) for T in Ts]
    w = np.ones(len(Ts)) if w is None else np.asarray(w, np.float64)
    w = w / w.sum()
    R = sum(wi * T[:3, :3] for wi, T in zip(w, Ts))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    C = sum(wi * (-T[:3, :3].T @ T[:3, 3]) for wi, T in zip(w, Ts))   # camera centre
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["chess", "office", "stairs"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--perturb", nargs=2, type=float, default=None)
    ap.add_argument("--jitter", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("M", "DEG"), help="render-pose jitter on rounds >= 1")
    ap.add_argument("--map-suffix", default="_ft")
    ap.add_argument("--out", default="results/acc_rounds/diag")
    args = ap.parse_args()
    import os
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    rend = GsplatRenderer()
    rows_all = {}
    for scene in args.scenes:
        ds = SevenScenes("data/7scenes_source", scene, split="train", sfm_pose_dir=PGT)
        idx = torch.linspace(0, len(ds) - 1, N_FRAMES).round().long().unique().tolist()
        cam = ds.scene_camera(640, 480)
        emap = load_gaussian_map(f"maps/{scene}{args.map_suffix}.pt")
        for f in ("means", "covs", "colors", "opacities", "gaussian_vacuity"):
            setattr(emap, f, getattr(emap, f).to(dev))
        corr = EvidentialMatchCorrector(emap, rend, cam, rounds=args.rounds,
                                        acc_debug=True, acc_jitter=args.jitter[0],
                                        acc_jitter_deg=args.jitter[1])
        rows = []
        for i in idx:
            img = ds.load_image(i).to(dev)
            T = ds[i].pose_cw.to(dev, torch.float32)
            T0 = T
            if args.perturb is not None:
                g = torch.Generator().manual_seed(7 * i)
                dt = torch.randn(3, generator=g); dt = dt / dt.norm() * args.perturb[0]
                dr = torch.randn(3, generator=g); dr = dr / dr.norm() * (args.perturb[1] * np.pi / 180)
                T0 = se3.apply_tangent(T.to(torch.float32),
                                       torch.cat([dt, dr]).to(dev, torch.float32))
            for attempt in range(30):
                try:
                    r = corr.correct(_to_gray(img), T0)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"    OOM frame {i} ({attempt + 1}); waiting 30 s", flush=True)
                    time.sleep(30)
            else:
                raise RuntimeError("persistent OOM")
            if not r.success or not r.round_T:
                continue
            gt = T.double().cpu().numpy()
            Ts = [np.asarray(x, np.float64) for x in r.round_T]
            Cs = np.stack([-P[:3, :3].T @ P[:3, 3] for P in Ts])
            spread = float(np.sqrt(((Cs - Cs.mean(0)) ** 2).sum(1).mean())) * 100
            Tm = mean_pose(Ts)
            Tw = mean_pose(Ts, r.round_inl)
            rows.append(dict(
                frame=int(i), n_rounds=len(Ts),
                err_best=t_cm(r.T.double().cpu().numpy(), gt),
                deg_best=r_deg(r.T.double().cpu().numpy(), gt),
                err_rounds=[t_cm(P, gt) for P in Ts],
                err_last=t_cm(Ts[-1], gt),
                spread_cm=spread,
                err_mean=t_cm(Tm, gt), deg_mean=r_deg(Tm, gt),
                err_wmean=t_cm(Tw, gt),
                inl=r.round_inl, rms=r.round_rms))
            print(f"  {scene:<8}f{i:<6} best {rows[-1]['err_best']:6.3f}  "
                  f"rounds " + " ".join(f"{e:6.3f}" for e in rows[-1]['err_rounds'])
                  + f"  spread {spread:6.3f}  mean {rows[-1]['err_mean']:6.3f}", flush=True)
        del corr, emap
        torch.cuda.empty_cache()
        rows_all[scene] = rows
        b = np.median([r["err_best"] for r in rows])
        m = np.median([r["err_mean"] for r in rows])
        w = np.median([r["err_wmean"] for r in rows])
        s = np.median([r["spread_cm"] for r in rows])
        l = np.median([r["err_last"] for r in rows])
        print(f"==> {scene:<9} best {b:.3f} cm | last {l:.3f} | mean-of-rounds {m:.3f} "
              f"| inlier-weighted mean {w:.3f} | round spread {s:.3f} cm "
              f"({100 * s / max(b, 1e-9):.0f}% of the floor)", flush=True)
    tag = "" if args.perturb is None else f"_p{args.perturb[0]:g}m{args.perturb[1]:g}deg"
    if any(args.jitter):
        tag += f"_j{args.jitter[0]:g}m{args.jitter[1]:g}deg"
    with open(out / f"round_indep_r{args.rounds}{tag}.json", "w") as f:
        json.dump(rows_all, f, indent=1)


if __name__ == "__main__":
    main()
