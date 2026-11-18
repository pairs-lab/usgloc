#!/usr/bin/env python3
"""ACE + PnP covariance on 12Scenes: the 12Scenes twin of ace_pnp_covariance.py (7Scenes).

At ACE's own dsacstar pose (results/ace_dsacstar_12scenes_100, unchanged), the predicted scene
coordinates whose reprojection error is below 10 px give Sigma = s^2 (J^T J)^-1 (same function).
The temperature is fitted per scene by our median rule on the OTHER half of the test frames
(2-fold, by parity), exactly as for 7Scenes; that choice favours ACE. Images go through ACE's
test transform (grayscale, height 480); the SfM focal and principal point are scaled with it.
Runs on CPU.

    python scripts/analysis/ace_pnp_covariance_12scenes.py --ace-repo ../third_party/ace \
        --heads ../third_party/ace_models/ace_models/12Scenes_pgt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/analysis"))
from usgloc import se3                          # noqa: E402
from usgloc.datasets import TwelveScenes        # noqa: E402
from ace_pnp_covariance import pnp_covariance   # noqa: E402

SCENES = ["apt1/kitchen", "apt1/living", "apt2/bed", "apt2/kitchen", "apt2/living", "apt2/luke",
          "office1/gates362", "office1/gates381", "office1/lounge", "office1/manolis", "office2/5a", "office2/5b"]
PGT = ROOT / "data/visloc_pseudo_gt_limitations/pgt/sfm/12scenes"


def preprocess(rgb):
    """ACE test transform (= scripts/analysis/run_ace_baseline.py): grayscale, height 480, normalize."""
    x = rgb.permute(2, 0, 1); h, w = x.shape[1:]
    if h != 480:
        x = TF.resize(x, [480, int(round(w * 480 / h))], antialias=True)
    x = TF.normalize(TF.rgb_to_grayscale(x), mean=[0.4], std=[0.25])
    return x[None], x.shape[1] / h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ace-repo", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--ace-run", default="results/ace_dsacstar_12scenes_100")
    ap.add_argument("--out", default="results/ace_pnpcov_12scenes_100")
    ap.add_argument("--threshold", type=float, default=10.0, help="dsacstar inlier threshold (px)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    sys.path.insert(0, args.ace_repo)
    from ace_network import Regressor
    torch.set_num_threads(args.threads)
    enc = torch.load(Path(args.ace_repo) / "ace_encoder_pretrained.pt", map_location="cpu", weights_only=False)
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    for s in SCENES:
        f_ = s.replace("/", "_")
        head = torch.load(Path(args.heads) / f"pgt_12scenes_{f_}.pt", map_location="cpu", weights_only=False)
        net = Regressor.create_from_split_state_dict(enc, head).eval()
        ds = TwelveScenes(str(ROOT / "data/12scenes_source"), s, split="test", sfm_pose_dir=str(PGT))
        where = {(fr.seq, fr.frame_id): i for i, fr in enumerate(ds.frames)}
        src = json.load(open(ROOT / args.ace_run / f"12scenes_{f_}.json"))["records"]
        recs = []
        for r in sorted(src, key=lambda r: (r["seq"], r["frame_id"])):
            i = where[(r["seq"], r["frame_id"])]
            rgb = ds.load_image(i)
            x, scale = preprocess(rgb)
            with torch.no_grad():
                sc = net(x).float()[0].numpy()
            cam = ds.scene_camera(rgb.shape[1], rgb.shape[0])
            T = np.array(r["T"]); T_gt = ds[i].pose_cw.double().numpy()
            Sig, n_inl, s2 = pnp_covariance(sc, T, cam.fx * scale, args.threshold, cx=cam.cx * scale, cy=cam.cy * scale)
            e = se3.relative_tangent(torch.tensor(T), torch.tensor(T_gt)).numpy()
            recs.append(dict(seq=r["seq"], frame_id=r["frame_id"], t_err=r["t_err"], r_err=r["r_err"],
                             inliers_dsacstar=r["inliers"], n_inl=n_inl, s2=s2,
                             d2=float(e @ np.linalg.solve(Sig, e)),
                             sigma_t=float(np.sqrt(np.trace(Sig[:3, :3]))), Sigma=Sig.tolist()))
        d2 = np.array([q["d2"] for q in recs]); med6 = chi2.ppf(0.5, 6)
        taus = [float(np.median(d2[1::2]) / med6), float(np.median(d2[0::2]) / med6)]
        for k, q in enumerate(recs):                     # fold = parity; tau from the other fold
            q["tau"] = taus[k % 2]
        json.dump(dict(scene=s, threshold=args.threshold, tau_folds=taus, records=recs),
                  open(out / f"12scenes_{f_}.json", "w"))
        dt = np.array([q["d2"] / q["tau"] for q in recs])
        ni = np.array([q["n_inl"] for q in recs]); nd = np.array([q["inliers_dsacstar"] for q in recs])
        cov = [100 * np.mean(dt <= chi2.ppf(l, 6)) for l in (0.683, 0.954, 0.997)]
        print(f"{s:17s} n={len(recs)} tau={taus[0]:.1f}/{taus[1]:.1f} cov {cov[0]:.0f}/{cov[1]:.0f}/{cov[2]:.0f}"
              f" | inliers mine vs dsacstar: median {np.median(ni):.0f} vs {np.median(nd):.0f},"
              f" corr {np.corrcoef(ni, nd)[0, 1]:.3f} | raw d2 median {np.median(d2):.1f}", flush=True)


if __name__ == "__main__":
    main()
