#!/usr/bin/env python3
"""ACE + PnP covariance: the covariance a scene-coordinate regressor gives for free.

At ACE's own dsacstar pose (results/ace_dsacstar_100, unchanged), the predicted
scene coordinates whose reprojection error is below dsacstar's inlier threshold
(10 px) give the Gauss-Newton covariance of the 6-DoF pose,
    Sigma = s^2 (J^T J)^-1,   s^2 = sum ||r||^2 / (2N - 6),
in the same chart as ours: T = exp(xi^) T_est, xi = (rho, phi), T world->camera.
The temperature is fitted per scene by our median rule, but on the OTHER half of
the test frames (2-fold), since ACE trains on every training frame and so has no
held-out train-side frames. That choice favours ACE.

Runs on CPU. ACE code: github.com/nianticlabs/ace (clone); heads: 7Scenes_pgt of
https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz
    python scripts/analysis/ace_pnp_covariance.py --ace-repo <clone> --heads <7Scenes_pgt dir>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import chi2, spearmanr
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from usgloc import se3  # noqa: E402

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
PGT = ROOT / "data/visloc_pseudo_gt_limitations/pgt/sfm/7scenes"
SRC = ROOT / "data/7scenes_source"
SUB = 8                                       # ACE output stride


def pgt_line(line):
    p = line.split(); q = [float(x) for x in p[1:5]]
    T = np.eye(4); T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = [float(x) for x in p[5:8]]
    return p[0], T, float(p[8])


def skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def pnp_covariance(sc, T, f, thr):
    """sc: (3,h,w) scene coordinates; T: 4x4 world->camera. Returns Sigma (6x6), n_inl, s2."""
    h, w = sc.shape[1:]
    u, v = np.meshgrid(np.arange(w) * SUB + SUB // 2, np.arange(h) * SUB + SUB // 2)
    Xw = sc.reshape(3, -1).T
    Xc = Xw @ T[:3, :3].T + T[:3, 3]
    Z = Xc[:, 2]; ok = Z > 0.1
    cx, cy = w * SUB / 2, h * SUB / 2
    pu = f * Xc[:, 0] / np.where(ok, Z, 1) + cx; pv = f * Xc[:, 1] / np.where(ok, Z, 1) + cy
    r = np.stack([pu - u.ravel(), pv - v.ravel()], 1)
    inl = ok & (np.linalg.norm(r, axis=1) < thr)
    X, Y, Z, r = Xc[inl, 0], Xc[inl, 1], Xc[inl, 2], r[inl]
    P = np.zeros((len(Z), 2, 3))
    P[:, 0, 0] = f / Z; P[:, 0, 2] = -f * X / Z**2
    P[:, 1, 1] = f / Z; P[:, 1, 2] = -f * Y / Z**2
    J = np.concatenate([P, -P @ skew(Xc[inl])], 2)      # d(residual)/d(rho, phi)
    H = np.einsum("nij,nik->jk", J, J)
    s2 = float((r**2).sum() / max(2 * len(Z) - 6, 1))
    return s2 * np.linalg.inv(H), int(inl.sum()), s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ace-repo", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--ace-run", default="results/ace_dsacstar_100")
    ap.add_argument("--out", default="results/ace_pnpcov_100")
    ap.add_argument("--threshold", type=float, default=10.0, help="dsacstar inlier threshold (px)")
    args = ap.parse_args()
    sys.path.insert(0, args.ace_repo)
    from ace_network import Regressor
    torch.set_num_threads(12)
    enc = torch.load(Path(args.ace_repo) / "ace_encoder_pretrained.pt", map_location="cpu")
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    for s in SCENES:
        head = torch.load(Path(args.heads) / f"pgt_7scenes_{s}.pt", map_location="cpu")
        net = Regressor.create_from_split_state_dict(enc, head).eval()
        gt = {n: (T, f) for n, T, f in map(pgt_line, open(PGT / f"{s}_test.txt"))}
        src = json.load(open(ROOT / args.ace_run / f"7scenes_{s}.json"))["records"]
        recs = []
        for r in sorted(src, key=lambda r: (r["seq"], r["frame_id"])):
            name = f"{r['seq']}/frame-{r['frame_id']:06d}.color.png"
            img = np.asarray(Image.open(SRC / s / name).convert("L"), np.float32) / 255.0
            x = (torch.from_numpy(img) - 0.4) / 0.25
            with torch.no_grad():
                sc = net(x[None, None]).float()[0].numpy()
            T = np.array(r["T"]); T_gt, f = gt[name]
            Sig, n_inl, s2 = pnp_covariance(sc, T, f, args.threshold)
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
                  open(out / f"7scenes_{s}.json", "w"))
        dt = np.array([q["d2"] / q["tau"] for q in recs])
        ni = np.array([q["n_inl"] for q in recs]); nd = np.array([q["inliers_dsacstar"] for q in recs])
        cov = [100 * np.mean(dt <= chi2.ppf(l, 6)) for l in (0.683, 0.954, 0.997)]
        print(f"{s:11s} n={len(recs)} tau={taus[0]:.1f}/{taus[1]:.1f} cov {cov[0]:.0f}/{cov[1]:.0f}/{cov[2]:.0f}"
              f" | inliers mine vs dsacstar: median {np.median(ni):.0f} vs {np.median(nd):.0f},"
              f" corr {np.corrcoef(ni, nd)[0, 1]:.3f} | raw d2 median {np.median(d2):.1f}", flush=True)


if __name__ == "__main__":
    main()
