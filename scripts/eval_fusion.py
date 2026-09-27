#!/usr/bin/env python3
"""What is a calibrated covariance actually WORTH downstream?

The letter motivates the posterior by the fact that a robot fuses each pose
into a Bayesian filter, and that a pose without a covariance forces that
filter to assume a fixed noise model. This script tests that claim instead
of asserting it. Over a continuous test sequence we run the same constant-
velocity EKF three ways and compare trajectory error:

  per-query   -- our tempered covariance as the measurement noise;
  fixed       -- one constant covariance, ORACLE-tuned by sweeping its scale
                 and keeping the best, i.e. the strongest possible version of
                 the baseline a point estimator permits;
  gated       -- per-query covariance, additionally rejecting measurements
                 whose predicted sigma exceeds a threshold.

Two further columns need no filter and matter to a practitioner:
  availability/accuracy trade-off -- error after discarding the least
  confident fraction, which is what a risk-aware system actually does.

    python3 scripts/eval_fusion.py results/traj_chess/7scenes_chess.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def se3_log_t(A, B):
    """Translation part of the relative error between two 4x4 poses (cm)."""
    d = A @ np.linalg.inv(B)
    return float(np.linalg.norm(d[:3, 3])) * 100


def centre(T):
    """Camera centre of a camera-from-world pose."""
    return -T[:3, :3].T @ T[:3, 3]


def centre_cov(T, Sigma):
    """Map a tangent-space se(3) covariance to the camera-centre covariance.

    Sigma is the covariance of xi with T(xi) = exp(xi^) T_mode, while the
    filter tracks the camera centre c = -R^T t. To first order
    dc = -R^T dv + R^T [t]_x dw, so Cov(c) = J Sigma J^T with that J. Using
    the scalar marginal instead would throw away the anisotropy, which is the
    part of a covariance a filter actually benefits from.
    """
    R, t = T[:3, :3], T[:3, 3]
    skew_t = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    J = np.zeros((3, 6))
    J[:, :3] = -R.T
    J[:, 3:] = R.T @ skew_t
    C = J @ Sigma @ J.T
    return 0.5 * (C + C.T)


def run_ekf_full(centres, covs, gt_centres, gate_idx=None):
    """Same filter, but with the full 3x3 measurement covariance."""
    n = len(centres)
    x = np.zeros(6); x[:3] = centres[0]
    P = np.eye(6) * 1e-2
    F = np.eye(6); F[:3, 3:] = np.eye(3)
    Q = np.eye(6) * 1e-4
    H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
    out = []
    for k in range(n):
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        if gate_idx is None or gate_idx[k]:
            R = covs[k] + np.eye(3) * 1e-12
            y = centres[k] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K @ H) @ P
        out.append(x[:3].copy())
    return np.array(out)


def run_ekf(centres, sigmas, gt_centres, meas_scale=1.0, gate=None):
    """Constant-velocity EKF on the camera centre.

    State [p, v]; process noise q tuned once and shared by every variant, so
    the only thing that differs between variants is the MEASUREMENT noise.
    """
    n = len(centres)
    x = np.zeros(6)
    x[:3] = centres[0]
    P = np.eye(6) * 1e-2
    F = np.eye(6)
    F[:3, 3:] = np.eye(3)
    q = 1e-4
    Q = np.eye(6) * q
    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)
    out = []
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        s = sigmas[k] * meas_scale
        if gate is None or sigmas[k] <= gate:
            R = np.eye(3) * max(s ** 2, 1e-12)
            y = centres[k] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(6) - K @ H) @ P
        out.append(x[:3].copy())
    return np.array(out)


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    tau = d.get("tau", 1.0)
    recs = [r for r in d["records"] if "T" in r and "T_gt" in r]
    if not recs:
        print(f"{path}: no poses saved (re-run with the current runner)")
        return
    recs.sort(key=lambda r: r["frame_id"])

    C = np.array([centre(np.array(r["T"])) for r in recs])
    G = np.array([centre(np.array(r["T_gt"])) for r in recs])
    # tempered marginal translation sigma (metres)
    sig = np.array([r["sigma_t"] * tau ** 0.5 for r in recs])
    raw_err = np.linalg.norm(C - G, axis=1) * 100

    name = Path(path).stem
    print(f"\n=== {name}   n={len(recs)} consecutive frames")
    print(f"  raw localization      median {np.median(raw_err):6.2f} cm   "
          f"mean {raw_err.mean():6.2f}   >5cm {100*(raw_err>5).mean():3.0f}%")

    # ---- (1) risk-aware rejection: no filter, just trust the covariance
    print("  -- accuracy vs availability (reject least-confident fraction)")
    for frac in (0.0, 0.1, 0.2, 0.3):
        keep = np.argsort(sig)[: int(round((1 - frac) * len(sig)))]
        by_unc = np.median(raw_err[keep])
        # the fair control: reject the same number of frames at random
        rng = np.random.default_rng(0)
        rand = np.median([np.median(raw_err[rng.permutation(len(sig))[:len(keep)]])
                          for _ in range(200)])
        print(f"     keep {100*(1-frac):3.0f}%   by our sigma {by_unc:6.2f} cm"
              f"   vs random {rand:6.2f} cm")

    # ---- (2) EKF fusion: per-query covariance vs the best fixed one
    ekf_ours = run_ekf(C, sig, G)
    e_ours = np.linalg.norm(ekf_ours - G, axis=1) * 100
    best = None
    for s_fixed in np.geomspace(sig.min() * 0.5, sig.max() * 2, 40):
        e = np.linalg.norm(run_ekf(C, np.full_like(sig, s_fixed), G) - G, axis=1) * 100
        if best is None or np.median(e) < best[1]:
            best = (s_fixed, float(np.median(e)))
    # full-covariance variant: the anisotropy is the point of a covariance
    Cs = [centre_cov(np.array(r["T"]), np.array(r["Sigma"])) * tau for r in recs]
    e_full = np.linalg.norm(run_ekf_full(C, Cs, G) - G, axis=1) * 100
    keep = sig <= np.quantile(sig, 0.8)
    e_fullg = np.linalg.norm(run_ekf_full(C, Cs, G, gate_idx=keep) - G, axis=1) * 100
    gate = float(np.quantile(sig, 0.8))
    e_gate = np.linalg.norm(run_ekf(C, sig, G, gate=gate) - G, axis=1) * 100
    print("  -- constant-velocity EKF, trajectory error")
    print(f"     fixed covariance (oracle-tuned)  median {best[1]:6.2f} cm")
    print(f"     our per-query covariance         median {np.median(e_ours):6.2f} cm")
    print(f"     + reject worst 20% by sigma      median {np.median(e_gate):6.2f} cm")
    print(f"     our FULL 3x3 covariance          median {np.median(e_full):6.2f} cm")
    print(f"     + reject worst 20% by sigma      median {np.median(e_fullg):6.2f} cm")
    best_ours = min(np.median(e_ours), np.median(e_full), np.median(e_fullg))
    gain = 100 * (1 - best_ours / best[1])
    print(f"     => best per-query variant vs oracle-tuned fixed: {gain:+.0f}%")


def cross_scene(paths):
    """The realistic baseline: a fixed covariance cannot be tuned on the
    trajectory it will run on. We therefore tune the constant sigma on the
    OTHER sequences and apply it here, which is what a deployed system with a
    hand-set noise model actually has. Our per-query covariance needs no such
    transfer."""
    data = {}
    for path in paths:
        d = json.load(open(path)); tau = d.get("tau", 1.0)
        recs = sorted([r for r in d["records"] if "T" in r], key=lambda r: r["frame_id"])
        if not recs:
            continue
        data[Path(path).stem] = dict(
            C=np.array([centre(np.array(r["T"])) for r in recs]),
            G=np.array([centre(np.array(r["T_gt"])) for r in recs]),
            sig=np.array([r["sigma_t"] * tau ** 0.5 for r in recs]),
            Cs=[centre_cov(np.array(r["T"]), np.array(r["Sigma"])) * tau for r in recs])

    def tuned_sigma(dd):
        best = None
        for sf in np.geomspace(dd["sig"].min() * 0.5, dd["sig"].max() * 2, 40):
            e = np.linalg.norm(run_ekf(dd["C"], np.full_like(dd["sig"], sf), dd["G"]) - dd["G"], axis=1)
            m = float(np.median(e))
            if best is None or m < best[1]:
                best = (sf, m)
        return best[0]

    print("\n=== TRANSFERRED fixed covariance (tuned on the OTHER scenes)")
    print(f"{'scene':<22}{'fixed(transfer)':>16}{'ours(full cov)':>16}{'gain':>8}")
    for name, dd in data.items():
        others = [tuned_sigma(v) for k, v in data.items() if k != name]
        s_tr = float(np.exp(np.mean(np.log(others))))  # geometric mean
        e_fx = np.linalg.norm(run_ekf(dd["C"], np.full_like(dd["sig"], s_tr), dd["G"]) - dd["G"], axis=1) * 100
        e_ou = np.linalg.norm(run_ekf_full(dd["C"], dd["Cs"], dd["G"]) - dd["G"], axis=1) * 100
        g = 100 * (1 - np.median(e_ou) / np.median(e_fx))
        print(f"{name:<22}{np.median(e_fx):16.2f}{np.median(e_ou):16.2f}{g:+7.0f}%")


if __name__ == "__main__":
    paths = sys.argv[1:]
    for p in paths:
        sys.argv = [sys.argv[0], p]
        main()
    if len(paths) > 1:
        cross_scene(paths)
