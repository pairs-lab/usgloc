#!/usr/bin/env python3
"""Does the reported covariance flag the queries the pipeline gets wrong?

The paper's claim is that the posterior is informative: a wrong pose should
come with a wide covariance. Measured over the stored per-query records, that
holds on every scene except the self-similar one -- on Stairs the reported
sigma ranks failures barely better than chance, while the correction's inlier
count ranks them almost perfectly.

So this script prices one change to the CONDENSATION, offline and exactly
(the records carry Sigma, d2, logdet, the inlier count and the ground truth):
scale each query's covariance by

    s_i = (I_ref / max(I_i, 1)) ** p,   I_ref = median inlier count (calib half)

before the usual single scalar temperature is fitted. p = 0 is the current
method. Fitting tau on one half of the queries and scoring the other (then
swapping) keeps the comparison honest.

    python3 scripts/analysis/exp_sigma_support.py results/valhard_base/*.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2

LEVELS = (0.683, 0.954, 0.997)


def auc(score, lab):
    """AUC of `score` ranking the boolean `lab` (True = failure)."""
    o = np.argsort(score, kind="mergesort")
    l = np.asarray(lab)[o]
    r = np.arange(1, len(l) + 1.0)
    n1, n0 = l.sum(), (~l).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ause(sigma, err):
    """Area under the sparsification-error curve (lower is better)."""
    sigma, err = np.asarray(sigma), np.asarray(err)
    n = len(err)
    fr = np.arange(0, n) / n
    o_s = np.argsort(-sigma, kind="mergesort")
    o_e = np.argsort(-err, kind="mergesort")
    cs = np.array([err[o_s[k:]].mean() if k < n else 0.0 for k in range(n)])
    ce = np.array([err[o_e[k:]].mean() if k < n else 0.0 for k in range(n)])
    denom = ce[0] if ce[0] > 0 else 1.0
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz((cs - ce) / denom, fr))


def metrics(d2, sig_t, logdet, t_err):
    d2 = np.asarray(d2)
    cov = {L: float((d2 <= chi2.ppf(L, 6)).mean()) for L in LEVELS}
    grid = np.linspace(0.05, 0.95, 19)
    ece = float(np.mean([abs((d2 <= chi2.ppf(L, 6)).mean() - L) for L in grid]))
    nll = float(np.mean(0.5 * (d2 + np.asarray(logdet) + 6 * math.log(2 * math.pi))))
    return cov, ece, nll, ause(sig_t, t_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--powers", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--good-cm", type=float, default=5.0)
    args = ap.parse_args()

    for f in args.results:
        blob = json.load(open(f))
        r = blob["records"]
        d2 = np.array([x["d2"] for x in r])
        sig = np.array([x["sigma_t"] for x in r])
        ld = np.array([x["logdet"] for x in r])
        te = np.array([x["t_err"] for x in r]) * 100
        inl = np.array([max(float(x["inliers"]), 1.0) for x in r])
        bad = te > args.good_cm
        print(f"\n=== {Path(f).parent.name}/{Path(f).name}  n={len(r)}  "
              f"median {np.median(te):.2f} cm  >{args.good_cm:g}cm "
              f"{100 * bad.mean():.0f}%")
        for p in args.powers:
            # two-fold: fit tau on one half, score the other, and swap
            idx = np.arange(len(r))
            parts = [idx[0::2], idx[1::2]]
            D2, SG, LD = np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r))
            for fit, ev in (parts, parts[::-1]):
                ref = np.median(inl[fit])
                s = (ref / inl) ** p
                tau = float(np.median(d2[fit] / s[fit]) / chi2.ppf(0.5, 6))
                tau = max(tau, 1e-12)
                D2[ev] = d2[ev] / (s[ev] * tau)
                SG[ev] = sig[ev] * np.sqrt(s[ev] * tau)
                LD[ev] = ld[ev] + 6 * np.log(s[ev] * tau)
            cov, ece, nll, au = metrics(D2, SG, LD, te)
            print(f"  p={p:<4g} cov {100 * cov[0.683]:5.1f}/{100 * cov[0.954]:5.1f}/"
                  f"{100 * cov[0.997]:5.1f}  ECE {ece:.3f}  AUSE {au:.3f}  "
                  f"NLL {nll:8.2f}  AUC(sigma flags failure) {auc(SG, bad):.2f}")


if __name__ == "__main__":
    main()
