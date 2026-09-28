#!/usr/bin/env python3
r"""What is the calibrated posterior WORTH? Quantitative, paired against ACE.

Coverage and ECE say the posterior is honest; they do not say it is useful.
These are the three questions a practitioner actually asks, each answered on
the SAME query frames for our sigma_t and for the only confidence signal a
scene-coordinate baseline exposes (its PnP inlier count):

(a) FAILURE DETECTION. Can the signal tell a gross failure (>5 cm) from a good
    pose?  Reported as ROC AUC and average precision, which are threshold-free,
    plus the false-positive rate at 95% recall of failures.

(b) SELECTIVE LOCALIZATION (risk-coverage). A robot that can decline to answer
    wants the most coverage subject to a risk budget: the largest fraction of
    queries it can ACCEPT while keeping 95% (or 99%) of accepted poses within
    5 cm.  READ THIS WITHIN A METHOD, NOT ACROSS METHODS: the achievable
    coverage depends mostly on how many failures a method has at all, so a more
    ROBUST method scores higher here regardless of how good its confidence
    signal is.  The meaningful contrast is against the RANDOM ordering, which
    isolates what the signal itself buys.  Use (a) and the AUSE/Spearman numbers
    of the paired-confidence table for cross-method claims.

(c) ERROR AT A GIVEN COVERAGE, ours vs the baseline signal vs random, so the
    ordering quality is visible independently of calibration.

    python3 scripts/analysis/uncertainty_benefit.py results/v2_seed0
"""
import argparse
import json
import os
import sys

import numpy as np

SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
ACE = "results/ace_dsacstar_100"


def load(d):
    """Paired arrays over frames present in BOTH our run and the ACE re-run."""
    ours_e, ours_s, ace_e, ace_i, scenes = [], [], [], [], []
    for s in SC:
        f, g = f"{d}/7scenes_{s}.json", f"{ACE}/7scenes_{s}.json"
        if not (os.path.exists(f) and os.path.exists(g)):
            continue
        j = json.load(open(f))
        tau = j.get("tau", 1.0)
        a = {(r["seq"], r["frame_id"]): r for r in json.load(open(g))["records"]}
        for r in j["records"]:
            k = (r["seq"], r["frame_id"])
            if k not in a:
                continue
            ours_e.append(100 * r["t_err"])
            ours_s.append(r["sigma_t"] * tau ** 0.5)
            ace_e.append(100 * a[k]["t_err"])
            ace_i.append(a[k]["inliers"])
            scenes.append(s)
    return (np.array(ours_e), np.array(ours_s), np.array(ace_e),
            np.array(ace_i, dtype=float), np.array(scenes))


def auroc(score, label):
    """P(score of a positive > score of a negative); ties count a half."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ap(score, label):
    o = np.argsort(-score, kind="mergesort")
    l = label[o]
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(l.sum(), 1))


def fpr_at_recall(score, label, target=0.95):
    o = np.argsort(-score, kind="mergesort")
    l = label[o]
    tp = np.cumsum(l)
    need = target * label.sum()
    k = int(np.searchsorted(tp, need) + 1)
    k = min(k, len(l))
    fp = k - tp[k - 1]
    return float(fp / max((~label).sum(), 1))


def max_coverage(score, err, thr=5.0, risk=0.05):
    """Largest accepted fraction whose (1-risk) quantile of error is <= thr.

    ``score`` is a CONFIDENCE (higher = more confident)."""
    o = np.argsort(-score, kind="mergesort")
    e = err[o]
    best = 0.0
    for k in range(1, len(e) + 1):
        if np.quantile(e[:k], 1 - risk) <= thr:
            best = k / len(e)
    return best


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("run", nargs="?", default="results/v2_seed0")
    a = ap_.parse_args()
    oe, os_, ae, ai, sc = load(a.run)
    if not len(oe):
        sys.exit(f"no paired records under {a.run}")
    print(f"paired queries: {len(oe)}   (ours: {a.run}; baseline: {ACE})\n")

    # (a) failure detection -------------------------------------------------
    print("(a) FAILURE DETECTION  (positive = translation error > 5 cm)")
    for tag, score, err in (("ours (sigma_t)", os_, oe),
                            ("ACE (-inliers)", -ai, ae)):
        lab = err > 5.0
        print(f"    {tag:<16} n_fail {lab.sum():3d}/{len(lab)}   "
              f"ROC AUC {auroc(score, lab):.3f}   AP {ap(score, lab):.3f}   "
              f"FPR@95%recall {100*fpr_at_recall(score, lab):.0f}%")

    # (b) selective localization -------------------------------------------
    print("\n(b) SELECTIVE LOCALIZATION: largest accepted fraction with")
    print("    95% / 99% of accepted poses within 5 cm")
    # NB: cross-method comparison here is confounded by the failure RATE
    # (ACE: 20/700 beyond 5 cm, ours: 96/700), so the row to read against ours
    # is 'random order', not ACE.
    for tag, conf, err in (("ours (sigma_t)", -os_, oe),
                           ("ACE (inliers)$", ai, ae)):
        c95 = max_coverage(conf, err, 5.0, 0.05)
        c99 = max_coverage(conf, err, 5.0, 0.01)
        print(f"    {tag:<16} {100*c95:5.1f}%      {100*c99:5.1f}%")
    rng = np.random.default_rng(0)
    r95 = np.median([max_coverage(rng.permutation(len(oe)).astype(float), oe, 5.0, 0.05)
                     for _ in range(50)])
    print(f"    {'random order':<16} {100*r95:5.1f}%      (ours vs random is the "
          f"value of the signal)")

    # (c) error at fixed coverage ------------------------------------------
    print("\n(c) MEDIAN / 90th-PERCENTILE error (cm) of the accepted set")
    print(f"    {'keep':<8}{'ours':>16}{'ACE-inliers':>18}{'random':>16}")
    for keep in (1.0, 0.9, 0.8, 0.7, 0.5):
        k = max(1, int(round(keep * len(oe))))
        o_sel = oe[np.argsort(os_, kind="mergesort")[:k]]
        a_sel = ae[np.argsort(-ai, kind="mergesort")[:k]]
        rmed = np.median([np.median(rng.permutation(oe)[:k]) for _ in range(50)])
        r90 = np.median([np.quantile(rng.permutation(oe)[:k], 0.9) for _ in range(50)])
        print(f"    {100*keep:3.0f}%   "
              f"{np.median(o_sel):7.2f}/{np.quantile(o_sel,0.9):6.2f}"
              f"{np.median(a_sel):11.2f}/{np.quantile(a_sel,0.9):6.2f}"
              f"{rmed:9.2f}/{r90:6.2f}")


if __name__ == "__main__":
    main()
