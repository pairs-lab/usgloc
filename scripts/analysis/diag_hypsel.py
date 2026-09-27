#!/usr/bin/env python3
"""Where is the correct pose lost? -- hypothesis-set autopsy + offline rescoring.

Input: the JSON written by ``scripts/run_7scenes.py --debug-json`` (which
records, per query, every corrected candidate and every kept mixture
component with its weight and the evidence behind it).

Part 1, DIAGNOSIS.  Every query whose reported pose is wrong is put in
exactly one bucket:

  (i)   never proposed   -- no corrected candidate lands within --good-cm of
                            the ground truth. Nothing downstream can help.
  (ii)a proposed, lost   -- a corrected candidate was good, but it never
                            became a mixture component (rejected by the
                            render-fidelity verification, merged into a
                            near-duplicate, overflowed top_k, or pruned).
  (ii)b out-weighted     -- a good component was kept but did not carry the
                            largest weight.
  (iii) condensation     -- the largest-weight component was good and the
                            REPORTED pose is not.

Part 2, RESCORING.  The weights are a function of quantities this dump
records, so alternative weighting rules can be evaluated exactly, offline,
without touching the GPU: each rule re-picks the argmax component per query
and the resulting median / >5 cm fraction / SR@5cm5deg is what the pipeline
would have printed with that rule.

    python3 scripts/analysis/diag_hypsel.py dump1.json [dump2.json ...]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- geometry
def _centre(T):
    T = np.asarray(T, dtype=np.float64)
    return -T[:3, :3].T @ T[:3, 3]


def t_cm(Ta, Tb):
    return float(np.linalg.norm(_centre(Ta) - _centre(Tb))) * 100


def r_deg(Ta, Tb):
    Ra = np.asarray(Ta, dtype=np.float64)[:3, :3]
    Rb = np.asarray(Tb, dtype=np.float64)[:3, :3]
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ------------------------------------------------------------------- features
def comp_feats(rows, good_cm, good_deg):
    """Per query: arrays of component errors and of the evidence per component."""
    out = []
    for r in rows:
        gt = r["T_gt"]
        k = r["kept"]
        if not k:
            continue
        st = [c.get("stats") or {} for c in k]
        q = dict(
            seq=r["seq"], frame_id=r["frame_id"], tag=r.get("tag", ""),
            rep_cm=t_cm(r["T_rep"], gt), rep_deg=r_deg(r["T_rep"], gt),
            sigma_t=r.get("sigma_t", float("nan")), d2=r.get("d2", float("nan")),
            logdet=r.get("logdet", float("nan")),
            cm=np.array([t_cm(c["T"], gt) for c in k]),
            deg=np.array([r_deg(c["T"], gt) for c in k]),
            w=np.array([float(c["weight"]) for c in k]),
            inl=np.array([float(c["inliers"]) for c in k]),
            sup=np.array([float(c["support"]) for c in k]),
            psnr=np.array([float(c["psnr"]) for c in k]),
            cov=np.array([float(s.get("cov", 0.0)) for s in st]),
            cov_w=np.array([float(s.get("cov_w", 0.0)) for s in st]),
            nmatch=np.array([max(float(s.get("n_matches", 1.0)), 1.0) for s in st]),
            rms=np.array([float(s.get("reproj_px", 0.0)) for s in st]),
            snaps=r.get("snaps", []),
            kept_T=[c["T"] for c in k],
            snap_cm=np.array([t_cm(s["T"], gt) for s in r.get("snaps", [])
                              if s.get("success")]),
            anchor_cm=np.array([t_cm(s["T0"], gt) for s in r.get("snaps", [])]),
        )
        q["ok"] = (q["cm"] <= good_cm) & (q["deg"] <= good_deg)
        out.append(q)
    return out


# ------------------------------------------------------------------ diagnosis
def diagnose(qs, good_cm, good_deg):
    cats = dict(never_proposed=0, proposed_lost=0, outweighted=0, condensation=0)
    detail = dict(anchor_far=0, verify_failed=0, evicted=0,
                  refine_drift=0, n_bad=0)
    for q in qs:
        bad = not (q["rep_cm"] <= good_cm and q["rep_deg"] <= good_deg)
        if not bad:
            continue
        detail["n_bad"] += 1
        snap_ok = q["snap_cm"].size and q["snap_cm"].min() <= good_cm
        if not snap_ok:
            cats["never_proposed"] += 1
            if not (q["anchor_cm"].size and q["anchor_cm"].min() <= 60.0):
                detail["anchor_far"] += 1
            continue
        if not q["ok"].any():
            cats["proposed_lost"] += 1
            if any(s.get("verify_failed") for s in q["snaps"]):
                detail["verify_failed"] += 1
            # Was the good snap evicted before it could be weighted, or did it
            # become a component that the refinement/polish then walked away
            # from? Match the best snap to the nearest kept component.
            j = int(np.argmin(q["snap_cm"]))
            good = [s for s in q["snaps"] if s.get("success")][j]
            d = [t_cm(c, good["T"]) for c in q["kept_T"]]
            if d and min(d) < 10.0:
                detail["refine_drift"] += 1
            else:
                detail["evicted"] += 1
            continue
        top = int(np.argmax(q["w"]))
        if not q["ok"][top]:
            cats["outweighted"] += 1
        else:
            cats["condensation"] += 1
    return cats, detail


# ------------------------------------------------------------------ rescoring
def _norm(x):
    m = x.max()
    return x / m if m > 0 else np.ones_like(x)


RULES = {}


def rule(name):
    def deco(f):
        RULES[name] = f
        return f
    return deco


def _support_term(q):
    """The inlier factor exactly as usgloc.pipeline computes it: the
    normalized, floored inlier count -- or, when nothing snapped at all
    (every count is zero), the render-fidelity softmax it falls back to."""
    if q["inl"].max() > 0:
        return np.clip(_norm(q["inl"]), 0.05, None)
    z = q["psnr"] / 2.0
    e = np.exp(z - z.max())
    return e / e.sum()


@rule("base (inliers x support)")
def _base(q):
    return _support_term(q) * q["sup"]


@rule("inliers only")
def _inl(q):
    return _support_term(q)


@rule("inliers x sqrt(support)")
def _inlsqs(q):
    return _support_term(q) * np.sqrt(q["sup"])


@rule("inliers^2 x support")
def _inl2s(q):
    return _support_term(q) ** 2 * q["sup"]


@rule("psnr softmax(tau=2)")
def _psnr(q):
    z = q["psnr"] / 2.0
    e = np.exp(z - z.max())
    return e / e.sum()


@rule("base x cov^1")
def _c1(q):
    return _base(q) * np.clip(q["cov"], 1e-3, None)


@rule("base x spread-free cov (cov_w^1)")
def _cw1(q):
    return _base(q) * np.clip(q["cov_w"], 1e-3, None)


@rule("base x cov^2")
def _c2(q):
    return _base(q) * np.clip(q["cov"], 1e-3, None) ** 2


@rule("base x cov^4")
def _c4(q):
    return _base(q) * np.clip(q["cov"], 1e-3, None) ** 4


@rule("base x cov_w^2")
def _cw2(q):
    return _base(q) * np.clip(q["cov_w"], 1e-3, None) ** 2


@rule("base x ratio^2")
def _r2(q):
    return _base(q) * np.clip(q["inl"] / q["nmatch"], 1e-3, None) ** 2


@rule("base x ratio^4")
def _r4(q):
    return _base(q) * np.clip(q["inl"] / q["nmatch"], 1e-3, None) ** 4


@rule("base x exp(-2 drms)")
def _rms2(q):
    return _base(q) * np.exp(-2.0 * np.clip(q["rms"] - q["rms"].min(), 0, None))


@rule("base x cov^2 x ratio^2")
def _c2r2(q):
    return _c2(q) * np.clip(q["inl"] / q["nmatch"], 1e-3, None) ** 2


@rule("cov-weighted inliers (inl*cov)")
def _ic(q):
    v = q["inl"] * q["cov"]
    if v.max() <= 0:
        return _support_term(q) * q["sup"]
    return np.clip(_norm(v), 0.05, None) * q["sup"]


@rule("inliers^4 x support")
def _inl4s(q):
    return _support_term(q) ** 4 * q["sup"]


@rule("hard gate 1500/1000 then base")
def _gate(q):
    w = _base(q)
    if q["inl"].max() >= 1500:
        w = np.where(q["inl"] >= 1000, w, w * 1e-3)
    return w


@rule("base x cov^2 x exp(-2 drms)")
def _c2rms(q):
    return _c2(q) * np.exp(-2.0 * np.clip(q["rms"] - q["rms"].min(), 0, None))


@rule("ORACLE best component")
def _oracle(q):
    return -q["cm"]          # argmax picks the smallest error


def score_rules(qs, good_cm, good_deg, rules=None):
    rows = []
    for name in (rules or RULES):
        f = RULES[name]
        cm, dg = [], []
        for q in qs:
            j = int(np.argmax(f(q)))
            cm.append(q["cm"][j])
            dg.append(q["deg"][j])
        cm, dg = np.array(cm), np.array(dg)
        rows.append(dict(rule=name, median=float(np.median(cm)),
                         over5=float((cm > good_cm).mean()),
                         sr=float(((cm <= good_cm) & (dg <= good_deg)).mean()),
                         cm=cm))
    return rows


# --------------------------------------------------- slot-selection simulator
def _dedup_slots(cands, top_k, t_th=0.01, r_th=2.0):
    """The pipeline's dedup + top_k slot filling, on corrected poses."""
    kept = []
    for c in cands:
        dup = None
        for j, k in enumerate(kept):
            if (np.linalg.norm(_centre(c["T"]) - _centre(k["T"])) < t_th
                    and r_deg(c["T"], k["T"]) < r_th):
                dup = j
                break
        if dup is not None:
            kept[dup]["sup"] += 1
            continue
        if len(kept) == top_k:
            continue
        kept.append(dict(c, sup=1))
    return kept


def simulate(rows, good_cm, good_deg, top_k=3, rank="psnr", weight="base",
             verify=True):
    """Re-run slot selection + weighting from the recorded SNAP poses.

    Approximate (it skips the refinement and the polish, each worth well under
    a centimetre) but it is the only way to price a change to the SLOT race
    without spending a GPU run on every candidate rule.
    """
    cm, dg = [], []
    for r in rows:
        gt = r["T_gt"]
        cands = [s for s in r.get("snaps", [])
                 if s.get("success") and not (verify and s.get("verify_failed"))]
        if not cands:
            continue
        for s in cands:
            s["_cov"] = float(s.get("st_cov", 0.0))
            s["_ratio"] = float(s["inliers"]) / max(float(s.get("n_matches", 1)), 1.0)
        if rank == "inliers":
            key = lambda s: -float(s["inliers"])                     # noqa: E731
        elif rank == "inl_cov":
            key = lambda s: -float(s["inliers"]) * s["_cov"]         # noqa: E731
        else:
            key = lambda s: -float(s.get("psnr", 0.0))               # noqa: E731
        kept = _dedup_slots(sorted(cands, key=key), top_k)
        inl = np.array([float(k["inliers"]) for k in kept])
        w = np.clip(inl / max(inl.max(), 1e-9), 0.05, None) * np.array(
            [float(k["sup"]) for k in kept])
        if weight == "cov2":
            w = w * np.clip([k["_cov"] for k in kept], 1e-3, None) ** 2
        elif weight == "ratio2":
            w = w * np.clip([k["_ratio"] for k in kept], 1e-3, None) ** 2
        elif weight == "oracle":
            w = -np.array([t_cm(k["T"], gt) for k in kept])
        j = int(np.argmax(w))
        cm.append(t_cm(kept[j]["T"], gt))
        dg.append(r_deg(kept[j]["T"], gt))
    cm, dg = np.array(cm), np.array(dg)
    return dict(n=len(cm), median=float(np.median(cm)),
                over5=float((cm > good_cm).mean()),
                sr=float(((cm <= good_cm) & (dg <= good_deg)).mean()), cm=cm)


def sigma_section(qs, good_cm, powers=(0.0, 0.5, 1.0, 2.0), ref=0.9):
    """Should the reported covariance know how well the correspondences
    closed? Inflate each query's Sigma by (ref / inlier ratio) ** p before the
    single scalar temperature is fitted, two-fold (fit on one half, score the
    other, then swap), and report what it does to coverage / ECE / AUSE / NLL
    and to the AUC with which sigma flags the queries that are wrong."""
    from scipy.stats import chi2

    d2 = np.array([float(q["d2"]) for q in qs])
    ld = np.array([float(q["logdet"]) for q in qs])
    sg = np.array([float(q["sigma_t"]) for q in qs])
    te = np.array([q["rep_cm"] for q in qs])
    ratio = []
    for q in qs:
        j = int(np.argmax(q["w"]))
        ratio.append(max(float(q["inl"][j]) / max(float(q["nmatch"][j]), 1.0), 1e-3))
    ratio = np.array(ratio)
    ok = np.isfinite(d2) & np.isfinite(ld)
    if ok.sum() < 8 or len(te) != len(d2):
        print("  sigma section: not enough finite records")
        return
    bad = te > good_cm
    idx = np.arange(len(d2))
    parts = [idx[0::2], idx[1::2]]
    for p_ in powers:
        s_all = (ref / ratio) ** p_
        D2, SG, LD = np.zeros_like(d2), np.zeros_like(d2), np.zeros_like(d2)
        for fit, ev in (parts, parts[::-1]):
            tau = max(float(np.median(d2[fit] / s_all[fit]) / chi2.ppf(0.5, 6)), 1e-12)
            D2[ev] = d2[ev] / (s_all[ev] * tau)
            SG[ev] = sg[ev] * np.sqrt(s_all[ev] * tau)
            LD[ev] = ld[ev] + 6 * np.log(s_all[ev] * tau)
        cov = {L: 100 * float((D2 <= chi2.ppf(L, 6)).mean())
               for L in (0.683, 0.954, 0.997)}
        grid = np.linspace(0.05, 0.95, 19)
        ece = float(np.mean([abs((D2 <= chi2.ppf(L, 6)).mean() - L) for L in grid]))
        nll = float(np.mean(0.5 * (D2 + LD + 6 * math.log(2 * math.pi))))
        print(f"    p={p_:<4g} cov {cov[0.683]:5.1f}/{cov[0.954]:5.1f}/{cov[0.997]:5.1f}"
              f"  ECE {ece:.3f}  NLL {nll:8.2f}  AUC(sigma flags failure) "
              f"{_auc(SG, bad):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--good-cm", type=float, default=5.0)
    ap.add_argument("--good-deg", type=float, default=5.0)
    ap.add_argument("--tag", default=None, help="keep only this tag (val/calib)")
    ap.add_argument("--per-query", action="store_true")
    ap.add_argument("--simulate", action="store_true",
                    help="also price slot-race changes (rank / top_k) from the "
                         "recorded snap poses")
    args = ap.parse_args()

    for d in args.dumps:
        blob = json.load(open(d))
        rows = blob["rows"]
        if args.tag:
            rows = [r for r in rows if r.get("tag") == args.tag]
        qs = comp_feats(rows, args.good_cm, args.good_deg)
        a = blob.get("args", {})
        print(f"\n=== {Path(d).name}  scene={a.get('scene')} "
              f"map={a.get('map')} ex-anchors={a.get('val_exclude_anchors')} "
              f"ex-seq={a.get('val_exclude_seq')}  n={len(qs)}"
              + (f"  tag={args.tag}" if args.tag else ""))
        rep = np.array([q["rep_cm"] for q in qs])
        repd = np.array([q["rep_deg"] for q in qs])
        print(f"reported: median {np.median(rep):.2f} cm  "
              f">{args.good_cm:g}cm {100 * (rep > args.good_cm).mean():.0f}%  "
              f"SR {100 * ((rep <= args.good_cm) & (repd <= args.good_deg)).mean():.0f}%")
        cats, det = diagnose(qs, args.good_cm, args.good_deg)
        n = max(det["n_bad"], 1)
        print(f"FAILURES n={det['n_bad']}")
        print(f"  (i)   never proposed    {cats['never_proposed']:3d} "
              f"({100 * cats['never_proposed'] / n:3.0f}%)  "
              f"[no anchor within 60 cm: {det['anchor_far']}]")
        print(f"  (ii)a proposed, lost    {cats['proposed_lost']:3d} "
              f"({100 * cats['proposed_lost'] / n:3.0f}%)  "
              f"[evicted before weighting {det['evicted']}, "
              f"refinement drifted off it {det['refine_drift']}; "
              f"a snap failed render-verification in {det['verify_failed']}]")
        print(f"  (ii)b out-weighted      {cats['outweighted']:3d} "
              f"({100 * cats['outweighted'] / n:3.0f}%)")
        print(f"  (iii) condensation      {cats['condensation']:3d} "
              f"({100 * cats['condensation'] / n:3.0f}%)")
        # uncertainty: does the reported covariance flag the failures?
        sig = np.array([q["sigma_t"] for q in qs])
        bad = rep > args.good_cm
        if bad.any() and (~bad).any():
            print(f"  sigma_t: good median {np.median(sig[~bad]):.4f} m, "
                  f"bad median {np.median(sig[bad]):.4f} m, "
                  f"AUC(sigma ranks error) "
                  f"{_auc(sig, bad):.2f}")
        print("  covariance vs. correspondence quality "
              "(Sigma *= (0.9/inlier ratio)^p, tau refit two-fold):")
        sigma_section(qs, args.good_cm)
        print("  rescoring (argmax over the SAME kept components):")
        base = None
        for row in score_rules(qs, args.good_cm, args.good_deg):
            if base is None:
                base = row["cm"]
            bet = int((row["cm"] < base - 1e-9).sum())
            wor = int((row["cm"] > base + 1e-9).sum())
            print(f"    {row['rule']:<30s} med {row['median']:5.2f}  "
                  f">{args.good_cm:g}cm {100 * row['over5']:3.0f}%  "
                  f"SR {100 * row['sr']:3.0f}%   better/worse {bet}/{wor}")
        if args.simulate:
            print("  SLOT-RACE simulation (snap poses, no refinement):")
            base_sim = None
            for rank in ("psnr", "inliers", "inl_cov"):
                for tk in (3, 5):
                    for wname in ("base", "cov2", "ratio2", "oracle"):
                        if wname != "base" and (rank != "psnr" or tk != 3):
                            continue
                        s = simulate(rows, args.good_cm, args.good_deg,
                                     top_k=tk, rank=rank, weight=wname)
                        if base_sim is None:
                            base_sim = s["cm"]
                        bet = int((s["cm"] < base_sim - 1e-9).sum())
                        wor = int((s["cm"] > base_sim + 1e-9).sum())
                        print(f"    rank={rank:<8s} top_k={tk} w={wname:<7s} "
                              f"n={s['n']:3d} med {s['median']:5.2f}  "
                              f">{args.good_cm:g}cm {100 * s['over5']:3.0f}%  "
                              f"SR {100 * s['sr']:3.0f}%  better/worse {bet}/{wor}")
        if args.per_query:
            for q in qs:
                if q["rep_cm"] > args.good_cm:
                    print(f"    {q['seq']}/{q['frame_id']:06d} rep {q['rep_cm']:8.2f} "
                          f"comp {np.array2string(q['cm'], precision=1)} "
                          f"w {np.array2string(q['w'], precision=2)} "
                          f"inl {np.array2string(q['inl'], precision=0)} "
                          f"cov {np.array2string(q['cov'], precision=2)} "
                          f"snap_min {q['snap_cm'].min() if q['snap_cm'].size else float('nan'):.1f}")


def _auc(score, label):
    """AUC of `score` ranking the boolean `label` (1 = failure)."""
    o = np.argsort(score)
    s, l = score[o], label[o]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = l.sum(), (~l).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    main()
