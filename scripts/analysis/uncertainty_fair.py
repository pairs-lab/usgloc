#!/usr/bin/env python3
r"""Stage-0 gate of docs/accuracy_strategy_2026-10.md: is the posterior a
contribution once it faces FAIR baselines?

The letter's uncertainty claims (AUSE 0.063 vs 0.166, Fig. "unc") rank OUR
errors with our sigma_t and ACE's errors with ACE's inlier count.  The two
error distributions differ (96/700 queries beyond 5 cm for us, 20/700 for
ACE), so that comparison partly measures bimodality, not signal quality.
This script asks the three questions a reviewer will ask, each on identical
query frames, numpy + scipy only (no GPU, no torch):

(A) SAME POSES.  Does sigma_t rank OUR errors better than the confidence any
    PnP-based pipeline gets for free -- the inlier count of the dominant
    hypothesis (``inliers``) or of the best one (``max_inliers``) -- both
    already stored in every run record?  Disagreement with ACE is shown as
    an external reference signal.

(B) ACE'S POSES.  How well do ACE's own signals rank ACE's errors: its inlier
    count, its PnP covariance (``Sigma_pnp``, written by the updated
    ``run_ace_baseline.py``), disagreement with our pose, and the Mahalanobis
    distance of ACE's pose under OUR posterior?

(C) CALIBRATION REPAIR.  Coverage at 68.3/95.4/99.7 %, ECE and the NLL of the
    ground truth for: raw, the stored train-side temperature (the paper's
    protocol), and three leave-one-scene-out (LOSO) schemes fitted on the six
    OTHER scenes' test queries -- the median-rule temperature, split-conformal
    on d^2, and a defensive mixture (1-eps) N(0, tau Sigma) + eps N(0, Sigma_w)
    with an isotropic failure component Sigma_w = diag(a^2 I3, b^2 I3).  ACE's
    PnP covariance gets the same LOSO temperature and the same mixture, so the
    NLL comparison is like-for-like.  LOSO uses other scenes' TEST labels:
    disclose it next to the train-side row.

sigma_t is ranked TEMPERED by each scene's stored train-side tau, as the
paper's AUSE 0.063 (pooled_calib.py) and AUC 0.952 (uncertainty_benefit.py)
are; the untempered row is printed alongside.  ACE's PnP sigma_t is tempered
by the same median rule, fitted leave-one-scene-out on ACE's own d^2.

Ranking metrics: AUSE exactly as the paper (first 90 % removed, 91-point
grid, each curve normalised by its full-set mean; ``pooled_calib.py``),
E-AURC (excess area under the risk-coverage curve; Geifman et al. 2019) at
1/2/5 cm and on the continuous error, Spearman rho (all queries and the
< 5 cm subset), ROC AUC for > 5 cm failures.  Ties (inlier counts) are broken
at random and averaged.  Confidence intervals: paired bootstrap stratified by
scene.  Relative improvements are (baseline - ours) / baseline, so positive
means sigma_t is better.

The pass/kill rules are spelled out in verdict(); the kill rules are
equivalence tests ("ACE is within 10 %"), so an insignificant advantage is
INCONCLUSIVE, never KILL.  Across methods, E-AURC is normalised by each
method's random-order E-AURC on its own errors.

NLL is compared as a paired DIFFERENCE in nats per query (a relative % is
undefined for log-likelihoods): "ours beats ACE" means the 95 % CI of
NLL_ACE - NLL_ours lies above zero.  The gate uses the LOSO defensive mixture
for BOTH methods: under a single tempered Gaussian the mean NLL is dominated
by gross failures and would mostly compare failure rates (the LOSO-tau
difference is still printed).

    python3 scripts/analysis/uncertainty_fair.py results/v2_seed0 \
        --ace results/ace_dsacstar_100_cov --json results/fair_v2_seed0.json
    python3 scripts/analysis/uncertainty_fair.py results/casc_rgbfix --ace ...

A run is either ``<dir>`` holding ``7scenes_<scene>.json`` or a pattern with
``%s`` for per-scene directories (as in ``pooled_calib.py``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_unc_np import (gaussian_logpdf, mahalanobis_sq, project_pd,  # noqa: E402
                         relative_tangent, translation_error)

SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
DOF = 6
LEVELS = (0.683, 0.954, 0.997)
ECE_GRID = np.linspace(0.01, 0.999, 120)        # pooled_calib.py's grid
AUSE_FR = np.linspace(0.0, 0.9, 91)             # pooled_calib.py's grid
EAURC_THR = (1.0, 2.0, 5.0, None)               # cm; None = continuous error
TIE_REPS = 32

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _run_path(run, scene):
    return (run % scene if "%s" in run else run) + f"/7scenes_{scene}.json"


def load_run(run, scenes):
    """Per-query arrays of one run, recomputing d2 / logdet / e from the
    stored pose, covariance and ground truth when they are present."""
    rows, meta, max_dev = [], {}, 0.0
    for s in scenes:
        p = _run_path(run, s)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        a = j.get("args", {})
        meta[s] = {"tau": float(j.get("tau", 1.0)), "n": len(j["records"]),
                   "wh": f"{a.get('width', '?')}x{a.get('height', '?')}",
                   "matcher": a.get("corr_matcher", "?"),
                   "warm": a.get("corr_stage_warm", "-"),
                   "seed": a.get("seed", "?"),
                   "corrector_query": j.get("corrector_query", "unrecorded"),
                   # >0: tau fitted on train-side frames (paper protocol);
                   # 0 / absent: the legacy --temperature-split, i.e. on this
                   # run's own test queries (in-sample for the stored tau row)
                   "tcf": int(a.get("temp_calib_frames", 0) or 0),
                   "dslam_gt": "sfm_pose_dir" in a and not a.get("sfm_pose_dir")}
        for r in j["records"]:
            row = {"scene": s, "key": (r["seq"], int(r["frame_id"])),
                   "t_err": 100.0 * float(r["t_err"]),
                   "sigma_t": float(r["sigma_t"]),
                   "d2": float(r["d2"]), "logdet": float(r["logdet"]),
                   # NaN when absent (older schema): the baseline is then
                   # reported as not evaluated instead of a constant ranking
                   "inliers": float(r["inliers"]) if "inliers" in r else float("nan"),
                   "max_inliers": (float(r["max_inliers"]) if "max_inliers" in r
                                   else float("nan")),
                   "tau_train": meta[s]["tau"], "T": None, "T_gt": None,
                   "Sigma": None, "e": None}
            if r.get("T") is not None and r.get("Sigma") is not None \
                    and r.get("T_gt") is not None:
                T, Tg = np.asarray(r["T"], float), np.asarray(r["T_gt"], float)
                S = project_pd(np.asarray(r["Sigma"], float))
                e = relative_tangent(T, Tg)
                d2 = mahalanobis_sq(e, S)
                max_dev = max(max_dev, abs(d2 - row["d2"]) / max(1.0, abs(row["d2"])))
                row.update(T=T, T_gt=Tg, Sigma=S, e=e, d2=d2,
                           logdet=float(np.linalg.slogdet(S)[1]))
            rows.append(row)
    return rows, meta, max_dev


def load_ace(ace_dir, scenes):
    out = {}
    for s in scenes:
        p = f"{ace_dir}/7scenes_{s}.json"
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["records"]:
            if r.get("failed") or r.get("T") is None:
                continue
            S = r.get("Sigma_pnp")
            out[(s, r["seq"], int(r["frame_id"]))] = {
                "t_err": 100.0 * float(r["t_err"]),
                "inliers": float(r.get("inliers", 0)),
                "T": np.asarray(r["T"], float),
                "Sigma": None if S is None else project_pd(np.asarray(S, float))}
    return out


# --------------------------------------------------------------------------
# ranking metrics (order = most uncertain first, random tie-break)
# --------------------------------------------------------------------------
def _order(unc, rng):
    return np.lexsort((rng.random(len(unc)), -np.asarray(unc, float)))


def _sparse_curve(err, order):
    e = err[order]
    n = len(e)
    suffix = np.cumsum(e[::-1])[::-1]
    ks = (AUSE_FR * n).astype(int)
    return suffix[ks] / (n - ks)


def ause_o(err, order):
    cp = _sparse_curve(err, order)
    co = _sparse_curve(err, np.argsort(-err, kind="stable"))
    return float(_trapz(cp / cp[0] - co / co[0], AUSE_FR))


def _aurc(loss_sorted):
    return float((np.cumsum(loss_sorted) / np.arange(1, len(loss_sorted) + 1)).mean())


def eaurc_o(err, order, thr):
    loss = (err > thr).astype(float) if thr is not None else err
    conf_first = order[::-1]                       # most confident first
    return _aurc(loss[conf_first]) - _aurc(np.sort(loss))


def neaurc_o(err, order, thr):
    """E-AURC divided by the E-AURC of a random order on the SAME errors
    (E[AURC | random] = mean loss exactly): 0 = oracle, ~1 = random.  Unlike
    raw E-AURC it is comparable across methods with different failure rates;
    NaN when the loss has no contrast (no query, or every query, above thr)."""
    loss = (err > thr).astype(float)
    ex_rand = loss.mean() - _aurc(np.sort(loss))
    return eaurc_o(err, order, thr) / ex_rand if ex_rand > 0 else float("nan")


def _tie_avg(fn, unc, err, rng, reps=TIE_REPS):
    unc = np.asarray(unc, float)
    reps = reps if len(np.unique(unc)) < len(unc) else 1
    return float(np.mean([fn(err, _order(unc, rng)) for _ in range(reps)]))


def auroc(unc, err, thr=5.0):
    lab = err > thr
    if lab.all() or not lab.any():
        return float("nan")
    r = rankdata(unc)
    npos, nneg = lab.sum(), (~lab).sum()
    return float((r[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def rank_metrics(unc, err, rng):
    unc, err = np.asarray(unc, float), np.asarray(err, float)
    m = {"ause": _tie_avg(ause_o, unc, err, rng)}
    for thr in EAURC_THR:
        k = "eaurc_cont" if thr is None else f"eaurc_{thr:g}cm"
        m[k] = _tie_avg(lambda e, o, t=thr: eaurc_o(e, o, t), unc, err, rng)
    m["spearman"] = float(spearmanr(unc, err).correlation)
    ok = err < 5.0
    m["spearman_lt5"] = (float(spearmanr(unc[ok], err[ok]).correlation)
                         if ok.sum() > 3 else float("nan"))
    m["auroc_5cm"] = auroc(unc, err)
    return m


def _strat_idx(groups, rng):
    return np.concatenate([rng.choice(np.where(groups == g)[0],
                                      (groups == g).sum(), replace=True)
                           for g in np.unique(groups)])


def boot_rel(unc_ours, unc_base, err_ours, err_base, groups, metric, B, rng):
    """Paired, scene-stratified bootstrap of (base - ours) / base."""
    def one(idx):
        if metric == "ause":
            mo = ause_o(err_ours[idx], _order(unc_ours[idx], rng))
            mb = ause_o(err_base[idx], _order(unc_base[idx], rng))
        else:
            thr = None if metric == "eaurc_cont" else float(metric[6:-2])
            mo = eaurc_o(err_ours[idx], _order(unc_ours[idx], rng), thr)
            mb = eaurc_o(err_base[idx], _order(unc_base[idx], rng), thr)
        return (mb - mo) / mb if mb > 0 else float("nan")
    vals = np.array([one(_strat_idx(groups, rng)) for _ in range(B)])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------
# calibration models.  Each returns (cov, nll): cov is an (n, L) boolean
# "inside the level-alpha credible region" matrix over LEVELS + ECE_GRID,
# nll the negative log density of the ground truth (NaN when undefined).
# --------------------------------------------------------------------------
ALPHAS = np.concatenate([np.array(LEVELS), ECE_GRID])


def gauss_model(d2, logdet, tau):
    u = chi2.cdf(d2 / tau, DOF)
    return u[:, None] <= ALPHAS[None, :], -gaussian_logpdf(d2, logdet, tau)


def tau_median(d2):
    return float(np.median(d2) / chi2.ppf(0.5, DOF))


def conformal_model(d2_train, d2):
    """Split-conformal on d^2: q_alpha = the ceil((n+1) alpha)-th smallest
    calibration score (finite-sample valid under exchangeability)."""
    s = np.sort(d2_train)
    n = len(s)
    k = np.ceil((n + 1) * ALPHAS).astype(int)
    q = np.where(k <= n, s[np.clip(k - 1, 0, n - 1)], np.inf)
    return d2[:, None] <= q[None, :], np.full(len(d2), np.nan)


def mix_terms(E, Sig):
    """Per-query sufficient statistics of the defensive mixture: the raw
    Mahalanobis d2 under Sigma, log|Sigma|, and the squared norms of the
    translation / rotation parts of e.  E (n, 6), Sig (n, 6, 6)."""
    d2 = np.einsum("ni,ni->n", E, np.linalg.solve(Sig, E[..., None])[..., 0])
    return (d2, np.linalg.slogdet(Sig)[1], (E[:, :3] ** 2).sum(1),
            (E[:, 3:] ** 2).sum(1))


def mix_logpdf(terms, tau, eps, a, b):
    """log[(1-eps) N(e; 0, tau Sigma) + eps N(e; 0, diag(a^2 I3, b^2 I3))]."""
    d2, ld, nt2, nr2 = terms
    l1 = gaussian_logpdf(d2, ld, tau)
    l2 = -0.5 * (nt2 / a ** 2 + nr2 / b ** 2 + 6 * np.log(a) + 6 * np.log(b)
                 + DOF * math.log(2 * math.pi))
    return np.logaddexp(np.log1p(-eps) + l1, np.log(eps) + l2)


def fit_mixture(terms, tau):
    """MLE of (eps, a, b) with the temperature held fixed."""
    def nll(th):
        eps = 1.0 / (1.0 + np.exp(-th[0]))
        return -mix_logpdf(terms, tau, eps, np.exp(th[1]), np.exp(th[2])).sum()
    best = None
    for a0 in (0.05, 0.3, 1.0):
        for e0 in (0.02, 0.15):
            r = minimize(nll, [math.log(e0 / (1 - e0)), math.log(a0), math.log(0.1)],
                         method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    th = best.x
    return (float(1.0 / (1.0 + np.exp(-th[0]))), float(np.exp(th[1])),
            float(np.exp(th[2])))


def mixture_model(E, Sig, tau, params, rng, mc):
    """HPD coverage by Monte Carlo: the truth lies in the level-alpha HPD
    region iff P(p(X) > p(e_true)) <= alpha (the exact chi2 rule for a single
    Gaussian)."""
    eps, a, b = params
    lp_true = mix_logpdf(mix_terms(E, Sig), tau, eps, a, b)
    scale_w = np.array([a] * 3 + [b] * 3)
    u = np.empty(len(E))
    for i in range(len(E)):
        L = np.linalg.cholesky(tau[i] * Sig[i])
        z = rng.standard_normal((mc, 6))
        wide = rng.random(mc) < eps
        X = np.where(wide[:, None], z * scale_w, z @ L.T)
        Si = np.linalg.inv(Sig[i])
        terms = (np.einsum("ni,ij,nj->n", X, Si, X),
                 np.full(mc, np.linalg.slogdet(Sig[i])[1]),
                 (X[:, :3] ** 2).sum(1), (X[:, 3:] ** 2).sum(1))
        u[i] = (mix_logpdf(terms, tau[i], eps, a, b) > lp_true[i]).mean()
    return u[:, None] <= ALPHAS[None, :], -lp_true


def summarize_cal(cov, nll):
    c = cov.mean(0)
    out = {f"cov{int(round(1000 * lv)) / 10:g}": 100.0 * float(c[i])
           for i, lv in enumerate(LEVELS)}
    out["ece"] = float(np.mean(np.abs(c[len(LEVELS):] - ECE_GRID)))
    out["nll_mean"] = float(np.nanmean(nll)) if np.isfinite(nll).any() else float("nan")
    out["nll_median"] = float(np.nanmedian(nll)) if np.isfinite(nll).any() else float("nan")
    return out


def loso_models(scenes_of, d2, logdet, E, Sig, tau_train, rng, mc, with_mix,
                train_label="train-side tau"):
    """Raw / stored tau / LOSO tau / LOSO conformal / LOSO mixture.  Rows a
    LOSO scheme cannot fit (a single scene: nothing to leave out) stay NaN, so
    their coverage reads NaN rather than a fabricated 0 %."""
    n, L = len(d2), len(ALPHAS)
    res = {"raw": gauss_model(d2, logdet, np.ones(n))}
    if tau_train is not None:
        res[train_label] = gauss_model(d2, logdet, tau_train)
    cov_t, nll_t = np.full((n, L), np.nan), np.full(n, np.nan)
    cov_c = np.full((n, L), np.nan)
    cov_m, nll_m = np.full((n, L), np.nan), np.full(n, np.nan)
    taus, mix_params = {}, {}
    for s in np.unique(scenes_of):
        te, tr = scenes_of == s, scenes_of != s
        if not tr.any():
            continue
        tau = tau_median(d2[tr])
        taus[s] = tau
        cov_t[te], nll_t[te] = gauss_model(d2[te], logdet[te], np.full(te.sum(), tau))
        cov_c[te] = conformal_model(d2[tr], d2[te])[0]
        if with_mix:
            p = fit_mixture(mix_terms(E[tr], Sig[tr]), tau)
            mix_params[s] = p
            cov_m[te], nll_m[te] = mixture_model(E[te], Sig[te], np.full(te.sum(), tau),
                                                 p, rng, mc)
    res["LOSO tau"] = (cov_t, nll_t)
    res["LOSO conformal"] = (cov_c, np.full(n, np.nan))
    if with_mix:
        res["LOSO mixture"] = (cov_m, nll_m)
    return res, taus, mix_params


def boot_diff(a, b, groups, B, rng):
    """CI of mean(a - b), paired, scene-stratified."""
    d = a - b
    if not np.isfinite(d).any():            # e.g. LOSO with a single scene
        return float("nan"), float("nan"), float("nan")
    vals = [np.nanmean(d[_strat_idx(groups, rng)]) for _ in range(B)]
    return (float(np.nanmean(d)), float(np.nanpercentile(vals, 2.5)),
            float(np.nanpercentile(vals, 97.5)))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
INLIER_BASES = ("-inliers (dominant hyp.)", "-max_inliers (best hyp.)")
NLL_EQ = math.log(1.1)      # "within 10 %" on NLL: likelihood ratio <= 1.1 per query
SIG = "sigma_t (ours)"


def _fmt_rank(tag, m):
    return (f"  {tag:<34} AUSE {m['ause']:.3f}  E-AURC 1/2/5cm "
            f"{m['eaurc_1cm']:.4f}/{m['eaurc_2cm']:.4f}/{m['eaurc_5cm']:.4f}  "
            f"cont {m['eaurc_cont']:7.3f}  rho {m['spearman']:+.2f} "
            f"(<5cm {m['spearman_lt5']:+.2f})  AUC {m['auroc_5cm']:.3f}")


def _fmt_cal(tag, m):
    return (f"  {tag:<30} cov {m['cov68.3']:5.1f}/{m['cov95.4']:5.1f}/{m['cov99.7']:5.1f}"
            f"  ECE {m['ece']:.3f}  NLL mean {m['nll_mean']:9.2f}  median {m['nll_median']:8.2f}")


def _neaurc2(unc, err, rng):
    return _tie_avg(lambda e, o: neaurc_o(e, o, 2.0), unc, err, rng)


def analyze(run, ace, args):
    rng = np.random.default_rng(args.seed)
    rows, meta, max_dev = load_run(run, args.scenes)
    if not rows:
        sys.exit(f"no records under {run}")
    rep = {"run": run, "meta": meta}
    print(f"\n{'=' * 100}\nRUN {run}   ({len(rows)} queries)")
    for s, m in meta.items():
        flag = ""
        if m["wh"] != "640x480":
            flag += "  <-- render NOT 640x480"
        if m["matcher"] == "stage" and m["warm"] == "mast3r" and m["corrector_query"] != "rgb":
            flag += "  <-- MASt3R warm-up fed luma (pre-fix run)"
        if m["tcf"] <= 0:
            flag += "  <-- tau fitted on this run's own test queries (legacy split)"
        if m["dslam_gt"]:
            flag += "  <-- DSLAM ground truth (no --sfm-pose-dir)"
        print(f"  {s:<11} n={m['n']:3d}  render {m['wh']:<8} matcher {m['matcher']}"
              f"/{m['warm']}  seed {m['seed']}  tau_train {m['tau']:.2f}{flag}")
    missing_sc = [s for s in args.scenes if s not in meta]
    if missing_sc:
        print(f"  WARNING: no run file for {missing_sc}")
    legacy = any(m["tcf"] <= 0 for m in meta.values())
    tau_label = "stored tau (IN-SAMPLE)" if legacy else "train-side tau"
    have_pose = all(r["e"] is not None for r in rows)
    print(f"  stored-vs-recomputed d2: max rel. deviation {max_dev:.2e}"
          + ("" if have_pose else "  (records without T/Sigma/T_gt: mixture and "
             "ACE comparisons skipped)"))

    sc = np.array([r["scene"] for r in rows])
    err = np.array([r["t_err"] for r in rows])
    tau_tr = np.array([r["tau_train"] for r in rows])
    sig_raw = np.array([r["sigma_t"] for r in rows])
    # The paper ranks by the TEMPERED sigma_t (pooled_calib.py,
    # uncertainty_benefit.py): a per-scene tau leaves each scene's order
    # unchanged but sets the pooled order across scenes.
    sig = sig_raw * np.sqrt(tau_tr)
    inl = np.array([r["inliers"] for r in rows])
    mxi = np.array([r["max_inliers"] for r in rows])
    d2 = np.array([r["d2"] for r in rows])
    ld = np.array([r["logdet"] for r in rows])
    print(f"  median {np.median(err):.2f} cm   beyond 5 cm: {(err > 5).sum()}/{len(err)}")

    # ---- (A) same poses -------------------------------------------------
    print("\n(A) RANKING OUR OWN ERRORS (identical poses; lower AUSE/E-AURC is better)")
    sigs = {SIG: sig, "sigma_t raw (untempered)": sig_raw}
    for name, v in zip(INLIER_BASES, (inl, mxi)):
        if not np.isfinite(v).all():
            print(f"  {name}: missing in some run records -- baseline NOT evaluated")
        elif len(np.unique(v)) == 1:
            print(f"  {name}: constant over the run -- baseline NOT evaluated")
        else:
            sigs[name] = -v
    bases = [b for b in INLIER_BASES if b in sigs]
    paired, dis = None, None
    if ace and have_pose:
        k = [(r["scene"],) + r["key"] for r in rows]
        has = np.array([kk in ace for kk in k])
        if has.any():
            dis = np.full(len(rows), np.nan)
            for i in np.where(has)[0]:
                dis[i] = 100.0 * translation_error(rows[i]["T"], ace[k[i]]["T"])
            paired = has
    A = {t: rank_metrics(u, err, rng) for t, u in sigs.items()}
    for t, m in A.items():
        print(_fmt_rank(t, m))
    if paired is not None:
        t = "disagreement w/ ACE (external)"
        A[t] = rank_metrics(dis[paired], err[paired], rng)
        print(_fmt_rank(t + f" n={paired.sum()}", A[t]))
    print(f"  {'random order (expected)':<34} AUSE "
          f"{np.mean([ause_o(err, rng.permutation(len(err))) for _ in range(64)]):.3f}")
    rel = {}
    for base in bases:
        for met in ("ause", "eaurc_2cm"):
            b, o = A[base][met], A[SIG][met]
            point = (b - o) / b if b > 0 else float("nan")   # undefined: nothing to rank
            lo, hi = boot_rel(sig, sigs[base], err, err, sc, met, args.boot, rng)
            rel[(base, met)] = (point, lo, hi)
            print(f"  sigma_t vs {base:<26} {met:<10} rel. improvement "
                  f"{100 * point:+6.1f}%  [95% CI {100 * lo:+6.1f}, {100 * hi:+6.1f}]")
    rep["A"] = {"metrics": A, "rel": {f"{b}|{m}": v for (b, m), v in rel.items()}}

    cols = [SIG] + bases
    print(f"\n  per scene: AUSE {' / '.join(cols)}   E-AURC(2cm) same order")
    for s in np.unique(sc):
        m = sc == s
        a = [_tie_avg(ause_o, sigs[c][m], err[m], rng) for c in cols]
        b = [_tie_avg(lambda e, o: eaurc_o(e, o, 2.0), sigs[c][m], err[m], rng) for c in cols]
        print(f"  {s:<11} {' / '.join(f'{x:.3f}' for x in a)}     "
              f"{' / '.join(f'{x:.4f}' for x in b)}")

    # ---- ACE pairing: completeness and its PnP covariance -----------------
    nll_cmp = None
    incomplete = []
    have_cov = False
    if paired is not None:
        idx = np.where(paired)[0]
        k = [(rows[i]["scene"],) + rows[i]["key"] for i in idx]
        sci = sc[idx]
        paired_sc = set(sci.tolist())
        incomplete = sorted(set(args.scenes) - paired_sc) + sorted(
            s for s in paired_sc if (sci == s).sum() < 0.9 * meta[s]["n"])
        if incomplete:
            print(f"\n  WARNING: ACE pairing incomplete (missing, or < 90% of frames "
                  f"paired): {incomplete} -- the external test will NOT be scored")
        aerr = np.array([ace[kk]["t_err"] for kk in k])
        ainl = np.array([ace[kk]["inliers"] for kk in k])
        have_cov = all(ace[kk]["Sigma"] is not None for kk in k)
        if have_cov:
            Ea = np.stack([relative_tangent(ace[kk]["T"], rows[i]["T_gt"])
                           for i, kk in zip(idx, k)])
            Sa = np.stack([ace[kk]["Sigma"] for kk in k])
            d2a = np.einsum("ni,ni->n", Ea, np.linalg.solve(Sa, Ea[..., None])[..., 0])
            lda = np.linalg.slogdet(Sa)[1]
            # ACE's sigma_t tempered by OUR rule, fitted leave-one-scene-out on
            # ACE's own d2 (it has no train-side calibration of its own)
            tau_a = np.array([tau_median(d2a[sci != s]) if (sci != s).any() else 1.0
                              for s in sci])
            asig = np.sqrt(np.trace(Sa[:, :3, :3], axis1=1, axis2=2) * tau_a)

    # ---- (B) ACE's poses ------------------------------------------------
    if paired is not None:
        print(f"\n(B) RANKING ACE'S OWN ERRORS (n={len(idx)}; ACE beyond 5 cm: "
              f"{(aerr > 5).sum()}, ours on these frames: {(err[idx] > 5).sum()})")
        bs = {"ACE -inliers": -ainl, "disagreement w/ ours": dis[idx]}
        bs["d2 of T_ACE under our posterior"] = np.array([
            mahalanobis_sq(relative_tangent(rows[i]["T"], ace[kk]["T"]),
                           rows[i]["tau_train"] * rows[i]["Sigma"])
            for i, kk in zip(idx, k)])
        if have_cov:
            bs["ACE PnP sigma_t (LOSO tau)"] = asig
        else:
            print("  (no Sigma_pnp in the ACE records: re-run run_ace_baseline.py "
                  "-- the PnP-covariance baseline is NOT evaluated)")
        B = {t: rank_metrics(u, aerr, rng) for t, u in bs.items()}
        B["ref: sigma_t on OUR errors, same frames"] = rank_metrics(sig[idx], err[idx], rng)
        for t, m in B.items():
            print(_fmt_rank(t, m))
        print("  NB: rows on ACE's errors and on ours rank DIFFERENT error "
              "distributions; compare AUSE/Spearman, read E-AURC with care.")
        rep["B"] = B

    # ---- (C) calibration ------------------------------------------------
    print("\n(C) CALIBRATION (condensed Gaussian; LOSO = fitted on the other scenes' queries)")
    if len(np.unique(sc)) < 2:
        print("  WARNING: one scene only -- leave-one-scene-out schemes are undefined (NaN)")
    E = np.stack([r["e"] for r in rows]) if have_pose else None
    Sig = np.stack([r["Sigma"] for r in rows]) if have_pose else None
    ours, taus, mixp = loso_models(sc, d2, ld, E, Sig, tau_tr, rng, args.mc,
                                   with_mix=have_pose and not args.no_mixture,
                                   train_label=tau_label)
    C = {t: summarize_cal(*v) for t, v in ours.items()}
    for t, m in C.items():
        print(_fmt_cal("ours " + t, m))
    print("  LOSO tau per scene: " + "  ".join(f"{s} {t:.1f}" for s, t in taus.items()))
    if mixp:
        print("  LOSO mixture (eps, a[m], b[rad]): " + "  ".join(
            f"{s} ({p[0]:.2f},{p[1]:.2f},{p[2]:.3f})" for s, p in mixp.items()))
    rep["C"] = {"ours": C, "loso_tau": taus,
                "loso_mixture": {s: list(p) for s, p in mixp.items()}}

    if paired is not None and have_cov:
        acem, ataus, _ = loso_models(sci, d2a, lda, Ea, Sa, None, rng, args.mc,
                                     with_mix=not args.no_mixture)
        oursp, _, _ = loso_models(sci, d2[idx], ld[idx], E[idx], Sig[idx], tau_tr[idx],
                                  rng, args.mc, with_mix=not args.no_mixture,
                                  train_label=tau_label)
        print(f"  --- paired with ACE (n={len(idx)}) ---")
        CA = {}
        for t in acem:
            CA[t] = summarize_cal(*acem[t])
            print(_fmt_cal("ACE PnP " + t, CA[t]))
            print(_fmt_cal("ours    " + t, summarize_cal(*oursp[t])))
        print("  ACE LOSO tau per scene: " + "  ".join(f"{s} {t:.1f}" for s, t in ataus.items()))
        nll_cmp = {"incomplete": incomplete}
        for t in ("LOSO tau", "LOSO mixture"):
            if t in acem:
                mean, lo, hi = boot_diff(acem[t][1], oursp[t][1], sci, args.boot, rng)
                nll_cmp[t] = (mean, lo, hi)
                print(f"  NLL_ACE - NLL_ours [{t}]: {mean:+.2f} nats/query "
                      f"[95% CI {lo:+.2f}, {hi:+.2f}]  (positive = ours better)")
        # E-AURC(2 cm) normalised by each method's random-order E-AURC on its
        # OWN errors, so different failure rates do not decide the contest;
        # scenes where either method has no 2 cm contrast are skipped.
        wins, n_eval, skipped = 0, 0, []
        for s in np.unique(sci):
            m = sci == s
            eo, ea = _neaurc2(sig[idx][m], err[idx][m], rng), _neaurc2(asig[m], aerr[m], rng)
            if not (np.isfinite(eo) and np.isfinite(ea)):
                skipped.append(str(s))
                continue
            n_eval += 1
            wins += eo < ea
        eo_p, ea_p = _neaurc2(sig[idx], err[idx], rng), _neaurc2(asig, aerr, rng)
        nll_cmp.update(eaurc2_scene_wins=int(wins), n_scenes=int(n_eval),
                       eaurc2_skipped=skipped,
                       neaurc2_ours=eo_p, neaurc2_ace=ea_p,
                       neaurc2_rel=((ea_p - eo_p) / ea_p if ea_p > 0 else float("nan")))
        print(f"  normalised E-AURC(2cm) (0 = oracle, 1 = random), each on its own errors: "
              f"ours {eo_p:.3f}, ACE PnP {ea_p:.3f}; per scene ours better on "
              f"{wins}/{n_eval} evaluable scenes"
              + (f" (no 2 cm contrast: {', '.join(skipped)})" if skipped else ""))
        rep["C"]["ace"] = CA
        rep["C"]["ace_loso_tau"] = ataus
        rep["C"]["ace_ours_paired"] = {t: summarize_cal(*v) for t, v in oursp.items()}
        rep["C"]["nll_ace_minus_ours"] = nll_cmp

    # ---- verdict ----------------------------------------------------------
    rep["verdict"] = verdict(rel, C, nll_cmp, rep.get("C", {}))
    return rep


def verdict(rel, C, nll_cmp, Cfull):
    """The success / kill rules of docs/accuracy_strategy_2026-10.md, S0-A.

    1. sigma_t vs the inlier count on OUR poses: PASS if >= 20 % better than
       both inlier signals on pooled AUSE and E-AURC(2cm); KILL if some single
       inlier signal comes within 10 % on both metrics.
    2. vs ACE + PnP covariance with the same LOSO repair: PASS if our NLL is
       significantly lower (CI of NLL_ACE - NLL_ours above 0) and our
       normalised E-AURC(2cm) wins >= 5/7 of the evaluable scenes (at least 4
       of them); KILL only if ACE is EQUIVALENT or better -- the CI upper bound
       stays below ln 1.1 nats/query, our pooled normalised E-AURC(2cm) is less
       than 10 % better, and ACE's 95.4 % coverage is closer to nominal.
       Not scored when ACE does not cover every scene.
    3. the best LOSO scheme puts pooled 95.4 % coverage within 5 points.
    PASS needs 1-3; KILL needs a kill rule; anything else is INCONCLUSIVE
    (INCOMPLETE when 1 or 2 could not be evaluated).
    """
    print("\n(V) STAGE-0 GATE")
    bases = [b for b in INLIER_BASES if (b, "ause") in rel]
    if not bases:
        print("  1. sigma_t vs inlier count: NOT EVALUATED (records lack usable inlier counts)")
        c1 = k1 = False
        g1 = False
    else:
        g1 = True
        vals = {met: [rel[(b, met)][0] for b in bases] for met in ("ause", "eaurc_2cm")}
        best = {met: (float("nan") if any(math.isnan(x) for x in v) else min(v))
                for met, v in vals.items()}
        c1 = best["ause"] >= 0.20 and best["eaurc_2cm"] >= 0.20
        k1_by = [b for b in bases
                 if rel[(b, "ause")][0] < 0.10 and rel[(b, "eaurc_2cm")][0] < 0.10]
        k1 = bool(k1_by)
        print(f"  1. sigma_t vs inlier count on OUR poses (worst case over the inlier signals): "
              f"AUSE {100 * best['ause']:+.1f}%, E-AURC(2cm) {100 * best['eaurc_2cm']:+.1f}%  -> "
              + ("PASS (>=20% on both)" if c1 else
                 f"KILL ({k1_by[0]} within 10% on both)" if k1 else "not met"))
    loso = [t for t in ("LOSO tau", "LOSO conformal", "LOSO mixture") if t in C]
    gap = {t: abs(C[t]["cov95.4"] - 95.4) for t in loso if np.isfinite(C[t]["cov95.4"])}
    if not gap:
        c3 = False
        print("  3. LOSO calibration: NOT EVALUATED (needs >= 2 scenes)")
    else:
        tbest = min(gap, key=gap.get)
        c3 = gap[tbest] <= 5.0
        print(f"  3. best LOSO scheme '{tbest}': 95.4% coverage {C[tbest]['cov95.4']:.1f}%  "
              f"-> {'PASS (within 5 points)' if c3 else 'not met'}")
    g2 = nll_cmp is not None and not nll_cmp.get("incomplete")
    if not g2:
        why = ("needs --ace with Sigma_pnp and poses in the run records" if nll_cmp is None
               else f"ACE incomplete for {nll_cmp['incomplete']}")
        print(f"  2. ACE + PnP covariance: NOT EVALUATED ({why})")
        c2 = k2 = False
    else:
        # The NLL test uses the defensive mixture when it was fitted: under a
        # single Gaussian the mean NLL is dominated by the gross failures, so
        # it would mostly compare failure RATES (ACE 20/700 vs ours 96/700),
        # not the quality of the covariance.  Both methods get the identical
        # LOSO repair, so the comparison stays like-for-like.
        tag = "LOSO mixture" if "LOSO mixture" in nll_cmp else "LOSO tau"
        mean, lo, hi = nll_cmp[tag]
        n_ev = nll_cmp["n_scenes"]
        need = math.ceil(5 * n_ev / 7)
        scene_ok = n_ev >= 4 and nll_cmp["eaurc2_scene_wins"] >= need
        c2 = bool(np.isfinite(lo) and lo > 0 and scene_ok)
        ace_cov = Cfull.get("ace", {}).get(tag, {}).get("cov95.4", float("nan"))
        our_cov = Cfull.get("ace_ours_paired", {}).get(tag, {}).get("cov95.4", float("nan"))
        rel_e = nll_cmp.get("neaurc2_rel", float("nan"))
        k2 = bool(np.isfinite(hi) and hi <= NLL_EQ
                  and np.isfinite(rel_e) and rel_e < 0.10
                  and abs(ace_cov - 95.4) < abs(our_cov - 95.4))
        print(f"  2. vs ACE PnP-cov, both with {tag}: NLL_ACE - NLL_ours {mean:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}] nats, normalised E-AURC(2cm) ours better by "
              f"{100 * rel_e:+.1f}% pooled and on {nll_cmp['eaurc2_scene_wins']}/{n_ev} scenes, "
              f"95.4% coverage ACE {ace_cov:.1f} vs ours {our_cov:.1f}  -> "
              f"{'PASS' if c2 else 'KILL (ACE equivalent or better)' if k2 else 'not met'}")
    if k1 or k2:
        v = "KILL"
    elif c1 and c2 and c3:
        v = "PASS"
    elif not (g1 and g2):
        v = "INCOMPLETE"
    else:
        v = "INCONCLUSIVE"
    print(f"  VERDICT: {v}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ace", default="results/ace_dsacstar_100",
                    help="ACE re-run directory (Sigma_pnp needed for the external test)")
    ap.add_argument("--scenes", nargs="+", default=SC)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--mc", type=int, default=2000,
                    help="Monte Carlo samples per query for mixture HPD coverage")
    ap.add_argument("--no-mixture", action="store_true", help="skip the mixture fits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write all numbers here")
    a = ap.parse_args()
    ace = load_ace(a.ace, a.scenes) if a.ace and os.path.isdir(a.ace) else {}
    if not ace:
        print(f"note: no ACE records under {a.ace}; sections (B) and the external test skipped")
    reps = [analyze(r, ace, a) for r in a.runs]
    if a.json:
        def _c(o):
            if isinstance(o, dict):
                return {str(k): _c(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_c(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(_c(reps), open(a.json, "w"), indent=1)
        print(f"\nsaved {a.json}")


if __name__ == "__main__":
    main()
