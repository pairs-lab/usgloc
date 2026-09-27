#!/usr/bin/env python3
"""Every number the two papers quote about the main 7Scenes runs, from one
place, so a configuration change is a re-run of this script rather than a
hunt through the prose.

    python3 scripts/analysis/regen_numbers.py results/final_seed0 [results/final_seed1 ...]

Per-scene medians are the median over the given seed directories (Table I
convention); pooled/paired statistics use the FIRST directory (seed 0), the
convention of every other table. ACE is results/ace_dsacstar_100 (identical
frames). Floors come from results/floor_final_seed0.json when present.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import chi2, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from make_tables import stats  # noqa: E402

SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
ACE = "results/ace_dsacstar_100"


def ause(sig, t):
    fr = np.linspace(0, 0.9, 91); n = len(t); op = np.argsort(-sig); oo = np.argsort(-t)
    cp = np.array([t[op[int(f * n):]].mean() for f in fr]); co = np.array([t[oo[int(f * n):]].mean() for f in fr])
    return float(np.trapz(cp / cp[0] - co / co[0], fr))


def load(d, s):
    return json.load(open(f"{d}/7scenes_{s}.json"))


def main():
    dirs = sys.argv[1:] or ["results/final_seed0"]
    d0 = dirs[0]
    # ---- Table I row (median over seeds) + per-seed rows
    per = {}
    for s in SC:
        st = [stats(load(d, s)) for d in dirs if os.path.exists(f"{d}/7scenes_{s}.json")]
        per[s] = dict(med_t=float(np.median([x["med_t"] for x in st])), med_r=float(np.median([x["med_r"] for x in st])),
                      sr55=float(np.median([x["sr55"] for x in st])), sr22=float(np.median([x["sr22"] for x in st])),
                      le1=float(np.median([x["le1"] for x in st])), n_seeds=len(st), seeds_t=[x["med_t"] for x in st])
    print(f"== Table I row (median over {len(dirs)} seed dir(s)):")
    print("  " + " & ".join(f"{per[s]['med_t']:.2f}/{per[s]['med_r']:.2f}" for s in SC)
          + f" & {np.mean([per[s]['med_t'] for s in SC]):.2f}/{np.mean([per[s]['med_r'] for s in SC]):.2f}")
    print("  SR@5/5: " + ", ".join(f"{s} {100*per[s]['sr55']:.0f}" for s in SC)
          + " | <=1cm: " + ", ".join(f"{s} {100*per[s]['le1']:.0f}" for s in SC))
    print("  per-seed medians: " + "; ".join(f"{s} {[round(v,2) for v in per[s]['seeds_t']]}" for s in SC))
    # ---- seed-0 per-scene calibration + diagnostics
    print(f"\n== per-scene (seed 0 = {d0}): scene med rot SR55 cov68/95/99.7 AUSE ECE snap inl <=1cm t_loc")
    S_all, T_all, D_all, tau_all = [], [], [], {}
    for s in SC:
        j = load(d0, s); st = stats(j); tau = j["tau"]; tau_all[s] = tau
        ece = [m for m in j["metrics"] if m.get("method") == "tempered"][0]["ece"]
        print(f"  {s:11s} {st['med_t']:.2f} {st['med_r']:.2f} {100*st['sr55']:.0f} "
              f"{100*st['cov'][0.683]:.0f}/{100*st['cov'][0.954]:.0f}/{100*st['cov'][0.997]:.0f} {st['ause']:.3f} {ece:.3f} "
              f"{100*st['snap']:.0f} {st['inl']:.0f} {100*st['le1']:.0f} {st['t_loc']:.1f}")
        for r in j["records"]:
            S_all.append(r["sigma_t"] * tau ** 0.5); T_all.append(100 * r["t_err"]); D_all.append(r["d2"] / tau)
    S_all, T_all, D_all = map(np.array, (S_all, T_all, D_all))
    cov = [100 * (D_all <= chi2.ppf(a, 6)).mean() for a in (0.683, 0.954, 0.997)]
    lv = np.linspace(0.01, 0.999, 120); ece = float(np.mean([abs((D_all <= chi2.ppf(a, 6)).mean() - a) for a in lv]))
    print(f"\n== pooled seed 0 (n={len(T_all)}): median {np.median(T_all):.2f} cm | cov {cov[0]:.1f}/{cov[1]:.1f}/{cov[2]:.1f} | AUSE {ause(S_all, T_all):.3f} | ECE {ece:.3f} | >5cm {100*(T_all>5).mean():.0f}% | <=0.5 {100*(T_all<=0.5).mean():.0f}% <=1 {100*(T_all<=1).mean():.0f}%")
    Sr, Dr = [], []
    for s in SC:
        for r in load(d0, s)["records"]:
            Sr.append(r["sigma_t"]); Dr.append(r["d2"])
    Sr, Dr = np.array(Sr), np.array(Dr)
    covr = [100 * (Dr <= chi2.ppf(a, 6)).mean() for a in (0.683, 0.954, 0.997)]
    ecer = float(np.mean([abs((Dr <= chi2.ppf(a, 6)).mean() - a) for a in lv]))
    print(f"   raw: cov {covr[0]:.1f}/{covr[1]:.1f}/{covr[2]:.1f} | AUSE {ause(Sr, T_all):.3f} | ECE {ecer:.3f}")
    # ---- paired ACE comparison (identical frames)
    O, A, So, Ia = [], [], [], []
    for s in SC:
        j = load(d0, s); a = load(ACE, s); tau = j["tau"]
        am = {(r["seq"], r["frame_id"]): r for r in a["records"]}
        for r in j["records"]:
            k = (r["seq"], r["frame_id"])
            if k in am:
                O.append(100 * r["t_err"]); A.append(100 * am[k]["t_err"]); So.append(r["sigma_t"] * tau ** 0.5); Ia.append(-am[k]["inliers"])
    O, A, So, Ia = map(np.array, (O, A, So, Ia))
    ace_med = {s: float(np.median([100 * r["t_err"] for r in load(ACE, s)["records"]])) for s in SC}
    ace_rot = {s: float(np.median([r["r_err"] for r in load(ACE, s)["records"]])) for s in SC}
    print(f"\n== paired vs ACE (n={len(O)}): pooled median ours {np.median(O):.2f} ACE {np.median(A):.2f} | >5cm ours {100*(O>5).mean():.0f}% ACE {100*(A>5).mean():.0f}% | <=0.5 ours {100*(O<=0.5).mean():.0f}% ACE {100*(A<=0.5).mean():.0f}% | <=1 ours {100*(O<=1).mean():.0f}% ACE {100*(A<=1).mean():.0f}%")
    print(f"   ACE per-scene med: " + " ".join(f"{s} {ace_med[s]:.2f}/{ace_rot[s]:.2f}" for s in SC) + f" | avg {np.mean(list(ace_med.values())):.2f}/{np.mean(list(ace_rot.values())):.2f}")
    print(f"   scene verdicts (|d|<0.1 tie): " + " ".join(f"{s}:{'tie' if abs(per[s]['med_t']-ace_med[s])<0.1 else ('ours' if per[s]['med_t']<ace_med[s] else 'ACE')}" for s in SC))
    print(f"   AUSE ours {ause(So, O):.3f} ACE(inliers) {ause(Ia.astype(float), A):.3f} | spearman ours {spearmanr(So, O)[0]:.3f} ACE {spearmanr(Ia, A)[0]:.3f}")
    qo = np.quantile(So, np.linspace(0, 1, 11)); qa = np.quantile(Ia, np.linspace(0, 1, 11))
    do = [float(np.median(O[(So >= qo[i]) & (So <= qo[i + 1])])) for i in range(10)]
    da = [float(np.median(A[(Ia >= qa[i]) & (Ia <= qa[i + 1])])) for i in range(10)]
    print(f"   deciles ours most-conf {do[0]:.2f} least {do[-1]:.2f} span {do[-1]/do[0]:.0f}x | ACE most {da[0]:.2f} least {da[-1]:.2f} span {da[-1]/da[0]:.1f}x")
    # failure/success sigma ratio per scene
    rat = {}
    for s in SC:
        j = load(d0, s); t = np.array([100 * r["t_err"] for r in j["records"]]); sg = np.array([r["sigma_t"] for r in j["records"]])
        if (t > 5).any() and (t <= 5).any():
            rat[s] = float(np.median(sg[t > 5]) / np.median(sg[t <= 5]))
    print("   sigma ratio failures/successes per scene: " + " ".join(f"{s} {v:.1f}x" for s, v in rat.items()))
    # failure sigma percentile
    thr = np.median(S_all[T_all > 5]) if (T_all > 5).any() else np.nan
    print(f"   median sigma of >5cm queries sits at the {100*np.mean(S_all <= thr):.0f}th percentile")
    # ---- headroom
    if os.path.exists("results/floor_final_seed0.json"):
        fl = {r["scene"]: r for r in json.load(open("results/floor_final_seed0.json"))}
        print("\n== headroom (test-frame floors, final maps): scene reported floor gap reg ACE")
        for s in SC:
            print(f"  {s:11s} {per[s]['med_t']:.2f} {fl[s]['floor']:.2f} {per[s]['med_t']-fl[s]['floor']:.2f} {fl[s]['reg']:.0f}% {ace_med[s]:.2f}")
        print(f"  average {np.mean([per[s]['med_t'] for s in SC]):.2f} {np.mean([fl[s]['floor'] for s in SC]):.2f}")
    # ---- timing
    E, L = [], []
    for s in SC:
        for r in load(d0, s)["records"]:
            E.append(r["t_encode_s"]); L.append(r["t_localize_s"])
    print(f"\n== timing seed 0: encode {np.median(E):.2f}s localize {np.median(L):.2f}s (per-scene localize " + " ".join(f"{s} {np.median([r['t_localize_s'] for r in load(d0, s)['records']]):.1f}" for s in SC) + ")")


if __name__ == "__main__":
    main()
