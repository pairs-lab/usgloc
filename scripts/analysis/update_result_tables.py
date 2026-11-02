#!/usr/bin/env python3
"""Rewrite the result tables of both papers from the result files.

Rows are replaced in place (matched by their scene / row label inside the
table block), so captions and layout stay as written. Run with --dry-run to
print every replaced row without writing.

    python3 scripts/analysis/update_result_tables.py results/final_seed0 results/final_seed1 results/final_seed2 [--dry-run]

Table I (letter) and tab:ace's Ours column use the median over the given
seed directories; every other table uses the first directory (seed 0).
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import chi2, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from make_tables import fix_record, stats  # noqa: E402

ROOT = "/home/thanhnc/gauss_ws/src/usgloc"
RAL = f"{ROOT}/paper/usgloc_ral_draft.tex"
FULL = f"{ROOT}/paper/usgloc_supplementary.tex"
SC = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
ACE = "results/ace_dsacstar_100"
# train-side floors of the selected maps (tab:finetune, bold column)
TRAIN_FLOOR = {"chess": 0.39, "fire": 0.37, "heads": 0.36, "office": 1.01, "pumpkin": 0.95, "redkitchen": 0.66, "stairs": 1.24}


def ause(sig, t):
    fr = np.linspace(0, 0.9, 91); n = len(t); op = np.argsort(-sig); oo = np.argsort(-t)
    cp = np.array([t[op[int(f * n):]].mean() for f in fr]); co = np.array([t[oo[int(f * n):]].mean() for f in fr])
    return float(np.trapz(cp / cp[0] - co / co[0], fr))


def load(d, s):
    return json.load(open(f"{ROOT}/{d}/7scenes_{s}.json"))


def replace_row(block, key, new_cells, label_re=None):
    pat = re.compile(r"^(" + (label_re or re.escape(key)) + r"\s*&)[^\n]*\\\\$", re.M)
    m = pat.search(block)
    assert m, f"row not found: {key}"
    return block[:m.start()] + m.group(1) + " " + new_cells + " \\\\" + block[m.end():], m.group(0)


def table_block(s, label):
    i = s.index("\\label{" + label + "}"); a = s.rfind("\\begin{table", 0, i); b = s.index("\\end{table", i)
    return a, b


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]; dry = "--dry-run" in sys.argv
    d0 = dirs[0]
    log = []
    # ---------- per-scene statistics
    per = {}
    for s in SC:
        st = [stats(load(d, s)) for d in dirs if os.path.exists(f"{ROOT}/{d}/7scenes_{s}.json")]
        per[s] = {k: float(np.median([x[k] for x in st])) for k in ("med_t", "med_r", "sr55", "sr22", "le1")}
        per[s]["s0"] = stats(load(d0, s)); per[s]["n"] = len(st)
    avg_t = np.mean([per[s]["med_t"] for s in SC]); avg_r = np.mean([per[s]["med_r"] for s in SC])
    acem = {s: float(np.median([100 * r["t_err"] for r in load(ACE, s)["records"]])) for s in SC}
    acer = {s: float(np.median([r["r_err"] for r in load(ACE, s)["records"]])) for s in SC}
    # pooled seed 0 (sanitised)
    S, T, D, So, O, A, Ia = [], [], [], [], [], [], []
    for s in SC:
        j = load(d0, s); tau = j["tau"]; am = {(r["seq"], r["frame_id"]): r for r in load(ACE, s)["records"]}
        for r0 in j["records"]:
            r = fix_record(r0)
            if r is None:
                continue
            S.append(r["sigma_t"] * tau ** 0.5); T.append(100 * r["t_err"]); D.append(r["d2"] / tau)
            k = (r["seq"], r["frame_id"])
            if k in am:
                So.append(r["sigma_t"] * tau ** 0.5); O.append(100 * r["t_err"]); A.append(100 * am[k]["t_err"]); Ia.append(-am[k]["inliers"])
    S, T, D, So, O, A, Ia = map(np.array, (S, T, D, So, O, A, Ia))
    cov = [100 * (D <= chi2.ppf(a, 6)).mean() for a in (0.683, 0.954, 0.997)]
    lv = np.linspace(0.01, 0.999, 120); ece = float(np.mean([abs((D <= chi2.ppf(a, 6)).mean() - a) for a in lv]))
    Sr, Dr = [], []
    for s in SC:
        for r0 in load(d0, s)["records"]:
            r = fix_record(r0)
            if r is not None:
                Sr.append(r["sigma_t"]); Dr.append(r["d2"])
    Sr, Dr = np.array(Sr), np.array(Dr)
    covr = [100 * (Dr <= chi2.ppf(a, 6)).mean() for a in (0.683, 0.954, 0.997)]
    ecer = float(np.mean([abs((Dr <= chi2.ppf(a, 6)).mean() - a) for a in lv]))
    au_o, au_a = ause(So, O), ause(Ia.astype(float), A)
    rc_o, rc_a = spearmanr(So, O)[0], spearmanr(Ia, A)[0]
    qo = np.quantile(So, np.linspace(0, 1, 11)); qa = np.quantile(Ia, np.linspace(0, 1, 11))
    do = [float(np.median(O[(So >= qo[i]) & (So <= qo[i + 1])])) for i in range(10)]
    da = [float(np.median(A[(Ia >= qa[i]) & (Ia <= qa[i + 1])])) for i in range(10)]
    verdict = {s: ("tie" if abs(per[s]["med_t"] - acem[s]) < 0.1 else ("ours" if per[s]["med_t"] < acem[s] else "ACE")) for s in SC}

    # ---------- letter: Table I Ours row, Table III rows
    s = open(RAL).read()
    row = " & ".join(f"{per[x]['med_t']:.2f}/{per[x]['med_r']:.2f}" for x in SC) + f" & {avg_t:.2f}/{avg_r:.2f}"
    s, old = replace_row(s, None, row, label_re=r"\\textbf\{\\method\{\} \(Ours, prior-free\)\}"); log.append(("RAL Table I", old, row))
    row = f"\\textbf{{{cov[0]:.1f}}} & \\textbf{{{cov[1]:.1f}}} & \\textbf{{{cov[2]:.1f}}} & \\textbf{{{ause(S, T):.3f}}} & \\textbf{{{ece:.3f}}}"
    a3, b3 = table_block(s, "tab:calib"); blk3 = s[a3:b3]
    blk3, old = replace_row(blk3, None, row, label_re=r"\\textbf\{\\method\{\} \(Ours\)\}"); s = s[:a3] + blk3 + s[b3:]; log.append(("RAL Table III ours", old, row))
    row = f"{covr[0]:.1f}  & {covr[1]:.1f}  & {covr[2]:.1f}  & {ause(Sr, T):.3f} & {ecer:.3f}"
    a3, b3 = table_block(s, "tab:calib"); blk3 = s[a3:b3]
    blk3, old = replace_row(blk3, None, row, label_re=r"Raw Laplace \(no \$\\tau\$\)"); s = s[:a3] + blk3 + s[b3:]; log.append(("RAL Table III raw", old, row))
    if not dry:
        open(RAL, "w").write(s)

    # ---------- report tables
    f = open(FULL).read()
    # tab:ace
    a, b = table_block(f, "tab:ace"); blk = f[a:b]
    for x in SC:
        ours = f"{per[x]['med_t']:.2f}/{per[x]['med_r']:.2f}"; ace = f"{acem[x]:.2f}/{acer[x]:.2f}"
        cells = ((f"\\textbf{{{ace}}}" if verdict[x] == "ACE" else ace) + " & " + (f"\\textbf{{{ours}}}" if verdict[x] == "ours" else ours) + f" & {verdict[x]}")
        blk, old = replace_row(blk, x, cells); log.append((f"tab:ace {x}", old, cells))
    aa_t, aa_r = np.mean(list(acem.values())), np.mean(list(acer.values()))
    cells = f"{'\\textbf{' if aa_t < avg_t else ''}{aa_t:.2f}/{aa_r:.2f}{'}' if aa_t < avg_t else ''} & {'\\textbf{' if avg_t < aa_t else ''}{avg_t:.2f}/{avg_r:.2f}{'}' if avg_t < aa_t else ''} & {'ACE' if aa_t < avg_t else 'ours'}"
    blk, old = replace_row(blk, "seven-scene avg", cells); log.append(("tab:ace avg", old, cells))
    pm_o, pm_a = np.median(O), np.median(A)
    cells = f"{'\\textbf{' if pm_a < pm_o else ''}{pm_a:.2f}{'}' if pm_a < pm_o else ''} & {'\\textbf{' if pm_o < pm_a else ''}{pm_o:.2f}{'}' if pm_o < pm_a else ''} & {'ACE' if pm_a < pm_o else ('ours' if pm_o < pm_a else 'tie')}"
    blk, old = replace_row(blk, "pooled median", cells); log.append(("tab:ace pooled", old, cells))
    fo, fa = 100 * (O > 5).mean(), 100 * (A > 5).mean()
    cells = f"\\textbf{{{fa:.0f}\\%}} & {fo:.0f}\\% & ACE"
    blk, old = replace_row(blk, None, cells, label_re=r"queries \$>5\$\\,cm"); log.append(("tab:ace >5", old, cells))
    f = f[:a] + blk + f[b:]
    # tab:headroom (test-frame floors of the final maps)
    fl = {r["scene"]: r for r in json.load(open(f"{ROOT}/results/floor_final_seed0.json"))}
    a, b = table_block(f, "tab:headroom"); blk = f[a:b]
    for x in SC:
        gap = per[x]["med_t"] - fl[x]["floor"]
        cells = f"{per[x]['med_t']:.2f} & {fl[x]['floor']:.2f} & {'$-$' if gap < 0 else ''}{abs(gap):.2f} & {fl[x]['reg']:.0f}\\% & {acem[x]:.2f}"
        blk, old = replace_row(blk, x, cells); log.append((f"tab:headroom {x}", old, cells))
    mf = np.mean([fl[x]["floor"] for x in SC])
    cells = f"{avg_t:.2f} & \\textbf{{{mf:.2f}}} & {avg_t - mf:.2f} & & {aa_t:.2f}"
    blk, old = replace_row(blk, "average", cells); log.append(("tab:headroom avg", old, cells))
    f = f[:a] + blk + f[b:]
    # tab:snaprate (seed 0; floor = train-side floor of the selected map)
    a, b = table_block(f, "tab:snaprate"); blk = f[a:b]
    for x in SC:
        st = per[x]["s0"]
        cells = f"{100*st['snap']:.0f}\\% & {st['med_t']:.2f} & {100*st['le1']:.0f}\\% & {TRAIN_FLOOR[x]:.2f} & {int(round(st['inl'])):,}".replace(",", "{,}")
        blk, old = replace_row(blk, x, cells); log.append((f"tab:snaprate {x}", old, cells))
    f = f[:a] + blk + f[b:]
    # tab:perscene-calib (seed 0, sanitised)
    a, b = table_block(f, "tab:perscene-calib"); blk = f[a:b]
    for x in SC:
        j = load(d0, x); tau = j["tau"]; recs = [fix_record(r) for r in j["records"]]; recs = [r for r in recs if r]
        d2 = np.array([r["d2"] / tau for r in recs]); sg = np.array([r["sigma_t"] * tau ** 0.5 for r in recs]); t = np.array([100 * r["t_err"] for r in recs])
        c = [100 * (d2 <= chi2.ppf(q, 6)).mean() for q in (0.683, 0.954, 0.997)]
        nll = np.mean([0.5 * (r["d2"] / tau + r["logdet"] + 6 * np.log(tau) + 6 * np.log(2 * np.pi)) for r in recs])
        cells = f"{c[0]:.0f} & {c[1]:.0f} & {c[2]:.0f} & {ause(sg, t):.3f} & {nll:.1f}"
        blk, old = replace_row(blk, x, cells); log.append((f"tab:perscene-calib {x}", old, cells))
    f = f[:a] + blk + f[b:]
    # tab:unc-compare
    a, b = table_block(f, "tab:unc-compare"); blk = f[a:b]
    for key, cells in (("AUSE", f"{au_a:.3f} & \\textbf{{{au_o:.3f}}}"),
                       ("rank corr", f"{rc_a:.3f} & \\textbf{{{rc_o:.3f}}}"),
                       ("span over deciles", f"{da[-1]/da[0]:.0f}$\\times$ & \\textbf{{{do[-1]/do[0]:.0f}$\\times$}}"),
                       ("median err., most-confident decile", f"{da[0]:.2f}\\,cm & \\textbf{{{do[0]:.2f}\\,cm}}"),
                       ("median err., least-confident decile", f"{da[-1]:.1f}\\,cm & {do[-1]:.1f}\\,cm")):
        pat = re.compile(r"^(" + re.escape(key) + r"[^&\n]*&)[^\n]*\\\\$", re.M); m = pat.search(blk); assert m, key
        blk = blk[:m.start()] + m.group(1) + " " + cells + " \\\\" + blk[m.end():]; log.append((f"tab:unc-compare {key}", m.group(0), cells))
    f = f[:a] + blk + f[b:]
    # tab:final (final column = Table I medians; +fine-tune from ablation_ftnosupport_* when present)
    a, b = table_block(f, "tab:final"); blk = f[a:b]
    for x in SC:
        m = re.search(r"^(" + x + r"\s*&\s*[0-9.]+\s*&\s*[^&]+&)\s*([^&]+)&\s*[^\n]*\\\\$", blk, re.M); assert m, x
        ft = m.group(2).strip()
        p = f"{ROOT}/results/ablation_ftnosupport_{x}/7scenes_{x}.json"
        if os.path.exists(p):
            ft = f"{stats(json.load(open(p)))['med_t']:.2f}"
        new = m.group(1) + f" {ft} & {per[x]['med_t']:.2f} \\\\"
        blk = blk[:m.start()] + new + blk[m.end():]; log.append((f"tab:final {x}", m.group(0), new))
    m = re.search(r"^(average\s*&\s*[0-9.]+\s*&)[^\n]*\\\\$", blk, re.M); assert m
    agree = [stats(json.load(open(f"{ROOT}/results/support1_seed0/7scenes_{x}.json")))["med_t"] for x in SC if os.path.exists(f"{ROOT}/results/support1_seed0/7scenes_{x}.json")]
    new = m.group(1) + f" {np.mean(agree):.2f} & -- & {avg_t:.2f} \\\\" if len(agree) == 7 else m.group(1) + f" \\res & -- & {avg_t:.2f} \\\\"
    blk = blk[:m.start()] + new + blk[m.end():]; log.append(("tab:final avg", m.group(0), new))
    f = f[:a] + blk + f[b:]
    if not dry:
        open(FULL, "w").write(f)
    for name, old, new in log:
        print(f"[{name}]\n   - {old.strip()[:150]}\n   + {new.strip()[:150]}")
    print(f"\n{'DRY RUN, nothing written' if dry else 'written'} | seeds used for Table I: {len(dirs)} | pooled n={len(T)} | cov {cov[0]:.1f}/{cov[1]:.1f}/{cov[2]:.1f} AUSE {ause(S,T):.3f} ECE {ece:.3f} | paired median ours {pm_o:.2f} ACE {pm_a:.2f} | >5cm {fo:.0f}% vs {fa:.0f}% | AUSE {au_o:.3f} vs {au_a:.3f} | spearman {rc_o:.3f} vs {rc_a:.3f} | deciles {do[0]:.2f}..{do[-1]:.2f} vs {da[0]:.2f}..{da[-1]:.2f}")


if __name__ == "__main__":
    main()
