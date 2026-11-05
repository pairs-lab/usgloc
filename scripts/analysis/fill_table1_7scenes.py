#!/usr/bin/env python3
"""Fill the USGLoc row (or, with --row posenet, the PoseNet row) of the letter's 7Scenes table (tab:7scenes).

USGLoc: per scene and seed run, median t (cm) / r (deg) and the 5cm/5deg and 2cm/2deg success rates
(scripts/make_tables.stats); each cell is the median over the given runs.
PoseNet: one run of scripts/analysis/train_posenet.py on the same 100 queries as our seed-0 run (checked);
success counts t < thr and r < thr, as for the other baselines (table1_sr.py, fill_table2_12scenes.py).
Avg. is the mean of the per-scene medians, the rates are the mean of the per-scene rates. The best/second
marks of the whole table are recomputed (fill_table2_12scenes.mark_best).
    python scripts/analysis/fill_table1_7scenes.py results/v2_7scenes_seed0 results/v2_7scenes_seed1 results/v2_7scenes_seed2
    python scripts/analysis/fill_table1_7scenes.py --row posenet results/posenet_ours_100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from make_tables import stats  # noqa: E402
from fill_table2_12scenes import mark_best  # noqa: E402

TEX = ROOT / "paper/usgloc_ral_draft.tex"
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
ROWS = {"ours": "\\textbf{\\method{} (Ours, prior-free)}", "posenet": "PoseNet~\\cite{posenet}"}
REF = ROOT / "results/v2_7scenes_seed0"        # the query list of every row


def keys(j):
    return [(r["seq"], r["frame_id"]) for r in j["records"]]


def ours(runs, s):
    st = [stats(json.load(open(r / f"7scenes_{s}.json"))) for r in runs]
    assert all(x["n"] == 100 for x in st), f"{s}: a run has fewer than 100 scored records"
    return [float(np.median([x[k] for x in st])) for k in ("med_t", "med_r", "sr55", "sr22")]


def posenet(runs, s):
    assert len(runs) == 1, "--row posenet takes one run"
    j = json.load(open(runs[0] / f"7scenes_{s}.json"))
    assert keys(j) == keys(json.load(open(REF / f"7scenes_{s}.json"))), f"{s}: not our 100 queries"
    t = np.array([100 * r["t_err"] for r in j["records"]]); r = np.array([r["r_err"] for r in j["records"]])
    return [float(np.median(t)), float(np.median(r)), float(np.mean((t < 5) & (r < 5))), float(np.mean((t < 2) & (r < 2)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="ours", choices=list(ROWS))
    ap.add_argument("runs", nargs="*", default=["results/v2_7scenes_seed0"])
    a = ap.parse_args()
    runs = [ROOT / d for d in a.runs]
    res = {s: (ours if a.row == "ours" else posenet)(runs, s) for s in SCENES}
    v = np.array([res[s] for s in SCENES])
    cells = [f"{t:.2f}/{r:.2f}" for t, r, _, _ in v]
    cells += [f"{v[:, 0].mean():.2f}/{v[:, 1].mean():.2f}", f"{100 * v[:, 2].mean():.2f}", f"{100 * v[:, 3].mean():.2f}"]
    tex = TEX.read_text()
    a_ = tex.rfind("\\begin{table*}", 0, tex.index("\\label{tab:7scenes}"))
    b_ = tex.index("\\end{table*}", a_)
    blk = tex[a_:b_]
    pos = blk.index(ROWS[a.row]); head = blk.index(" & ", pos) + 3; end = blk.index("\\\\", pos)
    old = blk[head:end]
    blk = blk[:head] + " & ".join(cells) + blk[end:]
    blk = mark_best(blk, higher_better=(8, 9))       # 7 scenes + Avg are medians; then SR 5/5, 2/2
    TEX.write_text(tex[:a_] + blk + tex[b_:])
    print(f"row {a.row}, {len(runs)} run(s): " + ", ".join(str(r.relative_to(ROOT)) for r in runs))
    print("old: " + old)
    print("new: " + " & ".join(cells))
    for s in SCENES:
        t, r, c, d = res[s]
        print(f"  {s:11s} {t:.3f}/{r:.3f}  SR5 {100 * c:.0f}  SR2 {100 * d:.0f}")


if __name__ == "__main__":
    main()
