#!/usr/bin/env python3
"""Fill the USGLoc row of the letter's 7Scenes table (tab:7scenes) with the per-scene median over seeds.

Per scene and seed run: median t (cm) / r (deg) and the 5cm/5deg and 2cm/2deg success rates
(scripts/make_tables.stats); each cell is the median over the given runs. Avg. is the mean of the
per-scene medians, the rates are the mean of the per-scene rates. The best/second marks of the whole
table are recomputed (fill_table2_12scenes.mark_best).
    python scripts/analysis/fill_table1_7scenes.py results/v2_7scenes_seed0 results/v2_7scenes_seed1 results/v2_7scenes_seed2
"""
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
ROW = "\\textbf{\\method{} (Ours, prior-free)}"


def main():
    runs = [ROOT / d for d in sys.argv[1:]] or [ROOT / "results/v2_7scenes_seed0"]
    res = {}
    for s in SCENES:
        st = [stats(json.load(open(r / f"7scenes_{s}.json"))) for r in runs]
        assert all(x["n"] == 100 for x in st), f"{s}: a run has fewer than 100 scored records"
        res[s] = [float(np.median([x[k] for x in st])) for k in ("med_t", "med_r", "sr55", "sr22")]
    v = np.array([res[s] for s in SCENES])
    cells = [f"{t:.2f}/{r:.2f}" for t, r, _, _ in v]
    cells += [f"{v[:, 0].mean():.2f}/{v[:, 1].mean():.2f}", f"{100 * v[:, 2].mean():.2f}", f"{100 * v[:, 3].mean():.2f}"]
    tex = TEX.read_text()
    a = tex.rfind("\\begin{table*}", 0, tex.index("\\label{tab:7scenes}"))
    b = tex.index("\\end{table*}", a)
    blk = tex[a:b]
    pos = blk.index(ROW); head = blk.index(" & ", pos) + 3; end = blk.index("\\\\", pos)
    old = blk[head:end]
    blk = blk[:head] + " & ".join(cells) + blk[end:]
    blk = mark_best(blk, higher_better=(8, 9))       # 7 scenes + Avg are medians; then SR 5/5, 2/2
    TEX.write_text(tex[:a] + blk + tex[b:])
    print(f"{len(runs)} run(s): " + ", ".join(str(r.relative_to(ROOT)) for r in runs))
    print("old: " + old)
    print("new: " + " & ".join(cells))
    for s in SCENES:
        t, r, c, d = res[s]
        print(f"  {s:11s} {t:.3f}/{r:.3f}  SR5 {100 * c:.0f}  SR2 {100 * d:.0f}")


if __name__ == "__main__":
    main()
