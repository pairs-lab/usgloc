#!/usr/bin/env python3
"""Fill the USGLoc row of the letter's per-scene 12Scenes table (tab:12scenes, one row).

Reads <run>/12scenes_<scene>.json (one per finished scene) and writes median t (cm) / r (deg)
per scene, then Avg. and the 5cm/5deg and 2cm/2deg success rates (mean of the per-scene rates).
A query the run skipped (no record) counts as a failure: infinite error, out of args.queries.
A scene without a result keeps the \\res placeholder; Avg. and the rates are filled only when
all 12 scenes are present.
    python scripts/analysis/fill_table2_12scenes.py --run results/v2_12scenes_seed0
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/usgloc_ral_draft.tex"
SCENES = ["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", "apt2_living", "apt2_luke",
          "office1_gates362", "office1_gates381", "office1_lounge", "office1_manolis",
          "office2_5a", "office2_5b"]
ROW = "\\textbf{\\method{} (Ours)} & "


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="results/v2_12scenes_seed0")
    run = ROOT / ap.parse_args().run
    res = {}
    for s in SCENES:
        f = run / f"12scenes_{s}.json"
        if f.exists():
            d = json.load(open(f)); R = d["records"]
            miss = max(0, int(d["args"]["queries"]) - len(R))
            t = np.array([100 * r["t_err"] for r in R] + [np.inf] * miss)
            r = np.array([r["r_err"] for r in R] + [np.inf] * miss)
            res[s] = (np.median(t), np.median(r), 100 * np.mean((t < 5) & (r < 5)),
                      100 * np.mean((t < 2) & (r < 2)), miss)
    tex = TEX.read_text()
    a = tex.rfind("\\begin{table*}", 0, tex.index("\\label{tab:12scenes}")); b = tex.index("\\end{table*}", a)
    blk = tex[a:b]
    pos = blk.index(ROW); end = blk.index("\\\\", pos)
    cells = [f"{res[s][0]:.2f}/{res[s][1]:.2f}" if s in res else "\\res" for s in SCENES]
    if len(res) == 12:
        v = np.array([res[s][:4] for s in SCENES])
        cells += [f"{v[:, 0].mean():.2f}/{v[:, 1].mean():.2f}", f"{v[:, 2].mean():.2f}", f"{v[:, 3].mean():.2f}"]
    else:
        cells += ["\\res"] * 3
    blk = blk[:pos] + ROW + " & ".join(cells) + blk[end:]
    TEX.write_text(tex[:a] + blk + tex[b:])
    print(f"{len(res)}/12 scenes filled")
    for s, (t, r, c, d, m) in res.items():
        print(f"  {s:18s} {t:.2f}/{r:.2f}  SR5 {c:.1f}  SR2 {d:.1f}" + (f"  ({m} skipped)" if m else ""))


if __name__ == "__main__":
    main()
