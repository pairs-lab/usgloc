#!/usr/bin/env python3
"""Fill the USGLoc row of the letter's per-scene 12Scenes table (tab:12scenes, one row).

Reads results/m3u8_12s_r1/12scenes_<scene>.json (one per finished scene; written by the
12Scenes chain) and writes median t (cm) / r (deg) per scene; a scene without a result
keeps the \\res placeholder, and Avg. is filled only when all 12 scenes are present.
    python scripts/analysis/fill_table2_12scenes.py [--run results/m3u8_12s_r1]
"""
import argparse, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/usgloc_ral_draft.tex"
BLOCKS = [["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", "apt2_living", "apt2_luke"],
          ["office1_gates362", "office1_gates381", "office1_lounge", "office1_manolis",
           "office2_5a", "office2_5b"]]
ROW = "\\textbf{\\method{} (Ours)} & "


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="results/m3u8_12s_r1")
    run = ROOT / ap.parse_args().run
    med = {}
    for s in sum(BLOCKS, []):
        f = run / f"12scenes_{s}.json"
        if f.exists():
            R = json.load(open(f))["records"]
            med[s] = (100 * np.median([r["t_err"] for r in R]), np.median([r["r_err"] for r in R]))
    tex = TEX.read_text()
    a = tex.rfind("\\begin{table*}", 0, tex.index("\\label{tab:12scenes}")); b = tex.index("\\end{table*}", a)
    blk = tex[a:b]
    pos = blk.index(ROW); end = blk.index("\\\\", pos)
    scenes = sum(BLOCKS, [])
    cells = [f"{med[s][0]:.2f}/{med[s][1]:.2f}" if s in med else "\\res" for s in scenes]
    cells.append(f"{np.mean([v[0] for v in med.values()]):.2f}/{np.mean([v[1] for v in med.values()]):.2f}"
                 if len(med) == 12 else "\\res")
    blk = blk[:pos] + ROW + " & ".join(cells) + blk[end:]
    TEX.write_text(tex[:a] + blk + tex[b:])
    print(f"{len(med)}/12 scenes filled: " + ", ".join(f"{s} {v[0]:.2f}/{v[1]:.2f}" for s, v in med.items()))


if __name__ == "__main__":
    main()
