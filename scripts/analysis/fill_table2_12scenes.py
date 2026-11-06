#!/usr/bin/env python3
"""Fill the USGLoc row (or, with --row posenet, the PoseNet row) of the letter's per-scene 12Scenes
table (tab:12scenes, one row).

Reads <run>/12scenes_<scene>.json (one per finished scene) and writes median t (cm) / r (deg)
per scene, then Avg. and the 5cm/5deg and 2cm/2deg success rates (mean of the per-scene rates).
A query the run skipped (no record) counts as a failure: infinite error, out of args.queries.
A scene without a result keeps the \\res placeholder; Avg. and the rates are filled only when
all 12 scenes are present.
With --row gsplatloc, the medians stay as printed (ULF-Loc) and only the two success rates are filled
from our runs of GSplatLoc's released code; with --partial MARK they are the mean over the scenes present,
followed by MARK (the caption must name the missing scenes).
    python scripts/analysis/fill_table2_12scenes.py --run results/v2_12scenes_seed0
    python scripts/analysis/fill_table2_12scenes.py --row posenet --run results/posenet_ours_100
    python scripts/analysis/fill_table2_12scenes.py --row gsplatloc --run results/gsplatloc_100 [--partial '$^\\ddagger$']
"""
import argparse, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper/usgloc_ral_draft.tex"
SCENES = ["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", "apt2_living", "apt2_luke",
          "office1_gates362", "office1_gates381", "office1_lounge", "office1_manolis",
          "office2_5a", "office2_5b"]
ROWS = {"ours": "\\textbf{\\method{} (Ours)} & ", "posenet": "PoseNet~\\cite{posenet}",
        "gsplatloc": "GSplatLoc~\\cite{gsplatloc}"}


def mark_best(blk, higher_better=()):
    """Bold the best and underline the second-best value of every column of the
    table body in ``blk`` (rows between \\midrule and \\bottomrule). A "t/r" cell
    is ranked per part, lower is better; columns listed in ``higher_better``
    (0-based, data columns only) are ranked higher-is-better. Ties share a mark
    (dense ranking); "--" and \\res cells are skipped; old marks are removed first."""
    a = blk.index("\\midrule") + len("\\midrule"); b = blk.index("\\bottomrule")
    unmark = lambda c: re.sub(r"\\(?:textbf|underline)\{([0-9.]+)\}", r"\1", c)
    rows = []
    for line in blk[a:b].strip("\n").split("\n"):
        head, _, tail = line.partition(" & ")
        body, _, end = tail.rpartition("\\\\")
        rows.append([head, [unmark(c.strip()) for c in body.split(" & ")], "\\\\" + end])
    num = re.compile(r"^(\d+\.\d+)(?:/(\d+\.\d+))?(.*)$")
    for j in range(len(rows[0][1])):
        parsed = [num.match(r[1][j]) for r in rows]
        marks = []                                   # per part: value -> mark, from the unmarked values
        for part in (1, 2):
            vals = sorted({float(m.group(part)) for m in parsed if m and m.group(part)},
                          reverse=j in higher_better)
            marks.append(dict(zip(vals[:2], ("textbf", "underline"))))
        for r, m in zip(rows, parsed):
            if not m:
                continue
            g = [m.group(1), m.group(2)]
            for part in (0, 1):
                mk = g[part] is not None and marks[part].get(float(g[part]))
                if mk:
                    g[part] = f"\\{mk}{{{g[part]}}}"
            r[1][j] = g[0] + (f"/{g[1]}" if g[1] is not None else "") + m.group(3)
    out = "\n".join(h + " & " + " & ".join(c) + e for h, c, e in rows)
    return blk[:a] + "\n" + out + "\n" + blk[b:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="results/v2_12scenes_seed0")
    ap.add_argument("--row", default="ours", choices=list(ROWS))
    ap.add_argument("--tex", default=str(TEX), help="letter to fill (a copy for a dry run)")
    ap.add_argument("--partial", help="gsplatloc only: success rates over the scenes present, followed by this mark")
    args = ap.parse_args()
    assert args.partial is None or args.row == "gsplatloc", "--partial is for the GSplatLoc row only"
    run = ROOT / args.run
    res = {}
    for s in SCENES:
        f = run / f"12scenes_{s}.json"
        if f.exists():
            d = json.load(open(f)); R = d["records"]
            miss = max(0, int(d.get("args", {}).get("queries", 100)) - len(R))
            t = np.array([100 * r["t_err"] for r in R] + [np.inf] * miss)
            r = np.array([r["r_err"] for r in R] + [np.inf] * miss)
            res[s] = (np.median(t), np.median(r), 100 * np.mean((t < 5) & (r < 5)),
                      100 * np.mean((t < 2) & (r < 2)), miss)
    tex_path = Path(args.tex); tex = tex_path.read_text()
    a = tex.rfind("\\begin{table*}", 0, tex.index("\\label{tab:12scenes}")); b = tex.index("\\end{table*}", a)
    blk = tex[a:b]
    pos = blk.index(ROWS[args.row]); head = blk.index(" & ", pos) + 3; end = blk.index("\\\\", pos)
    cells = [f"{res[s][0]:.2f}/{res[s][1]:.2f}" if s in res else "\\res" for s in SCENES]
    if len(res) == 12:
        v = np.array([res[s][:4] for s in SCENES])
        cells += [f"{v[:, 0].mean():.2f}/{v[:, 1].mean():.2f}", f"{v[:, 2].mean():.2f}", f"{v[:, 3].mean():.2f}"]
    elif args.partial is not None and res:
        v = np.array([res[s][:4] for s in SCENES if s in res])
        cells += ["\\res", f"{v[:, 2].mean():.2f}{args.partial}", f"{v[:, 3].mean():.2f}{args.partial}"]
    else:
        cells += ["\\res"] * 3
    if args.row == "gsplatloc":                        # keep the printed medians, replace the success rates
        old = blk[head:end].split(" & ")
        assert len(old) == 15, len(old)
        cells = old[:13] + cells[13:]
    blk = blk[:head] + " & ".join(cells) + blk[end:]
    blk = mark_best(blk, higher_better=(13, 14))       # 12 scenes + Avg are medians; then SR 5/5, 2/2
    tex_path.write_text(tex[:a] + blk + tex[b:])
    print(f"{len(res)}/12 scenes filled")
    for s, (t, r, c, d, m) in res.items():
        print(f"  {s:18s} {t:.2f}/{r:.2f}  SR5 {c:.1f}  SR2 {d:.1f}" + (f"  ({m} skipped)" if m else ""))


if __name__ == "__main__":
    main()
